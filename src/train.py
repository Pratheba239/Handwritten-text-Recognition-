import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from preprocess import load_labels, get_char_map, build_dataset


# 🔥 BUILD MODEL (this is safe to import)
def build_base_model(input_shape=(32, 128, 1), num_classes=63):
    inputs = layers.Input(shape=input_shape)

    # CNN
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,1))(x)
    x = layers.Dropout(0.25)(x)

    # 🔥 Reshape
    x = layers.Reshape((32, 4 * 128))(x)

    # RNN
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2)
    )(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


# 🔥 TRAINING CODE (only runs when file is executed directly)
if __name__ == "__main__":

    # Paths
    label_path = "../data/words_new.txt"
    image_base_path = "../data/iam_words/words/"

    # Load data
    labels = load_labels(label_path)
    char_to_num, num_to_char = get_char_map()

    X, Y = build_dataset(labels, image_base_path, char_to_num, max_samples=50000)

    # Add channel dimension
    X = np.expand_dims(X, axis=-1)

    # Pad labels
    max_label_length = max([len(y) for y in Y])

    Y_padded = np.ones((len(Y), max_label_length), dtype=np.int32) * -1

    for i, y in enumerate(Y):
        Y_padded[i, :len(y)] = y

    # Lengths
    input_length = np.ones((len(X), 1), dtype=np.int32) * (128 // 4)
    label_length = np.array([[len(y)] for y in Y], dtype=np.int32)

    # Build model
    base_model = build_base_model(num_classes=len(char_to_num) + 1)

    # CTC setup
    labels_input = layers.Input(name='labels', shape=(None,))
    input_length_input = layers.Input(name='input_length', shape=(1,))
    label_length_input = layers.Input(name='label_length', shape=(1,))

    loss_out = layers.Lambda(
        lambda args: tf.keras.backend.ctc_batch_cost(*args)
    )([labels_input, base_model.output, input_length_input, label_length_input])

    model = models.Model(
        inputs=[base_model.input, labels_input, input_length_input, label_length_input],
        outputs=loss_out
    )

    model.compile(
        optimizer='adam',
        loss=lambda y_true, y_pred: y_pred
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    # Train
    model.fit(
        [X, Y_padded, input_length, label_length],
        np.zeros(len(X)),
        epochs=60,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stop, lr_scheduler]
    )

    # Save
    base_model.save("../models/htr_model.h5")

    print("Model saved successfully!")