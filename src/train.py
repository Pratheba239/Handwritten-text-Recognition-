import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from preprocess import (
    load_line_labels,
    get_char_map,
    build_line_dataset
)

# =========================
# 🔥 BUILD MODEL
# =========================
def build_base_model(input_shape=(32, 512, 1), num_classes=63):

    inputs = layers.Input(shape=input_shape)

    # CNN
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,1))(x)

    # 🔥 Convert to sequence
    x = layers.Permute((2,1,3))(x)
    x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)

    # RNN
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.3)
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.3)
    )(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)


# =========================
# 🔥 TRAINING
# =========================
if __name__ == "__main__":

    label_path = "../data/iam_lines/labels.txt"
    image_base_path = "../data/iam_lines/images"

    char_to_num, num_to_char = get_char_map()

    labels = load_line_labels(label_path)

    X, Y = build_line_dataset(
        labels,
        image_base_path,
        char_to_num,
        max_samples=50000
    )

    if len(X) == 0:
        raise ValueError("❌ Dataset empty")

    X = np.expand_dims(X, axis=-1)

    print(f"✅ Dataset loaded: {len(X)} samples")

    # =========================
    # 🔥 FIX 3 — TIME STEPS
    # =========================
    time_steps = X.shape[2] // 4   # pooling effect

    # =========================
    # 🔥 FIX 5 — FILTER LONG LABELS
    # =========================
    MAX_LABEL_LENGTH = time_steps - 2

    filtered_X = []
    filtered_Y = []

    for x, y in zip(X, Y):
        if len(y) <= MAX_LABEL_LENGTH:
            filtered_X.append(x)
            filtered_Y.append(y)

    X = np.array(filtered_X)
    Y = filtered_Y

    print(f"✅ After filtering: {len(X)} samples")

    # =========================
    # PAD LABELS
    # =========================
    max_label_length = max([len(y) for y in Y])

    Y_padded = np.ones((len(Y), max_label_length), dtype=np.int32) * -1

    for i, y in enumerate(Y):
        Y_padded[i, :len(y)] = y

    # =========================
    # LENGTHS
    # =========================
    input_length = np.ones((len(X), 1), dtype=np.int32) * time_steps
    label_length = np.array([[len(y)] for y in Y], dtype=np.int32)

    # =========================
    # BUILD MODEL
    # =========================
    base_model = build_base_model(
        input_shape=(32, 512, 1),
        num_classes=len(char_to_num) + 1
    )

    # =========================
    # 🔥 FIX 4 — CTC SAFE
    # =========================
    labels_input = layers.Input(name='labels', shape=(None,))
    input_length_input = layers.Input(name='input_length', shape=(1,))
    label_length_input = layers.Input(name='label_length', shape=(1,))

    loss_out = layers.Lambda(
        lambda args: tf.keras.backend.ctc_batch_cost(*args),
        output_shape=(1,)
    )([labels_input, base_model.output, input_length_input, label_length_input])

    model = models.Model(
        inputs=[base_model.input, labels_input, input_length_input, label_length_input],
        outputs=loss_out
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: y_pred
    )

    # =========================
    # CALLBACKS
    # =========================
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    print("\n🚀 Training started...\n")

    model.fit(
        [X, Y_padded, input_length, label_length],
        np.zeros(len(X)),
        epochs=80,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop, lr_scheduler]
    )

    base_model.save("../models/htr_model.keras")

    print("✅ Model saved successfully!")