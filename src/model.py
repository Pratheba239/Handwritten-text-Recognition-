from tensorflow.keras import layers, models

def build_model(input_shape=(32, 128, 1), num_classes=63):
    inputs = layers.Input(shape=input_shape)

    # CNN part
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)

    # Prepare for RNN
    new_shape = ((input_shape[1] // 4), (input_shape[0] // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2))(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model