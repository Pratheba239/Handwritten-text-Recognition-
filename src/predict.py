import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from preprocess import load_labels, get_char_map, preprocess_image


# =========================
# PATHS
# =========================
label_path = "../data/words_new.txt"
image_base_path = "../data/iam_words/words/"
model_path = "../models/htr_model.h5"


# =========================
# LOAD DATA
# =========================
labels = load_labels(label_path)
char_to_num, num_to_char = get_char_map()


# =========================
# LOAD TRAINED MODEL
# =========================
print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded successfully!")


# =========================
# CTC DECODER (CORRECT WAY)
# =========================
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )[0][0]

    decoded = decoded.numpy()

    text = ""
    for char in decoded[0]:
        if char != -1:
            text += num_to_char.get(char, '')

    return text


# =========================
# OPTIONAL TEXT CLEANING
# =========================
def clean_text(text):
    import re
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text


# =========================
# TEST ON ONE SAMPLE
# =========================
for word_id in labels:

    parts = word_id.split('-')
    folder1 = parts[0]
    folder2 = parts[0] + "-" + parts[1]

    image_path = f"{image_base_path}/{folder1}/{folder2}/{word_id}.png"

    image = cv2.imread(image_path)

    if image is None:
        continue

    # ✅ Preprocess (NO augmentation during inference)
    processed = preprocess_image(image)

    processed = np.expand_dims(processed, axis=-1)
    processed = np.expand_dims(processed, axis=0)

    # =========================
    # PREDICT
    # =========================
    preds = model.predict(processed)

    decoded_text = decode_prediction(preds)

    # Optional cleaning
    cleaned_text = clean_text(decoded_text)

    # =========================
    # OUTPUT
    # =========================
    print("Actual   :", labels[word_id])
    print("Predicted:", decoded_text)
    print("Cleaned  :", cleaned_text)

    plt.imshow(processed[0].squeeze(), cmap='gray')
    plt.title(f"Pred: {cleaned_text}")
    plt.axis("off")
    plt.show()

    break