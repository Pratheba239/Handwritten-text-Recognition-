import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from preprocess import load_labels, get_char_map, preprocess_image


# Paths
label_path = "../data/words_new.txt"
image_base_path = "../data/iam_words/words/"
model_path = "../models/htr_model.h5"


# Load label + char map
labels = load_labels(label_path)
char_to_num, num_to_char = get_char_map()


# ✅ LOAD TRAINED MODEL (IMPORTANT FIX)
model = load_model(model_path, compile=False)


# ✅ PROPER CTC DECODING
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )[0][0]

    output_text = ""

    for res in results.numpy()[0]:
        if res != -1:
            output_text += num_to_char.get(res, "")

    return output_text


# Pick one sample
for word_id in labels:

    parts = word_id.split('-')
    folder1 = parts[0]
    folder2 = parts[0] + "-" + parts[1]

    image_path = f"{image_base_path}/{folder1}/{folder2}/{word_id}.png"

    image = cv2.imread(image_path)

    if image is None:
        continue

    processed = preprocess_image(image)
    processed = np.expand_dims(processed, axis=-1)
    processed = np.expand_dims(processed, axis=0)

    # Predict
    preds = model.predict(processed)

    decoded_text = decode_prediction(preds)

    print("Actual:", labels[word_id])
    print("Predicted:", decoded_text)

    plt.imshow(processed[0].squeeze(), cmap='gray')
    plt.title(f"Pred: {decoded_text}")
    plt.axis("off")
    plt.show()

    break