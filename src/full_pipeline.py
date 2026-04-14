import cv2
import numpy as np
import tensorflow as tf

from segment import segment_lines
from word_segment import segment_words
from preprocess import preprocess_image, get_char_map
from train import build_base_model


# =========================
# LOAD CHAR MAP
# =========================
char_to_num, num_to_char = get_char_map()


# =========================
# LOAD MODEL
# =========================
model = build_base_model(num_classes=len(char_to_num) + 1)
model.load_weights("../models/htr_model.h5")

print("✅ Model loaded successfully!")


# =========================
# CTC DECODER
# =========================
def decode_prediction(pred):

    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )[0][0]

    decoded = tf.keras.backend.get_value(decoded)

    text = ""

    for char in decoded[0]:
        if char != -1:
            text += num_to_char.get(char, '')

    return text


# =========================
# MAIN PIPELINE
# =========================
def process_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Image not found!")
        return ""

    lines = segment_lines(image_path)
    print(f"🟢 Lines detected: {len(lines)}")

    final_text = []

    for i, line in enumerate(lines):

        words = segment_words(line)
        print(f"   ➤ Line {i}: {len(words)} words detected")

        line_text = []

        for j, word in enumerate(words):

            processed = preprocess_image(word)
            processed = np.expand_dims(processed, axis=-1)
            processed = np.expand_dims(processed, axis=0)

            preds = model.predict(processed, verbose=0)

            text = decode_prediction(preds)

            print(f"      Word {j} → '{text}'")

            if text.strip() != "":
                line_text.append(text)

        final_text.append(" ".join(line_text))

    return "\n".join(final_text)


# =========================
# RUN
# =========================
image_path = r"D:/HTR_PROJECT_B/test_data.jpg"

output = process_image(image_path)

print("\n===== FINAL OUTPUT =====\n")
print(output)