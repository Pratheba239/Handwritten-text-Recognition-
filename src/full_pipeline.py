import cv2
import numpy as np
import tensorflow as tf

from segment import segment_lines
from preprocess import preprocess_image, get_char_map
from train import build_base_model


# =========================
# LOAD CHAR MAP
# =========================
char_to_num, num_to_char = get_char_map()


# =========================
# LOAD MODEL (CORRECT WAY)
# =========================
model = build_base_model(
    input_shape=(32, 512, 1),
    num_classes=len(char_to_num) + 1
)

model.load_weights("../models/htr_model.keras")

print("✅ Model loaded successfully!")


# =========================
# BEAM SEARCH DECODER
# =========================
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=False,
        beam_width=10
    )[0][0]

    decoded = tf.keras.backend.get_value(decoded)

    text = ""
    for char in decoded[0]:
        if char != -1:
            text += num_to_char.get(char, '')

    return text


# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    import re
    return re.sub(r'[^a-zA-Z0-9 ]', '', text)


# =========================
# POST PROCESS
# =========================
def post_process(text):
    corrections = {
        "meting": "meeting",
        "tomorow": "tomorrow",
        "teo": "to"
    }

    words = text.split()
    return " ".join([corrections.get(w, w) for w in words])


# =========================
# MAIN PIPELINE (FIXED)
# =========================
def process_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Image not found!")
        return ""

    # 🔥 STEP 1: Segment into lines
    lines = segment_lines(image_path)
    print(f"🟢 Lines detected: {len(lines)}")

    final_text = []

    # 🔥 STEP 2: Process each line
    for i, line in enumerate(lines):

        processed = preprocess_image(line)
        processed = np.expand_dims(processed, axis=-1)
        processed = np.expand_dims(processed, axis=0)

        preds = model.predict(processed, verbose=0)

        text = decode_prediction(preds)
        text = clean_text(text)
        text = post_process(text)

        print(f"   ➤ Line {i}: '{text}'")

        final_text.append(text)

    return "\n".join(final_text)


# =========================
# RUN
# =========================
image_path = r"D:\HTR_PROJECT_B\test_data_4.webp"

output = process_image(image_path)

print("\n===== FINAL OUTPUT =====\n")
print(output)