import numpy as np
import tensorflow as tf
import cv2

from preprocess import load_line_labels, get_char_map, preprocess_image


# =========================
# 🔥 CER (Character Error Rate)
# =========================
def cer(true, pred):
    dp = np.zeros((len(true)+1, len(pred)+1))

    for i in range(len(true)+1):
        dp[i][0] = i
    for j in range(len(pred)+1):
        dp[0][j] = j

    for i in range(1, len(true)+1):
        for j in range(1, len(pred)+1):
            if true[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )

    return dp[len(true)][len(pred)] / max(len(true), 1)


# =========================
# 🔥 WER (Word Error Rate)
# =========================
def wer(true, pred):
    true_words = true.split()
    pred_words = pred.split()

    dp = np.zeros((len(true_words)+1, len(pred_words)+1))

    for i in range(len(true_words)+1):
        dp[i][0] = i
    for j in range(len(pred_words)+1):
        dp[0][j] = j

    for i in range(1, len(true_words)+1):
        for j in range(1, len(pred_words)+1):
            if true_words[i-1] == pred_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1]
                )

    return dp[len(true_words)][len(pred_words)] / max(len(true_words), 1)


# =========================
# PATHS
# =========================
label_path = "../data/iam_lines/labels.txt"
image_base_path = "../data/iam_lines/images"
model_path = "../models/htr_model.keras"


# =========================
# LOAD DATA
# =========================
labels = load_line_labels(label_path)
char_to_num, num_to_char = get_char_map()


# =========================
# LOAD MODEL
# =========================
print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)
print("✅ Model loaded successfully!")


# =========================
# 🔥 DECODER (IMPROVED)
# =========================
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    decoded = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=False,
        beam_width=10
    )[0][0]

    decoded = decoded.numpy()

    text = ""
    for char in decoded[0]:
        if char != -1:
            text += num_to_char.get(char, '')

    return text


# =========================
# 🔥 CLEAN TEXT (MATCH TRAINING)
# =========================
def clean_text(text):
    import re

    text = text.lower()  # 🔥 IMPORTANT
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================
# TEST LOOP
# =========================
total_cer = 0
total_wer = 0
total_samples = 0

MAX_SAMPLES = 50

print("\n===== TESTING STARTED =====\n")

for img_name in labels:

    image_path = f"{image_base_path}/{img_name}"

    image = cv2.imread(image_path)
    if image is None:
        continue

    processed = preprocess_image(image)
    processed = np.expand_dims(processed, axis=-1)
    processed = np.expand_dims(processed, axis=0)

    preds = model.predict(processed, verbose=0)

    predicted = decode_prediction(preds)
    predicted = clean_text(predicted)

    actual = clean_text(labels[img_name])

    # skip empty
    if len(actual) == 0:
        continue

    c = cer(actual, predicted)
    w = wer(actual, predicted)

    total_cer += c
    total_wer += w
    total_samples += 1

    print(f"Sample {total_samples}")
    print("Actual   :", actual)
    print("Predicted:", predicted)
    print(f"CER: {c:.3f} | WER: {w:.3f}")
    print("------")

    if total_samples >= MAX_SAMPLES:
        break


# =========================
# FINAL RESULTS
# =========================
if total_samples > 0:
    print("\n===== FINAL RESULTS =====")
    print(f"Avg CER: {total_cer / total_samples:.4f}")
    print(f"Avg WER: {total_wer / total_samples:.4f}")
else:
    print("❌ No valid samples evaluated")