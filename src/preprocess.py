import cv2
import numpy as np
import os
import random


# =========================
# LOAD IAM WORD LABELS (OLD)
# =========================
def load_labels(label_path):
    labels = {}

    with open(label_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                continue

            parts = line.strip().split()

            if len(parts) < 9:
                continue

            word_id = parts[0]
            text = " ".join(parts[8:])

            labels[word_id] = text

    return labels


# =========================
# LOAD IAM LINE LABELS (NEW)
# =========================
def load_line_labels(label_path):
    labels = {}

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            img, text = line.strip().split("\t")
            labels[img] = text

    return labels


# =========================
# PREPROCESS IMAGE (FINAL)
# =========================
def preprocess_image(image, img_height=32, img_width=512, augment=False):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 🔥 Augmentation ONLY during training
    if augment:
        gray = augment_image(gray)

    # 🔥 OTSU threshold (clean text)
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # =========================
    # RESIZE WITH ASPECT RATIO
    # =========================
    h, w = thresh.shape

    if h == 0 or w == 0:
        return np.zeros((img_height, img_width), dtype=np.float32)

    scale = img_height / h
    new_w = int(w * scale)

    # Clamp width (VERY IMPORTANT)
    new_w = max(1, min(new_w, img_width))

    resized = cv2.resize(thresh, (new_w, img_height))

    # =========================
    # CENTER PADDING (BETTER)
    # =========================
    padded = np.zeros((img_height, img_width), dtype=np.uint8)

    start_x = (img_width - new_w) // 2
    padded[:, start_x:start_x + new_w] = resized

    # =========================
    # NORMALIZE
    # =========================
    padded = padded.astype("float32") / 255.0

    return padded


# =========================
# CHAR MAP (CTC SAFE)
# =========================
def get_char_map():
    import string

    chars = string.ascii_letters + string.digits + " .,?-\'"

    char_to_num = {char: idx for idx, char in enumerate(chars)}
    num_to_char = {idx: char for idx, char in enumerate(chars)}

    return char_to_num, num_to_char


# =========================
# CLEAN TEXT (IMPORTANT)
# =========================
def clean_text(text):
    import re

    # Keep letters, numbers, spaces
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================
# ENCODE TEXT
# =========================
def encode_text(text, char_to_num):
    return [char_to_num[c] for c in text if c in char_to_num]


# =========================
# DATASET BUILDER (LINES)
# =========================
def build_line_dataset(labels, image_base_path, char_to_num, max_samples=50000):
    X = []
    Y = []

    for i, img_name in enumerate(labels):

        image_path = os.path.join(image_base_path, img_name)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        # 🔥 SAME preprocessing as inference
        processed = preprocess_image(image, augment=True)

        # 🔥 CLEAN TEXT (CRITICAL FIX)
        text = clean_text(labels[img_name])

        encoded = encode_text(text, char_to_num)

        if len(encoded) == 0:
            continue

        X.append(processed)
        Y.append(encoded)

        if i >= max_samples:
            break

    return np.array(X), Y


# =========================
# AUGMENTATION (SAFE)
# =========================
def augment_image(image):

    h, w = image.shape

    # Rotation
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h), borderValue=255)

    # Noise
    noise = np.random.randint(0, 20, image.shape, dtype='uint8')
    image = cv2.add(image, noise)

    # Blur
    if random.random() < 0.2:
        image = cv2.GaussianBlur(image, (3,3), 0)

    return image