import cv2
import numpy as np
import string
import os
import random

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


def preprocess_image(image, img_height=32, img_width=128, augment=False):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 🔥 LIGHT normalization (DON'T over-process)
    gray = cv2.resize(gray, (img_width, img_height))

    # Normalize SAME as training
    gray = gray.astype("float32") / 255.0

    return gray


def get_char_map():
    import string

    chars = string.ascii_letters + string.digits

    # Start from 0 (NOT 1)
    char_to_num = {char: idx for idx, char in enumerate(chars)}
    num_to_char = {idx: char for idx, char in enumerate(chars)}

    return char_to_num, num_to_char


def encode_text(text, char_to_num):
    return [char_to_num[c] for c in text if c in char_to_num]


# 🔥 NEW: DATASET BUILDER

def build_dataset(labels, image_base_path, char_to_num, max_samples=1000):
    X = []
    Y = []

    count = 0

    for word_id in labels:

        parts = word_id.split('-')
        folder1 = parts[0]
        folder2 = parts[0] + "-" + parts[1]

        image_path = os.path.join(
            image_base_path,
            folder1,
            folder2,
            word_id + ".png"
        )

        if os.path.exists(image_path):
            image = cv2.imread(image_path)

            if image is None:
                continue

            processed = preprocess_image(image)

            text = labels[word_id]
            encoded = encode_text(text, char_to_num)

            if len(encoded) == 0:
                continue

            X.append(processed)
            Y.append(encoded)

            count += 1

            if count >= max_samples:
                break

    return np.array(X), Y


def augment_image(image):
    # Random rotation
    angle = random.uniform(-5, 5)
    h, w = image.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h), borderValue=255)

    # Random noise
    noise = np.random.randint(0, 30, image.shape, dtype='uint8')
    image = cv2.add(image, noise)

    # Random blur
    if random.random() < 0.3:
        image = cv2.GaussianBlur(image, (3,3), 0)

    return image