import cv2
import numpy as np
import string
import os

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


def preprocess_image(image, img_height=32, img_width=128):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape
    new_w = int(w * (img_height / h))

    image = cv2.resize(image, (new_w, img_height))

    padded = np.ones((img_height, img_width)) * 255

    if new_w > img_width:
        image = cv2.resize(image, (img_width, img_height))
        padded = image
    else:
        padded[:, :new_w] = image

    padded = padded.astype('float32') / 255.0

    return padded


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