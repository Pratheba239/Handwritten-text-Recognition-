import numpy as np
from preprocess import load_labels, get_char_map, build_dataset
from model import build_model

# Paths
label_path = "../data/words_new.txt"
image_base_path = "../data/iam_words/words/"

# Load data
labels = load_labels(label_path)
char_to_num, num_to_char = get_char_map()

X, Y = build_dataset(labels, image_base_path, char_to_num, max_samples=500)

# Add channel dimension
X = np.expand_dims(X, axis=-1)

# Build model
num_classes = len(char_to_num) + 1
model = build_model(num_classes=num_classes)

# Show summary
model.summary()