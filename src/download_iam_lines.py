from datasets import load_dataset
import os

# Load dataset
dataset = load_dataset("Teklia/IAM-line")

# Create folders
os.makedirs("../data/iam_lines/images", exist_ok=True)

label_file = "../data/iam_lines/labels.txt"

print("Downloading IAM line dataset...")

# Choose split
split = dataset["train"]

for i, sample in enumerate(split):

    image = sample["image"]
    text = sample["text"]

    img_name = f"img_{i}.png"
    img_path = os.path.join("../data/iam_lines/images", img_name)

    # Save image
    image.save(img_path)

    # Save label
    with open(label_file, "a", encoding="utf-8") as f:
        f.write(f"{img_name}\t{text}\n")

    if i % 100 == 0:
        print(f"Downloaded {i} images")

print("✅ Download complete!")