from segment import segment_lines
import matplotlib.pyplot as plt

lines = segment_lines(r"D:\HTR_PROJECT_B\test_data.jpg")

print("Number of lines detected:", len(lines))

# Show ALL lines together
fig, axes = plt.subplots(len(lines), 1, figsize=(8, 3 * len(lines)))

if len(lines) == 1:
    axes = [axes]

if len(lines)==0:
    print("NO lines detected!")
    exit()

for i, line in enumerate(lines):
    axes[i].imshow(line)
    axes[i].set_title(f"Line {i}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()