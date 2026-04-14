import cv2
import numpy as np

def segment_lines(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image not found!")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 🔥 Better binarization
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5
    )

    # 🔥 Horizontal projection
    projection = np.sum(thresh, axis=1)

    # Normalize
    projection = projection / np.max(projection)

    threshold=np.mean(projection)*0.4

    lines = []
    start = None

    for i in range(len(projection)):

        if projection[i] > threshold:   # 🔥 TEXT REGION
            if start is None:
                start = i
        else:                    # 🔥 GAP
            if start is not None:
                end = i

                # Filter small noise
                if end - start > 7:
                    line_img = image[start:end, :]
                    lines.append(line_img)

                start = None

    # Catch last line
    if start is not None:
        line_img = image[start:len(projection), :]
        lines.append(line_img)

    return lines