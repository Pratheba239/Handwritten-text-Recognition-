import cv2
import numpy as np

def segment_words(line_image):

    gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

    # Strong binary (text = white)
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 🔥 Horizontal projection (column-wise sum)
    projection = np.sum(thresh, axis=0)

    # Normalize
    projection = projection / np.max(projection)

    words = []
    start = None

    # 🔥 VERY IMPORTANT threshold tuning
    threshold = 0.15   # lower = more sensitive

    for i in range(len(projection)):

        if projection[i] > threshold:   # TEXT
            if start is None:
                start = i

        else:  # GAP
            if start is not None:
                end = i

                # 🔥 filter small noise
                if end - start > 20:
                    word = line_image[:, start:end]
                    words.append(word)

                start = None

    # Last word
    if start is not None:
        word = line_image[:, start:]
        words.append(word)

    return words