import os
import cv2
import face_recognition
from sklearn import svm

# Path to dataset folder
dataset_path = r"C:\Users\Prathyush\Documents\SEM 6\FoML\Lab\Ex6\dataset"

encodings = []
names = []

# Read all person folders
for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    # Read all images inside the person's folder
    for image_name in os.listdir(person_path):

        # Only use image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(person_path, image_name)

        try:
            image = face_recognition.load_image_file(image_path)

            # Detect face locations
            face_locations = face_recognition.face_locations(image)

            # Use only images with exactly one face
            if len(face_locations) != 1:
                continue

            # Extract face encoding
            encoding = face_recognition.face_encodings(image, face_locations)[0]

            encodings.append(encoding)
            names.append(person)

        except:
            continue

# Ensure at least 2 people exist
if len(set(names)) < 2:
    print("Need at least 2 different people in dataset.")
    exit()

# Train SVM classifier
clf = svm.SVC(gamma='scale', probability=True)
clf.fit(encodings, names)

print("Training Complete!")

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Convert frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        name = "Unknown"

        try:
            prediction = clf.predict([face_encoding])
            probabilities = clf.predict_proba([face_encoding])

            confidence = max(probabilities[0])

            # Accept only if confidence is high
            if confidence > 0.7:
                name = prediction[0]

        except:
            pass

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display name
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition using SVM", frame)

    # Exit when ESC is pressed
    if cv2.waitKey(1) == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
