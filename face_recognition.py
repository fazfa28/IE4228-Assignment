"""
Face Detection using Viola–Jones Haar Cascade (OpenCV)

Steps:
1. Capture live video from webcam.
2. Convert frame to grayscale.
3. Detect faces using pretrained Haar cascade.
4. Draw bounding boxes around detected faces.
5. Display result in real-time.
"""

import cv2

# Load pretrained Viola–Jones Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (required for Haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,     # image pyramid scaling
        minNeighbors=5,      # reduce false positives
        minSize=(60, 60)     # minimum face size
    )

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Viola-Jones Face Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()