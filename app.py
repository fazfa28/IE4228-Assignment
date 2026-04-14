import sys
import subprocess
#subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
import joblib

# Paths and model loading
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PCA_MODEL_PATH = os.path.join(BASE_DIR, "pca_model.pkl")
LDA_MODEL_PATH = os.path.join(BASE_DIR, "lda_model.pkl")
CLF_MODEL_PATH = os.path.join(BASE_DIR, "clf_model.pkl")

try:
    pca = joblib.load(PCA_MODEL_PATH)
    lda = joblib.load(LDA_MODEL_PATH)
    clf = joblib.load(CLF_MODEL_PATH)
    print("Loaded PCA, LDA, and KNN models from disk.")
except Exception as exc:
    pca = None
    lda = None
    clf = None
    print(f"Model load failed: {exc}")

# loading pretrained Haar Cascades
cascPathface = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_alt2.xml")
cascPatheyes = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_eye_tree_eyeglasses.xml")
faceCascade = cv2.CascadeClassifier(cascPathface)  # detect faces
eyeCascade = cv2.CascadeClassifier(cascPatheyes)  # detect eyes

app = Flask(__name__)

# Global variables to store UI settings
current_method = "PCA+LDA"
confidence_threshold = 0.75
pca_components = 50
camera = cv2.VideoCapture(0)

IMG_SIZE = (90, 90)


def normalize_lighting(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    normalized = cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def align_face(gray, x, y, w, h):
    face_roi = gray[y : y + h // 2, x : x + w]
    eyes = eyeCascade.detectMultiScale(
        face_roi,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(20, 20),
        maxSize=(w // 3, h // 3),
    )
    if len(eyes) < 2:
        return gray, x, y, w, h

    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])

    (ex1, ey1, ew1, eh1) = eyes[0]
    (ex2, ey2, ew2, eh2) = eyes[1]
    left_eye_center = (int(x + ex1 + ew1 // 2), int(y + ey1 + eh1 // 2))
    right_eye_center = (int(x + ex2 + ew2 // 2), int(y + ey2 + eh2 // 2))

    if abs(left_eye_center[1] - right_eye_center[1]) > h * 0.15:
        return gray, x, y, w, h
    if abs(left_eye_center[0] - right_eye_center[0]) < w * 0.2:
        return gray, x, y, w, h

    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))

    if abs(angle) > 30:
        return gray, x, y, w, h

    eye_mid = (
        int((left_eye_center[0] + right_eye_center[0]) // 2),
        int((left_eye_center[1] + right_eye_center[1]) // 2),
    )
    M = cv2.getRotationMatrix2D(eye_mid, angle, 1.0)
    aligned = cv2.warpAffine(
        gray,
        M,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, x, y, w, h


def preprocess_face(gray, x, y, w, h):
    forehead_extra = int(h * 0.15)
    chin_extra = int(h * 0.05)
    x1 = max(0, x)
    y1 = max(0, y - forehead_extra)
    x2 = min(gray.shape[1], x + w)
    y2 = min(gray.shape[0], y + h + chin_extra)

    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    face = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    face = normalize_lighting(face)
    face = face.astype(np.float32) / 255.0
    return face.flatten().reshape(1, -1)


def recognize_face(face_vec):
    if pca is None or clf is None:
        return "Unknown", 0.0

    pca_vec = pca.transform(face_vec)
    if current_method == "PCA":
        try:
            proba = clf.predict_proba(pca_vec)[0]
        except Exception:
            proba = np.array([1.0])
            return clf.predict(pca_vec)[0], 1.0
    elif current_method == "PCA+LDA" and lda is not None:
        lda_vec = lda.transform(pca_vec)
        proba = clf.predict_proba(lda_vec)[0]
    else:
        # fallback: PCA+LDA semantics if LDA is unavailable
        proba = clf.predict_proba(pca_vec)[0]

    label = clf.classes_[proba.argmax()]
    return label, float(proba.max())

def generate_frames():
    """Captures camera feed and processes faces"""

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (x, y, w, h) in faces:
            aligned_gray, x, y, w, h = align_face(gray, x, y, w, h)
            face_vec = preprocess_face(aligned_gray, x, y, w, h)

            label = "Unknown"
            confidence = 0.0
            if face_vec is not None:
                label, confidence = recognize_face(face_vec)
                if confidence < confidence_threshold:
                    label = "Unknown"

            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Serve your uploaded HTML file
    return render_template('face_recognition_gui.html')

@app.route('/video_feed')
def video_feed():
    # Stream the camera feed to the HTML <img> tag
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_settings', methods=['POST'])
def update_settings():
    # Receive parameters from HTML inputs
    global current_method, confidence_threshold, pca_components
    data = request.json
    current_method = data.get('method', 'PCA')
    confidence_threshold = float(data.get('confidence', 0.75))
    pca_components = int(data.get('pca_k', 50))
    return jsonify({"status": "Settings Updated"})

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    # Logic to trigger the "Start" button event
    return jsonify({"status": "Recognition Started"})

if __name__ == '__main__':
    app.run(debug=True)