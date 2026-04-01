import cv2
import numpy as np
import joblib
from collections import deque

# ── CONFIG ──────────────────────────────────────────────
IMG_SIZE       = (90, 90)
CONFIDENCE_THR = 0.5
SMOOTH_WINDOW  = 10
# ────────────────────────────────────────────────────────

# Load models
pca = joblib.load("pca_model.pkl")
lda = joblib.load("lda_model.pkl")
clf = joblib.load("clf_model.pkl")

# Guard
assert pca is not None, "pca_model.pkl failed to load"
assert lda is not None, "lda_model.pkl failed to load"
assert clf is not None, "clf_model.pkl failed to load"
print("✅ All models loaded.")
# Detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

buffers = {}

# ── Preprocessing (mirrors your script exactly) ──────────
def normalize_lighting(gray):
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img  = clahe.apply(gray)
    normalized = cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)
    return normalized

def align_face(gray, x, y, w, h):
    """Same align_face() from your preprocessing script."""
    face_roi = gray[y : y + h//2, x : x + w]
    eyes = eye_cascade.detectMultiScale(
        face_roi, scaleFactor=1.1, minNeighbors=8,
        minSize=(20, 20), maxSize=(w//3, h//3)
    )
    if len(eyes) < 2:
        return gray, x, y, w, h

    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])

    (ex1, ey1, ew1, eh1) = eyes[0]
    (ex2, ey2, ew2, eh2) = eyes[1]

    left_eye_center  = (int(x + ex1 + ew1 // 2), int(y + ey1 + eh1 // 2))
    right_eye_center = (int(x + ex2 + ew2 // 2), int(y + ey2 + eh2 // 2))

    if abs(left_eye_center[1] - right_eye_center[1]) > h * 0.15:
        return gray, x, y, w, h
    if abs(left_eye_center[0] - right_eye_center[0]) < w * 0.2:
        return gray, x, y, w, h

    dx    = right_eye_center[0] - left_eye_center[0]
    dy    = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))

    if abs(angle) > 30:
        return gray, x, y, w, h

    eye_mid = (
        int((left_eye_center[0] + right_eye_center[0]) // 2),
        int((left_eye_center[1] + right_eye_center[1]) // 2)
    )
    M       = cv2.getRotationMatrix2D(eye_mid, angle, 1.0)
    aligned = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return aligned, x, y, w, h

def preprocess_face(gray_crop, h_orig, w_orig, x, y, w, h):
    """Mirrors steps 4-6 of your preprocessing script."""
    # Step 4: Olivetti-style crop margins
    forehead_extra = int(h * 0.15)
    chin_extra     = int(h * 0.05)

    x1 = max(0, x)
    y1 = max(0, y - forehead_extra)
    x2 = min(w_orig, x + w)
    y2 = min(h_orig, y + h + chin_extra)

    crop = gray_crop[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Step 5: Resize
    face = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    # Step 6: CLAHE + normalize
    face = normalize_lighting(face)
    # load_dataset: float32 + flatten
    face = face.astype(np.float32) / 255.0
    return face.flatten().reshape(1, -1)

def predict_face(face_vec):
    pca_vec    = pca.transform(face_vec)
    lda_vec    = lda.transform(pca_vec)
    proba      = clf.predict_proba(lda_vec)[0]
    confidence = proba.max()
    label      = clf.classes_[proba.argmax()]
    return label, confidence

def smooth_prediction(face_id, label):
    if face_id not in buffers:
        buffers[face_id] = deque(maxlen=SMOOTH_WINDOW)
    buffers[face_id].append(label)
    return max(set(buffers[face_id]), key=buffers[face_id].count)

# ── Main Loop ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_img, w_img = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (mirrors your multi-scale loop)
    faces = []
    for scale in [1.2]:
        detected = face_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=7, minSize=(40, 40)
        )
        if len(detected) > 0:
            faces = detected
            break

    # Take largest face only (mirrors your script)
    if len(faces) > 0:
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)

    for i, (x, y, w, h) in enumerate(faces):
        aligned_gray, x, y, w, h = align_face(gray, x, y, w, h)
        vec = preprocess_face(aligned_gray, h_img, w_img, x, y, w, h)

        if vec is None:
            continue

        label, confidence = predict_face(vec)

        if confidence < CONFIDENCE_THR:
            label = "Unknown"

        label = smooth_prediction(face_id=i, label=label)

        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()