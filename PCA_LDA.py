"""
Live Face Recognition (Non-Deep Learning)
- Face detection: Viola-Jones Haar Cascade (OpenCV)
- Recognition: PCA (Eigenfaces) + Nearest Neighbor
- Unknown decision: distance threshold
- Smoothing: majority vote over last N frames

Requirements:
  pip install opencv-python numpy

Folder structure example:
dataset/
  train/
    alice/
      001.jpg ...
    bob/
      001.jpg ...
  test/   (optional)

Run:
  python live_pca_face_recog.py
"""

import os
import cv2
import time
import json
import numpy as np
from collections import deque, Counter
from dataclasses import dataclass

# -----------------------------
# Config
# -----------------------------
DATASET_DIR = "dataset/train"     # your gallery (training) images
MODEL_DIR   = "model"             # where to save PCA model
FACE_SIZE   = (120, 120)          # (W, H) >= (90,90)
MIN_FACE    = (80, 80)            # min face size for detection in frame
CAM_INDEX   = 0

# Haar cascades (OpenCV ships these)
HAAR_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_EYES = cv2.data.haarcascades + "haarcascade_eye.xml"

# PCA settings
PCA_COMPONENTS = 80               # tune: 30~150 depending on data
UNKNOWN_THR    = 3500.0           # tune by validation (distance threshold)

# Smoothing
VOTE_WINDOW = 7                   # number of recent predictions to vote over

# Pre-processing toggles
USE_CLAHE    = True
USE_ALIGN    = False              # eye alignment is optional (can be fragile)

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_images(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    return sorted(files)

def to_gray(img_bgr):
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def apply_clahe(gray):
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def detect_eyes(gray, eye_cascade, roi):
    x, y, w, h = roi
    face = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    # Convert to full-image coords
    eyes_full = [(x+ex, y+ey, ew, eh) for (ex, ey, ew, eh) in eyes]
    return eyes_full

def align_by_eyes(gray, face_roi, eyes_full):
    """
    Optional: align face using two eyes.
    This is a lightweight approach; for robustness, you'd need better landmarking.
    """
    x, y, w, h = face_roi
    if len(eyes_full) < 2:
        return gray[y:y+h, x:x+w]

    # pick two largest eyes (rough heuristic)
    eyes_sorted = sorted(eyes_full, key=lambda e: e[2]*e[3], reverse=True)[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes_sorted
    c1 = (x1 + w1//2, y1 + h1//2)
    c2 = (x2 + w2//2, y2 + h2//2)

    # Ensure left-right
    if c1[0] > c2[0]:
        c1, c2 = c2, c1

    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    if dx == 0:
        return gray[y:y+h, x:x+w]

    angle = np.degrees(np.arctan2(dy, dx))
    center = (x + w//2, y + h//2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR)

    return rotated[y:y+h, x:x+w]

def preprocess_face(frame_bgr, face_roi, eye_cascade=None):
    """
    Returns a normalized face image (gray, resized) and the processed ROI crop.
    """
    gray = to_gray(frame_bgr)
    x, y, w, h = face_roi

    # optional alignment
    if USE_ALIGN and eye_cascade is not None:
        eyes_full = detect_eyes(gray, eye_cascade, face_roi)
        crop = align_by_eyes(gray, face_roi, eyes_full)
    else:
        crop = gray[y:y+h, x:x+w]

    if crop.size == 0:
        return None

    # resize
    crop = cv2.resize(crop, FACE_SIZE, interpolation=cv2.INTER_AREA)

    # illumination normalization
    if USE_CLAHE:
        crop = apply_clahe(crop)
    else:
        crop = cv2.equalizeHist(crop)

    return crop

# -----------------------------
# PCA (Eigenfaces) Model
# -----------------------------
@dataclass
class PCAModel:
    mean: np.ndarray           # (D,)
    W: np.ndarray              # (D, k) eigenvectors
    labels: list               # list of label names (strings) for each training vector
    feats: np.ndarray          # (N, k) projected features for training set

def pca_train(X: np.ndarray, k: int):
    """
    X: (N, D) float32, each row a flattened face
    returns mean (D,), W (D,k)
    """
    # Mean center
    mean = X.mean(axis=0)
    Xc = X - mean

    # Compute SVD (stable)
    # Xc = U S Vt ; principal directions = Vt.T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:k].T  # (D,k)
    return mean, W

def project(X: np.ndarray, mean: np.ndarray, W: np.ndarray):
    return (X - mean) @ W  # (N,k)

def load_gallery(dataset_dir: str):
    """
    Reads dataset_dir/<person_name>/*.jpg, returns faces and labels.
    """
    persons = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    if not persons:
        raise RuntimeError(f"No person folders found in {dataset_dir}")

    face_cascade = cv2.CascadeClassifier(HAAR_FACE)
    eye_cascade  = cv2.CascadeClassifier(HAAR_EYES) if USE_ALIGN else None

    X_list = []
    y_list = []

    for person in persons:
        folder = os.path.join(dataset_dir, person)
        files = list_images(folder)
        if len(files) < 10:
            print(f"[WARN] {person} has only {len(files)} images (<10).")

        for fp in files:
            img = cv2.imread(fp)
            if img is None:
                continue

            # Detect face in the training image (so users can store uncropped images too)
            gray = to_gray(img)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            if len(faces) == 0:
                # If your dataset is already cropped, just use full image:
                # roi = (0,0,gray.shape[1], gray.shape[0])
                continue

            # Use the largest detected face
            faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            roi = faces[0]

            face = preprocess_face(img, roi, eye_cascade=eye_cascade)
            if face is None:
                continue

            X_list.append(face.flatten().astype(np.float32))
            y_list.append(person)

    if len(X_list) == 0:
        raise RuntimeError("No training faces loaded. Check dataset and detection.")

    X = np.stack(X_list, axis=0)  # (N,D)
    return X, y_list

def train_model():
    ensure_dir(MODEL_DIR)
    X, labels = load_gallery(DATASET_DIR)
    N, D = X.shape

    k = min(PCA_COMPONENTS, N - 1, D)  # safe
    mean, W = pca_train(X, k)
    feats = project(X, mean, W)

    model = PCAModel(mean=mean, W=W, labels=labels, feats=feats)

    # Save
    np.save(os.path.join(MODEL_DIR, "mean.npy"), model.mean)
    np.save(os.path.join(MODEL_DIR, "W.npy"), model.W)
    np.save(os.path.join(MODEL_DIR, "feats.npy"), model.feats)
    with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
        json.dump(model.labels, f)

    print(f"[OK] Trained PCA model: N={N}, D={D}, k={k}")
    return model

def load_model():
    mean = np.load(os.path.join(MODEL_DIR, "mean.npy"))
    W    = np.load(os.path.join(MODEL_DIR, "W.npy"))
    feats= np.load(os.path.join(MODEL_DIR, "feats.npy"))
    with open(os.path.join(MODEL_DIR, "labels.json"), "r") as f:
        labels = json.load(f)
    return PCAModel(mean=mean, W=W, labels=labels, feats=feats)

def predict_identity(face_img_gray, model: PCAModel):
    """
    face_img_gray: (H,W) uint8 normalized face
    Returns (name, distance, confidence-ish)
    """
    x = face_img_gray.flatten().astype(np.float32)[None, :]  # (1,D)
    z = project(x, model.mean, model.W)[0]                   # (k,)

    # Nearest neighbor
    diffs = model.feats - z[None, :]
    dists = np.sum(diffs * diffs, axis=1)                    # squared L2
    idx = int(np.argmin(dists))
    best_dist = float(dists[idx])
    best_name = model.labels[idx]

    if best_dist > UNKNOWN_THR:
        return "Unknown", best_dist

    return best_name, best_dist

# -----------------------------
# Live Demo
# -----------------------------
def main():
    # Train if no model exists
    if not os.path.exists(os.path.join(MODEL_DIR, "W.npy")):
        model = train_model()
    else:
        model = load_model()

    face_cascade = cv2.CascadeClassifier(HAAR_FACE)
    eye_cascade  = cv2.CascadeClassifier(HAAR_EYES) if USE_ALIGN else None

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    pred_history = deque(maxlen=VOTE_WINDOW)
    last_time = time.time()
    fps = 0.0

    print("[INFO] Press 'q' to quit, 'r' to retrain.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        gray = to_gray(frame)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE
        )

        # choose largest face for simplicity
        if len(faces) > 0:
            faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            (x, y, w, h) = faces[0]

            face_norm = preprocess_face(frame, (x, y, w, h), eye_cascade=eye_cascade)
            if face_norm is not None:
                name, dist = predict_identity(face_norm, model)
                pred_history.append(name)

                # majority vote smoothing
                vote = Counter(pred_history).most_common(1)[0][0]

                # draw
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{vote}  d={dist:.0f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2, cv2.LINE_AA)
        else:
            pred_history.clear()

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "q:quit  r:retrain", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Live PCA Face Recognition (Non-DL)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('r'):
            print("[INFO] Retraining...")
            model = train_model()
            pred_history.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()