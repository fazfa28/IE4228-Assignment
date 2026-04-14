# live_demo.py
# Live face recognition — OpenCV window only, no Tkinter/PyQt
# Controls:
#   Q        — quit
#   +  / =   — raise threshold (accept more → less rejection)
#   -        — lower threshold (stricter → more rejection)
#   R        — reset threshold to tuned value

import cv2
import pickle
import numpy as np

from preprocess_functions import (
    align_face,
    crop_face,
    resize_face,
    normalize_lighting,
    apply_ellipse_mask
)

# ── Define model classes (needed to unpickle scratch_model.pkl) ───
import numpy as np

class PCAReducer:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.eigenvalues = None
    def transform(self, X):
        return (X - self.mean) @ self.components.T
    def fit(self, X):
        return self
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class LDAReducer:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.global_mean = None
    def transform(self, X):
        return X @ self.components.T
    def fit(self, X, y):
        return self
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

class NearestCentroidClassifier:
    def __init__(self):
        self.centroids = None
        self.classes   = None
    def fit(self, X, y):
        self.classes   = np.unique(y)
        self.centroids = np.array([X[y == c].mean(axis=0) for c in self.classes])
        return self
    def predict_one(self, x, threshold=None):
        dists = np.sqrt(((self.centroids - x) ** 2).sum(axis=1))
        idx   = np.argmin(dists)
        conf  = dists[idx]
        if threshold is not None and conf > threshold:
            return -1, conf
        return self.classes[idx], conf
    def predict(self, X, threshold=None):
        return [self.predict_one(x, threshold) for x in X]

# ── Load model ────────────────────────────────────────────────────
with open("model/scratch_model.pkl", "rb") as f:
    saved = pickle.load(f)

pca           = saved["pca"]
lda           = saved["lda"]
clf           = saved["clf"]
names         = saved["names"]
BASE_THRESHOLD = saved["threshold"]
threshold     = BASE_THRESHOLD

# ── Face detector ─────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Colours (BGR) ─────────────────────────────────────────────────
GREEN  = (0,   200,  0)
RED    = (0,   0,  220)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREY   = (180, 180, 180)
YELLOW = (0,   220, 220)


# ── Predict ───────────────────────────────────────────────────────
def predict_face(face_90x90: np.ndarray, T: float):
    vec      = face_90x90.flatten().astype(float).reshape(1, -1)
    pca_proj = pca.transform(vec)
    lda_proj = lda.transform(pca_proj)
    label, conf = clf.predict_one(lda_proj[0], threshold=T)
    name = "Unknown" if label == -1 else names[label]
    return name, conf


# ── Process frame ─────────────────────────────────────────────────
def process_frame(frame: np.ndarray, T: float):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    results = []
    for (x, y, w, h) in faces:
        try:
            aligned, x2, y2, w2, h2 = align_face(gray, x, y, w, h)
            crop = crop_face(aligned, x2, y2, w2, h2)
            if crop is None:
                continue
            resized = resize_face(crop)
            normed  = normalize_lighting(resized)
            masked  = apply_ellipse_mask(normed)
            name, conf = predict_face(masked, T)
            results.append((x, y, w, h, name, conf))
        except Exception:
            results.append((x, y, w, h, "Error", 0.0))
    return results


# ── Draw bounding boxes ───────────────────────────────────────────
def draw_faces(frame: np.ndarray, results: list):
    for (x, y, w, h, name, conf) in results:
        color = RED if name in ("Unknown", "Error") else GREEN
        label = f"{name}  ({conf:.0f})"

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        (lw, lh), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        cv2.rectangle(frame, (x, y - lh - 14), (x + lw + 10, y), color, -1)
        cv2.putText(
            frame, label, (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2
        )


# ── Draw HUD overlay ──────────────────────────────────────────────
def draw_hud(frame: np.ndarray, T: float, n_faces: int, base_T: float):
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at bottom
    bar_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Threshold display
    t_color = YELLOW if abs(T - base_T) > 1 else WHITE
    cv2.putText(
        frame,
        f"Threshold: {T:.0f}  (tuned: {base_T:.0f})",
        (12, h - bar_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, t_color, 1
    )

    # Face count
    cv2.putText(
        frame,
        f"Faces detected: {n_faces}",
        (12, h - bar_h + 46),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1
    )

    # Controls hint
    cv2.putText(
        frame,
        "+/- : threshold    R : reset    Q : quit",
        (12, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREY, 1
    )

    # Title top-left
    cv2.putText(
        frame,
        "Face Recognition  |  PCA + LDA + Nearest Centroid",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1
    )


# ── Main loop ─────────────────────────────────────────────────────
def main():
    global threshold

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    print("Face Recognition running")
    print("  +/=  → raise threshold   (accept more faces)")
    print("  -    → lower threshold   (stricter rejection)")
    print("  R    → reset to tuned threshold")
    print("  Q    → quit")

    THRESHOLD_STEP = 20.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read error")
            break

        results = process_frame(frame, threshold)
        draw_faces(frame, results)
        draw_hud(frame, threshold, len(results), BASE_THRESHOLD)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:        # Q or ESC → quit
            break
        elif key in (ord("+"), ord("=")):        # raise threshold
            threshold = min(threshold + THRESHOLD_STEP, 900.0)
        elif key == ord("-"):                    # lower threshold
            threshold = max(threshold - THRESHOLD_STEP, 50.0)
        elif key == ord("r"):                    # reset
            threshold = BASE_THRESHOLD

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()