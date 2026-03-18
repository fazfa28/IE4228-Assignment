"""
STEP 3: Live Camera Demo
==========================
Real-time face detection (RetinaFace via InsightFace) + recognition
(ArcFace embeddings → MLP classifier) with a polished OpenCV GUI.

Features:
  - Green bounding box + name + confidence % for known faces
  - Red bounding box + "Unknown" for unrecognised faces
  - Live confidence bar chart sidebar for all 7 members
  - FPS counter
  - Press Q to quit, S to save a screenshot

Usage:
    python 3_live_demo.py
"""

import os
import pickle
import time
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH      = "models/mlp_classifier.pkl"
ENCODER_PATH    = "models/label_encoder.pkl"
NORMALIZER_PATH = "models/normalizer.pkl"

CONFIDENCE_THRESHOLD = 0.50   # Below → "Unknown"  (tune 0.4–0.7)
CAMERA_INDEX         = 0      # 0 = default webcam
SIDEBAR_WIDTH        = 260    # Width of the right-side confidence panel
SCREENSHOT_DIR       = "screenshots"
# ───────────────────────────────────────────────────────────────────────────────

# ── Colour palette ─────────────────────────────────────────────────────────────
GREEN       = (34,  197,  94)    # Known face box
RED         = (239,  68,  68)    # Unknown face box
SIDEBAR_BG  = (15,  23,  42)    # Dark navy sidebar
WHITE       = (255, 255, 255)
LIGHT_GREY  = (148, 163, 184)
ACCENT_BLUE = (59,  130, 246)
BAR_EMPTY   = (51,  65,  85)
# ───────────────────────────────────────────────────────────────────────────────


def load_models():
    """Load InsightFace app + MLP classifier + encoder + normalizer."""
    print("Loading InsightFace (RetinaFace + ArcFace)...")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],  # Use CUDAExecutionProvider for GPU
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("Loading MLP classifier...")
    for path in [MODEL_PATH, ENCODER_PATH, NORMALIZER_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found.\n"
                "Please run Steps 1 and 2 first."
            )

    with open(MODEL_PATH,      "rb") as f: mlp        = pickle.load(f)
    with open(ENCODER_PATH,    "rb") as f: encoder    = pickle.load(f)
    with open(NORMALIZER_PATH, "rb") as f: normalizer = pickle.load(f)

    print("All models loaded.\n")
    return app, mlp, encoder, normalizer


def predict_face(embedding, mlp, encoder, normalizer):
    """Return (name, confidence, all_probabilities)."""
    emb_norm = normalizer.transform(embedding.reshape(1, -1))
    proba    = mlp.predict_proba(emb_norm)[0]
    idx      = np.argmax(proba)
    conf     = proba[idx]
    name     = encoder.inverse_transform([idx])[0] if conf >= CONFIDENCE_THRESHOLD else "Unknown"
    return name, conf, proba


def draw_face_box(frame, bbox, name, conf, color):
    """Draw bounding box + label on frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label      = f"{name}" if name == "Unknown" else f"{name}  {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    pad = 6
    cv2.rectangle(frame, (x1, y1 - th - 2*pad), (x1 + tw + 2*pad, y1), color, -1)
    cv2.putText(frame, label,
                (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, WHITE, 1, cv2.LINE_AA)


def draw_sidebar(sidebar, class_names, latest_proba):
    """
    Draw the right-side confidence bar chart panel.
    sidebar: blank image of shape (H, SIDEBAR_WIDTH, 3)
    """
    h, w = sidebar.shape[:2]
    sidebar[:] = SIDEBAR_BG

    # Title
    cv2.putText(sidebar, "RECOGNITION", (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(sidebar, "CONFIDENCE", (12, 48),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, LIGHT_GREY, 1, cv2.LINE_AA)
    cv2.line(sidebar, (12, 56), (w - 12, 56), (51, 65, 85), 1)

    n        = len(class_names)
    row_h    = max(40, (h - 80) // (n + 1))
    bar_maxw = w - 24

    for i, name in enumerate(class_names):
        y_top  = 70 + i * row_h
        conf_i = float(latest_proba[i]) if latest_proba is not None else 0.0

        # Member name
        short = name if len(name) <= 12 else name[:11] + "…"
        cv2.putText(sidebar, short, (12, y_top + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)

        # Empty bar track
        bar_y = y_top + 20
        cv2.rectangle(sidebar, (12, bar_y), (12 + bar_maxw, bar_y + 10), BAR_EMPTY, -1)

        # Filled bar
        fill   = int(bar_maxw * conf_i)
        # Colour: green if top, blue otherwise
        is_top = (latest_proba is not None and i == int(np.argmax(latest_proba))
                  and conf_i >= CONFIDENCE_THRESHOLD)
        bar_col = GREEN if is_top else ACCENT_BLUE
        if fill > 0:
            cv2.rectangle(sidebar, (12, bar_y), (12 + fill, bar_y + 10), bar_col, -1)

        # Percentage label
        cv2.putText(sidebar, f"{conf_i:.0%}", (w - 44, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, LIGHT_GREY, 1, cv2.LINE_AA)


def main():
    app, mlp, encoder, normalizer = load_models()
    class_names = list(encoder.classes_)
    n_classes   = len(class_names)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAMERA_INDEX}")
        return

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    print("Live demo running...")
    print("  Q → quit   |   S → save screenshot\n")

    latest_proba = np.zeros(n_classes)   # Smoothed confidence bars
    fps_prev     = time.time()
    fps          = 0.0
    frame_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from camera.")
            break

        frame_count += 1

        # ── FPS calculation ──────────────────────────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_prev, 1e-6))
        fps_prev = now

        # ── Face detection + recognition ─────────────────────────────────────
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces   = app.get(img_rgb)

        detected_proba = None   # Will be set to proba of first/best face

        for face in faces:
            embedding = face.normed_embedding
            name, conf, proba = predict_face(embedding, mlp, encoder, normalizer)
            color = GREEN if name != "Unknown" else RED
            draw_face_box(frame, face.bbox, name, conf, color)

            # Use proba of most-confident detected face for sidebar
            if detected_proba is None or conf > float(np.max(detected_proba)):
                detected_proba = proba

        # Smooth sidebar bars (exponential moving average)
        if detected_proba is not None:
            latest_proba = 0.6 * latest_proba + 0.4 * detected_proba
        else:
            latest_proba = 0.85 * latest_proba   # Decay toward zero when no face

        # ── Build sidebar ────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        sidebar = np.zeros((h, SIDEBAR_WIDTH, 3), dtype=np.uint8)
        draw_sidebar(sidebar, class_names, latest_proba)

        # ── Overlay HUD on frame ─────────────────────────────────────────────
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        # Face count
        fc_label = f"Faces: {len(faces)}"
        cv2.putText(frame, fc_label, (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        # Model label (bottom-left)
        cv2.putText(frame, "RetinaFace + ArcFace + MLP", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, LIGHT_GREY, 1, cv2.LINE_AA)

        # ── Divider line between frame and sidebar ───────────────────────────
        cv2.line(frame, (w - 2, 0), (w - 2, h), (51, 65, 85), 2)

        # ── Combine frame + sidebar ──────────────────────────────────────────
        combined = np.hstack([frame, sidebar])

        cv2.imshow("Face Recognition Demo  |  Q: quit   S: screenshot", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.png")
            cv2.imwrite(path, combined)
            print(f"Screenshot saved → {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Demo closed.")


if __name__ == "__main__":
    main()
