import cv2
import numpy as np
from pathlib import Path

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

FACE_SIZE      = (90, 90)
CLAHE_CLIP     = 2.0
CLAHE_GRID     = (8, 8)
FOREHEAD_EXTRA = 0.15
CHIN_EXTRA     = 0.05

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [load_image] Could not read: {path}")
    return img

def detect_face(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for (scale, neighbors) in [(1.05, 4), (1.1, 4), (1.15, 3), (1.2, 2)]:
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(40, 40)
        )
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            return tuple(faces[0])
    return None

def align_face(gray, x, y, w, h):
    face_roi = gray[y: y + h // 2, x: x + w]
    eyes = _eye_cascade.detectMultiScale(
        face_roi, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15)
    )
    if len(eyes) < 2:
        return gray, x, y, w, h
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])
    (ex1, ey1, ew1, eh1) = eyes[0]
    (ex2, ey2, ew2, eh2) = eyes[1]
    left_eye_center  = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
    right_eye_center = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)
    eye_gap = abs(right_eye_center[0] - left_eye_center[0])
    if eye_gap < w * 0.20:
        return gray, x, y, w, h
    face_mid_y = y + h // 2
    if left_eye_center[1] > face_mid_y or right_eye_center[1] > face_mid_y:
        return gray, x, y, w, h
    vertical_gap = abs(left_eye_center[1] - right_eye_center[1])
    if vertical_gap > h * 0.10:
        return gray, x, y, w, h
    dx    = right_eye_center[0] - left_eye_center[0]
    dy    = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    if abs(angle) > 15.0:
        return gray, x, y, w, h
    eye_mid = (
        float((left_eye_center[0] + right_eye_center[0]) // 2),
        float((left_eye_center[1] + right_eye_center[1]) // 2)
    )
    M = cv2.getRotationMatrix2D(eye_mid, angle, 1.0)
    aligned = cv2.warpAffine(
        gray, M, (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return aligned, x, y, w, h

def crop_face(gray, x, y, w, h):
    img_h, img_w = gray.shape[:2]
    forehead = int(h * FOREHEAD_EXTRA)
    chin     = int(h * CHIN_EXTRA)
    x1 = max(0,     x)
    y1 = max(0,     y - forehead)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h + chin)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop

def resize_face(face_gray, size=FACE_SIZE):
    return cv2.resize(face_gray, size, interpolation=cv2.INTER_AREA)

def normalize_lighting(face_gray):
    clahe     = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    clahe_img = clahe.apply(face_gray)
    return cv2.normalize(clahe_img, None, 0, 255, cv2.NORM_MINMAX)

def apply_ellipse_mask(face_gray):
    h, w = face_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2),
                (int(w * 0.45), int(h * 0.48)),
                0, 0, 360, 255, -1)
    result = face_gray.copy()
    result[mask == 0] = 0
    return result

def save_face(face_gray, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), face_gray)
    if not ok:
        print(f"  [save_face] Failed to write: {out_path}")
    return ok
