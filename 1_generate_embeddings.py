"""
STEP 1: Generate ArcFace Embeddings
====================================
Uses InsightFace (buffalo_l model with ArcFace loss) to generate
512-d face embeddings from your processed face image folders.

Folder structure expected:
    data/processed_faces/
        member1/  (15 photos)
        member2/  (9 photos)
        ...

Output:
    embeddings/embeddings.pkl
"""

import os
import pickle
import numpy as np
from PIL import Image
import cv2
import insightface
from insightface.app import FaceAnalysis

# ── Configuration ──────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed_faces"   # Change if your folder path differs
OUTPUT_PATH   = "embeddings/embeddings.pkl"
IMG_SIZE      = (112, 112)               # ArcFace expects 112x112
# ───────────────────────────────────────────────────────────────────────────────

def load_insightface():
    """Load InsightFace buffalo_l model (ArcFace backbone)."""
    print("Loading InsightFace (buffalo_l) model...")
    app = FaceAnalysis(
        name="buffalo_l",           # ArcFace ResNet50 backbone
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]   # Change to CUDAExecutionProvider if GPU available
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded successfully.\n")
    return app

def get_embedding_from_processed_face(app, img_bgr):
    """
    Extract ArcFace embedding from an already-processed (cropped) face image.
    Since faces are pre-cropped, we try detection first; if it fails,
    we resize and embed directly using the recognition model.
    """
    faces = app.get(img_bgr)

    if faces:
        # Use the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding  # Already L2-normalised 512-d vector

    # Fallback: resize to 112x112 and run recognition model directly
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    embedding = app.models["recognition"].get_feat(img_resized)
    # L2-normalise manually
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.flatten()

def main():
    app = load_insightface()

    embeddings = []
    labels     = []
    skipped    = 0

    member_dirs = sorted([
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ])

    if not member_dirs:
        print(f"ERROR: No subdirectories found in '{PROCESSED_DIR}'")
        print("Make sure your folder structure is: data/processed_faces/member1/, member2/, ...")
        return

    for member_name in member_dirs:
        member_path = os.path.join(PROCESSED_DIR, member_name)
        img_files   = [
            f for f in os.listdir(member_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        print(f"Processing '{member_name}': {len(img_files)} images found...")
        member_count = 0

        for img_file in img_files:
            img_path = os.path.join(member_path, img_file)
            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f"  ⚠ Could not read: {img_file}")
                    skipped += 1
                    continue

                embedding = get_embedding_from_processed_face(app, img_bgr)
                embeddings.append(embedding)
                labels.append(member_name)
                member_count += 1

            except Exception as e:
                print(f"  ⚠ Skipping {img_file}: {e}")
                skipped += 1

        print(f"  ✓ {member_count}/{len(img_files)} embeddings extracted for '{member_name}'")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)

    print(f"\n{'='*50}")
    print(f"✅ Done! Saved {len(embeddings)} embeddings for {len(set(labels))} members.")
    if skipped:
        print(f"⚠  {skipped} images were skipped.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"{'='*50}")
    print("\nNext step → run: python 2_train_classifier.py")

if __name__ == "__main__":
    main()
