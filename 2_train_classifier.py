"""
STEP 2: Train MLP Classifier
==============================
Trains a 2-layer MLP (Multi-Layer Perceptron) neural network on top of
the ArcFace embeddings generated in Step 1.

Why MLP over SVM?
- Learnable decision boundaries specific to your 7-member database
- Outputs calibrated probabilities per class (needed for Unknown thresholding)
- More explainable as a "neural network classifier" for your report
- Still lightweight and fast — only trained on embeddings, not raw images

Input:  embeddings/embeddings.pkl
Output: models/mlp_classifier.pkl
        models/label_encoder.pkl
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.preprocessing      import LabelEncoder, Normalizer
from sklearn.neural_network     import MLPClassifier
from sklearn.model_selection    import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics            import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── Configuration ──────────────────────────────────────────────────────────────
EMBEDDINGS_PATH    = "embeddings/embeddings.pkl"
MODEL_OUTPUT_PATH  = "models/mlp_classifier.pkl"
ENCODER_PATH       = "models/label_encoder.pkl"
NORMALIZER_PATH    = "models/normalizer.pkl"
PLOTS_DIR          = "models/plots"

# MLP Hyperparameters
HIDDEN_LAYERS      = (256, 128)     # Two hidden layers: 512→256→128→N_classes
ACTIVATION         = "relu"
MAX_ITER           = 1000
LEARNING_RATE_INIT = 0.001
RANDOM_STATE       = 42
# ───────────────────────────────────────────────────────────────────────────────

def load_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"'{EMBEDDINGS_PATH}' not found.\n"
            "Please run Step 1 first: python 1_generate_embeddings.py"
        )
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)

    X = np.array(data["embeddings"])
    y = np.array(data["labels"])
    print(f"Loaded {len(X)} embeddings for {len(np.unique(y))} members.")

    # Per-member count
    for member in sorted(np.unique(y)):
        count = np.sum(y == member)
        flag  = "⚠ (<10)" if count < 10 else "✓"
        print(f"  {flag}  {member}: {count} samples")
    return X, y

def main():
    # 1. Load & normalise embeddings
    X_raw, y_str = load_embeddings()

    normalizer = Normalizer(norm="l2")
    X = normalizer.fit_transform(X_raw)

    # 2. Encode string labels → integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    class_names = encoder.classes_
    n_classes   = len(class_names)
    print(f"\nClasses: {list(class_names)}")

    # 3. Build MLP
    print("\nTraining MLP classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes = HIDDEN_LAYERS,
        activation         = ACTIVATION,
        solver             = "adam",
        learning_rate_init = LEARNING_RATE_INIT,
        max_iter           = MAX_ITER,
        early_stopping     = True,         # Prevents overfitting on small dataset
        validation_fraction= 0.15,
        random_state       = RANDOM_STATE,
        verbose            = False,
    )

    # 4. Cross-validation (stratified so every member is in each fold)
    n_splits = min(3, min(np.bincount(y)))   # Adjust folds if very few samples
    if n_splits >= 2:
        skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(mlp, X, y, cv=skf, scoring="accuracy")
        print(f"Cross-validation accuracy ({n_splits}-fold): "
              f"{scores.mean():.2%} ± {scores.std():.2%}")
    else:
        print("⚠ Too few samples per class for cross-validation — skipping.")

    # 5. Final train on full dataset
    mlp.fit(X, y)
    train_acc = mlp.score(X, y)
    print(f"Training accuracy (full dataset): {train_acc:.2%}")

    # 6. Classification report
    y_pred = mlp.predict(X)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=class_names))

    # 7. Save confusion matrix plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm  = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("MLP Classifier — Confusion Matrix (Training Set)", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved → {cm_path}")

    # 8. Save loss curve
    if hasattr(mlp, "loss_curve_"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(mlp.loss_curve_, label="Training Loss", color="#2563eb")
        if hasattr(mlp, "validation_scores_"):
            ax2 = ax.twinx()
            ax2.plot(mlp.validation_scores_, label="Val Accuracy",
                     color="#16a34a", linestyle="--")
            ax2.set_ylabel("Validation Accuracy")
            ax2.legend(loc="upper right")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("MLP Training Loss Curve")
        ax.legend(loc="upper left")
        plt.tight_layout()
        loss_path = os.path.join(PLOTS_DIR, "loss_curve.png")
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"Loss curve saved      → {loss_path}")

    # 9. Save model, encoder, normalizer
    os.makedirs("models", exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(mlp, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    with open(NORMALIZER_PATH, "wb") as f:
        pickle.dump(normalizer, f)

    print(f"\n{'='*50}")
    print(f"✅ Model saved  → {MODEL_OUTPUT_PATH}")
    print(f"✅ Encoder saved → {ENCODER_PATH}")
    print(f"✅ Normalizer saved → {NORMALIZER_PATH}")
    print(f"{'='*50}")
    print("\nNext step → run: python 3_live_demo.py")

if __name__ == "__main__":
    main()
