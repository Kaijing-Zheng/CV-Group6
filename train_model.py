import argparse
import json
from pathlib import Path
import itertools
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_npz(p: Path):
    d = np.load(p)
    return d["X"], d["y"]

def plot_confusion_matrix(cm, class_names, out_path):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)
    ticks = range(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90); ax.set_yticklabels(class_names)
    thresh = cm.max() / 2 if cm.size else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm[i, j]:d}", ha="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def build_clf():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
            verbose=True
        ))
    ])

def main():
    ap = argparse.ArgumentParser(description="Train ASL classifier on landmark vectors (train/val only)")
    ap.add_argument("--data_dir", type=str, default="Data/processed", help="Folder with train.npz, val.npz, labels.json")
    ap.add_argument("--model_out", type=str, default="model.joblib", help="Where to save the trained model")
    ap.add_argument("--cm_out", type=str, default="confusion_matrix.png", help="Where to save confusion matrix img")
    ap.add_argument("--refit_full", action="store_true",
                    help="After evaluation, refit on train+val without early stopping and save that model")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    X_train, y_train = load_npz(data_dir / "train.npz")
    X_val, y_val     = load_npz(data_dir / "val.npz")

    with open(data_dir / "labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    print("✅ Data loaded")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")
    print(f"Classes:       {len(labels)}")

    # Train on train set (internal early stopping)
    clf = build_clf()
    print("\n🚀 Training on train set...")
    clf.fit(X_train, y_train)

    # Evaluate on external val set
    print("\n🔎 Evaluating on val set...")
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n✅ Validation Accuracy: {acc:.4f}\n")
    print(classification_report(y_val, y_pred, target_names=labels))

    # Confusion matrix (ensure fixed class order)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(labels))))
    plot_confusion_matrix(cm, labels, args.cm_out)
    print(f"📊 Saved confusion matrix to {args.cm_out}")

    # Optionally refit on train+val to produce the final artifact
    if args.refit_full:
        print("\n🔁 Re-fitting on train+val (no early stopping) for final model...")
        X_full = np.vstack([X_train, X_val])
        y_full = np.hstack([y_train, y_val])
        final = Pipeline([
            ("scaler", clf.named_steps["scaler"]),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                batch_size=256,
                max_iter=80,
                early_stopping=False,
                random_state=42,
                verbose=True
            ))
        ])
        final.fit(X_full, y_full)
        joblib.dump(final, args.model_out)
    else:
        joblib.dump(clf, args.model_out)

    print(f"💾 Saved trained model to {args.model_out}")

if __name__ == "__main__":
    main()
