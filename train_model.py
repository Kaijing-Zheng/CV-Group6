import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import itertools
import joblib


def load_npz(npz_path):
    """Load X (features) and y (labels) from a .npz file"""
    data = np.load(npz_path)
    return data["X"], data["y"]


def plot_confusion_matrix(cm, class_names, out_path):
    """Save confusion matrix as an image file"""
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)

    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    fmt = "d"
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train ASL classifier on landmark vectors")
    parser.add_argument("--data_dir", type=str, default="processed", help="Folder with train/val/test .npz files")
    parser.add_argument("--model_out", type=str, default="model.joblib", help="Where to save the trained model")
    parser.add_argument("--cm_out", type=str, default="confusion_matrix.png", help="Where to save confusion matrix img")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load datasets
    X_train, y_train = load_npz(data_dir / "train.npz")
    X_val, y_val = load_npz(data_dir / "val.npz")
    X_test, y_test = load_npz(data_dir / "test.npz")

    # Load label names (A, B, C, ..., space)
    with open(data_dir / "labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    print("✅ Data loaded")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")
    print(f"Test samples:  {len(X_test)}")

    # Create classifier pipeline (scale → neural network)
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=256,
            max_iter=50,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42,
            verbose=True
        ))
    ])

    # Combine train + validation for final training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])

    print("\n🚀 Training model...")
    clf.fit(X_train_full, y_train_full)

    # Evaluate on untouched test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n==============================")
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("==============================")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, args.cm_out)
    print(f"📊 Saved confusion matrix to {args.cm_out}")

    # Save model
    joblib.dump(clf, args.model_out)
    print(f"💾 Saved trained model to {args.model_out}")


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import itertools
import joblib


def load_npz(npz_path):
    """Load X (features) and y (labels) from a .npz file"""
    data = np.load(npz_path)
    return data["X"], data["y"]


def plot_confusion_matrix(cm, class_names, out_path):
    """Save confusion matrix as an image file"""
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)

    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    fmt = "d"
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train ASL classifier on landmark vectors")
    parser.add_argument("--data_dir", type=str, default="processed", help="Folder with train/val/test .npz files")
    parser.add_argument("--model_out", type=str, default="model.joblib", help="Where to save the trained model")
    parser.add_argument("--cm_out", type=str, default="confusion_matrix.png", help="Where to save confusion matrix img")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load datasets
    X_train, y_train = load_npz(data_dir / "train.npz")
    X_val, y_val = load_npz(data_dir / "val.npz")
    X_test, y_test = load_npz(data_dir / "test.npz")

    # Load label names (A, B, C, ..., space)
    with open(data_dir / "labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    print("✅ Data loaded")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")
    print(f"Test samples:  {len(X_test)}")

    # Create classifier pipeline (scale → neural network)
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=256,
            max_iter=50,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42,
            verbose=True
        ))
    ])

    # Combine train + validation for final training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])

    print("\n🚀 Training model...")
    clf.fit(X_train_full, y_train_full)

    # Evaluate on untouched test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n==============================")
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("==============================")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, args.cm_out)
    print(f"📊 Saved confusion matrix to {args.cm_out}")

    # Save model
    joblib.dump(clf, args.model_out)
    print(f"💾 Saved trained model to {args.model_out}")


if __name__ == "__main__":
    main()