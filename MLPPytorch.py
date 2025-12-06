# Train MLP classifier on ASL landmark vectors (PyTorch version, Colab-friendly)
import argparse
import json
from pathlib import Path
import numpy as np
import itertools
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ✅ Detect if running in Google Colab
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def load_npz(npz_path: Path):
    """Load X (features) and y (labels) from a .npz file."""
    data = np.load(npz_path)
    return data["X"], data["y"]


def plot_confusion_matrix(cm, class_names, out_path: Path, title: str = "Confusion Matrix"):
    """Save confusion matrix as an image file."""
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    plt.colorbar(im)

    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    fmt = "d"
    thresh = cm.max() / 2 if cm.size else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j, i, format(cm[i, j], fmt),
            ha="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- PyTorch MLP model ----------
class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, input_dim)
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(
        description="Train ASL classifier on Mediapipe landmark vectors (PyTorch MLP)"
    )

    # 🔧 Default paths assume everything is on Google Drive when in Colab
    if IN_COLAB:
        base = "/content/drive/MyDrive/Duke University/CV-Group6"
        default_data_dir = f"{base}"
        default_model_out = f"{base}/asl_mlp_torch_model.joblib"
        default_val_cm_out = f"{base}/asl_confusion_matrix_val.png"
        default_test_cm_out = f"{base}/asl_confusion_matrix_test.png"
    else:
        # Fallback defaults for local runs
        default_data_dir = "/content/drive/MyDrive/Duke University/CV-Group6"
        default_model_out = "asl_mlp_torch_model.joblib"
        default_val_cm_out = "confusion_matrix_val.png"
        default_test_cm_out = "confusion_matrix_test.png"

    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Folder with train.npz, val.npz, test.npz, labels.json",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default=default_model_out,
        help="Where to save the trained model + scaler (joblib)",
    )
    parser.add_argument(
        "--val_cm_out",
        type=str,
        default=default_val_cm_out,
        help="Where to save the validation confusion matrix image (.png)",
    )
    parser.add_argument(
        "--test_cm_out",
        type=str,
        default=default_test_cm_out,
        help="Where to save the test confusion matrix image (.png)",
    )

    # 🧠 Fix argparse for Colab (avoid parsing notebook/kernel args)
    if IN_COLAB:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_out = Path(args.model_out)
    val_cm_out = Path(args.val_cm_out)
    test_cm_out = Path(args.test_cm_out)

    print(f"📂 Using data directory: {data_dir}")
    print(f"💾 Model will be saved to: {model_out}")
    print(f"📊 Val confusion matrix → {val_cm_out}")
    print(f"📊 Test confusion matrix → {test_cm_out}")

    # ---------- Load datasets ----------
    X_train, y_train = load_npz(data_dir / "train.npz")
    X_val, y_val     = load_npz(data_dir / "val.npz")
    X_test, y_test   = load_npz(data_dir / "test.npz")

    # Load label names (e.g., A, B, C, ..., space)
    with open(data_dir / "labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    print("✅ Data loaded")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
    print(f"  Test samples:  {len(X_test)}")

    # ---------- Scale features (like before) ----------
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # ---------- Convert to torch tensors ----------
    X_train_t = torch.from_numpy(X_train_scaled.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))

    X_val_t = torch.from_numpy(X_val_scaled.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))

    X_test_t = torch.from_numpy(X_test_scaled.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.int64))

    # ---------- Datasets & Dataloaders ----------
    # Optimal HP (from Optuna; mapped to PyTorch):
    # 'hidden_size': 343, 'num_layers': 2, 'dropout': 0.283..., 'lr': 0.001789..., 'batch_size': 64
    batch_size = 64

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t, y_val_t)
    test_ds  = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    input_dim   = X_train_t.shape[1]
    num_classes = len(labels)

    # ---------- Set up device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # ---------- Initialize PyTorch MLP ----------
    hidden_sizes = [343, 343]   # from 'hidden_size' and 'num_layers' in Optuna
    dropout = 0.28319417648906664
    lr = 0.0017897493066372295

    model = MLPNet(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 30
    best_val_f1 = -1.0
    best_state_dict = None

    # ---------- Training loop ----------
    print("\n🚀 Training PyTorch MLP on TRAIN set...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)

        train_loss = running_loss / total

        # ---- Validation each epoch (for monitoring) ----
        model.eval()
        all_val_preds = []
        all_val_labels = []
        val_loss = 0.0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * xb.size(0)
                val_total += xb.size(0)

                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(yb.cpu().numpy())

        val_loss /= val_total
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")

        # Track best model by macro F1
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_state_dict = model.state_dict()

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro F1: {val_f1_macro:.4f}"
        )

    # Load best weights (based on val macro F1)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\n✅ Loaded best model weights (Val macro F1 = {best_val_f1:.4f})")

    # ---------- Final evaluation on VALIDATION set ----------
    print("\n🔎 Evaluating on VALIDATION set...")
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t.to(device))
        val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()

    val_acc = accuracy_score(y_val, val_preds)
    val_f1_macro   = f1_score(y_val, val_preds, average="macro")
    val_f1_weighted = f1_score(y_val, val_preds, average="weighted")

    print("\n==============================")
    print(f"✅ Validation Accuracy:   {val_acc:.4f}")
    print(f"✅ Val macro F1:          {val_f1_macro:.4f}")
    print(f"✅ Val weighted F1:       {val_f1_weighted:.4f}")
    print("==============================")

    print("\n📋 Validation Classification Report:")
    print(classification_report(y_val, val_preds, labels=list(range(len(labels))), target_names=labels))

    cm_val = confusion_matrix(y_val, val_preds, labels=list(range(len(labels))))
    plot_confusion_matrix(cm_val, labels, val_cm_out, title="Validation Confusion Matrix")
    print(f"📊 Saved validation confusion matrix to {val_cm_out}")

    # ---------- Final evaluation on TEST set ----------
    print("\n🔎 Evaluating on TEST set...")
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t.to(device))
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()

    test_acc = accuracy_score(y_test, test_preds)
    test_f1_macro    = f1_score(y_test, test_preds, average="macro")
    test_f1_weighted = f1_score(y_test, test_preds, average="weighted")

    print("\n==============================")
    print(f"✅ Test Accuracy:         {test_acc:.4f}")
    print(f"✅ Test macro F1:         {test_f1_macro:.4f}")
    print(f"✅ Test weighted F1:      {test_f1_weighted:.4f}")
    print("==============================")

    print("\n📋 Test Classification Report:")
    print(classification_report(y_test, test_preds, labels=list(range(len(labels))), target_names=labels))

    cm_test = confusion_matrix(y_test, test_preds, labels=list(range(len(labels))))
    plot_confusion_matrix(cm_test, labels, test_cm_out, title="Test Confusion Matrix")
    print(f"📊 Saved test confusion matrix to {test_cm_out}")

    # ---------- Save model + scaler ----------
    model_out.parent.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_sizes": hidden_sizes,
        "num_classes": num_classes,
        "dropout": dropout,
        "scaler": scaler,
        "labels": labels,
    }
    joblib.dump(save_obj, model_out)
    print(f"💾 Saved trained PyTorch MLP + scaler to {model_out}")


if __name__ == "__main__":
    main()
