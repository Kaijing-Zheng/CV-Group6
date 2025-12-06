#LSTM Model from ChatGPT
# ============================================================
# 1. Imports
# ============================================================
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report

# ============================================================
# 2. GPU setup (works in Google Colab if GPU is enabled)
#    In Colab: Runtime -> Change runtime type -> Hardware accelerator: GPU
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# ============================================================
# 3. Dataset class for NPZ files: train.npz / val.npz / test.npz
# ============================================================
class ASLLandmarkDataset(Dataset):
    def __init__(self, npz_path):
        """
        Expects an .npz with:
          - X: (N, 42) where 42 = 21 landmarks * 2 (x,y) coordinates
          - y: (N,)
        """
        data = np.load(npz_path)
        self.X = data["X"]  # shape: (N, 42)
        self.y = data["y"]  # shape: (N,)

        assert len(self.X) == len(self.y), "X and y must have same length"
        # Assertions updated for (N, 42) shape
        assert self.X.ndim == 2, f"X must be 2-dimensional (N, features), but got {self.X.ndim} dimensions"
        assert self.X.shape[1] == 42, f"Expected 42 features per sample, but got {self.X.shape[1]}"

        # Reshape for LSTM: (N, 42) -> (N, T=1, features=42)
        # Each static image is treated as a sequence of length 1
        self.X = self.X[:, np.newaxis, :].astype(np.float32) # Shape becomes (N, 1, 42)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]              # (T=1, features=42)
        y = self.y[idx]              # scalar label
        x = torch.from_numpy(x)      # float32 tensor
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# ============================================================
# 4. LSTM model definition
# ============================================================
class ASLLSTMClassifier(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, num_layers=2, num_classes=26, dropout=0.3):
        """
        input_size: features per time step (21 landmarks * 2 coords = 42)
        hidden_size: LSTM hidden dimension
        num_layers: number of stacked LSTM layers
        num_classes: number of ASL classes
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: (batch, T, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        logits = self.fc(last_hidden)
        return logits

# ============================================================
# 5. Create datasets and dataloaders for train / val / test
# ============================================================
train_path = "/content/drive/MyDrive/Duke University/CV-Group6/train.npz"
val_path   = "/content/drive/MyDrive/Duke University/CV-Group6/val.npz"
test_path  = "/content/drive/MyDrive/Duke University/CV-Group6/test.npz"

train_dataset = ASLLandmarkDataset(train_path)
val_dataset   = ASLLandmarkDataset(val_path)
test_dataset  = ASLLandmarkDataset(test_path)

# Infer num_classes from training labels
num_classes = len(np.unique(train_dataset.y))
print("Number of classes:", num_classes)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# ============================================================
# 6. Initialize model, loss, optimizer
# ============================================================
model = ASLLSTMClassifier(
    input_size=42, # Changed from 63 to 42
    hidden_size=128,
    num_layers=2,
    num_classes=num_classes,
    dropout=0.3
).to(device)  # move model to GPU/CPU

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 30

# ============================================================
# 7. Training + validation loop (with F1 on val)
# ============================================================
for epoch in range(1, num_epochs + 1):
    # ---- Training ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            val_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(logits, dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)

            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(y_batch.cpu().numpy())

    val_loss /= val_total
    val_acc = val_correct / val_total

    # F1 scores on validation
    val_f1_macro = f1_score(all_val_labels, all_val_preds, average="macro")
    val_f1_weighted = f1_score(all_val_labels, all_val_preds, average="weighted")

    print(
        f"Epoch [{epoch}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} "
        f"F1(macro): {val_f1_macro:.3f} F1(weighted): {val_f1_weighted:.3f}"
    )

# ============================================================
# 8. Final evaluation on test set (with F1 + report)
# ============================================================
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        test_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(logits, dim=1)
        test_correct += (preds == y_batch).sum().item()
        test_total += y_batch.size(0)

        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(y_batch.cpu().numpy())

test_loss /= test_total
test_acc = test_correct / test_total

test_f1_macro = f1_score(all_test_labels, all_test_preds, average="macro")
test_f1_weighted = f1_score(all_test_labels, all_test_preds, average="weighted")

print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.3f}")
print(f"Test F1(macro): {test_f1_macro:.3f} | Test F1(weighted): {test_f1_weighted:.3f}")

print("\nClassification report:\n")
print(classification_report(all_test_labels, all_test_preds))

# ============================================================
# 9. Optional: helper to predict a single sequence
# ============================================================
def predict_single_sequence(model, single_sample_features_np):
    """
    single_sample_features_np: numpy array of shape (42,) for one example (x,y coords)
    returns: predicted class index (int)
    """
    model.eval()
    with torch.no_grad():
        # Reshape (42,) to (1, 1, 42) for batch=1, T=1, input_size=42
        x = torch.from_numpy(single_sample_features_np[np.newaxis, np.newaxis, :]).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred

# Example usage (uncomment when you have test data loaded):
# You can use a sample from the dataset's raw X array (which has shape (N, 42))
# For example, to predict the first sample:
# example_features = test_dataset.X_raw[0] # Assuming X_raw is the (N,42) array
# print("Predicted class:", predict_single_sequence(model, example_features))

# If you want to use the processed dataset output, you'd access it like:
# example_features_processed = test_dataset.X[0].squeeze().numpy() # This gives (42,)
# print("Predicted class:", predict_single_sequence(model, example_features_processed))