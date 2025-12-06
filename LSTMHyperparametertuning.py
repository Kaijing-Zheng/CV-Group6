#Hyperparameter Tuning for LSTM
# ==============================
# Hyperparameter tuning: LSTM
# ==============================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna

# ---------- Dataset ----------
class ASLLSTMDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]     # (N, 42)
        self.y = data["y"]     # (N,)

        assert self.X.ndim == 2, f"X must be 2-dimensional (N, features), but got {self.X.ndim} dimensions"
        assert self.X.shape[1] == 42, f"Expected 42 features per sample, but got {self.X.shape[1]}"
        assert len(self.X) == len(self.y), "X and y must have same length"

        # Reshape for LSTM: (N, 42) -> (N, T=1, features=42)
        # Each static image is treated as a sequence of length 1
        self.X = self.X[:, np.newaxis, :].astype(np.float32) # Shape becomes (N, 1, 42)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])          # (T=1, 42)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

# ---------- LSTM Model ----------
class ASLLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
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
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.fc(last_hidden)

# ---------- Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

train_dataset = ASLLSTMDataset("/content/drive/MyDrive/Duke University/CV-Group6/train.npz")
val_dataset   = ASLLSTMDataset("/content/drive/MyDrive/Duke University/CV-Group6/val.npz")

# Correctly derive input_size from the dataset's X shape
# self.X has shape (N, 1, 42), so X.shape[2] gives 42
_, _, input_size = train_dataset.X.shape
num_classes = len(np.unique(train_dataset.y))

# ---------- Optuna objective ----------
def objective_lstm(trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers  = trial.suggest_int("num_layers", 1, 3)
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model = ASLLSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 5  # small for tuning

    # ---- Train for a few epochs ----
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # ---- Validation loss (objective) ----
    model.eval()
    val_loss = 0.0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            val_total += X_batch.size(0)

    val_loss /= val_total
    return val_loss

# ---------- Run study ----------
study_lstm = optuna.create_study(direction="minimize")
study_lstm.optimize(objective_lstm, n_trials=20)

print("Best LSTM hyperparameters:", study_lstm.best_params)
print("Best LSTM validation loss:", study_lstm.best_value)
