#Hyperparameter tuning for MLP
# ==============================
# Hyperparameter tuning: MLP
# ==============================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna

# ---------- Dataset ----------
class ASLMLPDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]     # (N, 42)
        self.y = data["y"]     # (N,)

        # Assertions to ensure data is in the expected (N, 42) format
        assert self.X.ndim == 2, f"X must be 2-dimensional (N, features), but got {self.X.ndim} dimensions"
        assert self.X.shape[1] == 42, f"Expected 42 features per sample, but got {self.X.shape[1]}"
        assert len(self.X) == len(self.y), "X and y must have same length"

        # The X data is already flattened (N, 42) as required for an MLP input
        # No further reshaping needed for self.X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])          # (42,)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

# ---------- Model ----------
class ASLMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        layers = []
        in_dim = input_size

        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, input_size)
        return self.net(x)

# ---------- Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

train_dataset = ASLMLPDataset("/content/drive/MyDrive/Duke University/CV-Group6/train.npz")
val_dataset   = ASLMLPDataset("/content/drive/MyDrive/Duke University/CV-Group6/val.npz")

input_size  = train_dataset.X.shape[1]
num_classes = len(np.unique(train_dataset.y))

# ---------- Optuna objective ----------
def objective_mlp(trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_int("hidden_size", 64, 512)
    num_layers  = trial.suggest_int("num_layers", 1, 3)
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [32, 64, 128])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model = ASLMLPClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 5  # keep small for tuning speed

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
    return val_loss  # Optuna will MINIMIZE this

# ---------- Run study ----------
study_mlp = optuna.create_study(direction="minimize")
study_mlp.optimize(objective_mlp, n_trials=20)

print("Best MLP hyperparameters:", study_mlp.best_params)
print("Best MLP validation loss:", study_mlp.best_value)
