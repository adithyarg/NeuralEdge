"""
train_thermal.py — Train binary MLP on 32x32 thermal patches.

Network: 1024 → 64 (ReLU) → 32 (ReLU) → 2 (human / background)
Same topology as MNIST accelerator, just different input size and output classes.

Input:  training/thermal/data/patches_train.npz
        training/thermal/data/patches_test.npz
Output: training/thermal/weights/mlp_thermal.pt

Usage:
    python training/thermal/train_thermal.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ── Dataset ───────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        # Normalise to [0, 1] float32 — same as MNIST pipeline
        self.X = torch.from_numpy(data["X"].astype(np.float32) / 255.0)
        self.y = torch.from_numpy(data["y"].astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model — identical topology to MNIST MLP ───────────────────────────────────
class ThermalMLP(nn.Module):
    """
    784→64→32→10  (MNIST)
    1024→64→32→2  (thermal) — same RTL, different parameters
    """
    def __init__(self, in_features=1024, hidden1=64, hidden2=32, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ── Training ──────────────────────────────────────────────────────────────────
def train(epochs=20, lr=1e-3, batch_size=256,
          data_dir="training/thermal/data",
          out_dir="training/thermal/weights"):

    train_path = os.path.join(data_dir, "patches_train.npz")
    test_path  = os.path.join(data_dir, "patches_test.npz")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} not found.\n"
            "Run prepare_dataset.py first:\n"
            "  python training/thermal/prepare_dataset.py --llvip_dir /path/to/LLVIP"
        )

    train_set    = PatchDataset(train_path)
    test_set     = PatchDataset(test_path)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=1000)

    # Class weights to handle positive/negative imbalance
    labels     = train_set.y.numpy()
    n_pos      = int(labels.sum())
    n_neg      = len(labels) - n_pos
    w_pos      = n_neg / max(n_pos, 1)
    class_w    = torch.tensor([1.0, w_pos])

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = ThermalMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device))
    best_acc  = 0.0

    print(f"Training on {device}")
    print(f"Train: {len(train_set)} patches  |  Test: {len(test_set)} patches")
    print(f"Class balance — neg:{n_neg}  pos:{n_pos}  weight_pos:{w_pos:.2f}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()
        scheduler.step()

        # Evaluation
        model.eval()
        correct = tp = fp = fn = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                preds = model(X_batch.to(device)).argmax(1).cpu()
                correct += preds.eq(y_batch).sum().item()
                tp += ((preds == 1) & (y_batch == 1)).sum().item()
                fp += ((preds == 1) & (y_batch == 0)).sum().item()
                fn += ((preds == 0) & (y_batch == 1)).sum().item()

        acc       = 100.0 * correct / len(test_set)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)

        print(f"Epoch {epoch:02d}/{epochs}  "
              f"acc={acc:.1f}%  prec={precision:.3f}  "
              f"rec={recall:.3f}  F1={f1:.3f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(out_dir, "mlp_thermal.pt"))

    print(f"\nBest accuracy: {best_acc:.1f}%  →  {out_dir}/mlp_thermal.pt")


if __name__ == "__main__":
    train()
