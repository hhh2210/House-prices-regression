import argparse
import os
from pathlib import Path
import random

import numpy as np
import inspect
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        prefer = prefer.lower()
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # Auto-detect, prefer MPS on Apple silicon
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, layers: int = 3, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim]
        for _ in range(layers):
            dims.append(hidden_dim)
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


def build_preprocessor(df_train: pd.DataFrame, target_col: str):
    X = df_train.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Handle scikit-learn API change: sparse -> sparse_output
    ohe_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**ohe_kwargs)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = mse_loss(pred, y)
        bs = X.size(0)
        total_loss += loss.item() * bs
        total_count += bs
    mean_mse = total_loss / max(total_count, 1)
    rmse = float(np.sqrt(mean_mse))
    return mean_mse, rmse


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(patience // 2, 2)
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)
            count += X.size(0)
        train_mse = running / max(count, 1)

        val_mse, val_rmse = evaluate(model, val_loader, device)
        scheduler.step(val_mse)

        improved = val_mse < best_val - 1e-7
        if improved:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:03d} | train MSE: {train_mse:.6f} | val RMSE (log target): {val_rmse:.6f}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


def main():
    parser = argparse.ArgumentParser(description="House Prices - PyTorch MPS baseline")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data dir with train/test.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu|cuda|mps")
    parser.add_argument("--log_target", action="store_true", help="Model log1p(SalePrice) (recommended)")
    parser.add_argument("--no_log_target", dest="log_target", action="store_false")
    parser.set_defaults(log_target=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    assert train_path.exists() and test_path.exists(), "train.csv/test.csv not found under data_dir"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    target_col = "SalePrice"
    y = df_train[target_col].values.astype(np.float32)
    if args.log_target:
        y = np.log1p(y)

    preprocessor = build_preprocessor(df_train, target_col)
    X_all = preprocessor.fit_transform(df_train.drop(columns=[target_col]))
    X_all = X_all.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y, test_size=args.val_size, random_state=args.seed
    )

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    in_dim = X_all.shape[1]
    model = MLP(in_dim, hidden_dim=args.hidden_dim, layers=args.layers, dropout=args.dropout).to(device)

    train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=max(10, int(args.epochs * 0.15)),
    )

    # Predict on test and write submission
    X_test = preprocessor.transform(df_test).astype(np.float32)
    test_ds = TabularDataset(X_test, None)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()
    preds = []
    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)
            p = model(X).squeeze(1).detach().cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds, axis=0)
    if args.log_target:
        preds = np.expm1(preds)

    sub = pd.DataFrame({"Id": df_test["Id"], "SalePrice": preds})
    out_dir = Path("submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission_pytorch_mps.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
