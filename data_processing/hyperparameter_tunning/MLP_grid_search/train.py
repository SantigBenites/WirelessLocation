from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import torch
from torch import nn

from config import (
    USE_TIMESTAMP, BATCH_SIZE, PATIENCE, HIDDEN, EPOCHS, LR, WEIGHT_DECAY, DROPOUT,
)
from dataset import build_loaders
from model import MLPRegressor
from utils import StandardScaler, resolve_scale

@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(nn.functional.mse_loss(pred, target)).item()

def train_on_splits(
    train_recs: List[Dict],
    val_recs: List[Dict],
    database :str ,
    wandb_run: Any = None,
    wandb_prefix: str = "",
) -> Tuple[nn.Module, StandardScaler, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_ds = build_loaders(train_recs, val_recs, database, USE_TIMESTAMP, BATCH_SIZE)

    in_dim = train_ds.X.shape[1]
    model = MLPRegressor(in_dim, HIDDEN, DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr = LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=PATIENCE, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * Xb.size(0)

        model.eval()
        val_loss = 0.0
        preds, targs = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                p = model(Xb)
                val_loss += criterion(p, yb).item() * Xb.size(0)
                preds.append(p.cpu())
                targs.append(yb.cpu())
        train_loss = total / len(train_ds)
        val_loss = val_loss / len(val_loader.dataset)
        preds = torch.cat(preds, dim=0)
        targs = torch.cat(targs, dim=0)
        val_rmse = rmse(preds, targs)
        val_rmse_m = val_rmse * float(resolve_scale(database))
        sched.step(val_loss)

        if wandb_run is not None:
            wandb_run.log({
                f"{wandb_prefix}epoch": epoch,
                f"{wandb_prefix}train_loss": train_loss,
                f"{wandb_prefix}val_loss": val_loss,
                f"{wandb_prefix}val_rmse": val_rmse,
                f"{wandb_prefix}val_rmse_m": val_rmse_m,
                f"{wandb_prefix}lr": opt.param_groups[0]["lr"],
            }, step=epoch)

        #print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_RMSE={val_rmse:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_ds.scaler, val_rmse

def split_train_val(records: List[Dict], val_ratio: float = 0.2):
    import random
    idxs = list(range(len(records)))
    random.shuffle(idxs)
    cut = int(len(idxs) * (1 - val_ratio))
    train_idx = set(idxs[:cut])
    train, val = [], []
    for i, r in enumerate(records):
        (train if i in train_idx else val).append(r)
    return train, val

def fit_mlp(
    records: List[Dict],
    database_name: str,
    wandb_run: Any = None,
    wandb_prefix: str = "",
):
    if len(records) < 4:
        raise ValueError("Need at least a handful of samples to train. Provide more records.")
    train_recs, val_recs = split_train_val(records, 0.2)
    model, scaler, val_rmse = train_on_splits(
        train_recs, 
        val_recs,
        database_name,
        wandb_run=wandb_run,
        wandb_prefix=wandb_prefix,
    )
    return model, scaler, val_rmse
