from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from config import SEED, K_FOLDS, SHUFFLE_CV, USE_TIMESTAMP, BATCH_SIZE
from train import train_on_splits
from dataset import RssiLocationDataset
from wandb_utils import init_run


def _kfold_indices(n: int, k: int, shuffle: bool = True, seed: int = SEED) -> List[List[int]]:
    if k < 2:
        raise ValueError("k must be >= 2 for K-fold CV")
    k = min(k, n)
    idxs = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idxs)
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1
    folds: List[List[int]] = []
    start = 0
    for fs in fold_sizes:
        folds.append(idxs[start:start+fs].tolist())
        start += fs
    return folds


def cross_validate_mlp(
    records: List[Dict],
    database_name: str,
    log_to_wandb: bool = False,
    wandb_name: Optional[str] = None,
    wandb_group: Optional[str] = "cv",
) -> Dict[str, Any]:
    """
    Runs K-fold CV WITHOUT logging by default. If log_to_wandb=True, logs ONLY:
      - cv/mean_rmse, cv/std_rmse, cv/mean_mae
      - a single W&B Table with per-fold RMSE (no per-fold scalars)
    """
    n = len(records)
    if n < 4:
        raise ValueError("Not enough samples for cross-validation.")
    folds = _kfold_indices(n, K_FOLDS, SHUFFLE_CV, seed=SEED)

    rmse_list: List[float] = []
    mae_list: List[float] = []

    # No per-fold W&B logs — keep train_on_splits silent
    for val_idx in folds:
        train_idx = [j for j in range(n) if j not in val_idx]
        train_recs = [records[j] for j in train_idx]
        val_recs   = [records[j] for j in val_idx]

        model, train_ds, val_rmse = train_on_splits(
            train_recs,
            val_recs,
            database_name,
            wandb_run=None,            # <— silence fold-level logging
            wandb_prefix=None,
        )
        rmse_list.append(float(val_rmse))

        # Compute fold MAE by evaluating model on the fold's val set
        val_ds = RssiLocationDataset(
            val_recs, feature_keys=train_ds.feature_keys,
            use_timestamp=USE_TIMESTAMP, feature_scaler=train_ds.scaler
        )
        device = next(model.parameters()).device
        with torch.no_grad():
            preds = model(val_ds.X.to(device)).cpu()
        targs = val_ds.y
        mae = torch.mean(torch.abs(preds - targs)).item()
        mae_list.append(float(mae))

    rmse_arr = np.asarray(rmse_list, dtype=np.float64)
    mean_rmse = float(rmse_arr.mean())
    std_rmse = float(rmse_arr.std(ddof=1)) if len(rmse_arr) > 1 else 0.0
    mean_mae = float(np.asarray(mae_list, dtype=np.float64).mean())

    summary = {
        "fold_rmse": rmse_list,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "mean_mae": mean_mae,
    }

    if log_to_wandb:
        run = init_run(name=f"{wandb_name}_cv", config={"k_folds": len(folds)}, group=wandb_group)
        try:
            import wandb
            tbl = wandb.Table(columns=["fold", "rmse"])
            for i, r in enumerate(rmse_list, start=1):
                tbl.add_data(i, float(r))
            run.log({
                "cv/mean_rmse": mean_rmse,
                "cv/std_rmse": std_rmse,
                "cv/mean_mae": mean_mae,
                "cv/fold_count": len(rmse_list),
                "cv/folds": tbl,
            })
        finally:
            run.finish()

    return summary
