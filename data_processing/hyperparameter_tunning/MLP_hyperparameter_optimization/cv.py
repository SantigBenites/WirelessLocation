from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from config import SEED, K_FOLDS, SHUFFLE_CV, USE_TIMESTAMP, BATCH_SIZE
from train import train_on_splits, rmse
from dataset import RssiLocationDataset, _features_for_database 
from wandb_utils import init_run
from utils import resolve_scale


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

    rmse_list_m: List[float] = []
    mae_list_m:  List[float] = []

    scale = float(resolve_scale(database_name))
    feature_keys = _features_for_database(database_name)

    # No per-fold W&B logs — keep train_on_splits silent
    for val_idx in folds:
        train_idx = [j for j in range(n) if j not in val_idx]
        train_recs = [records[j] for j in train_idx]
        val_recs   = [records[j] for j in val_idx]

        model, scaler, val_rmse = train_on_splits(
            train_recs,
            val_recs,
            database_name,
            wandb_run=None,            # <— silence fold-level logging
            wandb_prefix=None,
        )
        
        val_ds = RssiLocationDataset(
            val_recs,
            feature_keys=feature_keys,
            use_timestamp=USE_TIMESTAMP,
            feature_scaler=scaler,
        )

        # Compute fold MAE by evaluating model on the fold's val set
        val_ds = RssiLocationDataset(
            val_recs,
            feature_keys=feature_keys,
            use_timestamp=USE_TIMESTAMP,
            feature_scaler=scaler,
        )
        device = next(model.parameters()).device
        with torch.no_grad():
            preds = model(val_ds.X.to(device)).cpu()
        targs = val_ds.y



        preds_m, targs_m = preds * scale, targs * scale
        rmse_m = rmse(preds_m, targs_m)
        mae_m  = torch.mean(torch.abs(preds_m - targs_m)).item()

        rmse_list_m.append(float(rmse_m))
        mae_list_m.append(float(mae_m))

    rmse_arr = np.asarray(rmse_list_m, dtype=np.float64)
    mean_rmse_m = float(rmse_arr.mean())
    std_rmse_m  = float(rmse_arr.std(ddof=1)) if len(rmse_arr) > 1 else 0.0
    mean_mae_m  = float(np.asarray(mae_list_m, dtype=np.float64).mean())

    summary = {
        "fold_rmse_m": rmse_list_m,
        "mean_rmse_m": mean_rmse_m,
        "std_rmse_m": std_rmse_m,
        "mean_mae_m": mean_mae_m,
    }

    if log_to_wandb:
        run = init_run(name=f"{wandb_name}_cv", config={"k_folds": len(folds)}, group=wandb_group)
        try:
            import wandb
            tbl = wandb.Table(columns=["fold", "rmse"])
            for i, r in enumerate(rmse_list_m, start=1):
                tbl.add_data(i, float(r))
            run.log({
                "cv/mean_rmse_m": mean_rmse_m,
                "cv/std_rmse_m": std_rmse_m,
                "cv/mean_mae_m": mean_mae_m,
                "cv/fold_count": len(rmse_list_m),
                "cv/folds_m": tbl,
            })
        finally:
            run.finish()

    return summary
