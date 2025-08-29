from __future__ import annotations
from typing import Dict, List, Any
import numpy as np

from config import (SEED, K_FOLDS, SHUFFLE_CV, HIDDEN, EPOCHS, LR, WEIGHT_DECAY, DROPOUT, PATIENCE, USE_TIMESTAMP, BATCH_SIZE)
from train import train_on_splits
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
    wandb_name: str | None = None,
    wandb_group: str | None = "cv",
) -> Dict[str, Any]:
    n = len(records)
    if n < 4:
        raise ValueError("Not enough samples for cross-validation.")
    folds = _kfold_indices(n, K_FOLDS, SHUFFLE_CV, seed=SEED)
    rmse_list: List[float] = []

    cfg = {
        "hidden": HIDDEN, "epochs": EPOCHS, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
        "patience": PATIENCE, "use_timestamp": USE_TIMESTAMP,
        "batch_size": BATCH_SIZE, "k_folds": K_FOLDS, "shuffle": SHUFFLE_CV,
    }
    run = init_run(name=wandb_name or "cv-run", config=cfg, group=wandb_group)

    print(f"Starting {len(folds)}-fold cross-validation on {n} samples…")
    for i, val_idx in enumerate(folds, start=1):
        train_idx = [j for j in range(n) if j not in val_idx]
        train_recs = [records[j] for j in train_idx]
        val_recs   = [records[j] for j in val_idx]
        print(f"Fold {i}/{len(folds)} | train={len(train_recs)} | val={len(val_recs)}")
        model, train_ds, val_rmse = train_on_splits(
            train_recs, val_recs,
            HIDDEN, EPOCHS, LR, WEIGHT_DECAY,
            DROPOUT, PATIENCE, USE_TIMESTAMP,
            BATCH_SIZE,
            wandb_run=run,
            wandb_prefix=f"fold{i}/",
        )
        rmse_list.append(val_rmse)
        run.log({f"fold{i}/val_rmse_final": val_rmse, "fold": i})

    rmse_arr = np.asarray(rmse_list, dtype=np.float64)
    mean_rmse = float(rmse_arr.mean())
    std_rmse = float(rmse_arr.std(ddof=1)) if len(rmse_arr) > 1 else 0.0

    print("CV summary:")
    for i, r in enumerate(rmse_list, start=1):
        print(f"  Fold {i}: RMSE={r:.4f}")
        run.log({f"fold{i}/val_rmse": r})
    print(f"  Mean RMSE={mean_rmse:.4f} ± {std_rmse:.4f} (std)")

    run.log({"cv/mean_rmse": mean_rmse, "cv/std_rmse": std_rmse})
    run.finish()

    return {"fold_rmse": rmse_list, "mean_rmse": mean_rmse, "std_rmse": std_rmse}
