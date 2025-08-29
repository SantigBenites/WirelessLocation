# gpu_fucntion.py
import os
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")

import ray, random, torch, numpy as np
from typing import Dict, List, Any

from config import USE_TIMESTAMP, HIDDEN, EPOCHS, LR, WEIGHT_DECAY, DROPOUT, BATCH_SIZE, PATIENCE, SEED
from data_processing import load_and_process_data
from grid import grid_search_mlp
from train import fit_mlp, rmse
from dataset import RssiLocationDataset, _features_for_database
from wandb_utils import init_run
from utils import resolve_scale 

@ray.remote(num_gpus=1)
def ray_function(model_name: str, collections: List[str], database: str, seed_offset: int = 0) -> Dict[str, Any]:
    random.seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)
    torch.manual_seed(SEED + seed_offset)
    torch.cuda.manual_seed_all(SEED + seed_offset)

    s = resolve_scale(database)  # meters per unit

    train_data, validation_data = load_and_process_data(collections, database)

    # Grid search (now CV in meters internally)
    try:
        results = grid_search_mlp(train_data, database)
        best_cfg = results[0]["config"] if results else {}
    except Exception as e:
        best_cfg, results = {}, []
        print(f"[worker] grid_search_mlp failed: {e}")

    run_cfg = {
        "hidden": best_cfg.get("hidden", HIDDEN) if best_cfg else HIDDEN,
        "epochs": best_cfg.get("epochs", EPOCHS) if best_cfg else EPOCHS,
        "lr": best_cfg.get("lr", LR) if best_cfg else LR,
        "weight_decay": best_cfg.get("weight_decay", WEIGHT_DECAY) if best_cfg else WEIGHT_DECAY,
        "dropout": best_cfg.get("dropout", DROPOUT) if best_cfg else DROPOUT,
        "batch_size": best_cfg.get("batch_size", BATCH_SIZE) if best_cfg else BATCH_SIZE,
        "use_timestamp": USE_TIMESTAMP,
        "patience": PATIENCE,
        "collections": collections,
        "database": database,
        "seed": SEED + seed_offset,
        "scale": s,  # visible in W&B config
    }
    run = init_run(name=model_name, config=run_cfg, group=database, tags=["final", "ray", "gpu"])

    # Final train (epoch val_rmse logged in meters via metric_scale)
    model, scaler, split_val_rmse_raw = fit_mlp(
        train_data, database,
        wandb_run=run, wandb_prefix="final/"
    )
    run.log({"final/holdout_from_split_rmse_m": float(split_val_rmse_raw * s)})

    # Holdout on true validation set — meters only
    device = next(model.parameters()).device
    try:
        feature_keys = _features_for_database(database)
        val_ds = RssiLocationDataset(validation_data, feature_keys, use_timestamp=USE_TIMESTAMP, feature_scaler=scaler)
        with torch.no_grad():
            preds = model(val_ds.X.to(device)).cpu()
        targs = val_ds.y

        preds_m, targs_m = preds * s, targs * s
        holdout_rmse_m = rmse(preds_m, targs_m)
        dists_m = torch.linalg.norm(preds_m - targs_m, dim=1)
        euclid_mean_m = float(dists_m.mean().item())
        mae_m = float(torch.mean(torch.abs(preds_m - targs_m)).item())
        num_samples = int(len(val_ds))

        run.log({
            "holdout/rmse_m": float(holdout_rmse_m),
            "holdout/euclidean_error_m": euclid_mean_m,
            "holdout/mae_m": mae_m,
            "holdout/num_samples": num_samples,
        })

        # Summary table (meters)
        try:
            import wandb
            table = wandb.Table(columns=["run_name", "rmse_m", "euclidean_error_m", "mae_m", "num_samples"])
            table.add_data(model_name, float(holdout_rmse_m), euclid_mean_m, mae_m, num_samples)
            run.log({"holdout/summary_table_m": table})
        except Exception as we:
            run.log({"holdout/table_error": str(we)})
    except ValueError as ve:
        run.log({"holdout/error": str(ve)})

    run.finish()
    return {
        "collections": collections,
        "database": database,
        "best_cfg": best_cfg,
        "split_val_rmse_m": float(split_val_rmse_raw * s),
    }
