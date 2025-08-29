import os
os.environ.setdefault("WANDB_START_METHOD", "thread")
# (optional noise reduction)
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")


import ray
import random
from typing import Dict, List, Any, Tuple
import torch

from config import USE_TIMESTAMP, HIDDEN, EPOCHS, LR, WEIGHT_DECAY, DROPOUT, BATCH_SIZE, PATIENCE, SEED
from data_processing import load_and_process_data
from cv import cross_validate_mlp
from grid import grid_search_mlp
from train import fit_mlp, rmse
from inference import predict_xy
from dataset import RssiLocationDataset, _features_for_database
from wandb_utils import init_run


@ray.remote(num_gpus=1)
def ray_function(model_name:str, collections: List[str], database: str, seed_offset: int = 0) -> Dict[str, Any]:
    """
    A Ray *task* that executes your end-to-end pipeline on a single GPU.
    Ray sets CUDA_VISIBLE_DEVICES for this worker, so torch.cuda() will pick the correct GPU.
    """
    # Make each worker deterministic-but-different
    import numpy as np, torch
    random.seed(SEED + seed_offset)
    np.random.seed(SEED + seed_offset)
    torch.manual_seed(SEED + seed_offset)
    torch.cuda.manual_seed_all(SEED + seed_offset)


    # 1) Load + preprocess (get both train and a true holdout validation set)
    train_data, validation_data = load_and_process_data(collections, database)

    # 2) CV (optional – comment out if you don’t want per-run CV)
    #cv_summary = None
    #try:
    #    cv_summary = cross_validate_mlp(train_data)
    #except TypeError as e:
    #    cv_summary = {"error": f"cross_validate_mlp failed: {e}"}

    # 3) Grid search (optional)
    best_cfg: Dict[str, Any] = {}
    try:
        results = grid_search_mlp(train_data,database)
        best_cfg = results[0]["config"] if results else {}
    except Exception as e:
        best_cfg = {}
        results = []
        print(f"[worker] grid_search_mlp failed: {e}")

    # 4) W&B run for final training + holdout eval
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
    }
    run = init_run(
        name=model_name,
        config=run_cfg,
        group=database,
        tags=["final", "ray", "gpu"],
    )

    # 5) Train final model (logs epochs to W&B via fit_mlp)
    model, scaler, split_val_rmse = fit_mlp(
        train_data,
        database,
        wandb_run=run,
        wandb_prefix="final/",
    )
    run.log({"final/holdout_from_split_rmse": float(split_val_rmse)})

    # 6) HOLDOUT EVAL on the *true* validation set (not a single sample)
    device = next(model.parameters()).device
    try:
        feature_keys = _features_for_database(database)
        val_ds = RssiLocationDataset(validation_data, feature_keys, use_timestamp=USE_TIMESTAMP, feature_scaler=scaler)
        with torch.no_grad():
            preds = model(val_ds.X.to(device)).cpu()
        targs = val_ds.y

        # Metrics
        holdout_rmse_xy = rmse(preds, targs)  # RMSE across both x,y targets
        diffs = preds - targs
        dists = torch.linalg.norm(diffs, dim=1)  # Euclidean error per point
        holdout_mean_dist = float(dists.mean().item())
        # "MAE" here: mean absolute error per-axis averaged across axes
        mae_xy = float(torch.mean(torch.abs(preds - targs)).item())
        num_samples = int(len(val_ds))

        # Log scalars
        run.log({
            "holdout/rmse_xy": float(holdout_rmse_xy),
            "holdout/euclidean_error": holdout_mean_dist,
            "holdout/mae_xy": mae_xy,
            "holdout/num_samples": num_samples,
        })

        # Log a W&B TABLE with the requested columns
        try:
            import wandb  # only needed to construct a Table
            run_name = model_name
            table = wandb.Table(columns=["run_name", "rmse", "euclidean_error", "mae", "num_samples"])
            table.add_data(run_name, float(holdout_rmse_xy), holdout_mean_dist, mae_xy, num_samples)
            run.log({"holdout/summary_table": table})
        except Exception as we:
            # If wandb isn't available or Table fails, continue without crashing
            run.log({"holdout/table_error": str(we)})

    except ValueError as ve:
        run.log({"holdout/error": str(ve)})

    run.finish()

    return {
        "collections": collections,
        "database": database,
        "best_cfg": best_cfg,
        "split_val_rmse": float(split_val_rmse),
    }