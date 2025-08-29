# grid.py
from __future__ import annotations
from typing import Dict, List, Any
from itertools import product

from config import GRID, TOP_N_SAVE, CHECKPOINT_DIR, USE_TIMESTAMP, PATIENCE, SEED
from cv import cross_validate_mlp
from train import fit_mlp
from checkpoint import save_model_checkpoint
from wandb_utils import init_run
from utils import resolve_scale 

def _grid_iter(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))

def grid_search_mlp(records: List[Dict], database_name: str) -> List[Dict[str, Any]]:
    if len(records) < 4:
        raise ValueError("Not enough samples for grid search.")

    s = resolve_scale(database_name)  # meters per unit

    combos = list(_grid_iter(GRID))
    print(f"Grid search over {len(combos)} configurations…")

    results: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(combos, start=1):
        cv_summary = cross_validate_mlp(
            records,
            database_name=database_name,
            log_to_wandb=False,
        )
        results.append({"config": cfg, "cv_summary": cv_summary})

    results.sort(key=lambda r: r["cv_summary"]["mean_rmse_m"])
    best = results[0]
    print(
        f"Best config: {best['config']} | "
        f"mean_rmse_m={best['cv_summary']['mean_rmse_m']:.4f} "
        f"± {best['cv_summary']['std_rmse_m']:.4f} | "
        f"mean_mae_m={best['cv_summary']['mean_mae_m']:.4f}"
    )

    # Log the best CV only (meters)
    best_cv_run = init_run(name=f"{database_name}_grid_best_cv",
                           config={**best["config"], "scale": s},
                           group=database_name)
    try:
        import wandb
        tbl = wandb.Table(columns=["fold", "rmse_m"])
        for i, r in enumerate(best["cv_summary"]["fold_rmse_m"], start=1):
            tbl.add_data(i, float(r))
        best_cv_run.log({
            "cv/mean_rmse_m": float(best["cv_summary"]["mean_rmse_m"]),
            "cv/std_rmse_m": float(best["cv_summary"]["std_rmse_m"]),
            "cv/mean_mae_m": float(best["cv_summary"]["mean_mae_m"]),
            "cv/fold_count": len(best["cv_summary"]["fold_rmse_m"]),
            "cv/folds_m": tbl,
        })
    finally:
        best_cv_run.finish()

    # Retrain best on full data; log holdout in meters
    retrain_run = init_run(name=f"{database_name}_grid-best-train",
                           config={**best["config"], "scale": s},
                           group=database_name)
    model, scaler, holdout_rmse_raw = fit_mlp(
        records,
        database_name,
        wandb_run=retrain_run,
        wandb_prefix="final/",
    )
    retrain_run.log({"final/holdout_rmse_m": float(holdout_rmse_raw * s)})
    retrain_run.finish()

    # Optionally save checkpoint (attaching meter-based CV summary)
    save_cfg = {**best["config"], "use_timestamp": USE_TIMESTAMP, "patience": PATIENCE, "seed": SEED, "scale": s}
    best["checkpoint_path"] = save_model_checkpoint(model, scaler, save_cfg, best["cv_summary"], CHECKPOINT_DIR)
    best["holdout_rmse_m"] = float(holdout_rmse_raw * s)

    return results
