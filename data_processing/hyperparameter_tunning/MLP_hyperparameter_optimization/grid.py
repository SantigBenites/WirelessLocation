from __future__ import annotations
from typing import Dict, List, Any
from itertools import product

from config import GRID, TOP_N_SAVE, CHECKPOINT_DIR, USE_TIMESTAMP, PATIENCE, SEED
from cv import cross_validate_mlp
from train import fit_mlp
from checkpoint import save_model_checkpoint
from wandb_utils import init_run


def _grid_iter(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def grid_search_mlp(
    records: List[Dict],
    database_name: str,
) -> List[Dict[str, Any]]:
    """
    Runs CV for all configs WITHOUT logging, ranks them, then:
      - logs a single CV summary for the BEST config
      - runs a full retrain on ALL data for the BEST config and logs training metrics
    """
    if len(records) < 4:
        raise ValueError("Not enough samples for grid search.")

    combos = list(_grid_iter(GRID))
    print(f"Grid search over {len(combos)} configurations…")

    results: List[Dict[str, Any]] = []
    for idx, cfg in enumerate(combos, start=1):
        # Evaluate silently (no W&B logging here)
        cv_summary = cross_validate_mlp(
            records,
            database_name=database_name,
            log_to_wandb=False,
            wandb_name=database_name,
            wandb_group=database_name,
        )
        results.append({"config": cfg, "cv_summary": cv_summary})

    # Rank by mean RMSE
    results.sort(key=lambda r: r["cv_summary"]["mean_rmse"])
    best = results[0]
    print(f"Best config: {best['config']} | mean_rmse={best['cv_summary']['mean_rmse']:.4f} "
          f"± {best['cv_summary']['std_rmse']:.4f} | mean_mae={best['cv_summary']['mean_mae']:.4f}")

    # Log a single CV summary for the BEST config
    best_cv_run = init_run(name=f"{database_name}_grid_best_cv", config=best["config"], group=database_name)
    # Only aggregated metrics + a compact table for fold RMSEs
    best_cv = best["cv_summary"]
    try:
        import wandb
        tbl = wandb.Table(columns=["fold", "rmse"])
        for i, r in enumerate(best_cv["fold_rmse"], start=1):
            tbl.add_data(i, float(r))
        best_cv_run.log({
            "cv/mean_rmse": float(best_cv["mean_rmse"]),
            "cv/std_rmse": float(best_cv["std_rmse"]),
            "cv/mean_mae": float(best_cv["mean_mae"]),
            "cv/fold_count": len(best_cv["fold_rmse"]),
            "cv/folds": tbl,
        })
    finally:
        best_cv_run.finish()

    # Retrain top-1 config on full dataset and save; this run logs ALL training metrics
    retrain_run = init_run(name=f"{database_name}_grid-best-train", config=best["config"], group=database_name)
    model, scaler, holdout_rmse = fit_mlp(
        records,
        database_name,
        wandb_run=retrain_run,
        wandb_prefix="final/",
    )
    # Save checkpoint + attach CV summary for provenance
    save_cfg = {**best["config"], "use_timestamp": USE_TIMESTAMP, "patience": PATIENCE, "seed": SEED}
    best["checkpoint_path"] = save_model_checkpoint(model, scaler, save_cfg, best["cv_summary"], CHECKPOINT_DIR)
    best["holdout_rmse"] = holdout_rmse
    retrain_run.log({"final/holdout_rmse": float(holdout_rmse)})
    retrain_run.finish()

    # Optionally keep TOP_N_SAVE behavior — here we already only retrained best (top-1)
    return results
