from __future__ import annotations
from typing import Dict, List, Any
from itertools import product

from config import (GRID, K_FOLDS, TOP_N_SAVE, CHECKPOINT_DIR, SHUFFLE_CV, USE_TIMESTAMP, PATIENCE, SEED,
                     HIDDEN, EPOCHS, LR, WEIGHT_DECAY, DROPOUT, BATCH_SIZE)
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
) -> List[Dict[str, Any]]:
    if len(records) < 4:
        raise ValueError("Not enough samples for grid search.")

    results: List[Dict[str, Any]] = []
    combos = list(_grid_iter(GRID))
    print(f"Grid search over {len(combos)} configurations…")

    for idx, cfg in enumerate(combos, start=1):
        print("" + "="*80)
        print(f"Config {idx}/{len(combos)}: {cfg}")
        run = init_run(name=f"grid-{idx}", config=cfg, group="grid")
        cv_summary = cross_validate_mlp(
            records,
            wandb_name=f"grid-{idx}-cv",
            wandb_group="grid/cv",
        )
        run.log({"cv/mean_rmse": cv_summary["mean_rmse"], "cv/std_rmse": cv_summary["std_rmse"]})
        run.finish()
        res = {"config": cfg, "cv_summary": cv_summary}
        results.append(res)

    results.sort(key=lambda r: r["cv_summary"]["mean_rmse"])  # type: ignore

    print("=== Grid search ranking (top to bottom) ===")
    for rank, r in enumerate(results, start=1):
        mean = r["cv_summary"]["mean_rmse"]
        std  = r["cv_summary"]["std_rmse"]
        print(f"#{rank:02d} mean={mean:.4f} ± {std:.4f} | cfg={r['config']}")

    for rank, r in enumerate(results[:max(1, TOP_N_SAVE)], start=1):
        cfg = r["config"]
        print(f"Retraining top-{rank} config on full dataset and saving…")
        retrain_run = init_run(name=f"retrain-top{rank}", config=cfg, group="grid/retrain")
        model, scaler, holdout_rmse = fit_mlp(
            records,
            wandb_run=retrain_run,
            wandb_prefix="final/",
        )
        save_cfg = {**cfg, "use_timestamp": USE_TIMESTAMP, "patience": PATIENCE, "seed": SEED}
        r["checkpoint_path"] = save_model_checkpoint(model, scaler, save_cfg, r["cv_summary"], CHECKPOINT_DIR)
        r["holdout_rmse"] = holdout_rmse
        retrain_run.log({"final/holdout_rmse": holdout_rmse})
        retrain_run.finish()

    return results
