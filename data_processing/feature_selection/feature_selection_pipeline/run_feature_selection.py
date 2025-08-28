from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import ray

from configs import Config
from user_config import CFG
from data_loader import load_collections_to_df, build_xy_groups
from feature_selection import fold_feature_selection, aggregate_across_folds
from utils import ensure_dir, atomic_write_json
from sklearn.model_selection import GroupKFold


def _init_ray(cfg: Config) -> None:
    if ray.is_initialized():
        return
    ray.init(address=cfg.ray.address or None, local_mode=cfg.ray.local_mode, ignore_reinit_error=True)


def main() -> None:
    cfg: Config = CFG

    ensure_dir(cfg.output.outdir)

    # 1) Load data
    df = load_collections_to_df(cfg)
    X, y, groups, feat_cols = build_xy_groups(cfg, df)

    # 2) GroupKFold splits
    gkf = GroupKFold(n_splits=cfg.cv.n_splits)
    splits = list(gkf.split(X, y, groups=groups))

    # 3) Ray parallel fold jobs
    _init_ray(cfg)

    @ray.remote(num_cpus=cfg.ray.num_cpus_per_task, num_gpus=cfg.ray.num_gpus_per_task)
    def run_one_fold(train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
        return fold_feature_selection(cfg, X, y, train_idx, val_idx)

    futures = [run_one_fold.remote(train_idx, val_idx) for train_idx, val_idx in splits]
    fold_results: List[dict] = ray.get(futures)

    # 4) Aggregate
    agg = aggregate_across_folds(fold_results, feat_cols)

    # 5) Final selection by frequency threshold
    freq = agg["frequency"]
    keep = freq[freq >= cfg.selection.min_fold_fraction].index.tolist()

    # 6) Persist outputs
    out = Path(cfg.output.outdir)

    # Per-artifact CSVs
    pd.DataFrame({"feature": freq.index, "selected_frequency": freq.values}).to_csv(out / "feature_frequencies.csv", index=False)
    agg["mi_mean"].rename("mi").reset_index(name=["feature"]).to_csv(out / "mi_scores.csv", index=False)
    agg["lasso_mean"].rename("lasso_importance").reset_index(name=["feature"]).to_csv(out / "lasso_importance.csv", index=False)
    agg["rfe_rank_mean"].rename("rfe_rank").reset_index(name=["feature"]).to_csv(out / "rfe_rank.csv", index=False)

    # Fold metrics
    metrics = {f"fold_{i}": fr["metrics"] for i, fr in enumerate(fold_results)}

    atomic_write_json(str(out / "cv_metrics.json"), metrics)
    atomic_write_json(str(out / "selected_features.json"), {"selected_features": keep})

    # Also keep a compact run summary
    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "targets": list(y.columns),
        "final_selected_count": len(keep),
        "min_fold_fraction": cfg.selection.min_fold_fraction,
        "n_splits": cfg.cv.n_splits,
    }
    atomic_write_json(str(out / "run_summary.json"), summary)

    print("=== Run summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {out}")


if __name__ == "__main__":
    main()