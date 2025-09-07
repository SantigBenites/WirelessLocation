# validate_models_pt.py

import os
import math
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn

import sys
import os
from pathlib import Path
# go up two levels: validate_single_model/ -> validation/ -> (join to data_processing/…)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_ROOT / "data_processing" / "hyperparameter_tunning" / "CNN_hyperparameter_optimization"
sys.path.append(str(MODULE_DIR))

# ---- your project imports (adjust relative paths if needed)
from model_generation import GeneratedModel                                                             # :contentReference[oaicite:5]{index=5}
from data_processing import (get_dataset, split_combined_data,combine_arrays, get_feature_list)         # :contentReference[oaicite:6]{index=6}

all_collections = [
    "equilatero_grande_garage",
    "equilatero_grande_outdoor",
    "equilatero_medio_garage",
    "equilatero_medio_outdoor",
    "isosceles_grande_indoor",
    "isosceles_grande_outdoor",
    "isosceles_medio_outdoor",
    "obtusangulo_grande_outdoor",
    "obtusangulo_pequeno_outdoor",
    "reto_grande_garage",
    "reto_grande_indoor",
    "reto_grande_outdoor",
    "reto_medio_garage",
    "reto_medio_outdoor",
    "reto_n_quadrado_grande_indoor",
    "reto_n_quadrado_grande_outdoor",
    "reto_n_quadrado_pequeno_outdoor",
    "reto_pequeno_garage",
    "reto_pequeno_outdoor",
]

def group_by_location(collections, locations):
    return [name for name in collections if any(loc in name for loc in locations)]

# ----------------------------
# Load the trained Torch model
# ----------------------------
def _load_checkpointed_model(model_path: str) -> nn.Module:
    """
    Expects files saved by your trainer:
      torch.save({
          "state_dict": model.state_dict(),
          "arch_config": cfg,
          "input_size": X_train.shape[1],
          "output_size": y_val.shape[1],
      }, path)
    """
    ckpt = torch.load(model_path, map_location="cpu")                 # :contentReference[oaicite:8]{index=8}

    missing = [k for k in ["state_dict", "arch_config", "input_size", "output_size"] if k not in ckpt]
    if missing:
        raise RuntimeError(f"Checkpoint is missing keys: {missing}")

    model = GeneratedModel(
        input_size=int(ckpt["input_size"]),
        output_size=int(ckpt["output_size"]),
        architecture_config=ckpt["arch_config"],
    )                                                                 # :contentReference[oaicite:9]{index=9}
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ----------------------------
# Prediction + metrics helpers
# ----------------------------
@torch.no_grad()
def _batched_predict(model: nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    preds = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i + batch_size])
        out = model(xb)
        if isinstance(out, (tuple, list)):
            out = out[0]
        yb = out.detach().cpu().numpy().astype(np.float32)
        if yb.ndim != 2 or yb.shape[1] != 2:
            raise ValueError(f"Model output shape {yb.shape} — expected [N, 2].")
        preds.append(yb)
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 2), dtype=np.float32)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.shape != y_pred.shape or y_true.shape[1] != 2:
        raise ValueError(f"Expected y_true/y_pred with shape [N, 2], got {y_true.shape} vs {y_pred.shape}")
    err = y_pred - y_true
    ex, ey = err[:, 0], err[:, 1]
    dist = np.sqrt(ex**2 + ey**2)

    def rmse(a: np.ndarray) -> float:
        return float(math.sqrt(np.mean(a**2))) if a.size else float("nan")

    return {
        "n_samples": int(y_true.shape[0]),
        "mae_x": float(np.mean(np.abs(ex))) if ex.size else float("nan"),
        "mae_y": float(np.mean(np.abs(ey))) if ey.size else float("nan"),
        "rmse_x": rmse(ex),
        "rmse_y": rmse(ey),
        "mae_dist": float(np.mean(dist)) if dist.size else float("nan"),
        "rmse_dist": rmse(dist),
        "median_dist": float(np.median(dist)) if dist.size else float("nan"),
        "p75_dist": float(np.percentile(dist, 75)) if dist.size else float("nan"),
    }


# ------------------------------------
# Build validation sets (combined data)
# ------------------------------------
def build_validation_sets(db_name: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns:
      {
        "indoor":  (X, y),
        "outdoor": (X, y),
        "garage":  (X, y),
      }
    Uses your all_collections + group_by_location and combines *all* collections per group.
    """
    groups = {
        "indoor":  group_by_location(all_collections, ["indoor"]),    # :contentReference[oaicite:10]{index=10}
        "outdoor": group_by_location(all_collections, ["outdoor"]),
        "garage":  group_by_location(all_collections, ["garage"]),
    }

    feature_list = get_feature_list(db_name)                          # :contentReference[oaicite:11]{index=11}

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, cols in groups.items():
        if not cols:
            continue
        datasets = [get_dataset(c, db_name, feature_list) for c in cols]   # :contentReference[oaicite:12]{index=12}
        combined = combine_arrays(datasets)                                 # :contentReference[oaicite:13]{index=13}
        X, y = split_combined_data(combined, feature_list)                  # :contentReference[oaicite:14]{index=14}
        out[name] = (X.astype(np.float32, copy=False), y.astype(np.float32, copy=False))
        print(f"📡 {name}: {len(cols)} collections -> X{X.shape}, y{y.shape}")
    return out


# --------------------------------------------------------
# Public API: validate a single .pt across the 3 locations
# --------------------------------------------------------
def validate_pt_model_across_groups(
    model_path: str,
    db_name: str,
    groups_to_eval: Iterable[str] = ("indoor", "outdoor", "garage"),
    batch_size: int = 8192,
) -> Dict[str, Any]:
    """
    Loads your .pt, evaluates on indoor/outdoor/garage (sample-count weighted overall),
    and returns all metrics.
    """
    model = _load_checkpointed_model(model_path)                      # uses your ckpt format
    val_sets = build_validation_sets(db_name)

    per_group: Dict[str, Dict[str, float]] = {}
    total_n = 0
    weighted = {}

    for g in groups_to_eval:
        if g not in val_sets:
            continue
        Xg, yg = val_sets[g]
        if Xg.size == 0:
            continue
        y_pred = _batched_predict(model, Xg, batch_size=batch_size)
        m = _regression_metrics(yg, y_pred)
        per_group[g] = m

        n = m["n_samples"]
        total_n += n
        for k, v in m.items():
            if k == "n_samples":
                continue
            weighted[k] = weighted.get(k, 0.0) + v * n

    overall = {"n_samples": total_n}
    for k, s in weighted.items():
        overall[k] = s / total_n if total_n > 0 else float("nan")

    return {
        "model_path": model_path,
        "db_name": db_name,
        "overall": overall,
        "per_group": per_group,
    }
