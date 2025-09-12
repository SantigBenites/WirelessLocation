import os, glob, csv
from typing import List, Optional, Tuple

import numpy as np
import torch

from CNN_model.model_generation import GeneratedModel as CNN_GeneratedModel
from NN_model.model_generation import GeneratedModel as NN_GeneratedModel
from MLP_model.model_generation import GeneratedModel as MLP_GeneratedModel
from utils.data_processing import (
    get_feature_list, get_dataset, combine_arrays, split_combined_data
)

def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)

def _batch_iter(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

def _load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    for k in ["state_dict", "arch_config", "input_size", "output_size"]:
        if k not in ckpt:
            raise ValueError(f"{os.path.basename(path)} missing key '{k}'")
    return ckpt

def _load_val_xy(collections: List[str], db_name: str, feature_names: Optional[List[str]] = None):
    feats = feature_names if feature_names is not None else get_feature_list(db_name)
    arrays = [get_dataset(c, db_name, feats) for c in collections]
    combo = combine_arrays(arrays)
    X, y = split_combined_data(combo, feats)
    return X, y, feats

def evaluate_model_on_collections(
    model_path: str,
    val_collections: List[str],
    db_name: str,
    batch_size: int,
    device: Optional[str],
    feat_names: List[str]
) -> Tuple[float, float, float, int]:
    """Returns (mae, mse, rmse, n_samples)."""
    ckpt = _load_ckpt(model_path)
    arch = ckpt["arch_config"]
    input_size = int(ckpt["input_size"])
    output_size = int(ckpt["output_size"])

    X_val_np, y_val_np, feats_used = _load_val_xy(val_collections, db_name, feature_names=feat_names)

    if X_val_np.shape[1] != input_size:
        raise RuntimeError(
            f"[{os.path.basename(model_path)}] Feature width mismatch: "
            f"X has {X_val_np.shape[1]}, checkpoint expects {input_size}. "
            f"Use a matching 'db_name' or save 'feature_names' in the checkpoint."
        )
    
    if "CNN" in model_path:
        model = CNN_GeneratedModel(input_size=input_size, output_size=output_size, architecture_config=arch)
    elif "MLP" in model_path:
        model = MLP_GeneratedModel(input_size=input_size, output_size=output_size, architecture_config=arch)
    elif "NN" in model_path:
        model = NN_GeneratedModel(input_size=input_size, output_size=output_size, architecture_config=arch)
    else:
        raise Exception
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[WARN] Partial state load for {os.path.basename(model_path)}. "
              f"missing={missing[:4]} unexpected={unexpected[:4]}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_val = _to_tensor(X_val_np).to(device)
    y_val = _to_tensor(y_val_np).to(device)

    mae_sum = 0.0
    mse_sum = 0.0
    count = 0

    with torch.no_grad():
        for xb, yb in _batch_iter(X_val, y_val, batch_size):
            preds = model(xb)
            batch_mae = torch.mean(torch.abs(preds - yb))
            batch_mse = torch.mean((preds - yb) ** 2)
            mae_sum += batch_mae.item() * xb.shape[0]
            mse_sum += batch_mse.item() * xb.shape[0]
            count += xb.shape[0]

    mae = mae_sum / max(count, 1)
    mse = mse_sum / max(count, 1)
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse, count