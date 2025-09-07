import os
import sys
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

"""
validate_models_simple.py

Validate all saved .pt models from a directory against four datasets
fetched from MongoDB: indoor, outdoor, garage, and all_datasets.

Configure variables in the section below.
"""

from pathlib import Path
import csv
import math
import os
import sys
from tqdm import tqdm
from typing import Dict, Tuple

import numpy as np
import torch

# ====== CONFIGURE HERE ======
MODEL_DIR = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/CNN_hyperparameter_optimization/model_storage_meters"                 # folder containing .pt bundles
DB_NAME   = "wifi_fingerprinting_data_meters"      # MongoDB database name
MONGO_URI = "mongodb://localhost:28910/"    # optional (informational)
DEVICE    = "cpu"                           # e.g., "cpu" or "cuda:0"
VAL_BATCH_SIZE = 2048
# ============================

import sys
import os
from pathlib import Path
# go up two levels: validate_single_model/ -> validation/ -> (join to data_processing/…)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_ROOT / "data_processing" / "hyperparameter_tunning" / "CNN_hyperparameter_optimization"
sys.path.append(str(MODULE_DIR))

# Project modules
from data_processing import get_dataset, combine_arrays, shuffle_array, split_combined_data, get_feature_list
from model_generation import GeneratedModel


all_collections = [
    #"equilatero_grande_garage",
    "equilatero_grande_outdoor",
    #"equilatero_medio_garage",
    #"equilatero_medio_outdoor",
    #"isosceles_grande_indoor",
    #"isosceles_grande_outdoor",
    #"isosceles_medio_outdoor",
    #"obtusangulo_grande_outdoor",
    #"obtusangulo_pequeno_outdoor",
    #"reto_grande_garage",
    #"reto_grande_indoor",
    #"reto_grande_outdoor",
    #"reto_medio_garage",
    #"reto_medio_outdoor",
    #"reto_n_quadrado_grande_indoor",
    #"reto_n_quadrado_grande_outdoor",
    #"reto_n_quadrado_pequeno_outdoor",
    #"reto_pequeno_garage",
    #"reto_pequeno_outdoor",
]


def group_by_location(collections, locations):
    return [name for name in collections if any(loc in name for loc in locations)]

def load_and_process_data(train_collections, db_name):
    # Resolve which features to use for this DB (preset name or explicit list)
    feature_list = get_feature_list(db_name)

    print(f"🧰 Database in use: {db_name}")
    # Uncomment to see the exact feature order:
    print("Features:", feature_list)

    # ---- Training data
    print(f"📡 Loading training datasets: {train_collections}")
    train_datasets = [get_dataset(name, db_name, feature_list) for name in train_collections]
    combined_train = combine_arrays(train_datasets)
    shuffled_train = shuffle_array(combined_train)
    X_train, y_train = split_combined_data(shuffled_train, feature_list)

    # ---- Validation data
    print("📡 Loading validation datasets: all collections")
    val_datasets = [get_dataset(name, db_name, feature_list) for name in all_collections]
    combined_val = combine_arrays(val_datasets)
    shuffled_val = shuffle_array(combined_val)
    X_val, y_val = split_combined_data(shuffled_val, feature_list)

    print(f"📊 Final shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, "
          f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    return X_train, y_train, X_val, y_val



def build_validation_sets(db_name: str, mongo_uri: str):
    os.environ.setdefault("MONGO_URI", mongo_uri)

    groups = {
        #"indoor": group_by_location(all_collections, ["indoor"]),
        "outdoor": group_by_location(all_collections, ["outdoor"]),
        #"garage": group_by_location(all_collections, ["garage"]),
        #"all_datasets": list(all_collections),
    }

    feature_list = get_feature_list(db_name)

    result = {}
    for group_name, collections in groups.items():
        print(f"📡 Loading validation group '{group_name}' with {len(collections)} collections...")
        datasets = [get_dataset(name, db_name, feature_list) for name in collections]
        #combined = combine_arrays(datasets)
        #shuffled = shuffle_array(combined)
        print(datasets)
        X, y = split_combined_data(datasets[0],feature_list)
        result[group_name] = (X.astype(np.float32), y.astype(np.float32))
        print(f"   -> {group_name}: X{X.shape}, y{y.shape}")
    return result


def load_model_bundle(pt_path: Path, map_location: str = "cpu"):
    bundle = torch.load(str(pt_path), map_location=map_location)
    required = {"state_dict", "arch_config", "input_size", "output_size"}
    if not required.issubset(bundle.keys()):
        raise ValueError(f"{pt_path.name} is missing required keys. Found: {list(bundle.keys())}")
    model = GeneratedModel(
        input_size=bundle["input_size"],
        output_size=bundle["output_size"],
        architecture_config=bundle["arch_config"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


def metrics(y_true: np.ndarray, y_pred: np.ndarray, batch_size: int | None = None):
    """Return both global metrics and a Lightning-like val_loss.

    - Global MSE/MAE/RMSE/R2 use the entire array at once.
    - val_loss emulates Lightning's epoch metric: mean of per-batch MSEs.
    """
    diff = y_pred - y_true

    # Global (over all samples at once)
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))
    ss_res = float(np.sum(np.square(diff)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true, axis=0, keepdims=True))))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Lightning-like: mean of batch losses (each batch loss is an MSE over that batch)
    if batch_size is None or batch_size <= 0:
        val_loss = mse  # fallback to global
    else:
        per_batch = []
        n = len(y_true)
        for i in range(0, n, batch_size):
            d = y_pred[i:i+batch_size] - y_true[i:i+batch_size]
            per_batch.append(float(np.mean(np.square(d))))
        val_loss = float(np.mean(per_batch))

    return {"val_loss": val_loss, "mse": mse, "mae": mae, "rmse": rmse, "r2": r2}



def evaluate_model(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device):
    with torch.no_grad():
        bs = VAL_BATCH_SIZE
        preds = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i+bs]).to(device)
            yb_pred = model(xb).cpu().numpy()
            preds.append(yb_pred)
        y_pred = np.vstack(preds).astype(np.float32)
    # Compute metrics; val_loss is Lightning-like mean of per-batch MSEs using the same bs
    return metrics(y, y_pred, batch_size=VAL_BATCH_SIZE)



def scan_models(model_dir: Path):
    pts = sorted(model_dir.glob("*.pt"))
    ckpts = sorted(model_dir.glob("*.ckpt"))
    if ckpts and not pts:
        print("⚠️ Found only .ckpt files. This validator expects .pt bundles. Please convert/resave your models.")
    elif ckpts and pts:
        print("ℹ️ Found both .pt and .ckpt files. Evaluating only .pt files.")
    return pts


# -------- helpers --------
def _set_single_thread():
    # Avoid N_procs × N_threads slowdown
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# Globals in each worker (set once per worker)
_VAL_SETS = None
_DEVICE = None

def _init_pool(val_sets, device_str):
    """Runs once per worker process."""
    global _VAL_SETS, _DEVICE
    _set_single_thread()
    _VAL_SETS = val_sets
    _DEVICE = device_str

def _evaluate_one_model(pt_path_str):
    """Evaluate a single model file across all val sets. Returns (rows, warnings)."""
    rows = []
    warns = []
    pt_path = Path(pt_path_str)
    pt_name = pt_path.name
    
    model, bundle = load_model_bundle(pt_path, map_location=_DEVICE)
    model.to(_DEVICE)

    try:
        in_dim = int(bundle["input_size"])
        out_dim = int(bundle["output_size"])
    except Exception as e:
        warns.append(f"⚠️ {pt_name}: missing/invalid input_size or output_size in bundle: {e}")
        return rows, warns

    for ds_name, (X, y) in _VAL_SETS.items():
        if X.shape[1] != in_dim or y.shape[1] != out_dim:
            warns.append(
                f"⚠️ Shape mismatch for {pt_name} on {ds_name}: expected ({in_dim}->{out_dim}), "
                f"got X:{X.shape[1]} y:{y.shape[1]}. Skipping."
            )
            continue
        try:
            # Ensure your evaluate_model uses torch.no_grad() internally if it's PyTorch
            result = evaluate_model(model, X, y, _DEVICE)
        except Exception as e:
            warns.append(f"⚠️ Evaluation failed for {pt_name} on {ds_name}: {e}")
            continue

        rows.append({
            "model_file": pt_name,
            "input_size": in_dim,
            "output_size": out_dim,
            "dataset": ds_name,
            **result,
        })

    return rows, warns


if __name__ == "__main__":


    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir.resolve()}")

    device = torch.device(DEVICE if torch.cuda.is_available() or not str(DEVICE).startswith("cuda") else "cpu")
    print(f"🖥️ Using device: {device}")

    val_sets = build_validation_sets(db_name=DB_NAME, mongo_uri=MONGO_URI)

    pt_files = scan_models(model_dir)
    if not pt_files:
        raise SystemExit(f"No .pt files found in {model_dir.resolve()}")

    fieldnames = ["model_file", "input_size", "output_size", "dataset", "mse", "mae", "rmse", "r2"]
    
    # ---- config ----
    device_str = str(device)         # e.g. "cpu"
    out_csv = Path("eval_results.csv")

    # Materialize the model list
    pt_file_list = [str(pt) for pt in pt_files]

    max_workers = min(len(pt_file_list), max(1, multiprocessing.cpu_count() - 1))

    all_rows = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_pool,
        initargs=(val_sets, device_str),
    ) as ex:
        futures = [ex.submit(_evaluate_one_model, p) for p in pt_file_list]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating models"):
            rows_i, warns_i = fut.result()
            for w in warns_i:
                print(w)
            all_rows.extend(rows_i)
            sys.stdout.flush()

    # ---- save to CSV (no extra deps) ----
    if all_rows:
        # Union of all keys to keep columns consistent
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"✅ Wrote {len(all_rows)} rows to {out_csv.resolve()}")
    else:
        print("⚠️ No results to write (all models failed or were skipped).")
