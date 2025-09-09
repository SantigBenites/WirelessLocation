from __future__ import annotations
from typing import Tuple, Dict, Any
import os, time, json, hashlib, torch

from config import USE_TIMESTAMP, HIDDEN, DROPOUT
from model import MLPRegressor
from utils import StandardScaler

def _config_fingerprint(cfg: Dict[str, Any]) -> str:
    txt = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:10]

def save_model_checkpoint(model, scaler: StandardScaler, cfg: Dict[str, Any], cv_summary, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fp = _config_fingerprint(cfg)
    mean = (cv_summary or {}).get("mean_rmse", None)
    mean_str = f"{mean:.4f}" if isinstance(mean, (int, float)) else "NA"
    fname = f"mlp_cv-{mean_str}_{fp}_{ts}.pt"
    path = os.path.join(out_dir, fname)
    payload = {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler": scaler.to_dict(),
        "config": cfg,
        "cv_summary": cv_summary,
        "saved_at": ts,
    }
    torch.save(payload, path)
    json_path = path.replace(".pt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "cv_summary": cv_summary, "saved_at": ts}, f, indent=2)
    print(f"Saved checkpoint: {path}")
    return path

def load_model_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    use_ts = cfg.get("use_timestamp", USE_TIMESTAMP)
    in_dim = 3 + (1 if use_ts else 0)
    hidden = cfg.get("hidden", HIDDEN)
    dropout = cfg.get("dropout", DROPOUT)
    model = MLPRegressor(in_dim, hidden, dropout)
    model.load_state_dict(ckpt["model_state"])  # type: ignore
    scaler = StandardScaler.from_dict(ckpt["scaler"])  # type: ignore
    return model, scaler, cfg
