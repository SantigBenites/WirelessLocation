from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import torch
from config import USE_TIMESTAMP
from utils import StandardScaler

def _extract_feats(sample: Dict, use_timestamp: bool) -> List[float]:
    feats = []
    for key_pair in [("friend1_rssi", "freind1_rssi"), ("friend2_rssi", "freind2_rssi"), ("friend3_rssi", "freind3_rssi")]:
        val = None
        for k in key_pair:
            if k in sample and sample[k] is not None:
                val = float(sample[k])
                break
        if val is None:
            raise ValueError(f"Missing RSSI for keys {key_pair}.")
        feats.append(val)
    if use_timestamp:
        feats.append(float(sample.get("timestamp", 0.0)))
    return feats

@torch.no_grad()
def predict_xy(model, scaler: StandardScaler, sample: Dict, use_timestamp: bool = USE_TIMESTAMP) -> Tuple[float, float]:
    device = next(model.parameters()).device
    feats = np.asarray([_extract_feats(sample, use_timestamp)], dtype=np.float32)
    X = scaler.transform(feats)
    pred = model(torch.from_numpy(X).to(device)).cpu().numpy()[0]
    return float(pred[0]), float(pred[1])
