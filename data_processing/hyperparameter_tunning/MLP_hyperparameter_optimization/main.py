#!/usr/bin/env python3
from __future__ import annotations
import random

from config import (
    USE_TIMESTAMP, DO_CV, K_FOLDS, SHUFFLE_CV, DO_GRID_SEARCH, GRID, TOP_N_SAVE, CHECKPOINT_DIR,PATIENCE,SEED
)
from data_processing import load_and_process_data
from cv import cross_validate_mlp
from grid import grid_search_mlp
from train import fit_mlp
from inference import predict_xy
from config import *

random.seed(SEED)

def main():
    records = load_and_process_data(["equilatero_grande_garage","equilatero_grande_outdoor"],"wifi_fingerprinting_data_meters")

    cross_validate_mlp(records)

    results = grid_search_mlp(records)
    best_cfg = results[0]["config"] if results else {}

    # 4) Train a final model with the best config (or defaults) and run a sample prediction
    cfg = best_cfg
    print(f"Training final model with config: {cfg if cfg else '[defaults]'}")
    model, scaler, val_rmse = fit_mlp(
        records,
        hidden=cfg.get("hidden", HIDDEN) if cfg else HIDDEN,
        epochs=cfg.get("epochs", EPOCHS) if cfg else EPOCHS,
        lr=cfg.get("lr", LR) if cfg else LR,
        weight_decay=cfg.get("weight_decay", WEIGHT_DECAY) if cfg else WEIGHT_DECAY,
        dropout=cfg.get("dropout", DROPOUT) if cfg else DROPOUT,
        patience=PATIENCE,
        use_timestamp=USE_TIMESTAMP,
        batch_size=cfg.get("batch_size", BATCH_SIZE) if cfg else BATCH_SIZE,
    )
    print(f"Validation RMSE (holdout): {val_rmse:.4f}")

    sample = {"freind1_rssi": -68, "freind2_rssi": -85, "freind3_rssi": -82}
    x, y = predict_xy(model, scaler, sample, use_timestamp=USE_TIMESTAMP)
    print({"pred_location_x": x, "pred_location_y": y})

if __name__ == "__main__":
    main()
