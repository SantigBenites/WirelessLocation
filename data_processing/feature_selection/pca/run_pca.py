from __future__ import annotations
import os, json
import pandas as pd
import numpy as np
import ray
from configs import MongoConfig, PCAConfig
from data_loader import load_all_collections
from pca_pipeline import run_pca

def main() -> None:
    # ---- Configure here (or edit configs.py defaults) ----
    mongo_cfg = MongoConfig(
        uri="mongodb://localhost:27017",
        db_name="wifi_fingerprinting_data_extra_features",
        # comment any you don't have; all will be merged
        collections=[
            "equilatero_grande_garage",
            "equilatero_grande_outdoor",
            "equilatero_medio_garage",
            "equilatero_medio_outdoor",
            "isosceles_grande_outdoor",
            "isosceles_medio_outdoor",
            "obtusangulo_grande_outdoor",
            "obtusangulo_pequeno_outdoor",
            "reto_grande_garage",
            "reto_grande_outdoor",
            "reto_medio_garage",
            "reto_medio_outdoor",
            "reto_n_quadrado_grande_outdoor",
            "reto_n_quadrado_pequeno_outdoor",
            "reto_pequeno_garage",
            "reto_pequeno_outdoor",
        ],
        query_filter={},
        sample=None,          # e.g., 200_000 for a test run
        batch_size=50_000,
    )
    pca_cfg = PCAConfig(
        drop_columns=["_id", "timestamp", "location_x", "location_y"],
        scale=True,
        whiten=False,
        random_state=42,
        variance_threshold=0.95,
        max_components=None,
        feature_selection_top_k=20,
        save_projection=True,
        output_dir="outputs/pca_feature_selection",
        use_cuml_if_available=True,
        ray_address=None,      # "auto" to connect to existing cluster
        num_reader_workers=8,
        use_gpu_for_pca=True,
    )
    # ------------------------------------------------------

    if pca_cfg.ray_address:
        ray.init(address=pca_cfg.ray_address, ignore_reinit_error=True)
    else:
        # local Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    print("Loading data from MongoDB (all collections merged)...")
    df = load_all_collections(mongo_cfg)
    if df.empty:
        raise RuntimeError("No data loaded. Check MongoDB URI, DB name, and collection list.")

    # Optional downsample for quick runs
    if mongo_cfg.sample is not None and mongo_cfg.sample < len(df):
        df = df.sample(n=mongo_cfg.sample, random_state=42).reset_index(drop=True)

    print(f"Merged dataset shape: {df.shape} (rows, cols)")
    res, feat_importance, selected_df = run_pca(df, pca_cfg)

    print(f"\nDone. Outputs saved to: {os.path.abspath(pca_cfg.output_dir)}")
    print("Top 10 features by PCA importance:")
    print(feat_importance.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
