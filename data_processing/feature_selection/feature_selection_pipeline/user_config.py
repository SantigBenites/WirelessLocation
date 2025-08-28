from __future__ import annotations
from configs import Config, DatabaseConfig, FeatureSpaceConfig, SelectionConfig, CVConfig, RayConfig, OutputConfig

# EDIT THIS FILE with your environment specifics

CFG = Config(
    database=DatabaseConfig(
        mongo_uri='mongodb://localhost:28910/',
        db_name="wifi_fingerprinting_data_extra_features",
        collections=[
            # Pick any subset you want to include in a single run
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
        projection=None,  # set {"_id": 0} to drop _id at source
        limit=None,
    ),
    feature_space=FeatureSpaceConfig(
        target_cols=["location_x", "location_y"],
        timestamp_col="timestamp",
        group_col=None,                # if you have a session column, set it here
        group_window_seconds=60,       # group by minute when deriving from timestamp
        drop_cols=["_id"],
        include_cols=None,
        exclude_cols=[],
        impute_strategy="median",
    ),
    selection=SelectionConfig(
        mi_top_k=50,
        mi_neighbors=3,
        lasso_alpha=0.01,
        lasso_max_iter=5000,
        lasso_top_k=40,
        rfe_n_features_to_select=20,
        rfe_step=1,
        min_fold_fraction=0.6,
    ),
    cv=CVConfig(
        n_splits=5,
        random_state=42,
    ),
    ray=RayConfig(
        address=None,       # set to "auto" if connecting to a Ray cluster
        local_mode=False,
        num_cpus_per_task=1,
        num_gpus_per_task=0.5,
    ),
    output=OutputConfig(
        outdir="./fs_outputs",
        overwrite=True,
    ),
)