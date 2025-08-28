from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DatabaseConfig:
    mongo_uri: str = 'mongodb://localhost:28910/'
    db_name: str = "wifi_fingerprinting_data_extra_features"
    collections: List[str] = field(default_factory=list)
    query_filter: Dict = field(default_factory=dict)  # e.g., {"timestamp": {"$gte": 0}}
    projection: Optional[Dict] = None  # e.g., {"_id": 0} to drop _id at the DB level
    limit: Optional[int] = None


@dataclass
class FeatureSpaceConfig:
    target_cols: List[str] = field(default_factory=lambda: ["location_x", "location_y"])
    timestamp_col: str = "timestamp"
    group_col: Optional[str] = None  # if None, derive from timestamp
    group_window_seconds: int = 60
    drop_cols: List[str] = field(default_factory=lambda: ["_id"])  # always drop
    include_cols: Optional[List[str]] = None  # if set, use only these (after excluding targets & drops)
    exclude_cols: List[str] = field(default_factory=list)
    impute_strategy: str = "median"  # or "zero"


@dataclass
class SelectionConfig:
    # Mutual information prefilter
    mi_top_k: Optional[int] = 50  # None = keep all
    mi_neighbors: int = 3
    # Lasso shrinkage
    lasso_alpha: float = 0.01
    lasso_max_iter: int = 5000
    lasso_top_k: Optional[int] = 40  # None = nonzero only
    # RFE pruning
    rfe_n_features_to_select: Optional[int] = 20  # final count after RFE (per target)
    rfe_step: int = 1
    # Aggregation across steps/folds
    min_fold_fraction: float = 0.0  # feature must appear in >= this fraction of folds to be kept


@dataclass
class CVConfig:
    n_splits: int = 5
    random_state: int = 42  # used in any randomized ops (not GroupKFold)


@dataclass
class RayConfig:
    address: Optional[str] = None  # e.g., "auto" to connect to cluster
    local_mode: bool = False
    num_cpus_per_task: int = 1
    num_gpus_per_task: float = 0.0  # sklearn doesn't need GPU; keep for future


@dataclass
class OutputConfig:
    outdir: str = "./fs_outputs"
    overwrite: bool = True


@dataclass
class Config:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    feature_space: FeatureSpaceConfig = field(default_factory=FeatureSpaceConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)