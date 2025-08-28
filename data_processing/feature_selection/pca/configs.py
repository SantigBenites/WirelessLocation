from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class MongoConfig:
    uri: str = "mongodb://localhost:27017"
    db_name: str = "wifi_fingerprinting_data_extra_features"
    collections: List[str] = field(default_factory=lambda: [
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
    ])
    query_filter: Dict[str, Any] = field(default_factory=dict)  # e.g., {"timestamp": {"$gte": 0}}
    sample: Optional[int] = None  # random sample size across the merged dataset (set None for all)
    batch_size: int = 50_000       # Mongo batch size per cursor

@dataclass
class PCAConfig:
    drop_columns: List[str] = field(default_factory=lambda: ["_id", "timestamp", "location_x", "location_y"])
    scale: bool = True
    whiten: bool = False
    random_state: int = 42
    variance_threshold: float = 0.95    # keep PCs until this cumulative variance
    max_components: Optional[int] = None
    feature_selection_top_k: Optional[int] = 20  # None = keep all ranked features
    save_projection: bool = True
    output_dir: str = "outputs/pca_feature_selection"
    use_cuml_if_available: bool = True
    ray_address: Optional[str] = None    # e.g., "auto" or "ray://<head>:10001"
    num_reader_workers: int = 8
    use_gpu_for_pca: bool = True         # requires RAPIDS/cuML if available
