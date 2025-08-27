from dataclasses import dataclass, field
import os
from typing import List


@dataclass
class Config:
    # --- Mongo ---
    mongo_uri: str = "mongodb://localhost:28910/"
    db_name: str = "wifi_fingerprinting_data_extra_features"


    # Collections to include (as provided)
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
    "reto_pequeno_outdoo", # note: if this is a typo, fix it here to match your DB
    ])


    # Feature discovery / streaming
    exclude_keys: List[str] = field(default_factory=lambda: ["_id", "location_x", "location_y", "timestamp"]) # labels excluded
    sample_per_collection: int = 200
    batch_size: int = 8192


    # PCA / selection
    variance_threshold: float = 0.95 # ignored if n_components > 0
    n_components: int = 0 # 0 => auto by variance_threshold
    top_m_features: int = 20
    seed: int = 1337


    # W&B
    wandb_enabled: bool = True
    wandb_project: str = "wifi-pca-feature-select"
    wandb_entity: str | None = None # set if you use a team/entity
    wandb_run_name: str | None = None
    wandb_notes: str | None = None
    wandb_mode: str = os.environ.get("WANDB_MODE", "online") # "online"|"offline"|"disabled"


    # Results write-back
    write_back_results: bool = True
    results_collection: str = "feature_selection_results"


    # Distributed/multi-GPU (no torchrun)
    use_spawn: bool = True
    world_size: int = 0 # 0 => auto: number of CUDA devices if available else 1
    master_addr: str = "127.0.0.1"
    master_port: int = 0 # 0 => auto-pick a free port