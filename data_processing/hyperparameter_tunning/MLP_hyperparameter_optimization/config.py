from __future__ import annotations
from typing import Any, Dict, List

# ------------------------------ CONFIG ------------------------------ #
USE_TIMESTAMP: bool = False  # set True to include a timestamp feature if available

# Cross-validation controls
DO_CV: bool = True
K_FOLDS: int = 5
SHUFFLE_CV: bool = True

# Grid search controls
DO_GRID_SEARCH: bool = True
TOP_N_SAVE: int = 3
CHECKPOINT_DIR: str = "checkpoints"
GRID: Dict[str, List[Any]] = {
    "hidden": [[64, 64], [128, 64], [128, 128, 64]],
    "lr": [3e-4, 1e-3],
    "dropout": [0.0, 0.1],
    "weight_decay": [0.0, 1e-4, 5e-4],
    "epochs": [150],
    "batch_size": [128],
}

#  Mongo connection
MONGO_URI: str = "mongodb://localhost:28910/"

# Default training hyperparams
SEED: int = 42
EPOCHS: int = 200
BATCH_SIZE: int = 128
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
DROPOUT: float = 0.1
PATIENCE: int = 20
HIDDEN: list[int] = [128, 128, 64]

# ------------------------------ W&B ------------------------------ #
WANDB_ENABLED: bool = True
WANDB_PROJECT: str = "rssi-mlp"
WANDB_ENTITY: str | None = None
WANDB_TAGS: list[str] = ["mlp", "grid", "cv"]
