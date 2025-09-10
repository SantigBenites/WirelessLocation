from dataclasses import dataclass, replace
from typing import Dict, Any

@dataclass
class TrainingConfig:
    # Data settings
    test_size: float = 0.2
    random_state: int = 42
    db_name: str = "error_db"

    # Training settings
    epochs: int = 50
    training_depth: int = 10
    models_per_depth: int = 12
    num_cpu: int = 24
    group_name: str = "error_group_name"
    num_dataloader_workers: int = 0

    # Model generation
    initial_variation_factor: float = 0.3
    variation_decay_rate: float = 0.02

    # Optimization
    default_batch_size: int = 2048
    default_learning_rate: float = 0.01
    default_weight_decay: float = 0.0

    # Global search configuration
    num_gradient_runs: int = 5

    # Model Storage / run metadata
    model_save_dir: str = "model_storage_hehe_error"
    experiment_name: str = "CNN_error"
    run_index: int = 0