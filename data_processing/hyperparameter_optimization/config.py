from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Data settings
    test_size: float = 0.2
    random_state: int = 42

    # Training settings
    epochs: int = 20
    training_depth: int = 10
    models_per_depth: int = 10
    num_gpus: int = 6
    num_cpu = 24
    group_name: str = "gradient_search_global"


    # Model generation
    initial_variation_factor: float = 0.3
    variation_decay_rate: float = 0.02

    # Optimization
    default_batch_size: int = 2048
    default_learning_rate: float = 0.01
    default_weight_decay: float = 0.0

    # Global search configuration
    num_gradient_runs: int = 10
    log_best_group: str = "gradient_search_best_models"

