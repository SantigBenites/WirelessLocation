from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Data settings
    test_size: float = 0.2
    random_state: int = 42

    # Training settings
    epochs: int = 20
    training_depth: int = 8
    models_per_depth: int = 6
    num_cpu = 24
    group_name: str = "LSTM"
    num_dataloader_workers = 95

    # Model generation
    initial_variation_factor: float = 0.3
    variation_decay_rate: float = 0.02

    # Optimization
    default_batch_size: int = 4096
    default_learning_rate: float = 0.01
    default_weight_decay: float = 0.0

    # Global search configuration
    num_gradient_runs: int = 5

    # Model Storage
    model_save_dir = "model_storage"
    experiment_name = "experiment"
    run_index = 0
