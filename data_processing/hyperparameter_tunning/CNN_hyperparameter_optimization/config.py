from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Data settings
    test_size: float = 0.2
    random_state: int = 42
    db_name="wifi_fingerprinting_data_raw"

    # Training settings
    epochs: int = 50
    training_depth: int = 10
    models_per_depth: int = 12
    num_cpu = 24
    group_name: str = "CNN_RAW_FINAL_outdoor"
    num_dataloader_workers = 0

    # Model generation
    initial_variation_factor: float = 0.3
    variation_decay_rate: float = 0.02

    # Optimization
    default_batch_size: int = 2048
    default_learning_rate: float = 0.01
    default_weight_decay: float = 0.0

    # Global search configuration
    num_gradient_runs: int = 5

    # Model Storage
    model_save_dir = "model_storage_RAW_FINAL_outdoor"
    experiment_name = "CNN_RAW_FINA_outdoor"
    run_index = 0
