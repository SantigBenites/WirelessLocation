# search_spaces.py — MLP-focused, simpler than the NN space

grid_search_space = {
    # MLP architecture
    'num_layers': [1, 2, 3, 4],
    'layer_size': [64, 128, 256, 512],
    'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'tanh'],

    # Light regularization knobs
    'batch_norm': [True, False],          # choose BatchNorm or none
    'use_dropout': [True, False],
    'dropout': [0.1, 0.2, 0.3, 0.5],

    # Optional extras
    'uncertainty_estimation': [False, True],
    'residual_connections': [False, True],

    # Optimizer / training params
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'weight_decay': [0.0, 1e-4, 1e-5],
    'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],
    'batch_size': [64, 128, 256],

    # Linear layer initialization
    'initialization': ['default', 'xavier', 'kaiming', 'orthogonal'],
}

# For random search, keep the same space
random_search_space = {
    **grid_search_space
}
