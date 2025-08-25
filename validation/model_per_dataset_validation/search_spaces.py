grid_search_space = {
            'num_layers': [1, 2, 3, 4],  # Deeper networks
            'layer_size': [64, 128, 256, 512],  # Wider networks
            'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'tanh'],
            'batch_norm': [True, False],
            'use_dropout': [True, False],
            'dropout': [0.2, 0.3, 0.4, 0.5],  # Different dropout rates
            'attention': [False, True],  # Option for attention mechanism
            'uncertainty_estimation': [False, True],  # Uncertainty estimation
            'learning_rate': [0.001, 0.0005, 0.0001],  # Learning rate variations
            'weight_decay': [0.0, 0.0001, 0.00001],  # L2 regularization
            'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],  # Different optimizers
            'batch_size': [64, 128, 256],  # Different batch sizes
            'normalization': ['none', 'batch', 'layer', 'instance'],  # Different norm types
            'residual_connections': [False, True],  # Skip connections
            'initialization': ['default', 'xavier', 'kaiming', 'orthogonal']
        }