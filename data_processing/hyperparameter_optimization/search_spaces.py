nn_search_space = {
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

cnn_search_space = {
    'input_channels': [1],
    'num_conv_blocks': [1, 2, 3, 4],
    'out_channels': [8, 16, 32, 64],
    'kernel_size': [3, 5, 7],
    'stride': [1],
    'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'tanh'],
    'batch_norm': [True, False],
    'use_dropout': [True, False],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
    'pool_type': ['none', 'max', 'avg'],
    'pool_size': [2, 3],
    'residual_connections': [False, True],
    'global_pool': ['avg', 'max', 'none'],
    'num_dense_layers': [0, 1, 2],
    'dense_size': [32, 64, 128, 256],
    'dense_dropout': [0.0, 0.1, 0.2],
}