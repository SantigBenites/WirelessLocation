cnn_search_space = {
    'num_conv_layers': [1, 2, 3, 4, 5],  # Number of convolutional layers
    'filters_per_layer': [16, 32, 64, 128, 256],  # Number of filters in each conv layer
    'kernel_size': [3, 5, 7],  # Convolution kernel sizes
    'stride': [1, 2],  # Stride for convolution
    'padding': ['same', 'valid'],  # Padding type
    'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'tanh'],  # Activation functions
    'batch_norm': [True, False],  # Apply batch normalization
    'use_dropout': [True, False],  # Apply dropout
    'dropout': [0.2, 0.3, 0.4, 0.5],  # Dropout rate
    'pooling_type': ['max', 'avg', 'none'],  # Pooling layer type
    'pool_size': [2, 3],  # Pooling window size
    'residual_connections': [False, True],  # Residual/skip connections
    'learning_rate': [0.001, 0.0005, 0.0001],  # Learning rate
    'weight_decay': [0.0, 0.0001, 0.00001],  # L2 regularization
    'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],  # Optimizers
    'batch_size': [64, 128, 256, 512],  # Mini-batch size
    'normalization': ['none', 'batch', 'layer', 'instance'],  # Normalization methods
    'initialization': ['default', 'xavier', 'kaiming', 'orthogonal']  # Weight init
}