cnn_search_space = {
    'num_conv_layers': [1, 2, 3, 4, 5],
    'filters_per_layer': [16, 32, 64, 128, 256],
    'kernel_size': [3, 5, 7],
    'stride': [1, 2],
    'padding': ['same', 'valid'],
    'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'tanh'],
    'batch_norm': [True, False],
    'use_dropout': [True, False],
    'dropout': [0.2, 0.3, 0.4, 0.5],
    'pooling_type': ['max', 'avg', 'none'],
    'pool_size': [2, 3],
    'residual_connections': [False, True],
    'global_pool': ['none', 'avg', 'max'],
    'num_dense_layers': [0, 1, 2],
    'dense_size': [32, 64, 128],
    'dense_dropout': [0.0, 0.3],
    'input_channels': [3]  # RSSI from 3 APs as channels
}

