mlp_random_search_space = {
    'num_layers': [1, 2, 3, 4, 5, 6],                 # depth
    'layer_size': [64, 128, 256, 512, 1024],          # base width (apply per layer or via a pattern)
    'width_pattern': ['constant', 'pyramid', 'inverse_pyramid'],  # how widths change across layers
    'activation': ['relu', 'leaky_relu', 'elu', 'selu', 'tanh', 'gelu', 'silu'],
    'leaky_relu_slope': [0.01, 0.1, 0.2],             # only used if activation == leaky_relu

    'use_bias': [True, False],
    'normalization': ['none', 'batch', 'layer'],      # pick one norm type
    'batch_norm_momentum': [0.9, 0.95, 0.99],         # only if normalization == 'batch'

    'use_dropout': [True, False],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],             # hidden-layer dropout
    'input_dropout': [0.0, 0.05, 0.1],                # optional input-layer dropout
    'dropout_type': ['standard', 'alpha'],            # AlphaDropout pairs well with SELU

    'residual_connections': [False, True],            # simple skip connections
    'attention': [False, True],                       # e.g., squeeze-excitation/gated layer
    'uncertainty_estimation': [False, True],          # e.g., MC dropout

    'learning_rate': [1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
    'weight_decay': [0.0, 1e-5, 5e-5, 1e-4],
    'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop', 'nadam'],

    'lr_scheduler': ['none', 'step', 'cosine', 'plateau', 'onecycle'],
    'scheduler_step_size': [5, 10, 20],               # only if 'step'
    'scheduler_gamma': [0.1, 0.5, 0.8],               # only if 'step'
    'plateau_patience': [3, 5, 10],                   # only if 'plateau'

    'batch_size': [32, 64, 128, 256, 512],
    'grad_clip_val': [0.0, 0.5, 1.0, 5.0],

    'initialization': ['default', 'xavier', 'kaiming', 'orthogonal'],
    'weight_init_gain': [1.0, 1.414, 2.0],            # optional scaling for init

    # task-dependent knobs (useful if you share one space for cls/reg)
    'label_smoothing': [0.0, 0.05, 0.1],              # classification only
    'loss': ['mse', 'mae', 'huber', 'cross_entropy', 'bce'],  # pick per task
}
