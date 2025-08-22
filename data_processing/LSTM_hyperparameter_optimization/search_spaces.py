lstm_search_space = {
    'num_layers': [1, 2, 3, 4],  # Number of stacked LSTM layers
    'hidden_size': [64, 128, 256, 512],  # Number of units in each LSTM layer
    'bidirectional': [False, True],  # Whether to use bidirectional LSTM
    'dropout': [0.0, 0.2, 0.3, 0.5],  # Dropout between LSTM layers
    'activation': ['tanh', 'relu', 'leaky_relu', 'selu'],  # Activation for fully connected layers after LSTM
    'batch_norm': [True, False],  # Whether to use batch normalization after LSTM outputs
    'use_attention': [False, True],  # Whether to apply attention after LSTM
    'learning_rate': [0.001, 0.0005, 0.0001],  # Learning rate variations
    'weight_decay': [0.0, 0.0001, 0.00001],  # L2 regularization
    'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop'],  # Different optimizers
    'batch_size': [32, 64, 128, 256],  # Batch sizes
    'sequence_length': [10, 20, 50, 100],  # Input sequence lengths
    'normalization': ['none', 'batch', 'layer'],  # Normalization strategy for outputs
    'initialization': ['default', 'xavier', 'kaiming', 'orthogonal'],  # Weight initialization method
    'clip_grad_norm': [None, 0.5, 1.0, 5.0],  # Gradient clipping thresholds
    'embedding_dim': [None, 50, 100, 200],  # If using embeddings for sequence input
}
