import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from itertools import product
from search_spaces import *


class GeneratedModel(nn.Module):
    def __init__(self, input_size, output_size, architecture_config):
        super(GeneratedModel, self).__init__()
        self.layers = nn.ModuleList()
        self.architecture_config = architecture_config
        self.activation_map = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True),
            'selu': nn.SELU(inplace=True),
            'tanh': nn.Tanh()
        }
        
        prev_size = input_size
        self.residual = architecture_config.get('residual_connections', False)
        
        # Build hidden layers
        for i, layer_spec in enumerate(architecture_config['hidden_layers']):
            layer_size = layer_spec['units']
            
            # Linear layer with optimized initialization
            linear_layer = nn.Linear(prev_size, layer_size)
            init_method = layer_spec.get('initialization', 'default')
            if init_method == 'xavier':
                nn.init.xavier_uniform_(linear_layer.weight)
            elif init_method == 'kaiming':
                nn.init.kaiming_uniform_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(linear_layer.weight)
                
            self.layers.append(linear_layer)
            
            # Normalization
            norm_type = layer_spec.get('normalization', 'none')
            if norm_type == 'batch':
                self.layers.append(nn.BatchNorm1d(layer_size))
            elif norm_type == 'layer':
                self.layers.append(nn.LayerNorm(layer_size))
            elif norm_type == 'instance':
                self.layers.append(nn.InstanceNorm1d(layer_size))
            
            # Activation with memory optimization
            activation = layer_spec.get('activation', 'relu')
            self.layers.append(self.activation_map[activation])
            
            # Dropout
            if layer_spec.get('use_dropout', False):
                self.layers.append(nn.Dropout(layer_spec['dropout']))
            
            prev_size = layer_size
        
        # Attention mechanism if enabled
        if architecture_config.get('attention', False):
            self.attention = nn.MultiheadAttention(prev_size, num_heads=4)
            self.attention_norm = nn.LayerNorm(prev_size)
        else:
            self.attention = None
        
        # Output layers
        self.output_layer = nn.Linear(prev_size, output_size)
        
        if architecture_config.get('uncertainty_estimation', False):
            self.uncertainty_layer = nn.Linear(prev_size, output_size)
        else:
            self.uncertainty_layer = None
    
    def forward(self, x):
        residual = x if self.residual else None
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if self.residual and residual is not None and residual.size(-1) == x.size(-1):
                    x = x + residual
                    residual = x
            else:
                x = layer(x)
        
        if self.attention is not None:
            attn_output, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = self.attention_norm(x + attn_output.squeeze(0))
        
        position = self.output_layer(x)
        uncertainty = torch.sigmoid(self.uncertainty_layer(x)) if self.uncertainty_layer else None
        
        return position, uncertainty

def generate_model_configs():
    keys, values = zip(*grid_search_space.items())
    configs = []
    
    for combination in product(*values):
        config = dict(zip(keys, combination))
        
        hidden_layers = []
        for i in range(config['num_layers']):
            layer_spec = {
                'units': config['layer_size'],
                'activation': config['activation'],
                'normalization': 'batch' if config['batch_norm'] else 'none',
                'use_dropout': config['use_dropout'],
                'dropout': config['dropout'] if config['use_dropout'] else None,
                'initialization': 'xavier'
            }
            hidden_layers.append(layer_spec)
        
        architecture_config = {
            'hidden_layers': hidden_layers,
            'attention': config['attention'],
            'uncertainty_estimation': config['uncertainty_estimation'],
            'residual_connections': config.get('residual', False),
            'use_checkpointing': config.get('checkpointing', False)
        }
        
        configs.append({
            'name': f"Model_{len(configs)+1}",
            'config': architecture_config,
            'params': config
        })
    
    return configs



import random

def generate_random_model_configs(search_space, num_samples=50):
    keys = list(search_space.keys())
    configs = []

    for i in range(num_samples):
        sample = {key: random.choice(search_space[key]) for key in keys}

        hidden_layers = []
        for _ in range(sample['num_layers']):
            layer_spec = {
                'units': sample['layer_size'],
                'activation': sample['activation'],
                'normalization': 'batch' if sample['batch_norm'] else 'none',
                'use_dropout': sample['use_dropout'],
                'dropout': sample['dropout'] if sample['use_dropout'] else None,
                'initialization': sample['initialization']
            }
            hidden_layers.append(layer_spec)

        architecture_config = {
            'hidden_layers': hidden_layers,
            'attention': sample['attention'],
            'uncertainty_estimation': sample['uncertainty_estimation'],
            'residual_connections': sample.get('residual_connections', False),
            'use_checkpointing': sample.get('checkpointing', False)
        }

        configs.append({
            'name': f"RandomModel_{i}",
            'config': architecture_config,
            'params': sample
        })

    return configs
