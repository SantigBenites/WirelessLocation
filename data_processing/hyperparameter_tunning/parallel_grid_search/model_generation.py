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

def generate_model_configs(search_space=grid_search_space):
    keys, values = zip(*search_space.items())
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

def generate_random_model_configs(search_space=random_search_space, number_of_models=10):
    keys = list(search_space.keys())
    configs = []

    for i in range(number_of_models):
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

import copy

def generate_similar_model_configs(base_model, search_space=random_search_space, number_of_models=10, variation_factor=0.2):
    """
    Generate similar model configurations based on an existing model with slight variations.
    
    Args:
        base_model: The original model configuration to base variations on
        search_space: The namespace defining possible values for each parameter
        number_of_models: How many similar models to generate
        variation_factor: How much to vary parameters (0.0-1.0)
        
    Returns:
        List of similar model configurations
    """
    configs = []
    base_params = base_model['params']
    
    for i in range(number_of_models):
        # Create a copy of the base params to modify
        new_params = copy.deepcopy(base_params)
        
        # Modify each parameter with some probability based on variation_factor
        for key in new_params:
            if key in search_space and random.random() < variation_factor:
                # For numerical parameters, add small random variation
                if isinstance(new_params[key], (int, float)):
                    current_val = new_params[key]
                    options = [x for x in search_space[key] if isinstance(x, (int, float))]
                    if options:
                        min_val = min(options)
                        max_val = max(options)
                        # Add random noise within 10% of the parameter range
                        noise = (random.random() - 0.5) * (max_val - min_val) * 0.1
                        new_val = current_val + noise
                        # Clip to valid range
                        new_val = max(min_val, min(max_val, new_val))
                        # Round if original was int
                        if isinstance(current_val, int):
                            new_val = int(round(new_val))
                        new_params[key] = new_val
                # For categorical parameters, sometimes pick a nearby option
                else:
                    options = search_space[key]
                    try:
                        current_index = options.index(new_params[key])
                        # Move up or down 1-2 positions with wrap-around
                        shift = random.choice([-2, -1, 1, 2])
                        new_index = (current_index + shift) % len(options)
                        new_params[key] = options[new_index]
                    except ValueError:
                        # Current value not in options, pick randomly
                        new_params[key] = random.choice(options)
        
        # Rebuild the hidden layers based on the new parameters
        hidden_layers = []
        for _ in range(new_params['num_layers']):
            layer_spec = {
                'units': new_params['layer_size'],
                'activation': new_params['activation'],
                'normalization': 'batch' if new_params['batch_norm'] else 'none',
                'use_dropout': new_params['use_dropout'],
                'dropout': new_params['dropout'] if new_params['use_dropout'] else None,
                'initialization': new_params['initialization']
            }
            hidden_layers.append(layer_spec)
        
        architecture_config = {
            'hidden_layers': hidden_layers,
            'attention': new_params['attention'],
            'uncertainty_estimation': new_params['uncertainty_estimation'],
            'residual_connections': new_params.get('residual_connections', False),
            'use_checkpointing': new_params.get('checkpointing', False)
        }
        
        configs.append({
            'name': f"SimilarModel_{base_model['name']}_{i}",
            'config': architecture_config,
            'params': new_params
        })
    
    return configs