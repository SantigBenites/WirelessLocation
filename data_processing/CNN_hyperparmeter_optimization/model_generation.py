
import torch, copy, random
import torch.nn as nn
from typing import List, Dict, Any
from search_spaces import cnn_search_space


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn=True, pool_kernel=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.pool = nn.MaxPool2d(pool_kernel) if pool_kernel else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class GeneratedModel(nn.Module):
    def __init__(self, input_shape, output_size, architecture_config):
        super().__init__()
        c, h, w = input_shape
        layers = []
        in_ch = c

        for block_conf in architecture_config.get('conv_blocks', []):
            out_ch = block_conf.get('out_channels', 32)
            k = block_conf.get('kernel_size', 3)
            s = block_conf.get('stride', 1)
            p = block_conf.get('padding', 1)
            pool_k = block_conf.get('pool_kernel', None)

            # Calculate output spatial size after conv
            h_out = (h + 2*p - k) // s + 1
            w_out = (w + 2*p - k) // s + 1

            # If pooling will cause < 2×2 before last layer, skip it
            if pool_k:
                ph, pw = (pool_k, pool_k) if isinstance(pool_k, int) else pool_k
                if h_out // ph < 2 or w_out // pw < 2:
                    pool_k = None

            layers.append(ConvBlock(in_ch, out_ch, k, s, p, True, pool_k))

            # Update size tracking
            h, w = h_out, w_out
            if pool_k:
                ph, pw = (pool_k, pool_k) if isinstance(pool_k, int) else pool_k
                h //= ph
                w //= pw

            in_ch = out_ch

        # Final adaptive pool to guarantee >= 2×2
        layers.append(nn.AdaptiveAvgPool2d((2, 2)))
        self.features = nn.Sequential(*layers)

        # Compute final flattened size
        final_flatten_size = in_ch * 2 * 2

        fc_layers = []
        in_fc = final_flatten_size
        for fc_units in architecture_config.get('fc_layers', [128]):
            fc_layers.append(nn.Linear(in_fc, fc_units))
            fc_layers.append(nn.ReLU())
            in_fc = fc_units
        fc_layers.append(nn.Linear(in_fc, output_size))

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_arch_from_params(p):
    return {
        'input_channels': p.get('input_channels', 3),
        'num_conv_layers': p['num_conv_layers'],
        'filters_per_layer': p['filters_per_layer'],
        'kernel_size': p['kernel_size'],
        'stride': p['stride'],
        'padding': p['padding'],
        'activation': p['activation'],
        'batch_norm': p['batch_norm'],
        'use_dropout': p['use_dropout'],
        'dropout': p['dropout'],
        'pooling_type': p['pooling_type'],
        'pool_size': p['pool_size'],
        'residual_connections': p['residual_connections'],
        'global_pool': p.get('global_pool', 'none'),
        'num_dense_layers': p['num_dense_layers'],
        'dense_size': p['dense_size'],
        'dense_dropout': p.get('dense_dropout', 0.0)
    }

def generate_random_model_configs(search_space=cnn_search_space, number_of_models=10):
    keys = list(search_space.keys())
    configs = []
    for i in range(number_of_models):
        p = {k: random.choice(search_space[k]) for k in keys}
        arch = _build_arch_from_params(p)
        configs.append({
            'name': f"RandomCNN2D_{i}",
            'config': arch,
            'params': p,
        })
    return configs

def generate_similar_model_configs(base_model, search_space=cnn_search_space, number_of_models=10, variation_factor=0.2):
    def vary_param(key, val):
        opts = search_space[key]
        try:
            idx = opts.index(val)
            shift = random.choice([-1, 1]) if random.random() < variation_factor else 0
            return opts[(idx + shift) % len(opts)]
        except ValueError:
            return random.choice(opts)

    base = copy.deepcopy(base_model['params'])
    configs = []
    for i in range(number_of_models):
        p = copy.deepcopy(base)
        for k in base.keys():
            if random.random() < variation_factor and k in search_space:
                p[k] = vary_param(k, base[k])
        arch = _build_arch_from_params(p)
        configs.append({
            'name': f"SimilarCNN2D_{base_model['name']}_{i}",
            'config': arch,
            'params': p,
        })
    return configs

