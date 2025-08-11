
import torch
import torch.nn as nn
import random
import copy
from typing import Tuple, Dict, Any
from search_spaces import cnn_search_space

def _infer_hw(input_size: int, arch_cfg: Dict[str, Any]) -> Tuple[int, int]:
    H = arch_cfg.get("input_height", None)
    W = arch_cfg.get("input_width", None)
    if isinstance(H, int) and isinstance(W, int) and H > 0 and W > 0 and H * W == input_size:
        return H, W
    r = int(input_size ** 0.5)
    if r * r == input_size:
        return r, r
    return 1, input_size

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, activation, batch_norm, use_dropout, dropout):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_norm)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        act_map = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True),
            'selu': nn.SELU(inplace=True),
            'tanh': nn.Tanh()
        }
        layers.append(act_map.get(activation, nn.ReLU(inplace=True)))
        if use_dropout and dropout and dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class GeneratedModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, architecture_config: Dict[str, Any]):
        super().__init__()
        self.cfg = architecture_config
        H, W = _infer_hw(input_size, architecture_config)
        self.input_hw = (H, W)

        in_ch = architecture_config.get('input_channels', 1)
        num_blocks = architecture_config.get('num_conv_blocks', 2)
        out_ch_base = architecture_config.get('out_channels', 16)
        k = architecture_config.get('kernel_size', 3)
        s = architecture_config.get('stride', 1)
        act = architecture_config.get('activation', 'relu')
        bn = architecture_config.get('batch_norm', True)
        use_do = architecture_config.get('use_dropout', False)
        p_do = architecture_config.get('dropout', 0.0)
        pool_type = architecture_config.get('pool_type', 'none')
        pool_sz = architecture_config.get('pool_size', 2)
        residual = architecture_config.get('residual_connections', False)
        global_pool = architecture_config.get('global_pool', 'avg')
        num_dense = architecture_config.get('num_dense_layers', 1)
        dense_size = architecture_config.get('dense_size', 128)
        dense_do = architecture_config.get('dense_dropout', 0.0)

        convs = []
        channels = [in_ch] + [out_ch_base * (2 ** i) for i in range(num_blocks)]
        self.skip_convs = nn.ModuleList() if residual else None

        current_h, current_w = H, W
        for i in range(num_blocks):
            convs.append(ConvBlock(channels[i], channels[i+1], k, s, act, bn, use_do, p_do))
            if residual:
                self.skip_convs.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=1))
            if pool_type == 'max' and current_h >= pool_sz and current_w >= pool_sz:
                convs.append(nn.MaxPool2d(kernel_size=pool_sz))
                current_h = max(1, current_h // pool_sz)
                current_w = max(1, current_w // pool_sz)
            elif pool_type == 'avg' and current_h >= pool_sz and current_w >= pool_sz:
                convs.append(nn.AvgPool2d(kernel_size=pool_sz))
                current_h = max(1, current_h // pool_sz)
                current_w = max(1, current_w // pool_sz)

        self.features = nn.Sequential(*convs)

        if global_pool == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            gp_out = channels[-1]
        elif global_pool == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
            gp_out = channels[-1]
        else:
            self.global_pool = nn.Identity()
            gp_out = channels[-1] * current_h * current_w

        dense_layers = []
        in_feat = gp_out
        act_map = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True),
            'selu': nn.SELU(inplace=True),
            'tanh': nn.Tanh()
        }
        for _ in range(max(0, num_dense)):
            dense_layers.append(nn.Linear(in_feat, dense_size))
            if bn:
                dense_layers.append(nn.BatchNorm1d(dense_size))
            dense_layers.append(act_map.get(act, nn.ReLU(inplace=True)))
            if dense_do and dense_do > 0:
                dense_layers.append(nn.Dropout(p=dense_do))
            in_feat = dense_size

        self.dense = nn.Sequential(*dense_layers)
        self.out = nn.Linear(in_feat, output_size)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        H, W = self.input_hw
        C = self.cfg.get('input_channels', 1)
        if x.shape[1] == C * H * W:
            return x.view(N, C, H, W)
        else:
            return x.view(N, 1, 1, x.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)
        out = x
        if self.cfg.get('residual_connections', False):
            skip_idx = 0
            for layer in self.features:
                if isinstance(layer, ConvBlock):
                    skip = out
                    out = layer(out)
                    out = out + self.skip_convs[skip_idx](skip)
                    skip_idx += 1
                else:
                    out = layer(out)
        else:
            out = self.features(x)

        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.dense(out)
        out = self.out(out)
        return out

def generate_random_model_configs(search_space=cnn_search_space, number_of_models=10):
    keys = list(search_space.keys())
    configs = []
    for i in range(number_of_models):
        sample = {key: random.choice(search_space[key]) for key in keys}
        arch = {k: sample[k] for k in sample}
        configs.append({
            'name': f"CNN_RandomModel_{i}",
            'config': arch,
            'params': sample
        })
    return configs

def generate_similar_model_configs(base_model, search_space=cnn_search_space, number_of_models=10, variation_factor=0.2):
    configs = []
    base_params = base_model['params']
    for i in range(number_of_models):
        new_params = copy.deepcopy(base_params)
        for key in new_params:
            if key in search_space and random.random() < variation_factor:
                options = search_space[key]
                try:
                    idx = options.index(new_params[key])
                    shift = random.choice([-2, -1, 1, 2])
                    new_params[key] = options[(idx + shift) % len(options)]
                except ValueError:
                    new_params[key] = random.choice(options)
        arch = {k: new_params[k] for k in new_params}
        configs.append({
            'name': f"CNN_Similar_{base_model['name']}_{i}",
            'config': arch,
            'params': new_params
        })
    return configs
