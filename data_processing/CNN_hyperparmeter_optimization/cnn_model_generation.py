
import torch
import torch.nn as nn
import random
import copy, math
from typing import Tuple, Dict, Any, List
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
        cfg = architecture_config

        in_channels = cfg.get('input_channels', 1)
        if input_size % in_channels != 0:
            raise ValueError(f"input_size {input_size} not divisible by input_channels {in_channels}")
        seq_len = input_size // in_channels

        blocks: List[Dict[str, Any]] = cfg['conv_blocks']
        self.residual = cfg.get('residual_connections', False)
        self.global_pool = cfg.get('global_pool', 'none')

        layers = []
        ch = in_channels
        cur_len = seq_len  # <— track current length
        for b in blocks:
            block = ConvBlock(
                in_ch=ch,
                out_ch=b['out_channels'],
                kernel_size=b['kernel_size'],
                stride=b.get('stride', 1),
                activation=b['activation'],
                use_bn=b.get('batch_norm', False),
                use_dropout=b.get('use_dropout', False),
                dropout=b.get('dropout', 0.0),
                pool_type=b.get('pool_type', 'none'),
                pool_size=b.get('pool_size', 2),
            )
            layers.append(block)
            ch = b['out_channels']

            # conv keeps length due to padding/stride=1; only pooling changes it
            if b.get('pool_type', 'none') in ('max', 'avg'):
                cur_len = math.floor(cur_len / b.get('pool_size', 2))
                cur_len = max(cur_len, 1)

        self.conv_layers = nn.ModuleList(layers)

        if self.global_pool == 'avg':
            self.global_pool_layer = nn.AdaptiveAvgPool1d(1)
        elif self.global_pool == 'max':
            self.global_pool_layer = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_pool_layer = None

        num_dense = cfg.get('num_dense_layers', 0)
        dense_size = cfg.get('dense_size', 64)
        dense_layers: List[nn.Module] = []

        # ✅ set correct input feature size for the dense head
        in_feat = ch if self.global_pool_layer is not None else ch * cur_len

        for _ in range(num_dense):
            dense_layers.append(nn.Linear(in_feat, dense_size))
            dense_layers.append(nn.ReLU(inplace=True))
            drop = cfg.get('dense_dropout', 0.0)
            if drop and drop > 0:
                dense_layers.append(nn.Dropout(drop))
            in_feat = dense_size

        self.dense = nn.Sequential(*dense_layers) if dense_layers else None
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
