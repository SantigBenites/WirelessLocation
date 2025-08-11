
import torch, copy, random
import torch.nn as nn
from typing import List, Dict, Any
from search_spaces import cnn_search_space


class ConvBlock(nn.Module):
    """
    A single 1D convolutional block with optional BatchNorm, Dropout and Pooling.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        activation: str,
        use_bn: bool,
        use_dropout: bool,
        dropout: float,
        pool_type: str,
        pool_size: int,
    ):
        super().__init__()
        padding = kernel_size // 2  # keep length when stride == 1
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,
        )
        self.bn = nn.BatchNorm1d(out_ch) if use_bn else None

        self.act = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True),
            'selu': nn.SELU(inplace=True),
            'tanh': nn.Tanh()
        }[activation]

        self.do = nn.Dropout(dropout) if use_dropout and dropout > 0 else None

        if pool_type == 'max':
            self.pool = nn.MaxPool1d(pool_size)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool1d(pool_size)
        else:
            self.pool = None

        # He init works well for most choices here
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        if self.do is not None:
            x = self.do(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class GeneratedModel(nn.Module):
    """
    CNN generated from a config dict. Accepts input as [B, F] and reshapes to [B, C, L].
    """
    def __init__(self, input_size: int, output_size: int, architecture_config: Dict[str, Any]):
        super().__init__()
        cfg = architecture_config

        # shapes: we accept [B, F]. We reshape to [B, C, L]. Default C=1, L=F.
        in_channels = cfg.get('input_channels', 1)
        if input_size % in_channels != 0:
            raise ValueError(f"input_size {input_size} not divisible by input_channels {in_channels}")
        seq_len = input_size // in_channels

        self.input_size = input_size
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.output_size = output_size

        blocks: List[Dict[str, Any]] = cfg['conv_blocks']
        self.residual = cfg.get('residual_connections', False)
        self.global_pool = cfg.get('global_pool', 'none')

        layers = []
        ch = in_channels
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
        self.conv_layers = nn.ModuleList(layers)

        # Projections for residual (channel-only). If length differs we skip residual.
        self.res_proj = nn.ModuleDict()

        # Global pooling
        if self.global_pool == 'avg':
            self.global_pool_layer = nn.AdaptiveAvgPool1d(1)
        elif self.global_pool == 'max':
            self.global_pool_layer = nn.AdaptiveMaxPool1d(1)
        else:
            self.global_pool_layer = None

        # Dense head
        num_dense = cfg.get('num_dense_layers', 0)
        dense_size = cfg.get('dense_size', 64)
        dense_layers: List[nn.Module] = []
        in_feat = ch  # after pooling we will flatten channel dimension
        for i in range(num_dense):
            dense_layers.append(nn.Linear(in_feat, dense_size))
            dense_layers.append(nn.ReLU(inplace=True))
            drop = cfg.get('dense_dropout', 0.0)
            if drop and drop > 0:
                dense_layers.append(nn.Dropout(drop))
            in_feat = dense_size
        self.dense = nn.Sequential(*dense_layers) if dense_layers else None

        self.out = nn.Linear(in_feat, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] -> [B, C, L]
        b = x.shape[0]
        x = x.view(b, self.in_channels, self.seq_len)

        for i, block in enumerate(self.conv_layers):
            x_in = x
            x = block(x)
            if self.residual:
                same_length = x.shape[-1] == x_in.shape[-1]
                if same_length:
                    if x.shape[1] != x_in.shape[1]:
                        key = str(i)
                        if key not in self.res_proj:
                            self.res_proj[key] = nn.Conv1d(x_in.shape[1], x.shape[1], kernel_size=1, bias=False)
                        x = x + self.res_proj[key](x_in)
                    else:
                        x = x + x_in

        if self.global_pool_layer is not None:
            x = self.global_pool_layer(x)  # [B, C, 1]
            x = x.squeeze(-1)  # [B, C]
        else:
            x = torch.flatten(x, start_dim=1)

        if self.dense is not None:
            x = self.dense(x)
        return self.out(x)


def _build_arch_from_params(p: Dict[str, Any]) -> Dict[str, Any]:
    blocks = []
    for _ in range(p['num_conv_blocks']):
        blocks.append({
            'out_channels': p['out_channels'],
            'kernel_size': p['kernel_size'],
            'stride': p['stride'],
            'activation': p['activation'],
            'batch_norm': p['batch_norm'],
            'use_dropout': p['use_dropout'],
            'dropout': p['dropout'],
            'pool_type': p['pool_type'],
            'pool_size': p['pool_size'],
        })
    arch = {
        'input_channels': p.get('input_channels', 1),
        'conv_blocks': blocks,
        'residual_connections': p['residual_connections'],
        'global_pool': p['global_pool'],
        'num_dense_layers': p['num_dense_layers'],
        'dense_size': p['dense_size'],
        'dense_dropout': p.get('dense_dropout', 0.0),
    }
    return arch


def generate_model_configs(search_space=cnn_search_space):
    keys, values = zip(*search_space.items())
    configs = []
    # Cartesian product
    from itertools import product
    for combo in product(*values):
        params = dict(zip(keys, combo))
        arch = _build_arch_from_params(params)
        configs.append({
            'name': f"CNN_{len(configs)+1}",
            'config': arch,
            'params': params,
        })
    return configs


def generate_random_model_configs(search_space=cnn_search_space, number_of_models=10):
    keys = list(search_space.keys())
    configs = []
    for i in range(number_of_models):
        p = {k: random.choice(search_space[k]) for k in keys}
        arch = _build_arch_from_params(p)
        configs.append({
            'name': f"RandomCNN_{i}",
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
            'name': f"SimilarCNN_{base_model['name']}_{i}",
            'config': arch,
            'params': p,
        })
    return configs
