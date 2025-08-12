
import math
import random
from copy import deepcopy
from typing import Dict, Any, List, Tuple
from search_spaces import cnn_search_space
import torch
import torch.nn as nn
import torch.nn.functional as F



# -------------------------
# Utilities
# -------------------------
def _choose(space: Dict[str, List[Any]], key: str):
    return random.choice(space[key])


def _maybe_change(value, options, variation_prob: float):
    if random.random() < max(0.0, min(1.0, variation_prob)):
        # Prefer a different value
        choices = [v for v in options if v != value]
        return random.choice(choices) if choices else value
    return value


def _activation(name: str):
    name = name.lower()
    return {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'tanh': nn.Tanh(),
    }[name]


class _LayerNorm1dAcrossChannels(nn.Module):
    """
    LayerNorm across channel dimension for 1D sequences.
    Expects input shape [N, C, L]. Applies LayerNorm over the last dim after permutation: [N, L, C].
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: [N, C, L] -> [N, L, C] -> LayerNorm(C) -> [N, C, L]
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2)
        return x


class _SafePool1d(nn.Module):
    """
    A pooling wrapper that clamps kernel_size to at most the current length.
    Uses stride=kernel_size (downsampling) and ceil_mode=True to be robust on short sequences.
    """
    def __init__(self, pool_type: str, kernel_size: int):
        super().__init__()
        self.pool_type = pool_type
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.pool_type == 'none':
            return x
        L = x.shape[-1]
        k = min(self.kernel_size, L) if L > 0 else self.kernel_size
        if self.pool_type == 'max':
            return F.max_pool1d(x, kernel_size=k, stride=k, ceil_mode=True)
        else:
            return F.avg_pool1d(x, kernel_size=k, stride=k, ceil_mode=True)


class _Norm1d(nn.Module):
    """
    Wrapper that applies the requested normalization but safely falls back when
    the temporal length is 1 (where InstanceNorm would error).
    """
    def __init__(self, norm_type: str, num_channels: int):
        super().__init__()
        self.norm_type = norm_type
        self.num_channels = num_channels
        if norm_type == 'batch':
            self.bn = nn.BatchNorm1d(num_channels)
        elif norm_type == 'instance':
            self.inorm = nn.InstanceNorm1d(num_channels, affine=True, track_running_stats=False)
            self.ln_fallback = _LayerNorm1dAcrossChannels(num_channels)
        elif norm_type == 'layer':
            self.ln = _LayerNorm1dAcrossChannels(num_channels)
        else:
            self.id = nn.Identity()

    def forward(self, x):
        if self.norm_type == 'batch':
            return self.bn(x)
        elif self.norm_type == 'instance':
            # InstanceNorm requires more than 1 spatial element at train time.
            if x.size(-1) > 1:
                return self.inorm(x)
            else:
                return self.ln_fallback(x)
        elif self.norm_type == 'layer':
            return self.ln(x)
        else:
            return self.id(x)


def _init_weights(module: nn.Module, scheme: str):
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        if scheme == 'xavier':
            nn.init.xavier_uniform_(module.weight)
        elif scheme == 'kaiming':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif scheme == 'orthogonal':
            nn.init.orthogonal_(module.weight)
        # 'default' -> leave as torch default
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# -------------------------
# Generated CNN
# -------------------------
class GeneratedModel(nn.Module):
    """
    CNN wrapper that accepts tensors of shape [N, F] (e.g., 3 RSSI features)
    and internally treats them as a 1D signal of length F with 1 input channel.
    Architecture is driven by "architecture_config".
    """
    def __init__(self, input_size: int, output_size: int, architecture_config: Dict[str, Any]):
        super().__init__()
        self.cfg = deepcopy(architecture_config)

        # Base hyperparams
        num_layers = int(self.cfg['num_conv_layers'])
        filters = int(self.cfg['filters_per_layer'])
        kernel = int(self.cfg['kernel_size'])
        stride = int(self.cfg['stride'])
        padding = self.cfg.get('padding', 'same')
        activation = _activation(self.cfg['activation'])
        use_dropout = bool(self.cfg['use_dropout'])
        dropout_p = float(self.cfg['dropout']) if use_dropout else 0.0
        pooling_type = self.cfg['pooling_type']
        pool_size = int(self.cfg['pool_size'])
        residual = bool(self.cfg['residual_connections'])
        normalization = self.cfg.get('normalization', 'none')
        batch_norm_flag = bool(self.cfg.get('batch_norm', False))  # legacy

        # We always start with 1 input channel and sequence length = input_size
        in_channels = 1
        layers: List[nn.Module] = []
        proj_layers: List[nn.Module] = []  # for residual projections when shapes mismatch
        current_len = input_size
        current_channels = in_channels

        for li in range(num_layers):
            # Decide effective padding for this layer to avoid zero/negative length for "valid" with large kernel
            eff_padding = padding
            if eff_padding == 'valid' and kernel > current_len:
                eff_padding = 'same'  # auto-correct to keep model valid on tiny inputs

            # Convert 'same'/'valid' to numeric padding to support stride>1
            if eff_padding == 'same':
                conv_padding = (kernel - 1) // 2  # assumes dilation=1 and odd kernels
                eff_mode = 'same'
            else:  # 'valid'
                conv_padding = 0
                eff_mode = 'valid'

            conv = nn.Conv1d(
                in_channels=current_channels,
                out_channels=filters,
                kernel_size=kernel,
                stride=stride,
                padding=conv_padding,
            )
            block: List[nn.Module] = [conv]

            # Normalization choices
            norm_choice = normalization if normalization != 'none' else ('batch' if batch_norm_flag else 'none')
            if norm_choice in ('batch','instance','layer'):
                block.append(_Norm1d(norm_choice, filters))

            block.append(deepcopy(activation))

            # Optional pooling
            if pooling_type != 'none':
                block.append(_SafePool1d(pooling_type, pool_size))

            # Optional dropout
            if use_dropout and dropout_p > 0.0:
                block.append(nn.Dropout(p=dropout_p))

            layers.append(nn.Sequential(*block))

            # For residuals, build a projection if needed
            if residual:
                needs_proj = (current_channels != filters) or (stride != 1) or (pooling_type != 'none')
                if needs_proj:
                    # 1x1 conv to match channels + stride/pool downsampling with SafePool if needed
                    proj_block: List[nn.Module] = [nn.Conv1d(current_channels, filters, kernel_size=1, stride=stride, padding=0)]
                    if pooling_type != 'none':
                        proj_block.append(_SafePool1d(pooling_type, pool_size))
                    proj_layers.append(nn.Sequential(*proj_block))
                else:
                    proj_layers.append(nn.Identity())

            # Update trackers (approximate current length under "same" or "valid")
            if eff_mode == 'same':
                current_len = math.ceil(current_len / stride)
            else:
                # valid padding
                current_len = math.floor((current_len - kernel + 1 + (stride - 1)) / stride)
                current_len = max(current_len, 1)
            if pooling_type != 'none':
                # using SafePool1d -> downsample by ~pool_size (ceil_mode)
                current_len = math.ceil(current_len / max(1, pool_size))
            current_channels = filters

        self.conv_blocks = nn.ModuleList(layers)
        self.residual = residual
        self.projections = nn.ModuleList(proj_layers) if residual else None

        # Head: global average pool to be independent of final length, then Linear
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),                     # [N, C, 1] -> [N, C]
            nn.Linear(current_channels, max(16, current_channels // 2)),
            nn.ReLU(),
            nn.Linear(max(16, current_channels // 2), output_size)
        )

        # Initialization
        init_scheme = self.cfg.get('initialization', 'default')
        if init_scheme != 'default':
            self.apply(lambda m: _init_weights(m, init_scheme))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: [N, F]. Convert to [N, 1, F]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != 1:
            # if user already provided [N, F, 1], permute to [N, 1, F]
            if x.shape[2] == 1:
                x = x.transpose(1, 2)
            else:
                # assume it's [N, C, L], keep as-is
                pass

        out = x
        if self.residual:
            for block, proj in zip(self.conv_blocks, self.projections):
                residual = proj(out)
                out = block(out)
                # lengths should match due to SafePool + projection stride/pool
                if residual.shape[-1] != out.shape[-1]:
                    # As a last resort, adapt with interpolation
                    out_len = out.shape[-1]
                    residual = F.interpolate(residual, size=out_len, mode="nearest")
                out = out + residual
        else:
            for block in self.conv_blocks:
                out = block(out)

        out = self.gap(out)
        out = self.head(out)
        return out


# -------------------------
# Config generators
# -------------------------
def _build_config(space: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Sample a valid architecture config from the provided search space.
    Adds small safety adjustments for tiny inputs (e.g., when padding='valid' with large kernels).
    """
    cfg = {
        'num_conv_layers': _choose(space, 'num_conv_layers'),
        'filters_per_layer': _choose(space, 'filters_per_layer'),
        'kernel_size': _choose(space, 'kernel_size'),
        'stride': _choose(space, 'stride'),
        'padding': _choose(space, 'padding'),
        'activation': _choose(space, 'activation'),
        'batch_norm': _choose(space, 'batch_norm'),
        'use_dropout': _choose(space, 'use_dropout'),
        'dropout': _choose(space, 'dropout'),
        'pooling_type': _choose(space, 'pooling_type'),
        'pool_size': _choose(space, 'pool_size'),
        'residual_connections': _choose(space, 'residual_connections'),
        'learning_rate': _choose(space, 'learning_rate'),
        'weight_decay': _choose(space, 'weight_decay'),
        'optimizer': _choose(space, 'optimizer'),
        'batch_size': _choose(space, 'batch_size'),
        'normalization': _choose(space, 'normalization'),
        'initialization': _choose(space, 'initialization'),
    }

    # If dropout is disabled, keep value but it won't be used.
    # If pooling is 'none' pool_size is irrelevant.
    return cfg


def generate_random_model_configs(number_of_models: int,
                                  space: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
    """
    Produce a list of configs with the shape expected by the training loop:
        { "name": "...", "config": <arch dict> }
    """
    if space is None:
        space = cnn_search_space
    configs = []
    for _ in range(number_of_models):
        cfg = _build_config(space)
        configs.append({
            "name": "cnn_random",
            "config": cfg
        })
    return configs


def _neighbors(options: List[Any], value: Any) -> List[Any]:
    # For discrete options, just allow any value != current
    return [o for o in options if o != value] or [value]


def generate_similar_model_configs(base_model: Dict[str, Any],
                                   number_of_models: int,
                                   variation_factor: float = 0.2,
                                   space: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
    """
    Create variations of a base model by randomly changing each hyperparameter with
    probability = variation_factor.
    """
    if space is None:
        space = cnn_search_space

    base_cfg = deepcopy(base_model.get("config", base_model))
    variants = []

    for _ in range(number_of_models):
        new_cfg = deepcopy(base_cfg)
        for k, options in space.items():
            if k not in new_cfg:
                # If the base model came from a different space, ensure key is present
                new_cfg[k] = _choose(space, k)
                continue
            new_cfg[k] = _maybe_change(new_cfg[k], options, variation_factor)

        variants.append({
            "name": f"{base_model.get('name', 'cnn_base')}_var",
            "config": new_cfg
        })
    return variants