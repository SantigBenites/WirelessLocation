
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any
from search_spaces import mlp_random_search_space
import torch
import torch.nn as nn
import torch.nn.functional as F



def _activation(name: str, leaky_slope: float = 0.01) -> nn.Module:
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=leaky_slope)
    if name == 'elu':
        return nn.ELU()
    if name == 'selu':
        return nn.SELU()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'gelu':
        return nn.GELU()
    if name == 'silu':
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class SqueezeExcite1D(nn.Module):

    def __init__(self, features: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, features // reduction)
        self.fc1 = nn.Linear(features, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate conditioned on features (no pooling over features; treat as MLP gating)
        g = self.fc2(F.relu(self.fc1(x)))
        g = torch.sigmoid(g)
        return x * g


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str,
        leaky_relu_slope: float,
        normalization: str,
        batch_norm_momentum: float,
        use_bias: bool,
        use_dropout: bool,
        dropout_p: float,
        dropout_type: str,
        attention: bool,
        residual: bool,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)

        norm = normalization.lower()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(out_features, momentum=batch_norm_momentum)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = nn.Identity()

        self.act = _activation(activation, leaky_slope=leaky_relu_slope)

        self.use_dropout = use_dropout and dropout_p > 0.0
        if self.use_dropout:
            if dropout_type == 'alpha' and activation.lower() == 'selu':
                self.dropout = nn.AlphaDropout(p=dropout_p)
            else:
                self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = nn.Identity()

        self.attention = SqueezeExcite1D(out_features) if attention else nn.Identity()

        # residual path: if dims mismatch use a projection
        self.use_residual = residual
        self.project = None
        if residual:
            if in_features != out_features:
                self.project = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.attention(x)
        if self.use_residual:
            res = identity if self.project is None else self.project(identity)
            x = x + res
        return x


def _make_widths(num_layers: int, base: int, pattern: str) -> List[int]:
    pattern = pattern.lower()
    widths: List[int] = []
    if pattern == 'constant':
        widths = [base for _ in range(num_layers)]
    elif pattern == 'pyramid':
        # grow by x2 until the last layer
        widths = [min(base * (2 ** i), 2048) for i in range(num_layers)]
    elif pattern == 'inverse_pyramid':
        widths = [max(base // (2 ** i), 8) for i in range(num_layers)][::-1]
    else:
        raise ValueError(f"Unknown width pattern: {pattern}")
    return widths


def _init_weights(module: nn.Module, strategy: str, gain: float):
    if isinstance(module, nn.Linear):
        if strategy == 'default':
            # leave as PyTorch default
            return
        elif strategy == 'xavier':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        elif strategy == 'kaiming':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif strategy == 'orthogonal':
            nn.init.orthogonal_(module.weight, gain=gain)
        else:
            raise ValueError(f"Unknown init: {strategy}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class GeneratedModel(nn.Module):

    def __init__(self, input_size: int, output_size: int, architecture_config: Dict[str, Any]):
        super().__init__()
        cfg = architecture_config.copy()

        self.cfg = cfg
        num_layers: int = int(cfg['num_layers'])
        base: int = int(cfg['layer_size'])
        pattern: str = cfg['width_pattern']

        activation: str = cfg['activation']
        leaky_slope: float = float(cfg.get('leaky_relu_slope', 0.01))
        use_bias: bool = bool(cfg['use_bias'])
        normalization: str = cfg['normalization']
        bn_momentum: float = float(cfg.get('batch_norm_momentum', 0.9))

        use_dropout: bool = bool(cfg['use_dropout'])
        dropout_p: float = float(cfg.get('dropout', 0.0))
        input_dropout_p: float = float(cfg.get('input_dropout', 0.0))
        dropout_type: str = cfg.get('dropout_type', 'standard')
        residual: bool = bool(cfg.get('residual_connections', False))
        attention: bool = bool(cfg.get('attention', False))

        self.input_dropout = nn.Identity() if input_dropout_p <= 0 else (
            nn.AlphaDropout(p=input_dropout_p) if (dropout_type == 'alpha' and activation == 'selu') else nn.Dropout(p=input_dropout_p)
        )

        widths = _make_widths(num_layers, base, pattern)

        layers: List[nn.Module] = []
        in_dim = input_size
        for w in widths:
            layers.append(
                MLPBlock(
                    in_features=in_dim,
                    out_features=int(w),
                    activation=activation,
                    leaky_relu_slope=leaky_slope,
                    normalization=normalization,
                    batch_norm_momentum=bn_momentum,
                    use_bias=use_bias,
                    use_dropout=use_dropout,
                    dropout_p=dropout_p,
                    dropout_type=dropout_type,
                    attention=attention,
                    residual=residual,
                )
            )
            in_dim = int(w)

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, output_size, bias=True)

        # init
        init_strategy = cfg.get('initialization', 'default')
        gain = float(cfg.get('weight_init_gain', 1.0))
        self.apply(lambda m: _init_weights(m, init_strategy, gain))

        # MC dropout flag
        self._mc_dropout = bool(cfg.get('uncertainty_estimation', False))

    def set_mc_dropout(self, enabled: bool = True):
        self._mc_dropout = enabled
        # Enable dropout layers during eval if requested
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.AlphaDropout)):
                m.train(enabled)  # keep only dropouts in 'train' mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        x = self.backbone(x)
        x = self.head(x)
        return x


# -----------------------------
# Config generation
# -----------------------------

def _choice(key: str, space: Dict[str, List[Any]]) -> Any:
    return random.choice(space[key])


def _sample_single(space: Dict[str, List[Any]]) -> Dict[str, Any]:
    cfg = {
        'num_layers': _choice('num_layers', space),
        'layer_size': _choice('layer_size', space),
        'width_pattern': _choice('width_pattern', space),
        'activation': _choice('activation', space),
        'use_bias': _choice('use_bias', space),
        'normalization': _choice('normalization', space),
        'use_dropout': _choice('use_dropout', space),
        'dropout': _choice('dropout', space),
        'input_dropout': _choice('input_dropout', space),
        'dropout_type': _choice('dropout_type', space),
        'residual_connections': _choice('residual_connections', space),
        'attention': _choice('attention', space),
        'uncertainty_estimation': _choice('uncertainty_estimation', space),
        'learning_rate': _choice('learning_rate', space),
        'weight_decay': _choice('weight_decay', space),
        'optimizer': _choice('optimizer', space),
        'lr_scheduler': _choice('lr_scheduler', space),
        'batch_size': _choice('batch_size', space),
        'grad_clip_val': _choice('grad_clip_val', space),
        'initialization': _choice('initialization', space),
        'weight_init_gain': _choice('weight_init_gain', space),
        'label_smoothing': _choice('label_smoothing', space),
        'loss': _choice('loss', space),
    }

    # conditionals
    if cfg['activation'] == 'leaky_relu':
        cfg['leaky_relu_slope'] = _choice('leaky_relu_slope', space)
    else:
        cfg['leaky_relu_slope'] = 0.01

    if cfg['normalization'] == 'batch':
        cfg['batch_norm_momentum'] = _choice('batch_norm_momentum', space)
    else:
        cfg['batch_norm_momentum'] = 0.9

    if cfg['lr_scheduler'] == 'step':
        cfg['scheduler_step_size'] = _choice('scheduler_step_size', space)
        cfg['scheduler_gamma'] = _choice('scheduler_gamma', space)
    elif cfg['lr_scheduler'] == 'plateau':
        cfg['plateau_patience'] = _choice('plateau_patience', space)
    else:
        # fill with defaults for consistency
        cfg['scheduler_step_size'] = 10
        cfg['scheduler_gamma'] = 0.5
        cfg['plateau_patience'] = 5

    # Ensure alpha dropout pairs with SELU (fallback to standard if not SELU)
    if cfg['dropout_type'] == 'alpha' and cfg['activation'] != 'selu':
        cfg['dropout_type'] = 'standard'

    return cfg


def generate_random_model_configs(number_of_models: int) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for i in range(number_of_models):
        cfg = _sample_single(mlp_random_search_space)
        configs.append({
            "name": f"mlp_random_{i}",
            "config": cfg
        })
    return configs


def _mutate_value(key: str, base_val: Any, space_vals: List[Any], variation: float) -> Any:
    # With probability proportional to variation, change the value; else keep base.
    if random.random() > max(0.05, min(1.0, variation)):
        return base_val
    # For discrete spaces, move to a neighbor or resample.
    if isinstance(space_vals, list):
        try:
            idx = space_vals.index(base_val)
            # pick adjacent or random neighbor
            candidates = list(set([max(0, idx-1), idx, min(len(space_vals)-1, idx+1)]))
            new_idx = random.choice(candidates)
            return space_vals[new_idx]
        except Exception:
            return random.choice(space_vals)
    return base_val


def _mutate_config(base: Dict[str, Any], space: Dict[str, List[Any]], variation: float) -> Dict[str, Any]:
    cfg = dict(base)  # shallow copy

    for key, vals in space.items():
        if key not in cfg:
            continue
        cfg[key] = _mutate_value(key, cfg[key], vals, variation)

    # Regenerate conditionals coherently
    if cfg['activation'] == 'leaky_relu':
        cfg['leaky_relu_slope'] = _mutate_value('leaky_relu_slope', cfg.get('leaky_relu_slope', 0.01), space['leaky_relu_slope'], variation)
    else:
        cfg['leaky_relu_slope'] = 0.01

    if cfg['normalization'] == 'batch':
        cfg['batch_norm_momentum'] = _mutate_value('batch_norm_momentum', cfg.get('batch_norm_momentum', 0.9), space['batch_norm_momentum'], variation)
    else:
        cfg['batch_norm_momentum'] = 0.9

    if cfg['lr_scheduler'] == 'step':
        cfg['scheduler_step_size'] = _mutate_value('scheduler_step_size', cfg.get('scheduler_step_size', 10), space['scheduler_step_size'], variation)
        cfg['scheduler_gamma'] = _mutate_value('scheduler_gamma', cfg.get('scheduler_gamma', 0.5), space['scheduler_gamma'], variation)
    elif cfg['lr_scheduler'] == 'plateau':
        cfg['plateau_patience'] = _mutate_value('plateau_patience', cfg.get('plateau_patience', 5), space['plateau_patience'], variation)
    else:
        cfg['scheduler_step_size'] = 10
        cfg['scheduler_gamma'] = 0.5
        cfg['plateau_patience'] = 5

    if cfg['dropout_type'] == 'alpha' and cfg['activation'] != 'selu':
        cfg['dropout_type'] = 'standard'

    return cfg


def generate_similar_model_configs(base_model: Dict[str, Any], number_of_models: int, variation_factor: float) -> List[Dict[str, Any]]:
    base_cfg = base_model['config']
    out: List[Dict[str, Any]] = []
    for i in range(number_of_models):
        mutated = _mutate_config(base_cfg, mlp_random_search_space, variation_factor)
        out.append({
            "name": f"{base_model.get('name','mlp_base')}_var{i}",
            "config": mutated
        })
    return out
