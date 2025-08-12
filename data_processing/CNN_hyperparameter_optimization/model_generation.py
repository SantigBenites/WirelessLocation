
import torch, random
import torch.nn as nn
from typing import Dict, List, Any
from dataclasses import dataclass
from search_spaces import cnn_search_space

# ---- Utilities ----

def _choose(space_key: str):
    """Pick a random value from the cnn_search_space for a given key."""
    values = cnn_search_space[space_key]
    return random.choice(values)

def _maybe_mutate(value, key: str, variation: float):
    """Slightly perturb a hyperparam based on variation factor."""
    space = cnn_search_space[key]
    if isinstance(value, (int, float)):
        # Pick a neighbor in the discrete set when available
        if value in space and len(space) > 1:
            idx = space.index(value)
            step = 1 if random.random() < 0.5 else -1
            new_idx = max(0, min(len(space) - 1, idx + step))
            return space[new_idx]
        else:
            return value
    else:
        # categorical: with small prob, switch to a different category
        if random.random() < variation:
            choices = [v for v in space if v != value]
            return random.choice(choices) if choices else value
        return value

# ---- Model building blocks ----

def activation_layer(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "selu":
        return nn.SELU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

def pooling_layer(kind: str, kernel_size: int) -> nn.Module:
    kind = kind.lower()
    if kind == "max":
        return nn.MaxPool2d(kernel_size)
    if kind == "avg":
        return nn.AvgPool2d(kernel_size)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown pooling type: {kind}")

class ConvBlock(nn.Module):
    """Conv -> (BN) -> Act -> (Dropout) with optional residual when in==out and stride=1."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: str,
        use_bn: bool,
        activation: str,
        use_dropout: bool,
        dropout_p: float,
        residual: bool,
    ):
        super().__init__()
        pad = kernel_size // 2 if padding == "same" else 0
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = activation_layer(activation)
        self.drop = nn.Dropout2d(dropout_p) if use_dropout and dropout_p > 0 else nn.Identity()
        self.residual = residual and (in_ch == out_ch) and (stride == 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        if self.residual:
            out = out + x
        return out

# ---- GeneratedModel ----

class GeneratedModel(nn.Module):
    """
    CNN that predicts 2D coordinates (location_x, location_y).
    The architecture is defined by a sampled 'architecture_config' dict.
    """
    def __init__(self, input_shape, num_classes, architecture_config: Dict[str, Any]):
        """
        input_shape: (C, H, W). We expect a grid (e.g., 3x32x32) but can be any size.
        num_classes: should be 2 (x,y), but kept generic.
        architecture_config: sampled from cnn_search_space.
        """
        super().__init__()
        C, H, W = input_shape
        cfg = architecture_config.copy()

        self.cfg = cfg
        self.save_hyperparameters = getattr(nn.Module, "save_hyperparameters", lambda *args, **kwargs: None)  # placeholder

        num_conv_layers = cfg["num_conv_layers"]
        filters_per_layer = cfg["filters_per_layer"]
        kernel_size = cfg["kernel_size"]
        stride = cfg["stride"]
        padding = cfg["padding"]
        activation = cfg["activation"]
        batch_norm = cfg["batch_norm"]
        use_dropout = cfg["use_dropout"]
        dropout = cfg["dropout"]
        pooling_type = cfg["pooling_type"]
        pool_size = cfg["pool_size"]
        residual_connections = cfg["residual_connections"]
        global_pool = cfg["global_pool"]
        num_dense_layers = cfg["num_dense_layers"]
        dense_size = cfg["dense_size"]
        dense_dropout = cfg["dense_dropout"]

        layers: List[nn.Module] = []

        in_ch = C
        # Build stacked conv blocks
        for i in range(num_conv_layers):
            out_ch = filters_per_layer
            layers.append(
                ConvBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    use_bn=batch_norm,
                    activation=activation,
                    use_dropout=use_dropout,
                    dropout_p=dropout,
                    residual=residual_connections,
                )
            )
            if pooling_type != "none":
                layers.append(pooling_layer(pooling_type, pool_size))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        # Global pooling to make the head shape-agnostic
        if global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == "max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            # Keep spatial dims; we'll flatten whatever remains
            self.global_pool = nn.Identity()

        # Head
        if global_pool in ("avg", "max"):
            head_in = in_ch
        else:
            # We don't know H/W after convs; infer at runtime with a dummy pass
            with torch.no_grad():
                dummy = torch.zeros(1, C, H, W)
                feat = self.global_pool(self.conv(dummy))
                head_in = feat.view(1, -1).shape[1]

        mlp: List[nn.Module] = []
        in_dim = head_in
        for _ in range(num_dense_layers):
            mlp += [nn.Linear(in_dim, dense_size), activation_layer(activation)]
            if dense_dropout and dense_dropout > 0:
                mlp += [nn.Dropout(dense_dropout)]
            in_dim = dense_size
        mlp.append(nn.Linear(in_dim, num_classes))
        self.head = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected as (N, C, H, W). If it's flat (N, C), lift to 1x1 map.
        if x.ndim == 2:
            N, C = x.shape
            x = x.view(N, C, 1, 1)
        feats = self.conv(x)
        feats = self.global_pool(feats)
        out = feats.view(feats.size(0), -1)
        return self.head(out)


# ---- Config Generators ----

def generate_random_model_configs(number_of_models: int):
    """Return a list of dicts with 'config' for each randomly generated architecture."""
    configs = []
    for i in range(number_of_models):
        cfg = {
            "num_conv_layers": _choose("num_conv_layers"),
            "filters_per_layer": _choose("filters_per_layer"),
            "kernel_size": _choose("kernel_size"),
            "stride": _choose("stride"),
            "padding": _choose("padding"),
            "activation": _choose("activation"),
            "batch_norm": _choose("batch_norm"),
            "use_dropout": _choose("use_dropout"),
            "dropout": _choose("dropout"),
            "pooling_type": _choose("pooling_type"),
            "pool_size": _choose("pool_size"),
            "residual_connections": _choose("residual_connections"),
            "global_pool": _choose("global_pool"),
            "num_dense_layers": _choose("num_dense_layers"),
            "dense_size": _choose("dense_size"),
            "dense_dropout": _choose("dense_dropout"),
            "input_channels": _choose("input_channels"),
        }

        # Sanity adjustments
        if cfg["pooling_type"] == "none":
            cfg["pool_size"] = 2  # unused, but keep valid
        if cfg["global_pool"] == "none" and cfg["num_dense_layers"] == 0:
            # ensure some capacity before output
            cfg["num_dense_layers"] = 1

        configs.append({
            "config": cfg,
            "name": f"rand_cnn_{i}"
        })
    return configs

def generate_similar_model_configs(base_model: Dict[str, Any], number_of_models: int, variation_factor: float = 0.2):
    """Mutate a base architecture slightly to explore nearby configs."""
    base_cfg = base_model.get("config", {})
    configs = []
    for i in range(number_of_models):
        cfg = {}
        for k, v in base_cfg.items():
            if k not in cnn_search_space:
                cfg[k] = v
                continue
            cfg[k] = _maybe_mutate(v, k, variation_factor)
        # Occasionally flip residuals/dropout together
        if random.random() < variation_factor * 0.5:
            cfg["residual_connections"] = not cfg.get("residual_connections", False)
        if random.random() < variation_factor * 0.5:
            cfg["use_dropout"] = not cfg.get("use_dropout", True)

        configs.append({
            "config": cfg,
            "name": f"{base_model.get('name','base')}_mut{i}"
        })
    return configs
