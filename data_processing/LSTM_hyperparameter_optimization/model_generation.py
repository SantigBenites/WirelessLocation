import math
import random
from copy import deepcopy
from typing import Dict, Any, List, Optional
from search_spaces import lstm_search_space
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================
# Utilities
# ===========================

def get_activation(name: str):
    name = (name or 'relu').lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.01)
    if name == 'selu':
        return nn.SELU()
    if name == 'tanh':
        return nn.Tanh()
    # default
    return nn.ReLU()


def init_weights(module: nn.Module, method: str = 'default'):
    method = (method or 'default').lower()
    if isinstance(module, (nn.Linear,)):
        if method == 'xavier':
            nn.init.xavier_uniform_(module.weight)
        elif method == 'kaiming':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif method == 'orthogonal':
            nn.init.orthogonal_(module.weight)
        else:
            # default
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LSTM,)):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                if method == 'xavier':
                    nn.init.xavier_uniform_(param.data)
                elif method == 'kaiming':
                    nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
                elif method == 'orthogonal':
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                if method == 'orthogonal':
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)


class TemporalAttention(nn.Module):
    """
    Simple additive attention over time.
    Input: (B, T, H)
    Output: (B, H) as weighted sum of time steps.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        scores = torch.tanh(self.proj(x)) @ self.query  # (B, T)
        weights = torch.softmax(scores, dim=1)          # (B, T)
        out = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, H)
        return out


class LSTMRegressor(nn.Module):
    """
    Flexible LSTM-based regressor with optional projection "embedding", attention,
    and normalization before the MLP head.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        cfg: Dict[str, Any],
    ):
        super().__init__()
        self.cfg = cfg = deepcopy(cfg)

        num_layers     = int(cfg.get('num_layers', 2))
        hidden_size    = int(cfg.get('hidden_size', 128))
        bidirectional  = bool(cfg.get('bidirectional', False))
        dropout        = float(cfg.get('dropout', 0.0))
        activation     = str(cfg.get('activation', 'relu'))
        batch_norm     = bool(cfg.get('batch_norm', False))
        use_attention  = bool(cfg.get('use_attention', False))
        normalization  = str(cfg.get('normalization', 'none')).lower()
        initialization = str(cfg.get('initialization', 'default'))
        embedding_dim  = cfg.get('embedding_dim', None)
        self.sequence_length = int(cfg.get('sequence_length', 1))

        # Optional learnable projection (works like an embedding for continuous features)
        feature_dim = input_size
        self.preproj = None
        if embedding_dim is not None:
            emb_dim = int(embedding_dim)
            if emb_dim > 0:
                self.preproj = nn.Linear(input_size, emb_dim)
                feature_dim = emb_dim

        # LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.attention = TemporalAttention(lstm_out_dim) if use_attention else None

        # Optional normalization just before the head
        if normalization == 'batch':
            self.prehead_norm = nn.BatchNorm1d(lstm_out_dim)
        elif normalization == 'layer':
            self.prehead_norm = nn.LayerNorm(lstm_out_dim)
        else:
            self.prehead_norm = None

        self.batch_norm_after_lstm = nn.BatchNorm1d(lstm_out_dim) if batch_norm else None

        self.act = get_activation(activation)

        # Head
        hidden_head = lstm_out_dim // 2 if lstm_out_dim >= 64 else max(16, lstm_out_dim)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_head),
            self.act,
            nn.Linear(hidden_head, output_size),
        )

        # Init
        init_weights(self.lstm, initialization)
        if self.preproj is not None:
            init_weights(self.preproj, initialization)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, initialization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts either:
          - (B, T, F)
          - (B, F)  (will be expanded to T=1, or repeated to sequence_length if configured)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.size(1) == 1 and self.cfg.get('sequence_length', 1) > 1:
            x = x.repeat(1, int(self.cfg['sequence_length']), 1)

        if self.preproj is not None:
            B, T, F = x.shape
            x = self.preproj(x.view(B * T, F)).view(B, T, -1)

        lstm_out, _ = self.lstm(x)  # (B, T, H*dir)

        if self.batch_norm_after_lstm is not None:
            B, T, H = lstm_out.shape
            lstm_out = lstm_out.contiguous().view(B * T, H)
            lstm_out = self.batch_norm_after_lstm(lstm_out)
            lstm_out = lstm_out.view(B, T, H)

        if self.attention is not None:
            feats = self.attention(lstm_out)   # (B, H)
        else:
            feats = lstm_out[:, -1, :]         # last timestep

        if self.prehead_norm is not None:
            feats = self.prehead_norm(feats)

        out = self.head(feats)
        return out


# ===========================
# Public API used by your pipeline
# ===========================

class GeneratedModel(nn.Module):
    """
    Thin wrapper to preserve your current import path.
    """
    def __init__(self, input_size: int, output_size: int, architecture_config: Dict[str, Any]):
        super().__init__()
        self.model = LSTMRegressor(input_size, output_size, architecture_config)

    def forward(self, x):
        return self.model(x)


def _sample_from_space(space: Dict[str, List[Any]]) -> Dict[str, Any]:
    cfg = {}
    for k, v in space.items():
        cfg[k] = random.choice(v)
    return cfg


def generate_random_model_configs(number_of_models: int = 8) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts, each with:
      { 'name': <filled by caller>,
        'config': <architecture + optimizer hyperparams> }
    """
    configs = []
    for _ in range(number_of_models):
        cfg = _sample_from_space(lstm_search_space)
        configs.append({
            "name": "LSTM_random",
            "config": cfg,
        })
    return configs


def _neighbor_value(value, choices):
    if value not in choices:
        return random.choice(choices)
    idx = choices.index(value)
    if len(choices) == 1:
        return value
    if idx == 0:
        return choices[1]
    if idx == len(choices) - 1:
        return choices[-2]
    return random.choice([choices[idx - 1], choices[idx + 1]])


def generate_similar_model_configs(base_model: Dict[str, Any], number_of_models: int = 8, variation_factor: float = 0.3) -> List[Dict[str, Any]]:
    """
    Make small perturbations around the base config. We interpret variation_factor in a simple way:
    - ~30% chance to tweak each hyperparameter to a neighbor choice.
    """
    base_cfg = deepcopy(base_model.get("config", {}))
    configs = []
    for _ in range(number_of_models):
        cfg = deepcopy(base_cfg)
        for k, choices in lstm_search_space.items():
            if random.random() < max(0.05, min(0.95, variation_factor)):
                if isinstance(choices, list):
                    cfg[k] = _neighbor_value(cfg.get(k, random.choice(choices)), choices)
        configs.append({
            "name": f"LSTM_variation_of_{base_model.get('name','base')}",
            "config": cfg,
        })
    return configs
