"""Physics-aware lighting prompt, compatible with the original encoder weights."""
import math
from pathlib import Path

import torch
from torch import nn


class SinPE(nn.Module):
    def __init__(self, dim, max_len=77):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(positions * frequencies)
        pe[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, tokens):
        return tokens + self.pe[:, :tokens.size(1)]


class ParamTower6D(nn.Module):
    """Map six physical lighting parameters to diffusion conditioning tokens."""
    def __init__(self, in_dim=6, token_dim=512, out_dim=1024, num_tokens=77,
                 depth=4, heads=8, train_last_layer=False):
        super().__init__()
        self.config = dict(
            in_dim=in_dim, token_dim=token_dim, out_dim=out_dim,
            num_tokens=num_tokens, depth=depth, heads=heads,
            train_last_layer=train_last_layer,
        )
        self.token_embed = nn.Parameter(torch.randn(1, num_tokens, token_dim) * 0.02)
        self.mapper = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(), nn.Linear(256, token_dim * 2),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=token_dim * 4,
            batch_first=True, dropout=0.0, activation='gelu', norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.pos = SinPE(token_dim, max_len=num_tokens)
        self.proj = nn.Linear(token_dim, out_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, condition):
        base = self.token_embed.repeat(condition.size(0), 1, 1)
        gamma, beta = self.mapper(condition).chunk(2, dim=-1)
        tokens = base * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return self.proj(self.encoder(self.pos(tokens)))


def load_lighting_encoder(path):
    """Restore architecture and weights strictly; never use a random fallback."""
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    model = ParamTower6D(**checkpoint.get('config', {}))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.eval().requires_grad_(False)
