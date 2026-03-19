"""
RW-Layer: Read-Write Layer for Neural Networks
===============================================

A learnable data correction layer that can be integrated into any neural network.
The RW-Layer computes z = W^k x with identity initialization and no activation.

Usage:
    from rw_layer import RWLayer, RWMLP

    # Add to any model
    rw = RWLayer(n_features=10, k=3)
    corrected_x = rw(x)

    # Or use the ready-made RW-MLP
    model = RWMLP(input_dim=10, k=3)
    prediction = model(x)
"""

import torch
import torch.nn as nn


class RWLayer(nn.Module):
    """
    Read-Write Layer with identity initialization.

    Computes z = x @ W^k.T + bias where W starts as identity matrix.
    No activation function is applied, preserving the input feature space.

    Args:
        n_features: Number of input features (d)
        k: Matrix power hyperparameter (default: 1)
    """
    def __init__(self, n_features: int, k: int = 1):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.eye(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_k = self.W
        for _ in range(self.k - 1):
            W_k = W_k @ self.W
        return x @ W_k.T + self.bias

    def get_correction_magnitude(self) -> float:
        """How much W has deviated from identity (Frobenius norm)."""
        with torch.no_grad():
            return (self.W - torch.eye(self.W.shape[0], device=self.W.device)).norm().item()


class MLP(nn.Module):
    """Standard MLP for regression."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RWMLP(nn.Module):
    """MLP with RW-Layer preprocessing."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2,
                 dropout: float = 0.1, k: int = 1):
        super().__init__()
        self.rw = RWLayer(input_dim, k=k)
        self.mlp = MLP(input_dim, hidden_dim, n_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rw(x)
        return self.mlp(x)


class FTTransformer(nn.Module):
    """FT-Transformer for tabular data."""
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        embeddings = [self.feature_embeddings[i](x[:, i:i+1]) for i in range(self.n_features)]
        x = torch.stack(embeddings, dim=1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0]).squeeze(-1)


class RWFTTransformer(nn.Module):
    """FT-Transformer with RW-Layer preprocessing."""
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1, k: int = 1):
        super().__init__()
        self.rw = RWLayer(n_features, k=k)
        self.ft = FTTransformer(n_features, d_model, n_heads, n_layers, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rw(x)
        return self.ft(x)


class RWModel(nn.Module):
    """Generic wrapper: adds RW-Layer to any PyTorch model."""
    def __init__(self, rw_layer: RWLayer, model: nn.Module):
        super().__init__()
        self.rw = rw_layer
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.rw(x))
