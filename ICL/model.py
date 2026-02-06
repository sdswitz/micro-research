"""Transformer model for in-context learning."""

import torch
import torch.nn as nn


class ICLTransformer(nn.Module):
    """
    Transformer encoder for in-context linear regression.

    Takes a sequence of [x, y] tokens and predicts y for the final query token.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_len: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_in: Input dimension (d + 1 for [x, y] tokens)
            d_model: Model hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_len: Maximum sequence length
            dropout: Dropout rate (default 0.0)
        """
        super().__init__()
        self.d_model = d_model

        self.in_proj = nn.Linear(d_in, d_model)
        self.type_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, seq: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            seq: (B, L, d_in) input sequence
            pad_mask: (B, L) boolean mask where True indicates padding

        Returns:
            (B,) predicted y values for the last (query) position
        """
        B, L, _ = seq.shape
        device = seq.device

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        types = torch.zeros(B, L, dtype=torch.long, device=device)
        types[:, -1] = 1

        h = self.in_proj(seq) + self.pos_emb(pos) + self.type_emb(types)
        h = self.encoder(h, mask=None, src_key_padding_mask=pad_mask)

        yhat = self.out(h[:, -1, :]).squeeze(-1)
        return yhat


def create_model(
    d: int,
    n_ctx_max: int = 128,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    dropout: float = 0.0,
    device: str = "cuda",
) -> ICLTransformer:
    """
    Factory function to create an ICLTransformer.

    Args:
        d: Input x dimension
        n_ctx_max: Maximum context length
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout rate
        device: Device to place model on

    Returns:
        Initialized ICLTransformer model
    """
    d_in = d + 1
    max_len = n_ctx_max + 1

    model = ICLTransformer(
        d_in=d_in,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=max_len,
        dropout=dropout,
    )
    return model.to(device)
