"""
Transformer model for in-context learning.

Implements a GPT-style decoder-only transformer with causal attention.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class ICLTransformer(nn.Module):
    """
    GPT-style decoder-only transformer for in-context linear regression.

    Uses causal attention so each token can only attend to previous tokens.
    The query token (last position) attends to all context tokens.
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
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Input projection
        self.in_proj = nn.Linear(d_in, d_model)

        # Learned positional embeddings
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer decoder layers (using encoder layers with causal mask)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                batch_first=True,
                activation="gelu",
                norm_first=True,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output head
        self.out = nn.Linear(d_model, 1)

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )

    def forward(
        self,
        seq: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with causal attention.

        Args:
            seq: (B, L, d_in) input sequence
            pad_mask: (B, L) boolean mask where True indicates padding

        Returns:
            (B,) predicted y values for the last (query) position
        """
        B, L, _ = seq.shape
        device = seq.device

        # Input embeddings
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        h = self.in_proj(seq) + self.pos_emb(pos)

        # Get causal mask for this sequence length
        causal_mask = self.causal_mask[:L, :L]

        # Apply transformer layers with causal masking
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask, src_key_padding_mask=pad_mask)

        h = self.norm(h)

        # Output prediction from last position
        yhat = self.out(h[:, -1, :]).squeeze(-1)
        return yhat

    def forward_with_hidden(
        self,
        seq: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass that also returns hidden states from each layer.

        Useful for probing internal representations.

        Returns:
            yhat: (B,) predictions
            hidden_states: List of (B, L, d_model) tensors, one per layer
        """
        B, L, _ = seq.shape
        device = seq.device

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        h = self.in_proj(seq) + self.pos_emb(pos)

        causal_mask = self.causal_mask[:L, :L]

        hidden_states = [h]  # Include input embeddings
        for layer in self.layers:
            h = layer(h, src_mask=causal_mask, src_key_padding_mask=pad_mask)
            hidden_states.append(h)

        h = self.norm(h)
        yhat = self.out(h[:, -1, :]).squeeze(-1)

        return yhat, hidden_states


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
