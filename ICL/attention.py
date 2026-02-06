"""Attention visualization utilities for ICL transformer."""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple


def get_attention_weights(
    model: nn.Module,
    seq: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Extract attention weights from all layers.

    Args:
        model: ICLTransformer model
        seq: (B, L, d_in) input sequence
        pad_mask: (B, L) optional padding mask

    Returns:
        pred: (B,) model predictions
        attentions: List of (B, n_heads, L, L) attention weights per layer
    """
    model.eval()
    attentions = []

    B, L, _ = seq.shape
    device = seq.device

    # Compute input embeddings
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    types = torch.zeros(B, L, dtype=torch.long, device=device)
    types[:, -1] = 1

    h = model.in_proj(seq) + model.pos_emb(pos) + model.type_emb(types)

    # Run through each layer manually to capture attention
    for layer in model.encoder.layers:
        # Pre-norm
        h_norm = layer.norm1(h)

        # Self-attention with weights
        attn_out, attn_weights = layer.self_attn(
            h_norm, h_norm, h_norm,
            key_padding_mask=pad_mask,
            need_weights=True,
            average_attn_weights=False,  # Get per-head weights
        )
        attentions.append(attn_weights.detach())

        # Residual + FFN
        h = h + attn_out
        h = h + layer.linear2(layer.activation(layer.linear1(layer.norm2(h))))

    # Final norm and output
    h = model.encoder.norm(h)
    pred = model.out(h[:, -1, :]).squeeze(-1)

    return pred, attentions


def plot_attention_heatmap(
    attn: torch.Tensor,
    layer: int = 0,
    head: int = 0,
    n_ctx: Optional[int] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "Blues",
):
    """
    Plot attention heatmap for a single head.

    Args:
        attn: (B, n_heads, L, L) attention weights
        layer: Layer index (for title)
        head: Head index
        n_ctx: Context length (for axis labels)
        title: Optional custom title
        ax: Matplotlib axes (creates new figure if None)
        cmap: Colormap
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Take first batch element
    weights = attn[0, head].cpu().numpy()
    L = weights.shape[0]

    im = ax.imshow(weights, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    if title is None:
        title = f"Layer {layer}, Head {head}"
    ax.set_title(title)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Mark context boundary (query is at last position L-1)
    if n_ctx is not None:
        ax.axhline(y=n_ctx, color="red", linestyle="--", alpha=0.5, label="Context end")
        ax.axvline(x=n_ctx, color="red", linestyle="--", alpha=0.5)

    return ax


def plot_query_attention(
    attn: torch.Tensor,
    layer: int = 0,
    n_ctx: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot what the query token attends to across all heads.

    Args:
        attn: (B, n_heads, L, L) attention weights
        layer: Layer index
        n_ctx: Context length (for marking on plot, not for finding query)
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Get query row (always last position) for first batch element
    n_heads = attn.shape[1]
    L = attn.shape[2]
    query_idx = L - 1  # Query is always at the last position

    query_attn = attn[0, :, query_idx, :].cpu().numpy()  # (n_heads, L)

    x = np.arange(L)
    width = 0.8 / n_heads

    for h in range(n_heads):
        ax.bar(x + h * width, query_attn[h], width, label=f"Head {h}", alpha=0.7)

    ax.set_xlabel("Key Position (context tokens)")
    ax.set_ylabel("Attention Weight")
    ax.set_title(f"Layer {layer}: Query Token Attention")
    ax.legend(loc="upper right", fontsize="small")

    if n_ctx is not None:
        ax.axvline(x=n_ctx, color="red", linestyle="--", alpha=0.5, label="Context end")

    return ax


def plot_all_layers_query_attention(
    attentions: List[torch.Tensor],
    n_ctx: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
):
    """
    Plot query attention patterns for all layers.

    Args:
        attentions: List of (B, n_heads, L, L) per layer
        n_ctx: Context length
        figsize: Figure size
    """
    n_layers = len(attentions)

    if figsize is None:
        figsize = (12, 3 * n_layers)

    fig, axes = plt.subplots(n_layers, 1, figsize=figsize, squeeze=False)

    for layer, attn in enumerate(attentions):
        plot_query_attention(attn, layer=layer, n_ctx=n_ctx, ax=axes[layer, 0])

    plt.tight_layout()
    return fig


def plot_attention_grid(
    attentions: List[torch.Tensor],
    n_ctx: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
):
    """
    Plot attention heatmaps for all layers and heads.

    Args:
        attentions: List of (B, n_heads, L, L) per layer
        n_ctx: Context length
        figsize: Figure size
    """
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]

    if figsize is None:
        figsize = (3 * n_heads, 3 * n_layers)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=figsize)

    for layer, attn in enumerate(attentions):
        for head in range(n_heads):
            ax = axes[layer, head] if n_layers > 1 else axes[head]

            weights = attn[0, head].cpu().numpy()
            im = ax.imshow(weights, cmap="Blues", aspect="auto", vmin=0, vmax=1)

            if layer == 0:
                ax.set_title(f"Head {head}", fontsize=10)
            if head == 0:
                ax.set_ylabel(f"Layer {layer}", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

            if n_ctx is not None:
                ax.axhline(y=n_ctx, color="red", linestyle="-", linewidth=0.5, alpha=0.5)
                ax.axvline(x=n_ctx, color="red", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    return fig


def analyze_attention_patterns(
    attentions: List[torch.Tensor],
    n_ctx: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute summary statistics of attention patterns.

    Args:
        attentions: List of (B, n_heads, L, L) per layer
        n_ctx: Context length (number of context tokens, not query position)

    Returns:
        Dictionary with analysis results
    """
    results = {}
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    L = attentions[0].shape[2]
    query_idx = L - 1  # Query is always at the last position

    # How much does query attend to context vs itself?
    query_to_context = torch.zeros(n_layers, n_heads)
    query_to_self = torch.zeros(n_layers, n_heads)

    # Average attention to each context position
    context_attention = torch.zeros(n_layers, n_heads, n_ctx)

    for layer, attn in enumerate(attentions):
        # attn: (B, n_heads, L, L)
        # Query is at last position, context is at positions 0..n_ctx-1
        query_attn = attn[0, :, query_idx, :]  # (n_heads, L)

        query_to_context[layer] = query_attn[:, :n_ctx].sum(dim=-1)
        query_to_self[layer] = query_attn[:, query_idx]
        context_attention[layer] = query_attn[:, :n_ctx]

    results["query_to_context"] = query_to_context  # (n_layers, n_heads)
    results["query_to_self"] = query_to_self  # (n_layers, n_heads)
    results["context_attention"] = context_attention  # (n_layers, n_heads, n_ctx)

    # Check for uniform attention (averaging over context)
    uniform = 1.0 / n_ctx
    context_attn_normalized = context_attention / context_attention.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    uniformity = 1.0 - (context_attn_normalized - uniform).abs().mean(dim=-1)
    results["uniformity"] = uniformity  # (n_layers, n_heads) - 1.0 means perfectly uniform

    return results


def print_attention_analysis(analysis: Dict[str, torch.Tensor], n_ctx: int):
    """Print a summary of attention analysis."""
    n_layers, n_heads = analysis["query_to_context"].shape

    print(f"Attention Analysis (n_ctx={n_ctx})")
    print("=" * 60)

    print("\nQuery attention to context (sum over context positions):")
    print("-" * 40)
    for layer in range(n_layers):
        vals = analysis["query_to_context"][layer]
        print(f"  Layer {layer}: " + " ".join(f"H{h}={v:.2f}" for h, v in enumerate(vals)))

    print("\nQuery self-attention:")
    print("-" * 40)
    for layer in range(n_layers):
        vals = analysis["query_to_self"][layer]
        print(f"  Layer {layer}: " + " ".join(f"H{h}={v:.2f}" for h, v in enumerate(vals)))

    print("\nAttention uniformity (1.0 = uniform over context):")
    print("-" * 40)
    for layer in range(n_layers):
        vals = analysis["uniformity"][layer]
        print(f"  Layer {layer}: " + " ".join(f"H{h}={v:.2f}" for h, v in enumerate(vals)))
