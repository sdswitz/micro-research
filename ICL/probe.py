"""
Linear probes to analyze what the ICL transformer learns internally.

Key question: Does the model internally represent the regression weights w?
If so, we can train a linear probe to extract w from the hidden states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import matplotlib.pyplot as plt

from data import make_batch_padded
from model import ICLTransformer


class LinearProbe(nn.Module):
    """
    Linear probe to extract regression weights from hidden states.

    Given hidden state h at some position, tries to predict w.
    """

    def __init__(self, d_model: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, d_model) hidden state

        Returns:
            (B, d_out) predicted weights
        """
        return self.proj(h)


def make_batch_with_weights(
    batch_size: int,
    d: int,
    n_ctx_max: int = 128,
    n_ctx_min: int = 2,
    noise_std: float = 0.05,
    w_std: float = 1.0,
):
    """
    Generate batch that also returns the true regression weights.

    Returns:
        seq: (B, L, d+1) input sequence
        y_query: (B,) target y values
        pad_mask: (B, L) padding mask
        n_ctx: (B,) context lengths
        w: (B, d) true regression weights
    """
    B = batch_size
    L = n_ctx_max + 1
    D_token = d + 1

    n_ctx = torch.randint(low=n_ctx_min, high=n_ctx_max + 1, size=(B,))
    w = torch.randn(B, d) * w_std

    X_ctx = torch.randn(B, n_ctx_max, d)
    y_ctx = (X_ctx * w[:, None, :]).sum(dim=-1) + noise_std * torch.randn(B, n_ctx_max)

    x_query = torch.randn(B, d)
    y_query = (x_query * w).sum(dim=-1) + noise_std * torch.randn(B)

    idx = torch.arange(n_ctx_max).unsqueeze(0)
    is_real = idx < n_ctx.unsqueeze(1)

    seq = torch.zeros(B, L, D_token)
    seq[:, :n_ctx_max, :d] = X_ctx * is_real.unsqueeze(-1)
    seq[:, :n_ctx_max, -1] = y_ctx * is_real
    seq[:, -1, :d] = x_query

    pad_mask = torch.ones(B, L, dtype=torch.bool)
    pad_mask[:, :n_ctx_max] = ~is_real
    pad_mask[:, -1] = False

    return seq, y_query, pad_mask, n_ctx, w


@torch.no_grad()
def extract_hidden_states(
    model: ICLTransformer,
    seq: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
    layer: int = -1,
    position: str = "last",
) -> torch.Tensor:
    """
    Extract hidden states from a specific layer and position.

    Args:
        model: ICLTransformer with forward_with_hidden method
        seq: (B, L, d_in) input sequence
        pad_mask: (B, L) padding mask
        layer: Which layer to extract from (-1 = last layer before output)
        position: "last" for query position, "mean" for mean over context

    Returns:
        (B, d_model) hidden states
    """
    model.eval()
    _, hidden_states = model.forward_with_hidden(seq, pad_mask=pad_mask)

    h = hidden_states[layer]  # (B, L, d_model)

    if position == "last":
        return h[:, -1, :]
    elif position == "mean":
        # Mean over non-padded positions (excluding query)
        if pad_mask is not None:
            mask = ~pad_mask[:, :-1]  # Exclude query position
            h_ctx = h[:, :-1, :]
            h_sum = (h_ctx * mask.unsqueeze(-1)).sum(dim=1)
            h_mean = h_sum / mask.sum(dim=1, keepdim=True).clamp(min=1)
            return h_mean
        else:
            return h[:, :-1, :].mean(dim=1)
    else:
        raise ValueError(f"Unknown position: {position}")


def train_probe(
    model: ICLTransformer,
    d: int,
    n_ctx_max: int = 128,
    layer: int = -1,
    position: str = "last",
    probe_steps: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    log_every: int = 500,
) -> tuple[LinearProbe, dict]:
    """
    Train a linear probe to predict regression weights from hidden states.

    Args:
        model: Trained ICLTransformer
        d: Input dimension (also output dimension of probe)
        n_ctx_max: Maximum context length
        layer: Which layer to probe (-1 = last)
        position: Which position to probe ("last" or "mean")
        probe_steps: Number of training steps
        batch_size: Batch size
        lr: Learning rate
        device: Device
        log_every: Logging frequency

    Returns:
        probe: Trained LinearProbe
        metrics: Dictionary with training metrics
    """
    model.eval()
    d_model = model.d_model

    probe = LinearProbe(d_model, d).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    losses = []
    r2_scores = []

    for step in range(1, probe_steps + 1):
        seq, y_query, pad_mask, n_ctx, w_true = make_batch_with_weights(
            batch_size=batch_size,
            d=d,
            n_ctx_max=n_ctx_max,
        )
        seq = seq.to(device)
        pad_mask = pad_mask.to(device)
        w_true = w_true.to(device)

        # Extract hidden states
        h = extract_hidden_states(model, seq, pad_mask, layer=layer, position=position)

        # Predict weights
        w_pred = probe(h)
        loss = F.mse_loss(w_pred, w_true)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

        # Compute R^2
        with torch.no_grad():
            ss_res = ((w_pred - w_true) ** 2).sum()
            ss_tot = ((w_true - w_true.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            r2_scores.append(r2.item())

        if step % log_every == 0:
            print(f"step {step:5d} | probe loss: {loss.item():.4f} | R²: {r2.item():.4f}")

    metrics = {
        "losses": losses,
        "r2_scores": r2_scores,
        "final_loss": losses[-1],
        "final_r2": r2_scores[-1],
    }

    return probe, metrics


@torch.no_grad()
def evaluate_probe(
    model: ICLTransformer,
    probe: LinearProbe,
    d: int,
    n_ctx_max: int = 128,
    layer: int = -1,
    position: str = "last",
    n_samples: int = 1024,
    device: str = "cuda",
) -> dict:
    """
    Evaluate probe accuracy on fresh data.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    probe.eval()

    seq, y_query, pad_mask, n_ctx, w_true = make_batch_with_weights(
        batch_size=n_samples,
        d=d,
        n_ctx_max=n_ctx_max,
    )
    seq = seq.to(device)
    pad_mask = pad_mask.to(device)
    w_true = w_true.to(device)

    h = extract_hidden_states(model, seq, pad_mask, layer=layer, position=position)
    w_pred = probe(h)

    # MSE
    mse = F.mse_loss(w_pred, w_true).item()

    # R^2
    ss_res = ((w_pred - w_true) ** 2).sum()
    ss_tot = ((w_true - w_true.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(w_pred, w_true, dim=-1).mean().item()

    # Per-dimension correlation
    correlations = []
    for i in range(d):
        corr = torch.corrcoef(torch.stack([w_pred[:, i], w_true[:, i]]))[0, 1]
        correlations.append(corr.item())

    return {
        "mse": mse,
        "r2": r2,
        "cosine_similarity": cos_sim,
        "per_dim_correlation": correlations,
        "mean_correlation": sum(correlations) / len(correlations),
    }


def probe_all_layers(
    model: ICLTransformer,
    d: int,
    n_ctx_max: int = 128,
    position: str = "last",
    probe_steps: int = 3000,
    batch_size: int = 256,
    device: str = "cuda",
) -> dict:
    """
    Train probes for all layers and compare.

    Returns:
        Dictionary mapping layer index to (probe, metrics)
    """
    n_layers = len(model.layers)
    results = {}

    # Probe each layer (including input embeddings at index 0)
    for layer_idx in range(n_layers + 1):
        print(f"\n{'='*50}")
        print(f"Probing layer {layer_idx} / {n_layers}")
        print(f"{'='*50}")

        probe, metrics = train_probe(
            model=model,
            d=d,
            n_ctx_max=n_ctx_max,
            layer=layer_idx,
            position=position,
            probe_steps=probe_steps,
            batch_size=batch_size,
            device=device,
        )

        eval_metrics = evaluate_probe(
            model=model,
            probe=probe,
            d=d,
            n_ctx_max=n_ctx_max,
            layer=layer_idx,
            position=position,
            device=device,
        )

        results[layer_idx] = {
            "probe": probe,
            "train_metrics": metrics,
            "eval_metrics": eval_metrics,
        }

        print(f"Layer {layer_idx} eval: R²={eval_metrics['r2']:.4f}, cos_sim={eval_metrics['cosine_similarity']:.4f}")

    return results


def plot_probe_results(results: dict, title: str = "Probe R² by Layer"):
    """Plot probe performance across layers."""
    layers = sorted(results.keys())
    r2_scores = [results[l]["eval_metrics"]["r2"] for l in layers]
    cos_sims = [results[l]["eval_metrics"]["cosine_similarity"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(layers, r2_scores)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("R²")
    ax1.set_title("Probe R² by Layer")
    ax1.set_xticks(layers)

    ax2.bar(layers, cos_sims)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Probe Cosine Similarity by Layer")
    ax2.set_xticks(layers)

    plt.tight_layout()
    return fig
