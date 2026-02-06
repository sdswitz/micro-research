"""Data generation utilities for in-context learning experiments."""

import torch


def make_batch(batch_size: int, n_ctx: int, d: int, noise_std: float = 0.05, w_std: float = 1.0):
    """
    Generate a batch with alternating x and y tokens.

    Returns:
        seq: (B, 2*n_ctx+1, d+1) sequence with alternating [x_i, 0...0] and [0...0, y_i] tokens
        types: (B, 2*n_ctx+1) token type indicators (0=x, 1=y)
        yq: (B,) target y for the query
    """
    B = batch_size
    n = n_ctx
    Din = d + 1
    L = 2 * n + 1

    w = torch.randn(B, d) * w_std

    X = torch.randn(B, n, d)
    y = (X * w[:, None, :]).sum(dim=-1) + noise_std * torch.randn(B, n)

    xq = torch.randn(B, d)
    yq = (xq * w).sum(dim=-1) + noise_std * torch.randn(B)

    seq = torch.zeros(B, L, Din)
    types = torch.zeros(B, L, dtype=torch.long)

    for i in range(n):
        xi_pos = 2 * i
        yi_pos = 2 * i + 1
        seq[:, xi_pos, :d] = X[:, i, :]
        seq[:, yi_pos, -1] = y[:, i]
        types[:, yi_pos] = 1

    seq[:, -1, :d] = xq
    types[:, -1] = 0
    return seq, types, yq


def make_batch_xy(batch_size: int, n_ctx: int, d: int, noise_std: float = 0.05, w_std: float = 1.0):
    """
    Generate a batch with combined [x_i, y_i] tokens.

    Returns:
        seq: (B, n_ctx+1, d+1) where tokens 0..n_ctx-1 are [x_i, y_i] and token n_ctx is [x_query, 0]
        yq: (B,) target y for the query
    """
    B = batch_size
    L = n_ctx + 1
    Din = d + 1

    w = torch.randn(B, d) * w_std

    X = torch.randn(B, n_ctx, d)
    y = (X * w[:, None, :]).sum(dim=-1) + noise_std * torch.randn(B, n_ctx)

    xq = torch.randn(B, d)
    yq = (xq * w).sum(dim=-1) + noise_std * torch.randn(B)

    seq = torch.zeros(B, L, Din)
    seq[:, :n_ctx, :d] = X
    seq[:, :n_ctx, -1] = y
    seq[:, -1, :d] = xq
    seq[:, -1, -1] = 0.0

    return seq, yq


def make_batch_xy_padded(
    batch_size: int,
    d: int,
    n_ctx_max: int = 128,
    noise_std: float = 0.05,
    w_std: float = 1.0,
    n_ctx_min: int = 2,
):
    """
    Generate a batch with variable context lengths (padded to n_ctx_max).

    Returns:
        seq: (B, n_ctx_max+1, d+1) padded sequence
        yq: (B,) target y for the query
        pad_mask: (B, n_ctx_max+1) boolean mask (True = padded position)
        n_ctx: (B,) actual context length per example
    """
    B = batch_size
    L = n_ctx_max + 1
    Din = d + 1

    n_ctx = torch.randint(low=n_ctx_min, high=n_ctx_max + 1, size=(B,))

    w = torch.randn(B, d) * w_std

    X = torch.randn(B, n_ctx_max, d)
    y = (X * w[:, None, :]).sum(dim=-1) + noise_std * torch.randn(B, n_ctx_max)

    idx = torch.arange(n_ctx_max).unsqueeze(0)
    real = idx < n_ctx.unsqueeze(1)

    seq = torch.zeros(B, L, Din)
    pad_mask = torch.ones(B, L, dtype=torch.bool)

    seq[:, :n_ctx_max, :d] = X
    seq[:, :n_ctx_max, -1] = y
    pad_mask[:, :n_ctx_max] = ~real

    xq = torch.randn(B, d)
    yq = (xq * w).sum(dim=-1) + noise_std * torch.randn(B)

    seq[:, -1, :d] = xq
    pad_mask[:, -1] = False

    seq[:, :n_ctx_max, :] *= real.unsqueeze(-1)

    return seq, yq, pad_mask, n_ctx


@torch.no_grad()
def make_batch_xy_fixed_padded(
    batch_size: int,
    n_ctx: int,
    d: int,
    n_ctx_max: int = 128,
    noise_std: float = 0.05,
    w_std: float = 1.0,
):
    """
    Generate a batch with fixed context length, padded to n_ctx_max.

    Returns:
        seq: (B, n_ctx_max+1, d+1) padded sequence
        yq: (B,) target y for the query
        pad_mask: (B, n_ctx_max+1) boolean mask (True = padded position)
    """
    assert n_ctx <= n_ctx_max
    B = batch_size
    L = n_ctx_max + 1
    Din = d + 1

    w = torch.randn(B, d) * w_std

    X = torch.randn(B, n_ctx, d)
    y = (X * w[:, None, :]).sum(dim=-1) + noise_std * torch.randn(B, n_ctx)

    xq = torch.randn(B, d)
    yq = (xq * w).sum(dim=-1) + noise_std * torch.randn(B)

    seq = torch.zeros(B, L, Din)
    pad_mask = torch.ones(B, L, dtype=torch.bool)

    seq[:, :n_ctx, :d] = X
    seq[:, :n_ctx, -1] = y
    pad_mask[:, :n_ctx] = False

    seq[:, -1, :d] = xq
    pad_mask[:, -1] = False

    return seq, yq, pad_mask
