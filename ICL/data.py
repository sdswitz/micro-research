"""
Data generation for in-context learning linear regression.

Token format:
    - Each token is (d+1)-dimensional: [x, y] where x is d-dim and y is scalar
    - Context tokens (positions 0 to n_ctx-1): [x_i, y_i] training examples
    - Query token (last position): [x_query, 0] - model predicts y_query

Sequence structure:
    [x_1, y_1]  [x_2, y_2]  ...  [x_n, y_n]  [x_query, 0]
    |___________ context ___________|        |__ query __|
"""

import torch


def make_batch(
    batch_size: int,
    n_ctx: int,
    d: int,
    noise_std: float = 0.05,
    w_std: float = 1.0,
):
    """
    Generate a batch of in-context linear regression tasks.

    Each task has a random weight vector w, and the model must predict
    y_query = x_query @ w from context examples {(x_i, y_i)}.

    Args:
        batch_size: Number of independent tasks
        n_ctx: Number of context (training) examples per task
        d: Input dimension
        noise_std: Gaussian noise added to y values
        w_std: Standard deviation of weight vectors

    Returns:
        seq: (B, n_ctx+1, d+1) - context tokens [x_i, y_i] then query [x_q, 0]
        y_query: (B,) - target y values to predict
    """
    B = batch_size
    L = n_ctx + 1  # context + query
    D_token = d + 1  # [x, y] per token

    # Random linear model per task
    w = torch.randn(B, d) * w_std

    # Context examples
    X_ctx = torch.randn(B, n_ctx, d)
    y_ctx = (X_ctx * w[:, None, :]).sum(dim=-1) + noise_std * torch.randn(B, n_ctx)

    # Query example
    x_query = torch.randn(B, d)
    y_query = (x_query * w).sum(dim=-1) + noise_std * torch.randn(B)

    # Build sequence: [x_1, y_1], [x_2, y_2], ..., [x_n, y_n], [x_query, 0]
    seq = torch.zeros(B, L, D_token)
    seq[:, :n_ctx, :d] = X_ctx      # context x values
    seq[:, :n_ctx, -1] = y_ctx      # context y values
    seq[:, -1, :d] = x_query        # query x value
    # seq[:, -1, -1] = 0            # query y slot (already zero)

    return seq, y_query


def make_batch_padded(
    batch_size: int,
    d: int,
    n_ctx_max: int = 128,
    n_ctx_min: int = 2,
    noise_std: float = 0.05,
    w_std: float = 1.0,
):
    """
    Generate a batch with variable context lengths, padded to n_ctx_max.

    Args:
        batch_size: Number of independent tasks
        d: Input dimension
        n_ctx_max: Maximum context length (sequence padded to this)
        n_ctx_min: Minimum context length
        noise_std: Gaussian noise added to y values
        w_std: Standard deviation of weight vectors

    Returns:
        seq: (B, n_ctx_max+1, d+1) - padded sequence
        y_query: (B,) - target y values to predict
        pad_mask: (B, n_ctx_max+1) - True for padded positions
        n_ctx: (B,) - actual context length per task
    """
    B = batch_size
    L = n_ctx_max + 1
    D_token = d + 1

    # Random context length per task
    n_ctx = torch.randint(low=n_ctx_min, high=n_ctx_max + 1, size=(B,))

    # Random linear model per task
    w = torch.randn(B, d) * w_std

    # Generate max context (will mask out extras)
    X_ctx = torch.randn(B, n_ctx_max, d)
    y_ctx = (X_ctx * w[:, None, :]).sum(dim=-1) + noise_std * torch.randn(B, n_ctx_max)

    # Query
    x_query = torch.randn(B, d)
    y_query = (x_query * w).sum(dim=-1) + noise_std * torch.randn(B)

    # Mask: True = padded (ignored), False = real token
    idx = torch.arange(n_ctx_max).unsqueeze(0)  # (1, n_ctx_max)
    is_real = idx < n_ctx.unsqueeze(1)          # (B, n_ctx_max)

    # Build sequence
    seq = torch.zeros(B, L, D_token)
    seq[:, :n_ctx_max, :d] = X_ctx * is_real.unsqueeze(-1)  # zero out padded x
    seq[:, :n_ctx_max, -1] = y_ctx * is_real                 # zero out padded y
    seq[:, -1, :d] = x_query

    # Padding mask
    pad_mask = torch.ones(B, L, dtype=torch.bool)
    pad_mask[:, :n_ctx_max] = ~is_real
    pad_mask[:, -1] = False  # query is never padded

    return seq, y_query, pad_mask, n_ctx


@torch.no_grad()
def make_batch_fixed_n_ctx(
    batch_size: int,
    n_ctx: int,
    d: int,
    n_ctx_max: int = 128,
    noise_std: float = 0.05,
    w_std: float = 1.0,
):
    """
    Generate a batch with fixed context length, padded to n_ctx_max.

    Useful for evaluation at specific context lengths.

    Args:
        batch_size: Number of independent tasks
        n_ctx: Context length (same for all tasks)
        d: Input dimension
        n_ctx_max: Sequence length for padding
        noise_std: Gaussian noise added to y values
        w_std: Standard deviation of weight vectors

    Returns:
        seq: (B, n_ctx_max+1, d+1) - padded sequence
        y_query: (B,) - target y values to predict
        pad_mask: (B, n_ctx_max+1) - True for padded positions
    """
    assert n_ctx <= n_ctx_max, f"n_ctx ({n_ctx}) must be <= n_ctx_max ({n_ctx_max})"

    B = batch_size
    L = n_ctx_max + 1
    D_token = d + 1

    # Random linear model per task
    w = torch.randn(B, d) * w_std

    # Context examples
    X_ctx = torch.randn(B, n_ctx, d)
    y_ctx = (X_ctx * w[:, None, :]).sum(dim=-1) + noise_std * torch.randn(B, n_ctx)

    # Query
    x_query = torch.randn(B, d)
    y_query = (x_query * w).sum(dim=-1) + noise_std * torch.randn(B)

    # Build padded sequence
    seq = torch.zeros(B, L, D_token)
    seq[:, :n_ctx, :d] = X_ctx
    seq[:, :n_ctx, -1] = y_ctx
    seq[:, -1, :d] = x_query

    # Padding mask: positions n_ctx to n_ctx_max-1 are padded
    pad_mask = torch.ones(B, L, dtype=torch.bool)
    pad_mask[:, :n_ctx] = False   # context is real
    pad_mask[:, -1] = False       # query is real

    return seq, y_query, pad_mask


# Aliases for backward compatibility
make_batch_xy = make_batch
make_batch_xy_padded = make_batch_padded
make_batch_xy_fixed_padded = make_batch_fixed_n_ctx
