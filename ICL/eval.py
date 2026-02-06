"""Evaluation utilities for ICL transformer."""

import torch
import torch.nn.functional as F

from data import make_batch, make_batch_fixed_n_ctx


@torch.no_grad()
def ols_predict(X: torch.Tensor, y: torch.Tensor, xq: torch.Tensor, ridge: float = 0.0) -> torch.Tensor:
    """
    Compute OLS (or ridge) predictions for a batch.

    Args:
        X: (B, n, d) context features
        y: (B, n) context targets
        xq: (B, d) query features
        ridge: Ridge regularization parameter

    Returns:
        (B,) predictions
    """
    B, n, d = X.shape
    device = X.device
    dtype = X.dtype

    yhat = torch.empty(B, device=device, dtype=dtype)
    I = torch.eye(d, device=device, dtype=dtype)

    for b in range(B):
        if ridge > 0:
            XtX = X[b].T @ X[b]
            Xty = X[b].T @ y[b]
            w_hat = torch.linalg.solve(XtX + ridge * I, Xty)
        else:
            w_hat = torch.linalg.lstsq(X[b], y[b]).solution
        yhat[b] = (xq[b] * w_hat).sum()

    return yhat


@torch.no_grad()
def ols_predict_from_seq(seq: torch.Tensor, d: int, n_ctx: int, ridge: float = 0.0) -> torch.Tensor:
    """
    Compute OLS predictions from a sequence tensor.

    Args:
        seq: (B, L, d+1) sequence with [x, y] tokens
        d: Input dimension
        n_ctx: Context length
        ridge: Ridge regularization parameter

    Returns:
        (B,) predictions
    """
    X = seq[:, :n_ctx, :d]
    y = seq[:, :n_ctx, -1]
    xq = seq[:, -1, :d]
    return ols_predict(X, y, xq, ridge=ridge)


@torch.no_grad()
def eval_rmse(model, d: int, n_ctx: int, batch_size: int = 2048, noise_std: float = 0.05, device: str = "cuda"):
    """
    Evaluate model RMSE against OLS baseline.

    Args:
        model: ICLTransformer model
        d: Input dimension
        n_ctx: Context length
        batch_size: Evaluation batch size
        noise_std: Label noise std
        device: Device

    Returns:
        (model_rmse, ols_rmse) tuple
    """
    model.eval()

    seq, yq = make_batch(batch_size, n_ctx, d, noise_std=noise_std)
    seq, yq = seq.to(device), yq.to(device)

    pred = model(seq)
    rmse_model = F.mse_loss(pred, yq).sqrt().item()

    pred_ols = ols_predict_from_seq(seq, d=d, n_ctx=n_ctx)
    rmse_ols = F.mse_loss(pred_ols, yq).sqrt().item()

    return rmse_model, rmse_ols


@torch.no_grad()
def sweep_context_lengths(
    model,
    d: int,
    n_ctx_list: list,
    batch_size: int = 4096,
    n_ctx_max: int = 128,
    noise_std: float = 0.05,
    device: str = "cuda",
):
    """
    Evaluate model across different context lengths.

    Args:
        model: ICLTransformer model
        d: Input dimension
        n_ctx_list: List of context lengths to evaluate
        batch_size: Evaluation batch size
        n_ctx_max: Maximum context length (for padding)
        noise_std: Label noise std
        device: Device

    Returns:
        List of (n_ctx, model_rmse, ols_rmse) tuples
    """
    model.eval()
    results = []

    for n_ctx in n_ctx_list:
        seq, yq, pad_mask = make_batch_fixed_n_ctx(
            batch_size, n_ctx, d, n_ctx_max=n_ctx_max, noise_std=noise_std
        )
        seq, yq, pad_mask = seq.to(device), yq.to(device), pad_mask.to(device)

        pred = model(seq, pad_mask=pad_mask)
        rm_model = F.mse_loss(pred, yq).sqrt().item()

        pred_ols = ols_predict_from_seq(seq, d=d, n_ctx=n_ctx)
        rm_ols = F.mse_loss(pred_ols, yq).sqrt().item()

        print(f"n_ctx={n_ctx:3d} | model RMSE={rm_model:.4f} | OLS RMSE={rm_ols:.4f}")
        results.append((n_ctx, rm_model, rm_ols))

    return results


@torch.no_grad()
def eval_suite(
    model,
    d: int,
    n_ctx: int,
    batch_size: int = 512,
    noise_std: float = 0.05,
    ridge: float = 0.0,
    device: str = "cuda",
):
    """
    Run comprehensive evaluation suite:
    - Normal evaluation
    - Shuffled context order
    - Permuted y values (breaks x-y pairing)
    - Wiped context
    - OLS/Ridge baseline

    Args:
        model: ICLTransformer model
        d: Input dimension
        n_ctx: Context length
        batch_size: Evaluation batch size
        noise_std: Label noise std
        ridge: Ridge regularization for OLS
        device: Device

    Returns:
        Dictionary of metrics
    """
    model.eval()

    seq, yq = make_batch(batch_size, n_ctx, d, noise_std=noise_std)
    seq, yq = seq.to(device), yq.to(device)

    # Normal evaluation
    pred = model(seq)
    rmse = F.mse_loss(pred, yq).sqrt().item()

    # Shuffle context order (permute tokens 0..n_ctx-1)
    perm = torch.randperm(n_ctx, device=device)
    seq_shuf = seq.clone()
    seq_shuf[:, :n_ctx, :] = seq_shuf[:, perm, :]
    pred_shuf = model(seq_shuf)
    rmse_shuf = F.mse_loss(pred_shuf, yq).sqrt().item()

    # Break x-y pairing: permute y among context tokens
    seq_bad = seq.clone()
    perm_y = torch.randperm(n_ctx, device=device)
    seq_bad[:, :n_ctx, -1] = seq_bad[:, perm_y, -1]
    pred_bad = model(seq_bad)
    rmse_bad = F.mse_loss(pred_bad, yq).sqrt().item()

    # Wipe context entirely
    seq_wipe = seq.clone()
    seq_wipe[:, :n_ctx, :] = 0.0
    pred_wipe = model(seq_wipe)
    rmse_wipe = F.mse_loss(pred_wipe, yq).sqrt().item()

    # OLS / ridge baseline
    yhat_ols = ols_predict_from_seq(seq, d=d, n_ctx=n_ctx, ridge=ridge)
    rmse_ols = F.mse_loss(yhat_ols, yq).sqrt().item()

    # Context sensitivity
    delta_wipe = (pred - pred_wipe).abs().mean().item()
    delta_bad = (pred - pred_bad).abs().mean().item()

    results = {
        "rmse": rmse,
        "rmse_shuf": rmse_shuf,
        "rmse_bad": rmse_bad,
        "rmse_wipe": rmse_wipe,
        "rmse_ols": rmse_ols,
        "delta_wipe": delta_wipe,
        "delta_bad": delta_bad,
    }

    print(f"RMSE (model, normal):          {rmse:.4f}")
    print(f"RMSE (model, shuffled ctx):    {rmse_shuf:.4f}")
    print(f"RMSE (model, y-permuted ctx):  {rmse_bad:.4f}")
    print(f"RMSE (model, wiped ctx):       {rmse_wipe:.4f}")
    print(f"RMSE (OLS{' ridge' if ridge > 0 else ''}):               {rmse_ols:.4f}")
    print()
    print(f"Mean |pred - pred_wipe|:       {delta_wipe:.4f}")
    print(f"Mean |pred - pred_yperm|:      {delta_bad:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    from model import create_model

    parser = argparse.ArgumentParser(description="Evaluate ICL Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--d", type=int, default=5, help="Input dimension")
    parser.add_argument("--n-ctx-max", type=int, default=128, help="Max context length")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--suite", action="store_true", help="Run full eval suite")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(
        d=args.d,
        n_ctx_max=args.n_ctx_max,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        device=device,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.suite:
        eval_suite(model, d=args.d, n_ctx=64, batch_size=args.batch_size, device=device)
    else:
        n_ctx_list = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        sweep_context_lengths(
            model, d=args.d, n_ctx_list=n_ctx_list,
            batch_size=args.batch_size, n_ctx_max=args.n_ctx_max, device=device
        )
