"""Training script for ICL transformer."""

import torch
import torch.nn.functional as F

from data import make_batch_xy_padded
from model import create_model


def train(
    d: int = 5,
    n_ctx_max: int = 128,
    n_ctx_min: int = 2,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    batch_size: int = 256,
    steps: int = 15000,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    noise_std: float = 0.05,
    log_every: int = 500,
    device: str = None,
):
    """
    Train an ICL transformer on random linear regression tasks.

    Args:
        d: Input dimension
        n_ctx_max: Maximum context length
        n_ctx_min: Minimum context length
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        batch_size: Training batch size
        steps: Number of training steps
        lr: Learning rate
        weight_decay: AdamW weight decay
        noise_std: Label noise standard deviation
        log_every: Print loss every N steps
        device: Device to use (auto-detected if None)

    Returns:
        Trained model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = create_model(
        d=d,
        n_ctx_max=n_ctx_max,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        device=device,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    use_bf16 = (
        device == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
    )
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    for step in range(1, steps + 1):
        seq, yq, pad_mask, n_ctx = make_batch_xy_padded(
            batch_size=batch_size,
            d=d,
            n_ctx_max=n_ctx_max,
            n_ctx_min=n_ctx_min,
            noise_std=noise_std,
        )
        seq = seq.to(device)
        yq = yq.to(device)
        pad_mask = pad_mask.to(device)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(device, dtype=dtype):
            pred = model(seq, pad_mask=pad_mask)
            loss = F.mse_loss(pred, yq)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0:
            print(f"step {step:5d} | train RMSE: {loss.sqrt().item():.4f} | mean n_ctx: {n_ctx.float().mean().item():.1f}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ICL Transformer")
    parser.add_argument("--d", type=int, default=5, help="Input dimension")
    parser.add_argument("--n-ctx-max", type=int, default=128, help="Max context length")
    parser.add_argument("--n-ctx-min", type=int, default=2, help="Min context length")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--steps", type=int, default=15000, help="Training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Label noise std")
    parser.add_argument("--save", type=str, default=None, help="Path to save model")

    args = parser.parse_args()

    model = train(
        d=args.d,
        n_ctx_max=args.n_ctx_max,
        n_ctx_min=args.n_ctx_min,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        noise_std=args.noise_std,
    )

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Model saved to {args.save}")
