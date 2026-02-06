#!/usr/bin/env python
"""Train and evaluate an ICL transformer end-to-end."""

import torch

from data import make_batch_xy_padded
from model import create_model
from eval import sweep_context_lengths, eval_suite
from train import train


def main():
    # Configuration
    d = 5
    n_ctx_max = 128
    d_model = 256
    n_heads = 8
    n_layers = 6
    batch_size = 256
    steps = 15000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("ICL Transformer Training")
    print("=" * 60)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"d={d}, n_ctx_max={n_ctx_max}, d_model={d_model}")
    print(f"n_heads={n_heads}, n_layers={n_layers}")
    print("=" * 60)
    print()

    # Train
    model = train(
        d=d,
        n_ctx_max=n_ctx_max,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        batch_size=batch_size,
        steps=steps,
        device=device,
    )

    print()
    print("=" * 60)
    print("Context Length Sweep")
    print("=" * 60)

    n_ctx_list = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    sweep_context_lengths(
        model,
        d=d,
        n_ctx_list=n_ctx_list,
        batch_size=4096,
        n_ctx_max=n_ctx_max,
        device=device,
    )

    print()
    print("=" * 60)
    print("Full Evaluation Suite (n_ctx=64)")
    print("=" * 60)

    eval_suite(model, d=d, n_ctx=64, batch_size=1024, device=device)


if __name__ == "__main__":
    main()
