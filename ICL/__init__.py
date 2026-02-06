"""In-context learning for linear regression."""

from .data import (
    make_batch,
    make_batch_padded,
    make_batch_fixed_n_ctx,
    # Aliases for backward compatibility
    make_batch_xy,
    make_batch_xy_padded,
    make_batch_xy_fixed_padded,
)
from .model import ICLTransformer, create_model
from .train import train, get_lr
from .eval import ols_predict, ols_predict_from_seq, eval_rmse, sweep_context_lengths, eval_suite
from .attention import (
    get_attention_weights,
    plot_attention_heatmap,
    plot_query_attention,
    plot_all_layers_query_attention,
    plot_attention_grid,
    analyze_attention_patterns,
    print_attention_analysis,
)
from .probe import (
    LinearProbe,
    make_batch_with_weights,
    extract_hidden_states,
    train_probe,
    evaluate_probe,
    probe_all_layers,
    plot_probe_results,
)

__all__ = [
    # Data
    "make_batch",
    "make_batch_padded",
    "make_batch_fixed_n_ctx",
    "make_batch_xy",
    "make_batch_xy_padded",
    "make_batch_xy_fixed_padded",
    # Model
    "ICLTransformer",
    "create_model",
    # Training
    "train",
    "get_lr",
    # Evaluation
    "ols_predict",
    "ols_predict_from_seq",
    "eval_rmse",
    "sweep_context_lengths",
    "eval_suite",
    # Attention
    "get_attention_weights",
    "plot_attention_heatmap",
    "plot_query_attention",
    "plot_all_layers_query_attention",
    "plot_attention_grid",
    "analyze_attention_patterns",
    "print_attention_analysis",
    # Probing
    "LinearProbe",
    "make_batch_with_weights",
    "extract_hidden_states",
    "train_probe",
    "evaluate_probe",
    "probe_all_layers",
    "plot_probe_results",
]
