"""In-context learning for linear regression."""

from .data import make_batch, make_batch_xy, make_batch_xy_padded, make_batch_xy_fixed_padded
from .model import ICLTransformer, create_model
from .eval import ols_predict, ols_predict_from_seq, eval_rmse, sweep_context_lengths, eval_suite

__all__ = [
    "make_batch",
    "make_batch_xy",
    "make_batch_xy_padded",
    "make_batch_xy_fixed_padded",
    "ICLTransformer",
    "create_model",
    "ols_predict",
    "ols_predict_from_seq",
    "eval_rmse",
    "sweep_context_lengths",
    "eval_suite",
]
