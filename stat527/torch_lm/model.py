from dataclasses import dataclass
import torch
from .algos import qr_solve
from .diagnostics import core_metrics, covariance_and_se, influence

from typing import List, Optional

def _coerce_xy(X, y, add_intercept, dtype, device):
    import torch
    import numpy as np

    # Get names if pandas
    feature_names = None
    target_name = None
    is_pandas = hasattr(X, "values") and hasattr(X, "columns")
    if is_pandas:
        feature_names = list(map(str, X.columns))
        Xv = X.values
    else:
        Xv = X

    if hasattr(y, "values") and hasattr(y, "name"):
        target_name = None if y.name is None else str(y.name)
        yv = y.values
    else:
        yv = y

    X_t = torch.as_tensor(Xv, dtype=dtype, device=device)
    y_t = torch.as_tensor(yv, dtype=dtype, device=device).reshape(-1)

    if add_intercept:
        ones = torch.ones((X_t.shape[0], 1), dtype=dtype, device=device)
        X_t = torch.cat([ones, X_t], dim=1)
        # Prepend intercept name
        feature_names = (["(Intercept)"] + feature_names) if feature_names else None

    return X_t, y_t, feature_names, target_name

@dataclass
class LMResults:
    beta: torch.Tensor
    se: torch.Tensor
    cov_beta: torch.Tensor
    sigma2: torch.Tensor
    r2: torch.Tensor
    adj_r2: torch.Tensor
    fitted: torch.Tensor
    residuals: torch.Tensor
    leverage: torch.Tensor
    stud_resid: torch.Tensor
    cooks_d: torch.Tensor
    # New (optional) metadata:
    feature_names: Optional[List[str]] = None   # includes "(Intercept)" if used
    target_name: Optional[str] = None
    n: Optional[int] = None
    p: Optional[int] = None

class LinearModel:
    def __init__(self, X=None, y=None, *, solver="qr", add_intercept=True,
                 device=None, dtype=torch.float64):
        self.solver = solver
        self.add_intercept = add_intercept
        self.device = device
        self.dtype = dtype
        self.results = None
        self._X = None
        self._y = None

        if (X is not None) and (y is not None):
            # Fit-on-init path
            self.fit(X, y)

    @torch.no_grad()
    def fit(self, X, y):
        # Use the coercion helper to get tensors and names
        X_t, y_t, feature_names, target_name = _coerce_xy(
            X, y, self.add_intercept, self.dtype, self.device
        )
        n, p = X_t.shape

        # --- choose solver (same as before) ---
        if self.solver == "qr":
            from .algos import qr_solve
            beta, xtx = qr_solve(X_t, y_t)
        elif self.solver == "svd":
            from .algos import svd_solve
            beta, xtx = svd_solve(X_t, y_t)
        else:
            from .algos import normal_equations
            beta, xtx = normal_equations(X_t, y_t)

        # --- diagnostics (same as before) ---
        from .diagnostics import core_metrics, covariance_and_se, influence
        m = core_metrics(X_t, y_t, beta)
        cov_beta, se = covariance_and_se(xtx, m["sigma2"])
        h, stud, cooks = influence(X_t, m["resid"], m["sigma2"])

        self.results = LMResults(
            beta=beta, se=se, cov_beta=cov_beta, sigma2=m["sigma2"],
            r2=m["r2"], adj_r2=m["adj_r2"],
            fitted=m["y_hat"], residuals=m["resid"],
            leverage=h, stud_resid=stud, cooks_d=cooks,
            feature_names=feature_names, target_name=target_name,
            n=n, p=p
        )
        self._X, self._y = X_t, y_t
        return self

    @torch.no_grad()
    def predict(self, X_new, add_intercept=None):
        assert self.results is not None, "Call fit() first (or pass X,y to the constructor)."
        if add_intercept is None:
            add_intercept = self.add_intercept

        # If user passes a pandas DataFrame, align columns to training order
        if hasattr(X_new, "values") and hasattr(X_new, "columns"):
            # Drop-in: assume same columns/order as training. If you want strict
            # alignment, you can store raw column names (without intercept)
            # and reorder here by name.
            Xv = X_new.values
        else:
            Xv = X_new

        Xn = torch.as_tensor(Xv, dtype=self._X.dtype, device=self._X.device)
        if add_intercept:
            ones = torch.ones((Xn.shape[0], 1), dtype=Xn.dtype, device=Xn.device)
            Xn = torch.cat([ones, Xn], dim=1)
        return Xn @ self.results.beta

    def summary(self):
        # defer to a small pretty-printer; keep math out of I/O
        from .summary import print_summary
        assert self.results is not None, "Call fit() first."
        print_summary(self.results)
