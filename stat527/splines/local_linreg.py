"""
Local Linear Regression (LLR) in PyTorch — from scratch, kept simple.

This implementation supports 1D or multi-dimensional features.
Given training data (X, y), we estimate f(x0) at each query point x0 by fitting
a *weighted* linear model in a small neighborhood around x0 using a kernel.

Core idea:
  For each query point x0, solve a *weighted* least squares problem:
    minimize_beta  sum_i w_i(x0) * [y_i - (beta0 + beta1^T (x_i - x0))]^2
  where w_i(x0) are kernel weights (e.g., Gaussian) that downweight points far from x0.

Key properties:
  - Prediction at x0 is the fitted intercept beta0 (because we parameterize using (x - x0)).
  - We add a tiny ridge term `lam` for numerical stability.
  - Uses torch.linalg.solve instead of matrix inverse for stability and clarity.
"""

import torch
from typing import Callable, Optional

# -----------------------------
# Kernels
# -----------------------------
def gaussian_kernel(r: torch.Tensor) -> torch.Tensor:
    """
    Gaussian kernel K(r) = exp(-0.5 * r^2).
    r should be nonnegative (typically distances scaled by bandwidth).
    """
    return torch.exp(-0.5 * r * r)


# -----------------------------
# Distance utilities
# -----------------------------
def pairwise_scaled_distance(X: torch.Tensor, x0: torch.Tensor, h: float) -> torch.Tensor:
    """
    Compute Euclidean distances from each row of X to the single point x0,
    scaled by the bandwidth h.

    Args:
      X:  (n, d) training features
      x0: (d,)   query point
      h:  float  bandwidth (>0)

    Returns:
      r: (n,) distances / h
    """
    diff = X - x0  # (n, d)
    # Euclidean norm per row
    dist = torch.linalg.norm(diff, dim=1)  # (n,)
    r = dist / float(h)
    return r


# -----------------------------
# Core LLR solver for one query point
# -----------------------------
def _local_linear_at_point(
    X: torch.Tensor,
    y: torch.Tensor,
    x0: torch.Tensor,
    h: float,
    kernel: Callable[[torch.Tensor], torch.Tensor],
    lam: float = 1e-10,
) -> torch.Tensor:
    """
    Compute the local linear regression estimate at a single query point x0.

    Model near x0: y ≈ beta0 + beta1^T (x - x0)
    => Design matrix Z has first column of 1s and remaining columns (X - x0).

    Weighted least squares solution:
        beta = (Z^T W Z + lam * I)^{-1} (Z^T W y)
    where W = diag(w_i) and w_i = K( ||x_i - x0|| / h )

    The estimate of f(x0) is beta0 (the intercept) because (x - x0) = 0 at x0.

    Args:
      X: (n, d) training features
      y: (n,)   training targets
      x0: (d,)  query point
      h:  float bandwidth
      kernel: function mapping (n,) -> (n,) nonnegative weights
      lam: small ridge regularization for stability

    Returns:
      yhat0: scalar tensor, prediction at x0
    """
    n, d = X.shape

    # 1) Compute kernel weights around x0
    r = pairwise_scaled_distance(X, x0, h)  # (n,)
    w = kernel(r)                            # (n,)

    # 2) Build local design matrix Z = [1, (X - x0)]
    ones = torch.ones((n, 1), dtype=X.dtype, device=X.device)
    Z = torch.cat([ones, X - x0], dim=1)  # (n, d+1)

    # 3) Form weighted normal equations
    # We can apply weights by multiplying each row of Z and y by sqrt(w):
    #   Z_tilde = sqrt(w) * Z,   y_tilde = sqrt(w) * y
    # so that (Z^T W Z) = (Z_tilde^T Z_tilde), etc.
    sqrt_w = torch.sqrt(torch.clamp(w, min=0))
    Zt = Z * sqrt_w.unsqueeze(1)   # (n, d+1)
    yt = y * sqrt_w                # (n,)

    # 4) Solve (Zt^T Zt + lam I) beta = Zt^T yt
    # Create regularizer with 0 on intercept if you want (here we keep it simple and regularize all)
    A = Zt.T @ Zt
    b = Zt.T @ yt
    A = A + lam * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

    beta = torch.linalg.solve(A, b)  # (d+1,)

    # 5) Prediction at x0 is the intercept beta0
    return beta[0]


# -----------------------------
# Simple Estimator Class
# -----------------------------
class LocalLinearRegressor:
    """
    Minimal Local Linear Regressor.

    Usage:
        llr = LocalLinearRegressor(bandwidth=0.5)
        llr.fit(X_train, y_train)
        yhat = llr.predict(X_test)  # returns predictions at each test point

    Notes:
      - X can be 1D (n,) or 2D (n, d). Internally we store as (n, d).
      - Bandwidth controls smoothness: smaller h = more local (wigglier), larger h = smoother.
      - lam is a tiny ridge penalty to stabilize the local solve when weights are very concentrated.
      - kernel can be swapped (Gaussian by default).
    """

    def __init__(
        self,
        bandwidth: float,
        kernel: Callable[[torch.Tensor], torch.Tensor] = gaussian_kernel,
        lam: float = 1e-10,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        assert bandwidth > 0.0, "bandwidth must be positive."
        self.h = float(bandwidth)
        self.kernel = kernel
        self.lam = float(lam)
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")
        self._X = None
        self._y = None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Store training data as tensors with consistent dtype/device.

        Args:
          X: (n,) or (n, d)
          y: (n,)
        """
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device)

        if X.ndim == 1:
            X = X[:, None]  # make (n, 1)
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows."

        self._X = X.contiguous()
        self._y = y.contiguous()
        return self

    @torch.no_grad()
    def predict(self, X_query: torch.Tensor) -> torch.Tensor:
        """
        Predict at one or many query points.

        Args:
          X_query: (m,) or (m, d)

        Returns:
          yhat: (m,) predictions
        """
        assert self._X is not None and self._y is not None, "Call fit() before predict()."

        Xq = torch.as_tensor(X_query, dtype=self.dtype, device=self.device)
        if Xq.ndim == 1:
            Xq = Xq[:, None]  # (m, 1)

        m = Xq.shape[0]
        preds = torch.empty((m,), dtype=self.dtype, device=self.device)

        # Loop over query points (clear and simple; easy to vectorize later if needed)
        for j in range(m):
            x0 = Xq[j]  # (d,)
            preds[j] = _local_linear_at_point(
                self._X, self._y, x0, self.h, kernel=self.kernel, lam=self.lam
            )

        return preds


# -----------------------------
# Minimal example (commented)
# -----------------------------
if __name__ == "__main__":
    # Example: 1D noisy data from a smooth function
    torch.manual_seed(0)
    n = 200
    X = torch.linspace(-3, 3, n)
    f_true = torch.sin(X)  # underlying function
    y = f_true + 0.3 * torch.randn(n)

    # Fit LLR with a reasonable bandwidth
    llr = LocalLinearRegressor(bandwidth=0.5).fit(X, y)

    # Predict on a grid
    X_test = torch.linspace(-3, 3, 201)
    y_hat = llr.predict(X_test)

    # At this point you could plot X/y and X_test/y_hat using matplotlib if desired.
    # The core algorithm above is the full local linear regression from scratch.
