import torch

@torch.no_grad()
def normal_equations(X, y):
    xtx = X.T @ X
    xty = X.T @ y
    beta = torch.linalg.solve(xtx, xty)  # stable than inv @
    return beta, xtx

@torch.no_grad()
def qr_solve(X, y):
    Q, R = torch.linalg.qr(X, mode="reduced")
    beta = torch.linalg.solve(R, Q.T @ y)
    xtx = X.T @ X  # for covariance; or compute via R.T @ R
    return beta, xtx

@torch.no_grad()
def svd_solve(X, y, rcond=None):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    if rcond is None:
        rcond = torch.finfo(S.dtype).eps * max(X.shape)
    Sinv = torch.where(S > rcond * S.max(), 1.0 / S, torch.zeros_like(S))
    beta = (Vh.T * Sinv) @ (U.T @ y)
    xtx = X.T @ X
    return beta, xtx
