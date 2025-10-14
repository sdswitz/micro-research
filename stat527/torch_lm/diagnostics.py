import torch

@torch.no_grad()
def core_metrics(X, y, beta):
    y_hat = X @ beta
    resid = y - y_hat
    rss = resid @ resid
    tss = (y - y.mean()) @ (y - y.mean())
    n, p = X.shape
    r2 = 1 - rss / tss
    adj_r2 = 1 - (rss/(n - p)) / (tss/(n - 1))
    sigma2 = rss / (n - p)
    return {"y_hat": y_hat, "resid": resid, "rss": rss, "tss": tss,
            "r2": r2, "adj_r2": adj_r2, "sigma2": sigma2}

@torch.no_grad()
def covariance_and_se(xtx, sigma2):
    xtx_inv = torch.linalg.inv(xtx)
    cov = sigma2 * xtx_inv
    se = torch.sqrt(torch.diag(cov))
    return cov, se

@torch.no_grad()
def influence(X, resid, sigma2):
    xtx_inv = torch.linalg.inv(X.T @ X)
    h = torch.sum((X @ xtx_inv) * X, dim=1)
    denom = (1 - h).clamp(min=1e-12)
    n, p = X.shape
    # deleted-variance estimate
    s2_i = ((n - p) * sigma2 - resid**2 / denom) / (n - p - 1)
    stud = resid / torch.sqrt(s2_i * denom)
    cooks = (resid**2 / (p * sigma2)) * (h / denom)
    return h, stud, cooks
