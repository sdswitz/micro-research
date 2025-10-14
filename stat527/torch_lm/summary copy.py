# summary.py
from __future__ import annotations

import math
from typing import Iterable, Optional
import torch

# Optional SciPy for p-values / CI
try:
    from scipy.stats import t as student_t  # type: ignore
    from scipy.stats import f as fdist      # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


# def _fmt_num(x: float, width: int, dig: int = 6) -> str:
#     return f"{x:.{dig}f}".rjust(width)

def _fmt_num(x, width: int = 12, dig: int = 6, style: str = "f",
             sci_below: float = 1e-4) -> str:
    """
    Format a number with dynamic width/precision.
    If |x| < sci_below (and x ≠ 0), use scientific notation.
    """
    try:
        xv = float(x)
    except Exception:
        return "n/a".rjust(width)

    use_style = "e" if (xv != 0.0 and abs(xv) < sci_below) else style
    return format(xv, f">{width}.{2 if use_style == 'e' else dig}{use_style}")



def print_summary(
    results,
    feature_names: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
    show_confint: bool = False,
    confint_alpha: Optional[float] = None,
) -> None:
    """
    Pretty-print an lm()-style summary for LinearModel results.

    Parameters
    ----------
    results : LMResults
        Must provide: beta, se, fitted, residuals, r2, adj_r2, sigma2.
    feature_names : iterable of str, optional
        Names for columns of X (excluding the intercept).
    alpha : float
        Significance level used when reporting model F-test p-value header.
    show_confint : bool
        If True, prints confidence intervals for coefficients.
    confint_alpha : float, optional
        If provided, overrides alpha for the confidence intervals only.
    """
    # Pull tensors to CPU for formatting
    beta = results.beta.detach().cpu().reshape(-1)
    se = results.se.detach().cpu().reshape(-1)
    resid = results.residuals.detach().cpu().reshape(-1)
    fitted = results.fitted.detach().cpu().reshape(-1)

    n = int(fitted.numel())
    p = int(beta.numel())
    df_resid = n - p
    df_model = max(p - 1, 0)

    # Names: intercept + provided feature names (or generic)
    if feature_names is None and results.feature_names is not None:
        feature_names = results.feature_names[1:] if results.feature_names and results.feature_names[0] == "(Intercept)" else results.feature_names

    names = ["(Intercept)", *list(feature_names)]
    # Adjust length to p
    if len(names) < p:
        names += [f"x{j}" for j in range(len(names), p)]
    elif len(names) > p:
        names = names[:p]

    # Core scalars
    rss = float(resid @ resid)
    # Reconstruct y and compute TSS safely
    y = fitted + resid
    y_centered = y - float(y.mean())
    tss = float(y_centered @ y_centered)
    r2 = float(results.r2)
    adj_r2 = float(results.adj_r2)
    sigma2 = float(results.sigma2)
    sigma = math.sqrt(max(sigma2, 0.0))

    # t-stats and optional p-values
    tvals = (beta / se).to(torch.float64)
    if _HAVE_SCIPY and df_resid > 0:
        pvals = torch.tensor(
            [2.0 * (1.0 - student_t.cdf(abs(float(t)), df=df_resid)) for t in tvals],
            dtype=torch.float64,
        )
    else:
        pvals = None

    # F-statistic for overall regression (excluding intercept)
    f_stat = None
    f_pval = None
    if df_model > 0 and df_resid > 0 and tss > 0:
        msr = (tss - rss) / df_model
        mse = rss / df_resid
        if mse > 0:
            f_stat = msr / mse
            if _HAVE_SCIPY:
                f_pval = 1.0 - fdist.cdf(f_stat, df_model, df_resid)

    # Header
    print("\nCall:  LinearModel.fit(X, y)\n")

    # Residual five-number summary
    q = torch.quantile(
        resid.to(torch.float64),
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64),
    )
    print("Residuals:")
    labels = ["Min", "1Q", "Median", "3Q", "Max"]
    print(" ".join(lbl.rjust(10) for lbl in labels))
    print(" ".join(_fmt_num(float(v), 10) for v in q.tolist()))
    print()

    # Coefficients table
    headers = ["", "Estimate", "Std. Error", "t value", "Pr(>|t|)"]
    colw = [18, 14, 14, 12, 12]
    print("".join(h.rjust(w) for h, w in zip(headers, colw)))
    print("-" * sum(colw))

    for j in range(p):
        name = names[j].rjust(colw[0])
        est = _fmt_num(float(beta[j]), colw[1], dig=6)
        se_j = _fmt_num(float(se[j]), colw[2], dig=6)
        t_j = _fmt_num(float(tvals[j]), colw[3], dig=3)
        if pvals is not None:
            pv = float(pvals[j])
            pstr = f"{_fmt_num(pv, dig=5)}".rjust(colw[4])
            star = _stars(pv)
            line = f"{name}{est}{se_j}{t_j}{pstr}" + (f" {star}" if star else "")
        else:
            line = f"{name}{est}{se_j}{t_j}{'n/a'.rjust(colw[4])}"
        print(line)

    if pvals is not None:
        print("\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")

    # Footer: RSE, R^2, F
    print(f"\nResidual standard error: {sigma:.6g} on {df_resid} degrees of freedom")
    print(f"Multiple R-squared: {r2:.6g},\tAdjusted R-squared: {adj_r2:.6g}")
    if f_stat is not None:
        if f_pval is not None:
            print(f"F-statistic: {f_stat:.6g} on {df_model} and {df_resid} DF,  p-value: {f_pval:.3g}")
        else:
            print(f"F-statistic: {f_stat:.6g} on {df_model} and {df_resid} DF")

    # Optional confidence intervals
    if show_confint:
        ci_alpha = confint_alpha if confint_alpha is not None else alpha
        if _HAVE_SCIPY and df_resid > 0:
            tcrit = float(student_t.ppf(1.0 - ci_alpha / 2.0, df=df_resid))
            level = int(round((1 - ci_alpha) * 100))
            print(f"\n{level}% confidence intervals:")
            for j in range(p):
                lo = float(beta[j]) - tcrit * float(se[j])
                hi = float(beta[j]) + tcrit * float(se[j])
                print(f"  {names[j]}: [{lo:.6g}, {hi:.6g}]")
        else:
            print("\nConfidence intervals require SciPy (student-t quantiles).")
