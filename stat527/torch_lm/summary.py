# summary.py
from __future__ import annotations

import math
from typing import Iterable, Optional, List
import torch

# Optional SciPy for p-values / CIs and F-test p-value
try:
    from scipy.stats import t as student_t  # type: ignore
    from scipy.stats import f as fdist      # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ---------- formatting helpers ----------
def _fmt_num(x, width: int = 12, dig: int = 6, style: str = "g",
             align: str = ">", sci_below: float = 1e-8) -> str:
    """
    Format number with dynamic width/precision. Switch to scientific if |x| < sci_below (and x != 0).
    style: 'f' fixed decimals, 'g' significant digits, 'e' scientific.
    """
    try:
        xv = float(x)
    except Exception:
        return "n/a".rjust(width)
    use_style = "e" if (xv != 0.0 and abs(xv) < sci_below) else style
    spec = f"{align}{width}.{dig}{use_style}"
    return format(xv, spec)


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


# ---------- main printer ----------
def print_summary(
    results,
    feature_names: Optional[Iterable[str]] = None,
    *,
    alpha: float = 0.05,
    show_confint: bool = False,
    confint_alpha: Optional[float] = None,
    # formatting knobs
    col_widths: List[int] = (18, 14, 14, 12, 12),
    digits: int = 6,
    sci_below: float = 1e-8,
) -> None:
    """
    Pretty-print an lm()-style summary for LinearModel results.

    Name resolution priority:
      1) feature_names argument (if provided)  -> adds '(Intercept)' automatically
      2) results.feature_names (if present)    -> used as-is (often already includes intercept)
      3) generic names: '(Intercept)', 'x1', 'x2', ...

    Parameters
    ----------
    results : LMResults
        Requires fields: beta, se, fitted, residuals, r2, adj_r2, sigma2.
        Optionally: feature_names, n, p.
    feature_names : iterable of str, optional
        Names for columns of X (excluding the intercept).
    alpha : float
        Significance level used for the model F-test header language and default CI level.
    show_confint : bool
        If True, prints confidence intervals for coefficients.
    confint_alpha : float, optional
        If provided, overrides alpha for the confidence intervals only.
    col_widths : list[int]
        Column widths for the printed table.
    digits : int
        Numeric precision used in the coefficient table.
    sci_below : float
        Use scientific notation when |x| < sci_below.
    """
    # Tensors to CPU
    beta = results.beta.detach().cpu().reshape(-1)
    se = results.se.detach().cpu().reshape(-1)
    resid = results.residuals.detach().cpu().reshape(-1)
    fitted = results.fitted.detach().cpu().reshape(-1)

    n = int(getattr(results, "n", fitted.numel()))
    p = int(getattr(results, "p", beta.numel()))
    df_resid = n - p
    df_model = max(p - 1, 0)

    # ---------- resolve names ----------
    names: List[str]
    if feature_names is not None:
        names = ["(Intercept)", *list(map(str, feature_names))]
    elif getattr(results, "feature_names", None):
        # Use saved names exactly as stored
        names = list(results.feature_names)  # type: ignore[attr-defined]
    else:
        # Generic
        names = ["(Intercept)", *[f"x{i}" for i in range(1, p)]]

    # Ensure correct length
    if len(names) < p:
        names += [f"x{j}" for j in range(len(names), p)]
    elif len(names) > p:
        names = names[:p]

    # ---------- core scalars ----------
    rss = float(resid @ resid)
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

    # Overall F
    f_stat = None
    f_pval = None
    if df_model > 0 and df_resid > 0 and tss > 0:
        msr = (tss - rss) / df_model
        mse = rss / df_resid
        if mse > 0:
            f_stat = msr / mse
            if _HAVE_SCIPY:
                f_pval = 1.0 - fdist.cdf(f_stat, df_model, df_resid)

    # ---------- print ----------
    print("\nCall:  LinearModel.fit(X, y)\n")

    # Residuals five-number summary
    q = torch.quantile(
        resid.to(torch.float64),
        torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64),
    )
    labels = ["Min", "1Q", "Median", "3Q", "Max"]
    print(" ".join(lbl.rjust(10) for lbl in labels))
    print(" ".join(_fmt_num(float(v), width=10, dig=digits, sci_below=sci_below) for v in q.tolist()))
    print()

    # Coefficients
    headers = ["", "Estimate", "Std. Error", "t value", "Pr(>|t|)"]
    w = list(col_widths)
    print("".join(h.rjust(wi) for h, wi in zip(headers, w)))
    print("-" * sum(w))

    for j in range(p):
        name = names[j].rjust(w[0])
        est = _fmt_num(float(beta[j]), width=w[1], dig=digits, sci_below=sci_below)
        se_j = _fmt_num(float(se[j]),   width=w[2], dig=digits, sci_below=sci_below)
        t_j = _fmt_num(float(tvals[j]), width=w[3], dig=min(digits, 6), sci_below=sci_below)
        if pvals is not None:
            pv = float(pvals[j])
            pstr = f"{pv:.3g}".rjust(w[4])
            star = _stars(pv)
            line = f"{name}{est}{se_j}{t_j}{pstr}" + (f" {star}" if star else "")
        else:
            line = f"{name}{est}{se_j}{t_j}{'n/a'.rjust(w[4])}"
        print(line)

    if pvals is not None:
        print("\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")

    # Footer
    print(f"\nResidual standard error: {_fmt_num(sigma, width=0, dig=digits)} on {df_resid} degrees of freedom")
    print(f"Multiple R-squared: {_fmt_num(r2, width=0, dig=min(digits, 8))},\tAdjusted R-squared: {_fmt_num(adj_r2, width=0, dig=min(digits, 8))}")
    if f_stat is not None:
        if f_pval is not None:
            print(f"F-statistic: {_fmt_num(f_stat, width=0, dig=min(digits, 6))} on {df_model} and {df_resid} DF,  p-value: {f_pval:.3g}")
        else:
            print(f"F-statistic: {_fmt_num(f_stat, width=0, dig=min(digits, 6))} on {df_model} and {df_resid} DF")

    # Optional CIs
    if show_confint:
        ci_alpha = confint_alpha if confint_alpha is not None else alpha
        if _HAVE_SCIPY and df_resid > 0:
            tcrit = float(student_t.ppf(1.0 - ci_alpha / 2.0, df=df_resid))
            level = int(round((1 - ci_alpha) * 100))
            print(f"\n{level}% confidence intervals:")
            for j in range(p):
                lo = float(beta[j]) - tcrit * float(se[j])
                hi = float(beta[j]) + tcrit * float(se[j])
                print(f"  {names[j]}: [{_fmt_num(lo, width=0, dig=digits)}, {_fmt_num(hi, width=0, dig=digits)}]")
        else:
            print("\nConfidence intervals require SciPy (student-t quantiles).")
