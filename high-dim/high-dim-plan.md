# 0) Prereqs to be crisp on (1–2 weeks, in parallel)

* Linear algebra: SVD, eigendecomposition, projections, pseudoinverse, subspaces/nullspace, conditioning.
* Probability: LLN/CLT intuition, tail bounds (Markov/Chebyshev), martingale basics (optional).
* Optimization: convex sets/functions, Lagrange multipliers, KKT, proximal gradient (just the idea).

Mini-lab: simulate $p \gg n$ regression; try to invert $X^\top X$ (fail), then solve with QR/SVD and ridge. Compare fits and coefficient norms.

# 1) Geometry + concentration (week 1–2)

**Core ideas:** Curse/blessing of dimensionality, concentration of measure, sub-Gaussian/sub-exponential tails, Johnson–Lindenstrauss, basic empirical process terms (Rademacher complexity—at least the vibe).

**Why:** These explain *why* regularization and simple algorithms work surprisingly well even when $p$ explodes.

Mini-lab: draw i.i.d. sub-Gaussian rows $x_i\in\mathbb{R}^p$. Show $\|x_i\|_2^2/p$ concentrates, and pairwise inner products $\langle x_i,x_j\rangle$ cluster near 0.

# 2) Random matrix intuition (week 2–3)

**Core ideas:** Sample covariance $\hat\Sigma$ in high-dim; Marchenko–Pastur bulk for eigenvalues, spiked covariance models, PCA “phase transitions”.

**Why:** PCA, covariance regularization, and whitening behave differently when $p/n$ is not tiny.

Mini-lab: simulate a “spiked” covariance model; show when the top eigenvector recovers the spike as $n,p$ grow with $p/n$ fixed.

# 3) Sparse models + regularization (week 3–5)

**Core ideas:** Lasso / $\ell_1$, elastic net, group lasso; restricted eigenvalue (RE) and compatibility conditions; oracle inequalities; bias–variance and the shrinkage view; ridge vs lasso; early stopping as implicit regularization.

**Why:** This is the workhorse toolbox for $p\gg n$.

Mini-labs:

* Compare OLS (QR pseudoinverse), ridge, lasso when only $s$ of $p$ features are nonzero. Track test error and support recovery vs SNR and correlation.
* Stability paths: show how lasso support changes with $\lambda$. Try cross-validation vs one-standard-error rule.

# 4) Multiple testing + FDR control (week 5)

**Core ideas:** Why familywise error is too strict in $p$-large worlds; BH/BY procedures; empirical nulls; knockoffs (intuition).

**Why:** Genomics/finance/feature screening all need principled discovery control.

Mini-lab: Simulate thousands of hypotheses with a small nonzero fraction of signals; compare BH to Bonferroni; visualize true/false discoveries vs target FDR.

# 5) High-dimensional inference (week 6)

**Core ideas:** Inference *after* selection; debiased / desparsified lasso; selective inference vs post-selection pitfalls; compatibility vs identifiability.

**Why:** Getting intervals/p-values is nontrivial once you select with lasso.

Mini-lab: Fit lasso, then compute debiased estimates for a few coordinates; check coverage under sparsity.

# 6) Beyond vectors: structured sparsity & low-rank (week 7)

**Core ideas:** Compressed sensing; RIP intuition; nuclear-norm for matrix completion; robust PCA.

**Why:** Many modern problems are better modeled as sparse/low-rank structures, not just sparse vectors.

Mini-lab: Matrix completion on a synthetic low-rank matrix with missing entries; compare naive mean-impute SVD vs nuclear-norm solver.

# 7) Covariance & graphical models (week 8)

**Core ideas:** High-dim covariance estimation (tapering, thresholding); Gaussian graphical models; graphical lasso; neighborhood selection.

**Why:** Networks and conditional independence require well-behaved precision matrices in $p$ large.

Mini-lab: Simulate a sparse precision matrix; estimate with graphical lasso; recover the edge set ROC.

# 8) Modern overparameterization (week 9)

**Core ideas:** Interpolation, min-norm interpolator, double descent, benign overfitting, ridgeless regression and implicit bias of gradient descent.

**Why:** Connect classical HD stats with what happens in very over-parametrized ML.

Mini-lab: Fit min-norm least squares (pseudoinverse) with $p \gg n$; sweep $p/n$ and noise; show test error double descent. Compare to small ridge.

# 9) Robustness + distribution shift (week 10)

**Core ideas:** Heavy tails and median-of-means; adversarial contamination; trimmed estimators; out-of-distribution generalization basics.

Mini-lab: Build a regression with $\alpha$-stable noise; compare OLS, Huber loss, and median-of-means regression.

# 10) Glue everything with SVD/QR first (always)

**Rule of thumb:** In code, *never* invert $X^\top X$. Use QR/SVD; use ridge or lasso if $p$ is not tiny. For inference, remember selection effects.

---

## Short reading list (high-yield)

* **Foundational, broad:** *The Elements of Statistical Learning* (Hastie–Tibshirani–Friedman), ch. 3, 7, 18–19 (lasso/EN, high-dim).
* **Focused, rigorous:** *High-Dimensional Statistics* (Wainwright). Perfect for sparsity, oracle bounds, RE conditions.
* **Probability backbone:** *High-Dimensional Probability* (Vershynin). Sub-Gaussian tools, concentration, random matrices—clear and applied.
* **Sparsity & inference:** *Statistics for High-Dimensional Data* (Bühlmann–van de Geer). Lasso theory, GLMs, graphical models.
* **Compressed sensing (taste):** Survey/tutorial papers by Candès/Tao/Donoho.

(You don’t need all of these to start; Wainwright + Vershynin + ESL is a great trifecta.)

---

## Concrete starter pack (you can do tonight)

1. **Reproduce p≫n failure + fixes**

* Generate $n=100, p=1000$ with $s=10$ nonzeros. Fit: pseudoinverse (min-norm), ridge (CV), lasso (CV). Plot test MSE & coefficient sparsity.
* Inspect $\text{rank}(X)$, spectrum of $X^\top X$, and condition numbers.

2. **PCA in high-dim**

* $n=300, p=600$. Add a rank-1 spike; plot leading eigenvalue vs $p/n$. See when PCA “finds” the spike.

3. **FDR**

* 10,000 t-tests; 5% truly non-null. Compare BH q=0.1 to Bonferroni.

If you want, I can drop in ready-to-run R or Python notebooks for these three—each \~50 lines—so you can iterate quickly.

---

## Mental model to carry with you

* **Estimation vs prediction:** When $p\gg n$, coefficients are often not identifiable without structure; predictions (projections) can still be excellent.
* **Structure is everything:** Sparsity, low rank, smoothness—pick a plausible inductive bias and regularize accordingly.
* **Don’t invert; factorize:** QR/SVD > normal equations. If you must stabilize, ridge a little goes a long way.
* **Inference needs care:** Selection changes the null; use selective/debiased procedures when you need error bars.

If you tell me whether you prefer R or Python for the mini-labs, I’ll hand you a compact starter notebook for “(1) p≫n regression: min-norm vs ridge vs lasso” tailored to your setup.
