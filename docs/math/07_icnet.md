# 7. IC-Net: When the Loss Function Is the Economics

This chapter contains the mathematical heart of the custom model
(`src/models/icnet.py`): a theorem showing precisely what mean-squared error
wastes, the objective that wastes none of it, and the complete gradient
derivation that the NumPy code implements line by line.

## 7.1 The motivating theorem: what MSE actually optimizes

Fix one date. Predictions $p$, forward returns $y$, $n$ stocks. Write
$\bar p, \bar y$ for the means, $s_p, s_q$ for the (population) standard
deviations, $\rho$ for the Pearson correlation.

**Theorem (exact per-date MSE decomposition).**
$$
\frac1n\sum_{i=1}^n (p_i - y_i)^2 \;=\; \underbrace{(\bar p - \bar y)^2}_{\text{level error}} \;+\; \underbrace{(s_p - s_y)^2}_{\text{scale error}} \;+\; \underbrace{2\, s_p\, s_y\,(1 - \rho)}_{\text{ordering error}} .
$$

**Proof.** Split each error into a mean part and a demeaned part:
$p_i - y_i = (\bar p - \bar y) + (\tilde p_i - \tilde y_i)$. Squaring and
averaging over $i$, the cross term vanishes because demeaned quantities
average to zero:
$$
\tfrac1n\sum (p_i - y_i)^2 = (\bar p - \bar y)^2 + \tfrac1n\sum(\tilde p_i - \tilde y_i)^2 .
$$
Expand the second piece: $\tfrac1n\sum \tilde p_i^2 - \tfrac2n\sum\tilde p_i\tilde y_i + \tfrac1n\sum\tilde y_i^2
= s_p^2 - 2 s_p s_y \rho + s_y^2$ (the middle term is the covariance,
$= s_p s_y \rho$ by definition of $\rho$). Finally regroup:
$s_p^2 + s_y^2 - 2s_ps_y\rho = (s_p - s_y)^2 + 2 s_p s_y(1 - \rho)$. ∎

**Read the three terms against chapters 2–3.** The portfolio is dollar-
neutral, so it is *provably indifferent* to the level term (§3.2), and the
quantile sort uses only ordering, so it is indifferent to the scale term
too. Only the third term — the correlation — maps to money. Yet an
MSE-trained model spends capacity on all three, and worse: the third term is
weighted by $s_y$, the month's return dispersion, so **volatile months
dominate MSE training** even though (Theorem B, §2.2) they carry no extra
information about ordering skill. Minimizing MSE optimizes the right thing
only *after* wasting effort on two irrelevancies, with a volatility-skewed
sample weighting.

## 7.2 The IC-Net objective

Delete the waste. Train the network $f(\cdot\,; W)$ to directly **maximize**
$$
J(W) \;=\; \frac{1}{T}\sum_{t=1}^{T} \rho_t\big(f(X_t; W),\, y_t\big) \;-\; \lambda \|W\|^2 ,
$$
the average per-date cross-sectional correlation between predictions and
forward returns, minus a ridge penalty. On rank-normalized inputs this is a
differentiable stand-in for the Spearman IC — the exact statistic every
signal in the repo is judged by. The invariances proved in §2.2 now read as
*designed-in economics*: the objective cannot reward market timing
(translation invariance = §3.2's dollar neutrality) and cannot be bribed by
volatile months (scale invariance).

## 7.3 The gradient of a correlation (full derivation)

The only nontrivial calculus is $\partial \rho / \partial p$ for one date.
Use the cosine form: with $M = I - \tfrac1n\mathbf{1}\mathbf{1}^\top$ the
demeaning matrix ($Mp = \tilde p$; note $M^\top = M$ and $MM = M$),
$$
\rho(p) = \frac{A}{B}, \qquad A = \langle \tilde p, \tilde y\rangle, \qquad B = \|\tilde p\|\,\|\tilde y\|.
$$

**Piece 1: gradient of $A$.** $A = (Mp)^\top \tilde y = p^\top (M\tilde y) = p^\top \tilde y$
(demeaning the already-demeaned $\tilde y$ changes nothing). So
$\nabla_p A = \tilde y$.

**Piece 2: gradient of $\|\tilde p\|$.** $\|\tilde p\|^2 = p^\top M p$, so
$\nabla_p \|\tilde p\|^2 = 2Mp = 2\tilde p$, and by the chain rule
$\nabla_p \|\tilde p\| = \tilde p / \|\tilde p\|$.

**Combine with the quotient rule.**
$$
\nabla_p \rho = \frac{\nabla_p A}{B} - \frac{A\, \nabla_p B}{B^2}
= \frac{\tilde y}{\|\tilde p\|\|\tilde y\|} - \frac{A\,\|\tilde y\|\,\tilde p/\|\tilde p\|}{\|\tilde p\|^2\|\tilde y\|^2}
$$
$$
\boxed{\;\nabla_p \rho \;=\; \frac{\tilde y}{\|\tilde p\|\,\|\tilde y\|} \;-\; \rho\,\frac{\tilde p}{\|\tilde p\|^2}\;}
$$
— exactly the two-term expression in `_objective_and_grad`. A pleasing
sanity property falls out free: both $\tilde y$ and $\tilde p$ have zero
mean, so the gradient has zero mean — nudging all predictions up together
cannot improve the objective, the calculus rediscovering Theorem A. The
per-date gradients are averaged over dates ($1/T$) and, because we maximize,
applied as gradient **ascent**.

## 7.4 Backpropagation through the network (matching the code line by line)

The network is deliberately small: $Z = XW_1 + b_1$, $H = \tanh(Z)$,
$p = Hw_2 + b_2$.

**Lemma.** $\tanh'(z) = 1 - \tanh^2(z)$.

**Proof.** $\tanh z = \frac{e^z - e^{-z}}{e^z + e^{-z}}$. Quotient rule with
$u = e^z - e^{-z}$, $v = e^z + e^{-z}$ (note $u' = v$, $v' = u$):
$\tanh' = \frac{v\cdot v - u\cdot u}{v^2} = 1 - (u/v)^2 = 1 - \tanh^2 z$. ∎

Let $g = \partial J / \partial p$ (the §7.3 vector, stacked over dates). The
chain rule, one layer at a time — each line is a line of `icnet.py`:

| Math | Code |
|---|---|
| $\partial J/\partial w_2 = H^\top g$ | `dw2 = H.T @ g_p` |
| $\partial J/\partial b_2 = \mathbf 1^\top g$ | `db2 = g_p.sum()` |
| $\partial J/\partial H = g\, w_2^\top$ | `dH = np.outer(g_p, self.w2)` |
| $\partial J/\partial Z = (g\,w_2^\top) \odot (1 - H^2)$ | `dZ = dH * (1 - H*H)` |
| $\partial J/\partial W_1 = X^\top\, \partial J/\partial Z$ | `dW1 = X.T @ dZ` |
| $\partial J/\partial b_1 = \mathbf 1^\top \partial J/\partial Z$ | `db1 = dZ.sum(axis=0)` |

each with $-2\lambda W$ appended for the ridge term (derivative of
$-\lambda\|W\|^2$). "Backpropagation" is nothing more mysterious than this
table: the chain rule, organized so every intermediate is reused.

## 7.5 Adam, and why the bias correction exists

Plain gradient ascent, $W \leftarrow W + \eta g$, is fragile when gradient
scales differ across parameters. Adam keeps two exponential moving averages —
$m \leftarrow \beta_1 m + (1-\beta_1) g$ (direction, smoothed) and
$v \leftarrow \beta_2 v + (1-\beta_2) g^2$ (per-parameter magnitude) — and
steps $W \leftarrow W + \eta\, \hat m / (\sqrt{\hat v} + \epsilon)$: a
momentum-smoothed direction with each coordinate normalized by its own
typical size.

**The correction, derived.** Both averages start at 0, so early values are
biased low. Unroll: $m_k = (1-\beta_1)\sum_{j=1}^{k} \beta_1^{\,k-j} g_j$. If
gradients hover around a constant $g$, the weights sum to a geometric series:
$E[m_k] \approx g\,(1-\beta_1)\frac{1-\beta_1^k}{1-\beta_1} = g\,(1 - \beta_1^k)$.
Dividing by $(1-\beta_1^k)$ — and likewise $v_k$ by $(1-\beta_2^k)$ — removes
the startup bias exactly. ∎ That division is the otherwise-cryptic pair of
lines in the training loop.

## 7.6 Early stopping, and the discipline that makes it legal

Neural training will eventually memorize noise (chapter 5's variance term,
unleashed). The fix is **early stopping**: monitor the objective on held-out
data and keep the weights from the best epoch — a form of implicit
regularization (training paths start near zero weights and grow; stopping
early caps effective complexity, cousin to ridge's explicit cap). The
non-negotiable detail is *which* data: a **chronological tail of the
training dates**, never anything from the test fold — the identical rule
chapter 8 imposes on all tuning. Randomly sampled validation rows would leak
(a random month's neighbors sit in training with overlapping information).

## 7.7 What the theory predicts — and what was measured

- **Ordering skill:** the objective optimizes the tested statistic directly.
  Measured OOS IC: **0.070** vs LightGBM 0.060 and elastic net 0.043 (168
  OOS months, identical inputs, identical walk-forward, identical portfolio
  constructor).
- **Turnover (heuristic, not a theorem):** the network is a smooth function
  of slowly-moving AR(1) features, so predictions — and hence sorts — evolve
  gradually; boosted trees are staircases whose predictions *jump* when a
  stock crosses a split boundary. Measured one-way turnover: IC-Net **0.48**
  vs LightGBM **0.91** — the smoothness dividend, worth ~0.5 net Sharpe at
  the configured costs.
- **Attribution:** first-layer path importances
  $\mathrm{imp}_k = \sum_h |W_{1,kh}||w_{2,h}|$ put the planted interaction
  pair on top (value 0.32, momentum 0.28) and the placebo last (0.07).
- **Caveats the math also predicts:** the objective is non-convex (multiple
  local optima ⇒ seed dependence — backlog item 19), and correlation ignores
  *magnitude*, which is exactly right for quantile sorts but would be
  incomplete for weight-proportional construction; the documented extension
  (a turnover penalty, then MSRR-style weight learning) addresses that.
