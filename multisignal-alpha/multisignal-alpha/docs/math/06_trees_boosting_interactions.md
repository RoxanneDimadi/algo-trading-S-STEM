# 6. Trees, Boosting, and the Interaction Theorem

## 6.1 Decision trees in one paragraph

A regression tree predicts by asking yes/no questions about the features
("is momentum rank $> 0.3$?"), routing each stock down branches to a
**leaf**, and predicting the leaf's average target. Trees are step functions:
piecewise-constant surfaces over feature space. One tree is crude; the power
comes from adding many small ones.

## 6.2 Theorem: gradient boosting with squared loss = repeatedly fitting residuals

**Setup.** Build a model in stages: $F_M(x) = \sum_{m=1}^M \eta\, h_m(x)$,
where each $h_m$ is a small tree and $\eta$ is a learning rate. Stage $m$
chooses $h_m$ to reduce the loss $L(F) = \tfrac12\sum_i \big(y_i - F(x_i)\big)^2$.

**Claim.** The steepest-descent direction at stage $m$ is the vector of
**residuals** $y_i - F_{m-1}(x_i)$, so each new tree is fit to what the
current model still gets wrong.

**Proof.** Treat the model's predictions at the training points,
$F(x_1),\dots,F(x_N)$, as free variables. Then
$$
\frac{\partial L}{\partial F(x_i)} = -\big(y_i - F(x_i)\big),
$$
so the negative gradient — the direction that decreases the loss fastest —
is exactly the residual vector. Gradient descent "in function space" means:
fit the next tree $h_m$ to approximate these residuals, then take a small
step $F_m = F_{m-1} + \eta h_m$. ∎

That is the whole conceptual content of gradient boosting; LightGBM
(`make_lgbm`) adds industrial optimizations — histogram-based split search,
leaf-wise growth, per-tree row/column subsampling — which change speed and
regularization, not the mathematics. For other losses the recipe is
identical with "residual" replaced by "negative gradient of that loss," a
fact chapter 7 exploits from the opposite direction: *choose the loss so
that its gradient is the thing you care about*.

## 6.3 The interaction theorem: what additive models cannot say

The planted DGP contains $\beta_{\text{int}}\, z_1 z_2$ (momentum × value):
momentum works *better among cheap stocks*. Linear models — indeed anything
of the **additive** form $f_1(z_1) + f_2(z_2)$, however nonlinear each piece
— cannot represent this.

**Theorem.** There exist no functions $f, g$ with
$z_1 z_2 = f(z_1) + g(z_2)$ for all $z_1, z_2$.

**Proof.** Set $z_2 = 0$: $0 = f(z_1) + g(0)$ for every $z_1$, so $f$ is the
constant $-g(0)$. By symmetry ($z_1 = 0$), $g$ is the constant $-f(0)$. Then
$f + g$ is constant while $z_1 z_2$ is not. Contradiction. ∎

(Equivalent calculus fingerprint: interactions are exactly where the mixed
derivative $\partial^2 F/\partial z_1 \partial z_2 \ne 0$; additive functions
have it identically zero.)

**How trees escape.** A tree that first splits on $z_1 > 0$ and then, *within
that branch*, on $z_2 > 0$ assigns the four quadrants of $(z_1, z_2)$ four
different values — a staircase approximation to the saddle-shaped surface
$z_1 z_2$, refined further by boosting rounds. Depth ≥ 2 plus sequential
splits *is* interaction capacity.

**Check (repo).** This theorem is the entire explanation of the model
comparison table. Out of sample: elastic net IC **0.043** (it harvests the
additive part perfectly and the interaction not at all), LightGBM **0.060**,
IC-Net **0.070**. The test suite enforces the ordering's sign
(`test_models_backtest_runs_and_lgbm_beats_linear_on_interaction`), and it is
robust precisely because we *planted* the non-additive term.

## 6.4 Reading feature importances (and their fine print)

LightGBM's **gain importance** for a feature sums, over every split made on
that feature, the reduction in squared error the split achieved — "how much
work did this feature do." The repo configures `importance_type="gain"`
deliberately: the default (counting splits) measures how *often* a feature
was used, which is diffuse and nearly uninformative here.

**Check (repo).** Gain importances put the interaction pair on top
(`sig_value` 0.199, `sig_momentum` 0.189) with the placebo near the bottom —
but note the placebo still shows 0.153, not 0: trees will happily split on
noise a little, and gain accounting credits those splits. Contrast IC-Net's
path importances in chapter 7 (0.32 / 0.28 / placebo 0.07): part of the
custom model's appeal is a *cleaner attribution*, not just a higher IC.
A general lesson travels with this: importances describe **what the model
used**, which is evidence about — but not identical to — **what is true**.
