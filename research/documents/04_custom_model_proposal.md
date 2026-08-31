# Custom Model Proposal: IC-Net — Training on the Objective You Actually Trade

*Answers two questions: (1) do professional quant researchers build their own models rather than importing them, and when is that justified? (2) A concrete proposal — implemented, tested, and validated in this repo at `src/models/icnet.py` — for a from-scratch model that encodes the cross-sectional trading problem in its loss function. Extends backlog item 12 (completing the model ladder) and goes one step past it.*

---

## 1. Do researchers build their own models? Yes — where the financial prior lives

The pattern across the serious literature is consistent: **packages for commodity components, custom code where the economics is.** Gu–Kelly–Xiu built their own neural networks rather than using an off-the-shelf tabular learner; Kelly–Pruitt–Su's IPCA is a custom alternating-least-squares estimator that exists nowhere in sklearn; Chen–Pelger–Zhu built an adversarial network to impose no-arbitrage moment conditions; Kelly–Malamud's maximum-Sharpe-ratio regression (MSRR) maps signals directly to portfolio weights; Kelly–Malamud–Zhou's virtue-of-complexity experiments are random Fourier features plus closed-form ridge — assembled from primitives, not imported.

What these have in common is *where* the custom work sits. Nobody hand-rolls a gradient-boosting library — that's commodity. The custom capacity goes into three places: the **objective function** (the cheapest and highest-leverage place to encode an economic prior), the **structure** the model is allowed to exploit (cross-sectional grouping, no-arbitrage constraints, latent factors), and the **estimation discipline** around it. A candidate who builds a bespoke transformer for six features has misread the lesson; a candidate who changes the *loss* to match the *use* has understood it.

## 2. The mismatch the proposal targets

Every packaged regressor in this repo's ladder (ElasticNet, LightGBM) minimizes pointwise MSE on forward returns. But the project never monetizes the level of a forecast — it sorts the cross-section each month and trades the top-against-bottom spread. The quantity that maps to P&L is the per-date cross-sectional correlation between forecast and realized forward return: the IC. MSE spends model capacity on precisely the components the long-short portfolio nets out — whether next month is an up month (the market level) and how volatile it is (the scale). Both are removed by the sort.

## 3. The proposal: IC-Net

A small multilayer perceptron — **implemented entirely in NumPy**: forward pass, hand-derived backpropagation, and Adam, with no torch and no sklearn — trained by gradient *ascent* on

maximize (1/T) · Σₜ corrₜ( f(Xₜ; W), yₜ ) − λ‖W‖²

where corrₜ is the Pearson correlation computed cross-sectionally within formation date t (on rank-normalized inputs this is a differentiable Spearman/IC surrogate). Three properties fall out of the objective, each with a financial meaning:

1. **Market-neutral by construction.** corrₜ demeans within each date, so the objective is invariant to adding any constant to a date's predictions — the model *cannot* waste capacity on market timing.
2. **Volatility-scale invariance.** corrₜ is unchanged if a month's returns are rescaled, so high-volatility months don't dominate training the way they dominate MSE.
3. **Listwise, not pointwise.** The training signal is "did you order this month's cross-section correctly" — which is literally the job the quantile constructor performs downstream.

Estimation discipline matches the rest of the repo: early stopping on a *chronological* tail of the training dates (never the test fold — same rule as optuna tuning), deterministic seed, deliberately small capacity (one hidden layer of 16 tanh units; the contribution is the objective, not the size). The gradient of the per-date correlation is derived in the module docstring — it is worth being able to reproduce on a whiteboard. The backtest engine passes formation dates to any model declaring `requires_dates = True`, so the model slots into the identical purged walk-forward and identical portfolio constructor as every other model: the comparison cannot be an artifact of different plumbing.

## 4. Verified results (synthetic panel, shipped config)

All three models on identical inputs, purged walk-forward, 168 OOS months:

| model | OOS IC | ICIR | net Sharpe | one-way turnover | alpha t (net) |
|---|---|---|---|---|---|
| elasticnet | 0.043 | 0.93 | 2.66 | 0.47 | 8.3 |
| lightgbm | 0.060 | 1.38 | 4.05 | 0.91 | 18.7 |
| **icnet** | **0.070** | **1.50** | **4.57** | **0.48** | 19.6 |

Two observations worth making in an interview:

- **IC-Net wins with half LightGBM's turnover.** The smooth correlation objective plus early stopping produces stabler forecasts than greedy tree splits, so more of the gross edge survives costs. Turnover is a *property of the objective*, not just a cost line — that connection is the practical payoff of custom loss design.
- **It wins for the right reason.** Its first-layer path importances isolate the planted interaction pair — sig_value 0.32 and sig_momentum 0.28 on top, the zero-beta placebo last at 0.07 — a cleaner attribution than LightGBM's gain importances (0.20/0.19, placebo 0.15). The test suite additionally requires it to detect planted structure OOS (t > 2) before anything else is believed.

Standing caveat: synthetic Sharpes are pedagogically inflated (the betas are planted). The claim this table supports is that the model and harness are *correct*, not that any of these numbers survive contact with real data — where the three-way ranking is an open empirical question the pipeline will report as it falls.

## 5. Honest limitations

- **Non-convex and seed-dependent.** Unlike ElasticNet (convex) and LightGBM (deterministic given seed), a neural objective has local optima; the shipped result is one seed. Backlog item 19 (seed-robustness sweep) applies with extra force here.
- **Correlation ignores magnitude.** The objective rewards ordering, not spread size. That matches quantile construction exactly, but a weight-based construction would want magnitude information — see the extension path.
- **One more trial.** Adding a model expands the selection surface; it is counted in the deflated-Sharpe trial register (n_trials is now 9), as every configuration must be.
- **Small-capacity choice is deliberate but untested at scale.** With ~200 real signals rather than 6, capacity, regularization, and feature screening all become live questions.

## 6. Alternatives considered, and the extension path

**Considered and deferred:** RFF-ridge in the Kelly–Malamud–Zhou style (kept as backlog P3 — it is an *experiment about complexity*, not a candidate production model); IPCA (answers a latent-factor question this project doesn't pose); transformers/sequence models (capacity mismatch at six features — exactly the misreading §1 warns against).

**Natural next steps, in order:** (1) add a differentiable turnover penalty to the objective — corrₜ minus λ·‖Δforecast‖ — making the cost-awareness explicit rather than incidental; (2) the MSRR move: output portfolio weights directly and maximize in-sample Sharpe with a ridge penalty, which the Kelly–Xiu survey frames as the endpoint of this line of thinking; (3) a monotonicity check of IC-Net's learned response surfaces against the planted DGP, closing the interpretability loop.

---

*Relation to the rest of the project: this module is the answer to "show me something that isn't a tutorial." It demonstrates the three things custom modeling is actually for — an objective that encodes the economics, estimation discipline that survives audit, and validation against planted truth before any real-data claim — while leaving the packaged baselines in place as the benchmark discipline that keeps the comparison honest.*
