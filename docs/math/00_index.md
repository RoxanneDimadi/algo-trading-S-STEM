# Mathematical Foundations of `multisignal-alpha`
## A proof-driven lesson plan

This folder derives, from first principles, every important formula in this
repository — what each loss function and statistic *is*, why it was chosen,
and what it does. It assumes **no finance background** and only **beginner
statistics** (you know what an average is; everything else is built here).

**The one idea that organizes everything:** this repo's demo data is
*synthetic with planted truth* — we chose the return-generating equation
ourselves. So nearly every derivation here ends with a falsifiable
**prediction** of a number, and a **Check** box comparing it to what
`make demo` actually measured. When a proof predicts a pipeline output to
within sampling error, you have verified both the math *and* the code. The
flagship example: chapter 3 predicts the momentum strategy's gross annual
return to be **10.14%** from four config numbers; the pipeline measures
**10.17%**.

### Reading order

| # | Chapter | What is proved there |
|---|---------|----------------------|
| 1 | `01_foundations.md` | Returns, expectation, variance, correlation; \|corr\| ≤ 1 (Cauchy–Schwarz); Sharpe ratio and the √12 annualization rule |
| 2 | `02_correlation_and_the_ic.md` | Correlation as a cosine; invariance theorems; Spearman's 1 − 6Σd²/(n(n²−1)) formula derived; what the IC is and why it is computed per date |
| 3 | `03_from_signal_to_portfolio.md` | Portfolio return as a dot product; dollar-neutrality theorem; the truncated-normal mean E[Z\|Z>a] = φ(a)/(1−Φ(a)); the 10.14% prediction; turnover and cost algebra; AR(1) staleness law IC(k) ≈ ρᵏ·IC(0) |
| 4 | `04_inference_time_series.md` | Why t-statistics; Var(mean) = σ²/T proof; the autocorrelation correction (Newey–West) derived; Fama–MacBeth as one regression per date |
| 5 | `05_linear_models_and_shrinkage.md` | Least squares from scratch (normal equations); what "alpha" is; bias–variance decomposition proved; ridge shrinks by 1/(1+λ), lasso soft-thresholds (proved in the orthonormal case) |
| 6 | `06_trees_boosting_interactions.md` | Gradient boosting = fitting residuals (proved for squared loss); theorem: a product z₁z₂ cannot be written as f(z₁)+g(z₂) — why linear models miss interactions and trees don't |
| 7 | `07_icnet.md` | The MSE decomposition MSE = level² + scale² + 2σₚσᵧ(1−ρ) — the motivating theorem; the IC objective's invariance proofs; the full gradient derivation matching the code line-by-line; tanh backprop; Adam |
| 8 | `08_backtest_validity_and_leakage.md` | Information sets; the purging theorem (purge ≥ horizon); predicting the leak-demo IC (≈0.43) before measuring it (0.395); the placebo that crossed t = 2 and what it teaches |
| 9 | `09_multiple_testing_and_dsr.md` | Why the best of N tries is biased; E[max] ≤ √(2 ln N) proved; the Probabilistic and Deflated Sharpe Ratio, term by term; decay as estimation |
| 10 | `10_pulse_state_space.md` | The Kalman filter derived from Bayes' rule (one completed square); theorem: its steady state is an EWMA with a data-chosen half-life; point-in-time proof for the PULSE forecast |
| 11 | `11_differentiable_trading.md` | The trading policy as an EWMA of aims (proved); the exact gradient of the Sharpe ratio; forward-mode sensitivities through the trading recursion; why exact gradients beat model-free RL |
| 12 | `12_msrr_and_execution.md` | MSRR closed form: max-Sharpe blend is Σ⁻¹μ (via Cauchy–Schwarz); the Sherman–Morrison identity behind the "Regression" name; the capture-ratio theorem κ = γ/(1−(1−γ)ρ); square-root impact |
| 13 | `13_symmetries_identifiability_composition.md` | The null-direction theorem (symmetry ⇒ gradient ⊥ θ) that predicted a real bug; why Adam random-walks flat directions; effective exposure and identifiability; composition is point-in-time by induction; the Pearson formula for a proportional book, checked to 2% |

### Notation (used throughout)

- $r_{i,t}$ — return of stock $i$ over month $t$ (a 5% gain is $r = 0.05$).
- $z_{i,t}$ — a **signal**: a number known at the *end* of month $t$ that we
  hope predicts $r_{i,t+1}$ (the *forward* return, `fwd_ret` in the code).
- $E[X]$ — expectation (long-run average); $\bar{x}$ — sample average.
- $\mathrm{Var}(X) = E[(X-E[X])^2]$; $\sigma = \sqrt{\mathrm{Var}}$ (standard deviation).
- $\tilde{x} = x - \bar{x}$ — a *demeaned* quantity (its average subtracted).
- $n$ — number of stocks on one date (the **cross-section**); $T$ — number of dates.
- Vectors are per-date cross-sections: $p = (p_1,\dots,p_n)$ are the model's
  predictions for the $n$ stocks on one date; $y$ the forward returns.
- $\langle a,b\rangle = \sum_i a_i b_i$ (dot product); $\|a\| = \sqrt{\langle a,a\rangle}$.

**Honesty labels.** Everything marked **Proof** is complete on the page.
Results marked **Stated** are quoted with a reference because a full proof
needs tools beyond this plan's scope — the label tells you which is which.

**The planted truth (used by every Check box).** The synthetic generator
(`src/data/synthetic.py`, parameters in `configs/config.yaml`) draws returns
as
$$
r_{i,t+1} \;=\; \sum_k \beta_k(t)\, z_{k,i,t} \;+\; \beta_{\text{int}}\, z_{a,i,t} z_{b,i,t} \;+\; b_i\, m_{t+1} \;+\; \varepsilon_{i,t+1},
$$
with Gaussian signals $z$ (standardized each date), market factor $m \sim N(0.006, 0.045^2)$,
market betas $b_i \sim N(1, 0.3^2)$, idiosyncratic noise $\varepsilon \sim N(0, 0.08^2)$,
and $\beta_k(t)$ stepping down by ×0.70 after each signal's "sample end" and to
×0.45 after its "publication." One signal (`sig_dead`) has $\beta = 0$ forever.
Because we wrote this equation, we can *compute* what every statistic in the
pipeline should be — and then check.
