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

This file is the whole lesson plan as one document. Each chapter also
exists on its own under `math/`, linked in the table below.

### Reading order

| # | Chapter | What is proved there |
|---|---------|----------------------|
| 1 | `math/01_foundations.md` | Returns, expectation, variance, correlation; \|corr\| ≤ 1 (Cauchy–Schwarz); Sharpe ratio and the √12 annualization rule |
| 2 | `math/02_correlation_and_the_ic.md` | Correlation as a cosine; invariance theorems; Spearman's 1 − 6Σd²/(n(n²−1)) formula derived; what the IC is and why it is computed per date |
| 3 | `math/03_from_signal_to_portfolio.md` | Portfolio return as a dot product; dollar-neutrality theorem; the truncated-normal mean E[Z\|Z>a] = φ(a)/(1−Φ(a)); the 10.14% prediction; turnover and cost algebra; AR(1) staleness law IC(k) ≈ ρᵏ·IC(0) |
| 4 | `math/04_inference_time_series.md` | Why t-statistics; Var(mean) = σ²/T proof; the autocorrelation correction (Newey–West) derived; Fama–MacBeth as one regression per date |
| 5 | `math/05_linear_models_and_shrinkage.md` | Least squares from scratch (normal equations); what "alpha" is; bias–variance decomposition proved; ridge shrinks by 1/(1+λ), lasso soft-thresholds (proved in the orthonormal case) |
| 6 | `math/06_trees_boosting_interactions.md` | Gradient boosting = fitting residuals (proved for squared loss); theorem: a product z₁z₂ cannot be written as f(z₁)+g(z₂) — why linear models miss interactions and trees don't |
| 7 | `math/07_icnet.md` | The MSE decomposition MSE = level² + scale² + 2σₚσᵧ(1−ρ) — the motivating theorem; the IC objective's invariance proofs; the full gradient derivation matching the code line-by-line; tanh backprop; Adam |
| 8 | `math/08_backtest_validity_and_leakage.md` | Information sets; the purging theorem (purge ≥ horizon); predicting the leak-demo IC (≈0.43) before measuring it (0.395); the placebo that crossed t = 2 and what it teaches |
| 9 | `math/09_multiple_testing_and_dsr.md` | Why the best of N tries is biased; E[max] ≤ √(2 ln N) proved; the Probabilistic and Deflated Sharpe Ratio, term by term; decay as estimation |
| 10 | `math/10_pulse_state_space.md` | The Kalman filter derived from Bayes' rule (one completed square); theorem: its steady state is an EWMA with a data-chosen half-life; point-in-time proof for the PULSE forecast |
| 11 | `math/11_differentiable_trading.md` | The trading policy as an EWMA of aims (proved); the exact gradient of the Sharpe ratio; forward-mode sensitivities through the trading recursion; why exact gradients beat model-free RL |
| 12 | `math/12_msrr_and_execution.md` | MSRR closed form: max-Sharpe blend is Σ⁻¹μ (via Cauchy–Schwarz); the Sherman–Morrison identity behind the "Regression" name; the capture-ratio theorem κ = γ/(1−(1−γ)ρ); square-root impact |
| 13 | `math/13_symmetries_identifiability_composition.md` | The null-direction theorem (symmetry ⇒ gradient ⊥ θ) that predicted a real bug; why Adam random-walks flat directions; effective exposure and identifiability; composition is point-in-time by induction; the Pearson formula for a proportional book, checked to 2% |

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


---
---

# 1. Foundations: Returns, Averages, Spread, and the Sharpe Ratio

## 1.1 What a return is, and what the game is

If a stock costs 100 at the end of one month and 103 at the end of the next,
its **return** over that month is $r = 103/100 - 1 = 0.03$. Returns are the
only quantity we ultimately care about; everything else in this repo exists
to *predict* them.

The specific game is **cross-sectional**: on each date we look across many
stocks at once and ask *which* will do better than *which* — not whether the
market will go up. A **signal** $z_{i,t}$ is any number computable from
information available at the end of month $t$ (past returns, accounting data,
…). A signal is useful exactly to the extent that stocks with higher $z$
today tend to have higher returns *next* month.

## 1.2 Expectation and variance (the two numbers behind everything)

For a random quantity $X$ observed repeatedly, the **expectation** $E[X]$ is
its long-run average and the **variance**
$\mathrm{Var}(X) = E[(X - E[X])^2]$ measures spread: the average *squared*
distance from the average. Its square root $\sigma$ (**standard deviation**)
is spread in the original units. Two rules we use constantly, both provable
in one line from the definition of expectation as a (weighted) sum:

- **Linearity:** $E[aX + bY] = aE[X] + bE[Y]$, always.
- **Variance of a sum of independent quantities:**
  $\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$ when $X, Y$ are
  independent.

**Proof of the second.** Demean: let $\tilde X = X - E[X]$, $\tilde Y = Y-E[Y]$.
Then $\mathrm{Var}(X+Y) = E[(\tilde X + \tilde Y)^2] = E[\tilde X^2] + 2E[\tilde X\tilde Y] + E[\tilde Y^2]$.
The middle term $E[\tilde X \tilde Y]$ is the **covariance**; for independent
variables the average of a product is the product of averages, so
$E[\tilde X\tilde Y] = E[\tilde X]E[\tilde Y] = 0$. ∎

Covariance, $\mathrm{Cov}(X,Y) = E[\tilde X \tilde Y]$, is positive when $X$
and $Y$ tend to be above (or below) their averages *together*.

## 1.3 Correlation, and a real proof that it lives in [−1, 1]

Covariance has awkward units (return × return). Dividing by both standard
deviations gives the unit-free **correlation**:
$$
\rho(X,Y) \;=\; \frac{\mathrm{Cov}(X,Y)}{\sigma_X\, \sigma_Y}.
$$

**Theorem (Cauchy–Schwarz).** $|\rho| \le 1$ always.

**Proof.** For any number $t$, the quantity $(\tilde X + t\tilde Y)^2$ is a
square, so its expectation is $\ge 0$:
$$
0 \;\le\; E[(\tilde X + t\tilde Y)^2] \;=\; \underbrace{E[\tilde Y^2]}_{c}\,t^2 + \underbrace{2E[\tilde X\tilde Y]}_{b}\,t + \underbrace{E[\tilde X^2]}_{a}.
$$
A quadratic $ct^2 + bt + a$ that is never negative cannot have two distinct
real roots, so its discriminant satisfies $b^2 - 4ca \le 0$, i.e.
$4\,\mathrm{Cov}(X,Y)^2 \le 4\,\mathrm{Var}(Y)\mathrm{Var}(X)$. Divide by
$4\sigma_X^2\sigma_Y^2$ and take square roots: $|\rho| \le 1$. ∎

So $\rho = 1$ means a perfect increasing straight-line relationship,
$\rho = -1$ perfect decreasing, $\rho = 0$ no *linear* relationship. Keep the
proof's trick in mind — "a square has nonnegative expectation" — it is the
engine behind half of statistics.

**Calibration for this project (memorize this):** honest return-prediction
correlations are *tiny*. A monthly cross-sectional correlation of **0.05**
between a signal and next month's returns is *strong* in this business
(chapter 2 makes this precise as the "IC"). If you ever compute 0.3+, the
correct reaction is not excitement but a hunt for the timestamp error
(chapter 8 demonstrates this deliberately).

## 1.4 Standardization and why we compare apples to apples

A signal measured in dollars and one measured in percent can't be combined
raw. The **z-score** $x \mapsto (x - \bar x)/\sigma_x$ recenters to average 0
and rescales to spread 1. This repo goes one step further and uses **ranks**
(chapter 2), which also kill outliers: the biggest value becomes "rank $n$"
no matter *how* big it is. The exact map used everywhere in the code
(`rank_normalize_cross_section`) sends each date's values to evenly spaced
points in $[-1, 1]$ — an (approximately) **uniform** distribution, a fact
chapter 3 exploits in a computation.

## 1.5 The Sharpe ratio, and a proof of the √12 rule

A strategy's monthly returns $r_1, \dots, r_T$ have average $\mu$ and
standard deviation $\sigma$. The **Sharpe ratio** is
$$
\mathrm{SR}_{\text{monthly}} = \mu / \sigma :
$$
reward per unit of risk. It is the right *ratio* because a strategy can
always be scaled (bet twice as much: both $\mu$ and $\sigma$ double, SR is
unchanged) — SR measures quality independent of sizing.

Convention reports SR **annualized**. The rule is: multiply by $\sqrt{12}$.

**Proof (with its assumption visible).** Assume monthly returns are
independent with the same $\mu, \sigma$, and approximate the annual return
as the sum of 12 monthly returns. By linearity, the annual mean is $12\mu$;
by independence (§1.2), the annual variance is $12\sigma^2$, so the annual
standard deviation is $\sqrt{12}\,\sigma$. Hence
$$
\mathrm{SR}_{\text{annual}} = \frac{12\mu}{\sqrt{12}\,\sigma} = \sqrt{12}\;\frac{\mu}{\sigma}. \qquad \blacksquare
$$
Both assumptions are approximations for real returns (compounding is not a
sum; months are not perfectly independent). Chapter 4 is entirely about what
happens to inference when independence fails — and the code's use of
Newey–West standard errors is the repair.

**Check (repo).** `src/utils/stats.py::annualized_stats` implements exactly
$12\mu$, $\sqrt{12}\sigma$, and their ratio, and additionally records the
skewness and kurtosis of the monthly series — not for decoration: chapter 9's
Deflated Sharpe Ratio needs those higher moments, and this is where they are
computed.

## 1.6 Where we are going

Chapters 2–3 build the two evaluation pillars: the **IC** (a correlation —
the *statistical* test of a signal) and the **long-short portfolio** (a dot
product — the *economic* test). Chapter 4 makes the t-statistics honest.
Chapters 5–7 are the three models, culminating in IC-Net, whose loss function
*is* the correlation from this chapter, maximized directly. Chapters 8–9 are
the immune system: leakage and selection bias, the two ways backtests lie.


---
---

# 2. Correlation as Geometry, Ranks, and the Information Coefficient

## 2.1 Correlation is a cosine

Take one date's cross-section: predictions $p = (p_1,\dots,p_n)$ and forward
returns $y = (y_1,\dots,y_n)$, one entry per stock. Demean each
($\tilde p = p - \bar p\mathbf{1}$, $\tilde y = y - \bar y\mathbf{1}$, where
$\mathbf 1$ is the all-ones vector). The **sample correlation** is
$$
\rho(p, y) \;=\; \frac{\langle \tilde p, \tilde y\rangle}{\|\tilde p\|\,\|\tilde y\|},
$$
which is *literally the cosine of the angle* between the two demeaned vectors
in $n$-dimensional space (that is the definition of the angle between
vectors). Perfectly aligned: $\rho = 1$. Opposite: $-1$. Perpendicular: $0$.
This geometric picture makes the next two theorems one-line proofs, and it is
the exact expression IC-Net differentiates in chapter 7.

## 2.2 Two invariance theorems (small proofs, large consequences)

**Theorem A (translation invariance).** $\rho(p + c\mathbf 1,\, y) = \rho(p, y)$
for any constant $c$.

**Proof.** Demeaning kills constants: $(p + c\mathbf 1) - \overline{(p+c\mathbf 1)}\,\mathbf 1
= p + c\mathbf 1 - (\bar p + c)\mathbf 1 = \tilde p$. The formula only sees
$\tilde p$. ∎

**Theorem B (positive scale invariance).** $\rho(p,\, \lambda y) = \rho(p, y)$
for any $\lambda > 0$ (and likewise in $p$).

**Proof.** $\widetilde{\lambda y} = \lambda\tilde y$, so the numerator gains a
factor $\lambda$ and the denominator gains $\|\lambda \tilde y\| = \lambda\|\tilde y\|$;
they cancel. ∎

*Why these matter here:* Theorem A says correlation cannot reward predicting
the *level* of a date's returns ("is this an up month?") — only the ordering
across stocks. Theorem B says a violently volatile month, with all its
returns stretched by a common factor, contributes the same correlation as a
calm one. Chapter 3 proves the long-short portfolio has exactly the same two
indifferences — which is precisely why chapter 7 argues correlation is the
*right training objective* and mean-squared-error is the wrong one.

## 2.3 Ranks, and deriving Spearman's famous formula

**Spearman correlation** = Pearson correlation applied to the *ranks* of the
data (smallest value gets rank 1, …, largest gets rank $n$; assume no ties).
Robustness is the reason: replacing the largest return by something 10× as
large changes Pearson a lot but changes ranks not at all.

Textbooks state Spearman's shortcut $\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$
where $d_i$ is the difference between stock $i$'s two ranks. Here is where it
comes from — a pleasant exercise in the definitions.

**Step 1: mean and variance of ranks.** Ranks are the numbers $1,\dots,n$ in
some order, so $\bar R = \frac1n\sum_{k=1}^n k = \frac{n+1}{2}$ and, using
$\sum k^2 = \frac{n(n+1)(2n+1)}{6}$,
$$
\mathrm{Var}(R) = \frac1n\sum k^2 - \bar R^2
= \frac{(n+1)(2n+1)}{6} - \frac{(n+1)^2}{4}
= \frac{(n+1)\big(2(2n+1) - 3(n+1)\big)}{12}
= \frac{n^2-1}{12}.
$$

**Step 2: covariance via the $d_i$'s.** Let $R_i, S_i$ be the two rank lists
and $d_i = R_i - S_i$. Expanding $d_i^2 = R_i^2 - 2R_iS_i + S_i^2$ and summing,
$$
\sum R_iS_i = \tfrac12\Big(\sum R_i^2 + \sum S_i^2 - \sum d_i^2\Big)
= \frac{n(n+1)(2n+1)}{6} - \tfrac12\sum d_i^2 ,
$$
because both rank lists contain the same numbers $1..n$. Then
$$
\mathrm{Cov}(R,S) = \frac1n\sum R_iS_i - \bar R\,\bar S
= \frac{(n+1)(2n+1)}{6} - \frac{\sum d_i^2}{2n} - \frac{(n+1)^2}{4}
= \frac{n^2-1}{12} - \frac{\sum d_i^2}{2n}.
$$

**Step 3: divide.** $\rho_s = \mathrm{Cov}/\mathrm{Var} = 1 - \dfrac{6\sum d_i^2}{n(n^2-1)}$. ∎

The code never uses the shortcut (`scipy.stats.spearmanr` computes Pearson on
ranks directly, ties handled by average ranks), but the derivation teaches
the real content: **Spearman is nothing new — it is Pearson after the rank
transform**, inheriting §2.2's invariances and adding outlier immunity.

## 2.4 The Information Coefficient (IC): definition and design choices

$$
\mathrm{IC}_t \;=\; \text{Spearman correlation, across stocks } i \text{, between } z_{i,t} \text{ and } r_{i,t+1}.
$$

One number per date; the headline statistic is the time-series mean
$\overline{\mathrm{IC}}$, with dispersion $\sigma_{\mathrm{IC}}$ and the
stability ratio $\mathrm{ICIR} = \overline{\mathrm{IC}}/\sigma_{\mathrm{IC}}$
(implemented in `src/evaluation/ic.py`). Two design choices carry the theory:

**(a) Forward, never contemporaneous.** The pair is (signal known at $t$,
return over $t\!\to\!t{+}1$). Correlating a signal with the *same* month's
return measures description, not prediction — chapter 8 quantifies exactly
how spectacular that mistake looks.

**(b) Per date, then average — never pooled.** Suppose you instead pooled all
$(z, r)$ pairs across dates into one giant correlation. Write each variable
as (its date average) + (deviation from the date average). The pooled
covariance then splits into a *between-dates* part — do months with a
high average signal have high average returns? — and a *within-date* part,
which is the only part a market-neutral portfolio can harvest. Computing the
correlation within each date and averaging (the IC) isolates the within-date
part by construction: every date's demeaning removes that date's market
level (Theorem A), so market-direction effects contribute exactly zero.

**Sampling noise under "no skill" (used repeatedly later).** If a signal is
pure noise, its per-date IC is a correlation between $n$ independent pairs;
a standard result (Stated; see any mathematical-statistics text) is that its
standard deviation is $\approx 1/\sqrt{n-1}$. With $n = 500$ stocks:
$\sigma_{\mathrm{IC}} \approx 0.0448$ per date. Averaging $T = 298$
independent dates divides the noise by $\sqrt{T}$ (chapter 4 proves this):
the mean IC of a *useless* signal fluctuates with standard error
$0.0448/\sqrt{298} \approx 0.0026$.

**Check (repo).** The placebo `sig_dead` measured $\overline{\mathrm{IC}} = 0.0048$
— about 1.9 of those standard errors from zero, i.e. exactly the size of
fluctuation the null predicts, and correctly *not* significant. The planted
signals measured 0.010–0.037, i.e. 4–11 standard errors: cleanly detected.
Chapter 3 predicts the 0.037 itself from the config.


---
---

# 3. From Signal to Portfolio: The Economics as Algebra

The IC is the statistical test. The **long-short portfolio** is the economic
one: can the signal's ordering be turned into money? This chapter derives the
portfolio's properties and then does the most satisfying thing in the whole
lesson plan: it **predicts the pipeline's measured 10.17% annual return from
four numbers in the config file.**

## 3.1 A portfolio is a dot product

Assign each stock a **weight** $w_i$ (fraction of capital; negative =
**short**, i.e. borrow the stock, sell it, profit if it falls). The
portfolio's return over the month is
$$
r_p \;=\; \sum_i w_i\, y_i \;=\; \langle w, y\rangle ,
$$
because each dollar of weight earns that stock's return. The repo's
constructor (`src/evaluation/portfolio.py::score_to_weights`) sorts each
date's stocks into 5 **quintiles** by score and sets
$$
w_i = \begin{cases} +1/n_{\text{top}} & i \in \text{top quintile}\\[2pt]
-1/n_{\text{bot}} & i \in \text{bottom quintile}\\[2pt] 0 & \text{else,}\end{cases}
$$
i.e. \$1 long the best fifth, \$1 short the worst fifth.

## 3.2 Theorem (dollar neutrality): the market drops out

**Theorem.** If $\sum_i w_i = 0$, then adding any constant $c$ to *every*
stock's return leaves the portfolio return unchanged:
$\langle w,\, y + c\mathbf 1\rangle = \langle w, y\rangle$.

**Proof.** $\langle w, y + c\mathbf 1\rangle = \langle w,y\rangle + c\sum_i w_i = \langle w,y\rangle$. ∎

The weights above sum to zero by construction ($+1$ and $-1$). So a month in
which *everything* rises 8% contributes nothing: the long leg's gain is the
short leg's loss. Compare Theorem A of chapter 2: the portfolio is
indifferent to the return *level* in exactly the way correlation is — the
first half of the argument that correlation is the natural training
objective (completed in chapter 7).

## 3.3 A tool: the mean of the top slice of a bell curve

To predict the strategy's return we need: *what is the average signal value
inside the top quintile?* The synthetic returns are generated from Gaussian
(bell-curve) signals, and sorting preserves order, so we need the mean of a
standard normal *conditional on being in its top 20%*.

**Lemma.** For standard normal $Z$ with density $\phi(z) = \tfrac{1}{\sqrt{2\pi}}e^{-z^2/2}$
and CDF $\Phi$: $\;E[Z \mid Z > a] = \dfrac{\phi(a)}{1 - \Phi(a)}$.

**Proof.** The key is that $\phi'(z) = -z\,\phi(z)$ (differentiate the
exponential). Therefore
$\int_a^\infty z\,\phi(z)\,dz = \big[-\phi(z)\big]_a^\infty = \phi(a)$.
Dividing by the probability of the event, $P(Z>a) = 1-\Phi(a)$, gives the
conditional mean. ∎

Top quintile: $a = \Phi^{-1}(0.8) = 0.8416$, so
$E[Z \mid \text{top } 20\%] = \phi(0.8416)/0.2 = 0.2800/0.2 = \mathbf{1.400}$.
By symmetry the bottom quintile averages $-1.400$: the **long–short spread in
signal units is 2.80**.

## 3.4 The flagship prediction: 10.14% vs measured 10.17%

The planted model (index page) says $r_{i,t+1} = \beta\, z_{i,t} + (\text{terms
with mean 0 in both legs})$: the interaction has mean zero given the sort
(the *other* signal is independent), market exposure $b_i$ is independent of
$z$ so both legs average $b \approx 1$ and cancel by §3.2, and noise averages
out. So the expected monthly long–short return is
$$
E[r_p] \;=\; \beta \cdot \big(E[z\mid\text{top}] - E[z\mid\text{bot}]\big) \;=\; 2.80\,\beta .
$$

Now the config numbers for `sig_momentum`: $\beta = 0.0040$ in-sample,
stepped to $0.70\beta$ after 2012-12 and $0.45\beta$ after 2014-12. The panel
spans 299 months of which about 155 are in-sample, 24 post-sample, 120
post-publication, so the *time-averaged* multiplier is
$$
\frac{155(1.0) + 24(0.70) + 120(0.45)}{299} \;=\; 0.755 .
$$
Prediction: $12 \times 2.80 \times 0.0040 \times 0.755 = \mathbf{0.1014}$,
i.e. **10.14% per year, gross**.

**Check (repo).** `results/tables/signal_evaluation.csv` measures
`sig_momentum` gross annual return $= \mathbf{0.1017}$. The derivation and
the code agree to 0.3% — with *zero* fitted parameters. This one check
exercises the generator, the rank sort, the weight constructor, the forward-
return alignment, and the annualization all at once. (Volatility can be
predicted the same way — noise averaging $\sigma/\sqrt{n_{\text{leg}}}$ per
leg plus a small market-beta mismatch term — giving monthly $\approx 0.0115$
and hence in-sample Sharpe $\approx \sqrt{12}\cdot(2.8\times0.004)/0.0115 \approx 3.4$;
measured in-sample Sharpe: 3.64. Same machinery, ~7% agreement; the gap is
sampling error plus the small terms we dropped.)

**Predicting the IC too.** The per-date correlation between $z$ and $r$ under
the planted model is $\beta$ divided by the cross-sectional return spread:
$\rho \approx \beta / \sqrt{\beta_{\text{all}}^2\text{-terms} + b\text{-dispersion}^2 E[m^2] + \sigma_\varepsilon^2}
= 0.0040/0.0813 = 0.049$ in-sample (Pearson). Two adjustments: Spearman on
near-Gaussian data is slightly smaller (factor $\tfrac{6}{\pi}\arcsin(\rho/2) \approx 0.955\rho$
for small $\rho$ — Stated, classical result for bivariate normals), and the
decay multiplier 0.755 applies. Prediction:
$0.049 \times 0.955 \times 0.755 = \mathbf{0.0355}$. Measured: **0.0374**.
Within sampling error (chapter 2 put the standard error at $\approx 0.0026$).

## 3.5 Turnover and costs: the algebra of friction

Weights change each month; trading costs money. With weight vectors $w_t$
(pivoted stock-by-date), the **traded notional** is
$\text{traded}_t = \sum_i |w_{t,i} - w_{t-1,i}|$ — every dollar bought or
sold — and the net return is
$$
r^{\text{net}}_t \;=\; r^{\text{gross}}_t \;-\; \text{traded}_t \times \frac{\text{cost}_{\text{bps}}}{10{,}000}.
$$
"One-way turnover" $= \text{traded}_t/2$ (a \$1 sale funding a \$1 purchase is
\$2 traded, one repositioning). This is deliberately the simplest defensible
cost model; `docs/03_improvement_backlog.md` lists its two honest
refinements (drift correction; size-dependent costs).

## 3.6 Persistence, staleness, and the law IC(k) ≈ ρᵏ · IC(0)

The synthetic signals follow an **AR(1)**:
$z_t = \rho z_{t-1} + \sqrt{1-\rho^2}\,\epsilon_t$ with fresh noise
$\epsilon_t$ (variance 1).

**Claim 1 (variance is stable at 1).** If $\mathrm{Var}(z_{t-1}) = 1$ then
$\mathrm{Var}(z_t) = \rho^2\cdot 1 + (1-\rho^2)\cdot 1 = 1$ (independence of
$\epsilon_t$, §1.2). ∎

**Claim 2 (correlation across k months is ρᵏ).**
$\mathrm{Cov}(z_t, z_{t-1}) = \rho\,\mathrm{Var}(z_{t-1}) = \rho$
(the noise term is uncorrelated with the past); iterate $k$ times to get
$\mathrm{Corr}(z_t, z_{t-k}) = \rho^k$. ∎

**Consequence.** A $k$-month-old signal is (in the correlation sense) a
$\rho^k$-strength copy of today's plus unrelated noise, so its predictive
correlation is scaled: $\mathrm{IC}(k\text{-stale}) \approx \rho^k\,\mathrm{IC}(\text{fresh})$.
This is the mathematical content of the "Anomaly Time" lesson (stale
formation hides real signal), turned into a *rate*.

**Check (repo).** `results/tables/staleness_profile.csv`, momentum
($\rho = 0.90$): measured IC ratios at lags 1–4 are **0.93, 0.77, 0.65, 0.52**
vs predicted $0.90, 0.81, 0.73, 0.66$ — matching at short lags and decaying
slightly faster at long lags (the rank transform and the time-varying decay
multipliers both shave long-lag persistence; each measured ratio carries a
standard error of roughly $\pm 0.09$). The qualitative law — geometric decay
at rate set by the signal's persistence — is exactly what the table shows,
and it is also why slow signals (`sig_value`, $\rho = 0.98$) trade little
(measured one-way turnover 0.23) while fast ones trade a lot (momentum 0.51):
**turnover is persistence, seen from the portfolio's side.**


---
---

# 4. Honest Uncertainty: t-Statistics, Newey–West, and Fama–MacBeth

Every headline in this repo is a **time-series average** — mean IC, mean
long-short return, alpha. Averages computed from finite noisy data are
themselves noisy. This chapter derives how noisy, why naive formulas
understate the noise for market data, and how the code corrects for it.

## 4.1 The standard error of an average (proved)

**Theorem.** If $x_1,\dots,x_T$ are uncorrelated with common mean $\mu$ and
variance $\sigma^2$, then the sample mean $\bar x = \frac1T\sum x_t$
satisfies $\mathrm{Var}(\bar x) = \sigma^2/T$.

**Proof.** $\mathrm{Var}(\bar x) = \frac{1}{T^2}\mathrm{Var}\big(\sum x_t\big)
= \frac{1}{T^2}\sum_t \mathrm{Var}(x_t) = \frac{T\sigma^2}{T^2}$, using
§1.2's variance-of-a-sum rule (cross terms vanish because the $x_t$ are
uncorrelated). ∎

The **standard error** is $\mathrm{se}(\bar x) = \sigma/\sqrt T$, and the
**t-statistic** $t = \bar x / \mathrm{se}(\bar x)$ counts how many standard
errors the average sits from zero. By the Central Limit Theorem (Stated:
sums of many small independent effects are approximately bell-curved), under
the null "true mean $= 0$" the t-stat is approximately standard normal, so
$|t| > 2$ happens by luck only $\approx 5\%$ of the time. That is the entire
content of "$t > 2$ = significant" — and chapter 9 is about how this
guarantee *dissolves* when you test many things.

## 4.2 What autocorrelation does to that formula (derived)

Strategy returns and IC series are **autocorrelated**: a good month is more
likely after a good month. Then the cross terms in the proof above do *not*
vanish. Redo the computation keeping them, with $\rho_k$ the correlation
between observations $k$ apart:
$$
\mathrm{Var}(\bar x) = \frac{1}{T^2}\Big[\sum_{t}\mathrm{Var}(x_t) + 2\!\!\sum_{t<s}\!\mathrm{Cov}(x_t,x_s)\Big]
= \frac{\sigma^2}{T}\Big[1 + 2\sum_{k=1}^{T-1}\Big(1-\frac{k}{T}\Big)\rho_k\Big],
$$
since there are exactly $T-k$ pairs at distance $k$. **If the $\rho_k$ are
positive, the true variance of your average is *larger* than $\sigma^2/T$** —
the naive t-stat divides by too small a number and overstates significance.
Intuition: with persistence, $T$ correlated months contain fewer than $T$
independent pieces of information.

## 4.3 Newey–West: estimating the correction robustly

Newey–West (1987) plugs *estimated* autocovariances into the bracket,
truncated at a lag $L$ and damped by **Bartlett weights** $1 - \frac{k}{L+1}$:
$$
\widehat{\mathrm{Var}}_{\text{NW}}(\bar x) = \frac{1}{T}\Big[\hat\gamma_0 + 2\sum_{k=1}^{L}\Big(1-\frac{k}{L+1}\Big)\hat\gamma_k\Big],
$$
where $\hat\gamma_k$ is the sample autocovariance at lag $k$. The triangular
weights are not cosmetic: they guarantee the estimate is never negative
(Stated — the weighted sum is a smoothed spectral density at frequency zero,
which is nonnegative), whereas raw truncation can produce a "negative
variance." The repo uses $L = 6$ for monthly data throughout — enough to
absorb within-half-year persistence without drowning the estimate in noise.

**Implementation identity worth knowing** (`src/utils/stats.py::nw_mean_test`):
regressing a series on a constant gives intercept $=\bar x$; asking the
regression for HAC (Newey–West) standard errors gives exactly the formula
above. One tested code path serves every mean in the project. The same
machinery with real regressors is the factor-controlled **alpha** test of
chapter 5.

## 4.4 Fama–MacBeth: one regression per date (1973, still the workhorse)

Question: does signal $z_a$ predict returns *beyond* what signals
$z_b, z_c,\dots$ already predict? (Marginal, not standalone, power.)

**Pass 1.** On each date $t$, run one cross-sectional regression across the
$n$ stocks:
$$
r_{i,t+1} = c_t + \lambda_{a,t} z_{a,i,t} + \lambda_{b,t} z_{b,i,t} + \cdots + e_{i,t}.
$$
This yields a *time series* of estimated coefficients $\lambda_{a,t}$
($t = 1..T$).

**Pass 2.** Report the time-series mean $\bar\lambda_a$ with a Newey–West
t-statistic (§4.3). That's it — Fama–MacBeth **is** "the mean of per-date
regression slopes, tested honestly."

**Why this construction is clever.** Stocks are heavily correlated with each
other *within* a month (they share the market). A pooled regression over all
stock-months would pretend it has $n \times T$ independent observations and
produce absurdly small standard errors. Fama–MacBeth compresses each date
into *one* observation per coefficient, so the cross-stock correlation is
absorbed inside each $\lambda_{a,t}$ and never contaminates the inference —
which then only has to handle time-series dependence, which Newey–West does.

**Check (repo), three ways** (`results/tables/fama_macbeth.csv`):

1. **The intercept is the market.** $c_t$ estimates the average stock return
   on date $t$ (signals average zero), so $\bar c$ should be the average
   market drift: the config plants $E[m] = 0.006$ with average beta 1.
   Measured $\bar c = 0.0066$. ✓
2. **The placebo fails, jointly.** `sig_dead`: $t = 1.95 < 2$ even in the
   multivariate regression. ✓
3. **Coefficient sizes are computable** *(optional, uses one classical
   identity)*. The regression is run on rank-normalized signals
   $\tilde z \approx$ uniform on $[-1,1]$ (variance $\tfrac13$), while
   returns were generated from the Gaussian $z$. The slope of $r$ on
   $\tilde z$ is $\beta\,\mathrm{Cov}(z, \tilde z)/\mathrm{Var}(\tilde z)$.
   With $\tilde z = 2\Phi(z) - 1$, $\mathrm{Cov} = 2E[z\Phi(z)] = 1/\sqrt{\pi}$
   (Stated; one-line via Stein's lemma $E[zf(z)] = E[f'(z)]$, giving
   $E[\phi(z)] = \tfrac{1}{2\sqrt\pi}$). So the predicted momentum
   coefficient is $0.004 \times \frac{1/\sqrt\pi}{1/3} \times 0.755
   \text{ (decay)} = \mathbf{0.0051}$. Measured: **0.0054**. ✓ (6% apart,
   inside its own standard error.)

## 4.5 The reading rule for every table in `results/`

A mean without its Newey–West t-stat is an anecdote. The repo's convention:
mean, NW-t, and the number of periods travel together, and the lag length is
fixed in config (never tuned to make a result significant — that would be a
chapter 9 sin).


---
---

# 5. Linear Models: Least Squares, Alpha, and Why Shrinkage Wins in Noise

## 5.1 Least squares from scratch (the normal equations)

Given features $x_i$ (vector, includes a leading 1 for the intercept) and
targets $y_i$, ordinary least squares (OLS) picks coefficients $\beta$
minimizing the sum of squared errors
$L(\beta) = \sum_i (y_i - x_i^\top \beta)^2 = \|y - X\beta\|^2$.

**Derivation.** $L$ is a smooth bowl in $\beta$; at the minimum its gradient
is zero. Expanding $L = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta$
and differentiating: $\nabla L = -2X^\top y + 2X^\top X\beta = 0$, so
$$
\boxed{\;\hat\beta = (X^\top X)^{-1} X^\top y\;}
$$
(the **normal equations**). In the one-feature case this collapses, after a
little algebra, to two formulas worth memorizing:
$$
\hat\beta_1 = \frac{\mathrm{Cov}(x,y)}{\mathrm{Var}(x)}, \qquad \hat\beta_0 = \bar y - \hat\beta_1 \bar x .
$$
So a regression slope *is* a rescaled covariance — regression, correlation,
and the IC are one family. And regressing on a constant alone gives
$\hat\beta_0 = \bar y$: §4.3's "mean test as regression" identity, proved.

## 5.2 Alpha: the intercept with a job title

Chapter 3's strategy returns might secretly be repackaged *known* risk
premia. The test (`src/evaluation/factor_controls.py`): regress strategy
returns on factor returns $f_t$ (market, size, value, …),
$$
r^{\text{strat}}_t = \alpha + \beta^\top f_t + e_t ,
$$
with Newey–West errors. $\beta^\top f_t$ is the part explained by *rentable
exposures anyone can buy*; the intercept $\alpha$ is the average return left
over — the claim to genuine information. $R^2$ (the fraction of variance the
factors explain) completes the picture: **surviving $\alpha$ with low $R^2$
is the credible pattern; vanished $\alpha$ means the "signal" was a known
factor in disguise** (worth reporting, honestly, either way). One timing trap
the code handles: returns indexed by *formation* date must be paired with
factor returns over the *same holding window* (`align="formation"` shifts the
factor panel accordingly).

## 5.3 The bias–variance decomposition (proved)

Why not always fit the most flexible model? Let the truth be
$y = f(x) + \varepsilon$ with noise variance $\sigma^2$, and let $\hat f$ be
a model fit on a random training sample. For a fixed test point $x$:

**Theorem.** $\;E\big[(y - \hat f(x))^2\big] = \underbrace{\big(E[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}\big(\hat f(x)\big)}_{\text{variance}} + \sigma^2 .$

**Proof.** Add and subtract $E[\hat f]$ inside the square:
$y - \hat f = \varepsilon + (f - E[\hat f]) + (E[\hat f] - \hat f)$.
Square and take expectations. The three cross terms die: $\varepsilon$ is
independent of everything with mean zero (kills two), and
$E\big[(f - E[\hat f])(E[\hat f] - \hat f)\big] = (f - E[\hat f])\cdot E[E[\hat f] - \hat f] = 0$
(the first factor is a constant, the second has mean zero). What survives is
the three claimed terms. ∎

**Why finance sits at the extreme of this tradeoff.** In return prediction
the signal-to-noise ratio is brutal — chapter 3 computed the *true* model's
correlation ceiling at about 0.05–0.10 monthly, i.e. $R^2$ well under 1%.
When $\sigma^2$ dwarfs the signal, the variance term dominates the bias term,
so **deliberately biased, low-variance estimators win**. That is the entire
case for shrinkage.

## 5.4 Ridge and lasso, solved exactly in the clean case

The **elastic net** (the repo's linear benchmark, `make_linear`) minimizes
$\|y - X\beta\|^2 + \lambda_2\|\beta\|^2 + \lambda_1\|\beta\|_1$ — squared
error plus an L2 ("ridge") and an L1 ("lasso") penalty. In the *orthonormal*
case ($X^\top X = I$; think uncorrelated standardized features) both pieces
solve in closed form, and the closed forms teach exactly what each penalty
does. Let $\hat\beta = X^\top y$ be the OLS solution.

**Ridge (L1 off).** Minimize $\|y - X\beta\|^2 + \lambda\|\beta\|^2$.
Gradient: $-2X^\top y + 2\beta + 2\lambda\beta = 0$, so
$$
\hat\beta^{\text{ridge}} = \frac{\hat\beta}{1+\lambda}:
$$
**every coefficient is shrunk toward zero by the same factor.** Pure
variance reduction, paid for with a little bias — §5.3's trade made explicit.

**Lasso (L2 off).** The problem separates coordinate by coordinate: minimize
$g(\beta_j) = (\beta_j - \hat\beta_j)^2 + \lambda|\beta_j|$ (up to constants).
For $\beta_j > 0$: $g' = 2(\beta_j - \hat\beta_j) + \lambda = 0 \Rightarrow
\beta_j = \hat\beta_j - \lambda/2$, valid only if positive; symmetric case
for negative; otherwise the minimum is at the kink $\beta_j = 0$. Compactly:
$$
\hat\beta_j^{\text{lasso}} = \mathrm{sign}(\hat\beta_j)\,\max\!\big(|\hat\beta_j| - \tfrac{\lambda}{2},\, 0\big)
$$
— **soft thresholding**: small coefficients are set *exactly* to zero
(automatic signal selection), large ones shrunk by a constant. The elastic
net blends both behaviors; the repo keeps its penalties tiny because the
features are already few and rank-normalized — the benchmark's job is "best
honest linear combination," not aggressive selection.

## 5.5 What the linear model *cannot* do — the cliffhanger

However shrunk, $\hat y = \beta^\top z$ is **additive**: momentum's
contribution is the same regardless of the value signal's level. The
synthetic truth contains a term $\beta_{\text{int}}\, z_{\text{mom}}\, z_{\text{val}}$
whose whole point is that it is *not* additive. Chapter 6 proves no additive
model can represent it — and shows why trees can. That gap is exactly the
measured OOS difference between the elastic net (IC 0.043) and the
tree/net models (0.060–0.070).


---
---

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


---
---

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


---
---

# 8. Backtest Validity: Information Sets, Purging, and the Anatomy of a Leak

Backtests lie in two ways: by **using information from the future** (this
chapter) and by **selection among many tries** (chapter 9). Both failure
modes are provable, predictable, and — in this repo — deliberately
demonstrated.

## 8.1 Information sets: the bookkeeping that everything hangs on

Let $\mathcal{F}_t$ denote everything knowable at the end of month $t$. The
rules of the game, stated once:

1. A signal dated $t$ must be a function of $\mathcal F_t$ only.
2. The label paired with it, $r_{t\to t+1}$, is realized during $(t, t+1]$ —
   it belongs to $\mathcal F_{t+1}$, *not* $\mathcal F_t$.
3. Any statistic that pairs a time-$t$ feature with information outside
   $\mathcal F_t$-measurable inputs and $(t,t+1]$-realized labels, in a way
   that lets fitting see test-period information, is **leaked**.

## 8.2 The purging theorem

Walk-forward evaluation trains on early dates, tests on later ones. The
subtlety: the *label* attached to training date $t$ lives in the interval
$(t, t+1]$ (generally $(t, t+h]$ for horizon $h$).

**Theorem.** Let the last training date be $t^\*$ and the first test date be
$\tau$. Training-label windows and test-fold information are disjoint if and
only if $t^\* + h < \tau$ — i.e. at least $h$ whole periods must separate
train from test ("purge $\ge h$").

**Proof.** The union of training-label windows is $(\,\cdot\,,\, t^\* + h]$;
the test fold's features and labels begin at $\tau$. The two overlap exactly
when $t^\* + h \ge \tau$. Removing the $h$ dates between them
($\tau - h \le t \le \tau - 1$ dropped from training) is precisely the
condition $t^\* + h < \tau$. ∎

With $h = 1$ month, purge $= 1$: hence the hard `ValueError` in
`walkforward.py` when anyone requests purge 0, and the per-fold assertions
that recompute the gap. If you ever extend `fwd_ret` to $h$ months, the
theorem tells you the config change that must travel with it. (The optional
`embargo` widens the gap further as insurance against dependence *beyond*
the label window — defensible to leave at 0 for monthly data, but say so.)
One more rule rides on the same logic: **all tuning — optuna searches,
IC-Net's early stopping — must consume only training-window data**, because
a hyperparameter chosen with test information is a leak wearing a suit.

## 8.3 Predicting a leak before measuring it

The repo ships a deliberate leak (`demonstrate_lookahead`): a fake feature
$\ell = a\,y + e$ — a half-strength copy of the *very return it claims to
predict* plus independent noise ($a = 0.5$, $\sigma_e = \sigma_y$). This
simulates a timestamp error that lets next month's information into today's
feature. What IC *should* it produce? Compute, don't guess:
$$
\rho(\ell, y) = \frac{\mathrm{Cov}(ay + e,\, y)}{\sigma_\ell\,\sigma_y}
= \frac{a\,\sigma_y^2}{\sigma_y\sqrt{a^2\sigma_y^2 + \sigma_e^2}}
= \frac{a\,\sigma_y}{\sqrt{a^2\sigma_y^2 + \sigma_e^2}}
= \frac{0.5}{\sqrt{1.25}} = 0.447 .
$$
Converting Pearson to Spearman for near-Gaussian data
($\rho_s = \tfrac{6}{\pi}\arcsin(\rho/2)$ — Stated, classical): predicted IC
$\approx \mathbf{0.431}$.

**Check (repo).** Measured: **0.395** (t = 179). Prediction and measurement
agree to within the approximations made (returns are not exactly Gaussian;
the noise is calibrated on the pooled rather than per-date spread — each
assumption shaves a little). The pedagogical point survives any of that
slack: an honest single signal in this universe tops out near 0.04
(chapter 3 *derived* that ceiling), so **a 0.4 IC is not a discovery — it is
a diagnosis.** Ten times too good is not "great," it is "broken." The leak
table exists so you know the signature on sight.

## 8.4 The placebo that crossed the line (a gift of a teaching moment)

The same table evaluates `honest_noise`, a feature of pure random numbers:
measured IC 0.0059 with $t = 2.12$ — *nominally significant.* Nothing is
broken. Chapter 2 computed the null standard error of a mean IC here as
$\approx 0.0026$; a draw of $2.3$ standard errors happens by chance a couple
of percent of the time, and this run drew one. Three lessons, cheaply bought:

1. $t = 2$ is a *probability statement*, not a certificate — 1-in-20 pure
   noise features cross it (§4.1).
2. This is why the repo's placebo **test** (`test_placebo…`) uses the planted
   `sig_dead` under a fixed seed and a threshold with margin, and why
   conclusions should never hang on a single borderline t-stat.
3. Scale the thought: evaluate 50 candidate signals and a couple of them
   *will* look significant by luck alone. Quantifying that inflation is
   chapter 9's entire subject.

## 8.5 Staleness is the mirror-image leak

Forward leakage uses information too early; **staleness** uses it too late
and *destroys* real signal: chapter 3 proved IC decays like $\rho^k$ with
signal age, and the measured profile matched. The practical moral (the
"Anomaly Time" result): the *formation timestamp is part of the strategy*.
An evaluation on stale features can "fail to replicate" a perfectly real
anomaly. Point-in-time correctness is therefore two-sided — the repo's leak
report checks one side, the staleness profile the other.


---
---

# 9. Multiple Testing and the Deflated Sharpe Ratio

Chapter 8 closed with a pure-noise feature crossing $t = 2$. This chapter is
the mathematics of that phenomenon at industrial scale — the *selection*
lie — and the estimator the repo uses to price it.

## 9.1 The best of N tries is biased, provably

Run $N$ strategy variants that are all, in truth, worthless: each backtest
Sharpe is a mean-zero random draw. You then report the **best** one. The
best of $N$ mean-zero draws is not mean-zero — its expectation grows with
$N$.

**Theorem (the √(2 ln N) bound).** If $X_1,\dots,X_N$ are standard normal
(any dependence allowed), then $E[\max_i X_i] \le \sqrt{2\ln N}$.

**Proof.** For any $s > 0$, Jensen's inequality ($e^{sx}$ is convex) gives
$$
e^{\,s\,E[\max X_i]} \;\le\; E\big[e^{\,s \max X_i}\big] \;=\; E\big[\max_i e^{sX_i}\big] \;\le\; \sum_{i=1}^N E[e^{sX_i}] \;=\; N e^{s^2/2},
$$
using $\max \le \text{sum}$ for nonnegative terms and the normal
moment-generating function $E[e^{sX}] = e^{s^2/2}$ (Stated; one Gaussian
integral). Take logs and divide by $s$:
$E[\max X_i] \le \frac{\ln N}{s} + \frac{s}{2}$. Minimize the right side over
$s$ (calculus: $s = \sqrt{2\ln N}$) to get the bound. ∎

Read it as a *price list*: testing $N = 20$ worthless variants buys an
expected best t-stat of up to $\sqrt{2\ln 20} \approx 2.45$ — past the
"significant" line — for free. **Selection manufactures significance.** The
only defenses are (a) count your trials honestly and (b) raise the hurdle
accordingly. The DSR does both.

## 9.2 The sampling noise of a Sharpe ratio

A backtest Sharpe $\widehat{SR}$ (per-period, e.g. monthly) computed from $T$
observations is an estimate with error bars. Its approximate variance
(Stated — Lo 2002 / Bailey & López de Prado 2014, via the delta method):
$$
\mathrm{Var}(\widehat{SR}) \;\approx\; \frac{1 - \gamma_3\, SR + \frac{\gamma_4 - 1}{4} SR^2}{T - 1},
$$
where $\gamma_3$ is skewness and $\gamma_4$ *full* kurtosis (normal = 3).
The intuitions to keep: more months ⇒ tighter ($1/(T{-}1)$); left-skewed
strategies ($\gamma_3 < 0$, e.g. steady gains punctuated by crashes) have
*noisier* Sharpes than the normal formula suggests; fat tails likewise. This
is why `annualized_stats` records skew and kurtosis for every series — they
are inputs here, not decorations.

## 9.3 The Probabilistic Sharpe Ratio (PSR)

Standardize the estimate against a benchmark $SR^\*$:
$$
\mathrm{PSR}(SR^\*) \;=\; \Phi\!\left( \frac{(\widehat{SR} - SR^\*)\,\sqrt{T-1}}{\sqrt{\,1 - \gamma_3 \widehat{SR} + \frac{\gamma_4-1}{4}\widehat{SR}^2\,}} \right)
$$
— the probability the *true* Sharpe exceeds $SR^\*$, given the estimate, its
sample size, and its non-normality. With $SR^\* = 0$ this is a
moment-corrected one-sided test; everything is per-period (mixing a monthly
$\widehat{SR}$ with annual anything is the classic implementation bug the
module docstring warns about).

## 9.4 Deflation: setting the benchmark to "the luck of the best trial"

The Deflated Sharpe Ratio makes one substitution: the benchmark is the
Sharpe that *pure selection luck* would hand the best of your $N$ trials,
$$
SR^\* \;=\; \sqrt{\mathrm{Var}(\widehat{SR}_n)}\;\Big[(1-\gamma_E)\,\Phi^{-1}\!\big(1 - \tfrac1N\big) + \gamma_E\,\Phi^{-1}\!\big(1 - \tfrac{1}{Ne}\big)\Big],
$$
where $\mathrm{Var}(\widehat{SR}_n)$ is the variance of Sharpe estimates
*across the trials you ran* and $\gamma_E \approx 0.5772$ is the
Euler–Mascheroni constant. This is the extreme-value refinement of §9.1's
crude bound (Stated — the expected maximum of $N$ Gaussians, Bailey & López
de Prado 2014): same $\sqrt{\ln N}$ growth, correct constants. Then
$$
\mathrm{DSR} = \mathrm{PSR}(SR^\*) :
$$
*the probability the true Sharpe is positive, after charging for the search
that found it.* Note the two dials: deflation grows with the **number** of
trials and with their **dispersion** — trying many wildly different things
costs more than trying near-duplicates, exactly as intuition wants.

**The honesty condition (where DSR is gamed).** $N$ must count *every*
configuration examined — each single-signal portfolio, every model, every
tuning study — not just the finalists. The repo wires this in
(`count_single_signal_trials: true` ⇒ $N = 9$ currently) and the backlog's
"trial register" (item 6) makes the count an audit trail on real data.

**Check (repo), with its caveat.** The demo prints DSR $= 1.000$ for the best
model. Correct arithmetic, uninformative number: planted betas give the
strategy a monthly Sharpe ≈ 1.1 over 168 months, so certainty saturates.
The statistic earns its keep on real data, where Sharpes of 0.1–0.3 monthly
meet trial counts in the dozens — there the deflation routinely moves
conclusions from "significant" to "maybe."

## 9.5 Decay is estimation, not disappointment

McLean–Pontiff's finding — anomaly returns fall ~26% after the sample ends
and ~58% after publication — is, mathematically, just a **difference of
segment means** with all of this chapter's caveats attached: short segments
have huge standard errors (§4.1's $\sigma/\sqrt{T}$ with small $T$).

**Check (repo).** The generator plants post-publication retention 0.45.
Measured across the five real signals: 0.40, 0.46, 0.68, 0.12, 0.24 — noisy
around the truth, tightest for the strongest signal (momentum: 0.396), and
the *post-sample* retentions (24-month windows!) scatter absurdly (0.68 to
2.04). That scatter is not a bug in the decay module; it is §4.1 telling you
that a 24-month Sharpe carries a standard error of roughly
$\sqrt{12/24}\approx 0.7$ annualized. The mature reading of any decay table
— this repo's or a paper's — is point estimates *with* their error bars, and
the placebo row (retention $-0.03$ of a premium that never existed) as the
reminder of what pure noise looks like in the same format.

## 9.6 The one-paragraph philosophy of the whole lesson plan

Every chapter ended by predicting a number and checking it. That habit *is*
the project: a claim you cannot turn into a falsifiable number is not yet
research, and a number you did not deflate for search, protect from leakage,
and equip with honest error bars is not yet a result. The mathematics here
is not decoration on top of the code — it is the reason the code is
trustworthy, and now you can verify that statement yourself, one proof at a
time.


---
---

# 10. PULSE and the Kalman Filter: Tracking a Moving Truth

Chapters 1–9 treated signal efficacy as a constant to be *estimated*. The
decay evidence (chapters 3, 9) says it is a *moving target*. This chapter
derives, from Bayes' rule and one completed square, the optimal tracker of a
moving Gaussian target — the Kalman filter — and proves that its steady
state is the exponentially weighted average practitioners use by instinct,
with the weight chosen by the data instead of by hand.

## 10.1 The state-space model, in words and symbols

For one coefficient (PULSE runs one such filter per expanded feature):

$$
\beta_t = a\,\beta_{t-1} + w_t,\quad w_t \sim N(0, q)
\qquad\qquad
\lambda_t = \beta_t + v_t,\quad v_t \sim N(0, r_t)
$$

$\beta_t$ is the *true, unobservable* efficacy this month. The first
equation says it drifts slowly ($q$ small) and relaxes toward zero at rate
$a \le 1$ — the decay prior. The second says what we *can* see: the per-date
cross-sectional regression coefficient $\lambda_t$ (Fama–MacBeth pass 1,
chapter 4) equals the truth plus estimation noise whose variance $r_t$ the
regression itself reports (§5.1's machinery: $\mathrm{Var}(\hat\beta) =
s^2 (Z^\top Z)^{-1}$, diagonal). Known observation noise is the luxury that
makes everything below exact rather than heuristic.

## 10.2 The one lemma everything rests on: multiplying two Gaussians

**Lemma (Bayesian update for a Gaussian).** If prior belief is
$\beta \sim N(m, P)$ and we observe $\lambda \mid \beta \sim N(\beta, r)$,
then the posterior is
$$
\beta \mid \lambda \;\sim\; N\!\Big(m + K(\lambda - m),\; (1-K)\,P\Big),
\qquad K = \frac{P}{P + r}.
$$

**Proof.** Bayes' rule multiplies densities; work with exponents (log
densities), dropping constants:
$$
-\tfrac{(\beta - m)^2}{2P} - \tfrac{(\lambda - \beta)^2}{2r}
= -\tfrac12\Big[\beta^2\big(\tfrac1P + \tfrac1r\big) - 2\beta\big(\tfrac{m}{P} + \tfrac{\lambda}{r}\big)\Big] + \text{const}.
$$
A quadratic in $\beta$ is the exponent of a Gaussian with
precision (= 1/variance) equal to the $\beta^2$ coefficient and mean equal
to the linear coefficient divided by the precision:
$$
P_{\text{post}} = \Big(\tfrac1P + \tfrac1r\Big)^{-1} = \frac{Pr}{P + r} = (1-K)P,
\qquad
m_{\text{post}} = P_{\text{post}}\Big(\tfrac{m}{P} + \tfrac{\lambda}{r}\Big) = m + K(\lambda - m). \;\blacksquare
$$

Read the mean: **posterior = prior + gain × surprise.** The gain
$K = P/(P+r)$ is a precision-weighted compromise — trust the observation
more when your prior is uncertain ($P$ large) or the measurement is clean
($r$ small). Every "learning rate" you have ever met is this formula with
the uncertainties hidden.

## 10.3 The filter = predict, then update (that's the whole algorithm)

Between observations the state moves, so belief must too. If
$\beta_{t-1}\mid\text{data} \sim N(m_{t-1}, P_{t-1})$, then by linearity of
the state equation and §1.2's variance rules:
$$
\textbf{Predict:}\quad \beta_t \mid \text{data}_{t-1} \sim N\big(a\,m_{t-1},\; a^2 P_{t-1} + q\big).
$$
Then fold in $\lambda_t$ with the Lemma:
$$
\textbf{Update:}\quad m_t = a\,m_{t-1} + K_t\big(\lambda_t - a\,m_{t-1}\big),
\qquad K_t = \frac{a^2P_{t-1} + q}{a^2P_{t-1} + q + r_t}.
$$
Those two lines are, verbatim, the loop in `pulse.py::_filter_1d`. The
forecast PULSE actually trades on is the *predicted* mean $a\,m_{t-1}$ —
belief about *this* month's efficacy given data through last month. The
one-step **innovation** $\lambda_t - a\,m_{t-1} \sim N(0, a^2P_{t-1}+q+r_t)$
also hands us a model-selection criterion for free: the hyperparameters
$(a, q)$ that maximize the summed innovation log-likelihood are the ones
whose *predictions* explain the observed coefficient series best — an
out-of-sample criterion computed entirely inside the training window.

## 10.4 Theorem: the steady-state filter is an EWMA (with a data-chosen weight)

Practitioners smooth factor returns with exponentially weighted averages.
The filter explains *why that works and what the weight should be*.

**Theorem.** With constant $r_t = r$ and $a = 1$ (pure random walk), the
filter variance converges to a fixed point $P^\* = \tfrac{-q + \sqrt{q^2 + 4qr}}{2}$,
the gain to a constant $K^\* = \tfrac{P^\* + q}{P^\* + q + r}$, and the state
estimate becomes exactly
$$
m_t = (1 - K^\*)\, m_{t-1} + K^\*\,\lambda_t
\;=\; K^\*\sum_{j\ge0} (1-K^\*)^j\, \lambda_{t-j} :
$$
an exponentially weighted moving average with half-life
$\ln 2 / \big|\ln(1 - K^\*)\big|$.

**Proof.** At a fixed point the post-update variance reproduces itself
through one predict–update cycle: $P = \frac{(P + q)\,r}{P + q + r}$.
Cross-multiplying: $P^2 + qP - qr = 0$, whose positive root is $P^\*$
(quadratic formula). Constant $P^\*$ gives constant $K^\*$; substituting
into the update recursion and unrolling the geometric recursion gives the
EWMA form. ∎

The moral: EWMA smoothing of factor efficacy is not ad hoc — it is the
*optimal* tracker under random-walk dynamics — **and** the right smoothing
constant is determined by the signal-to-noise ratio $q/r$: noisy monthly
coefficients (large $r$) ⇒ small $K^\*$ ⇒ long memory; genuinely mobile
efficacy (large $q$) ⇒ fast adaptation. PULSE estimates that ratio per
signal by likelihood instead of guessing one half-life for all.

## 10.5 Proposition (point-in-time correctness of the PULSE forecast)

**Claim.** The forecast for formation date $t$ uses only
$\mathcal F_t$-measurable quantities.

**Proof.** The observation $\lambda_s$ is computed from signals at $s$ and
returns over $(s, s+1]$, hence $\lambda_s \in \mathcal F_{s+1}$ (chapter 8's
bookkeeping). At formation date $t$ the observations available are exactly
$\lambda_1, \dots, \lambda_{t-1}$; the filter state $m_{t-1}$ is a function
of those alone, and the traded forecast $\sum_k a\,m_{k,t-1}\, z_{k,i,t}$
additionally uses only signals dated $t$. Inside a walk-forward test fold no
update steps run (states propagate as $a^h m$), so no test-fold return ever
touches the weights. ∎

## 10.6 Checks (repo)

The planted-truth test (`test_pulse_tracks_planted_decay_and_predicts_oos`)
verifies the *mechanism*: filtered paths correlate with the true planted
beta staircase, register each post-publication step-down, and hold the
placebo near zero — the figure `results/figures/pulse_efficacy_paths.png`
shows the walk down the stairs. The likelihood grid chose $a = 0.99$ with
the smallest state noise: "persistent efficacy, gentle decay pull," selected
by data, not assumed. Performance: OOS IC 0.072 / net Sharpe 5.05, first
among the four models — with chapter 9's standing caveat that a synthetic
world built from time-varying betas is home turf for a time-varying-beta
model. The mathematics here proves the tracker is *correct*; only the
real-data run (docs/05, §5's pre-registered predictions) can prove it is
*useful*.


---
---

# 11. Differentiable Trading: Exact Gradients Through a Market

Chapters 1–10 evaluated *forecasts*. The agent (docs/06) optimizes a
*policy* — holdings as a function of history — and this chapter derives the
three pieces of calculus that make that optimization exact rather than
estimated: the unrolled form of the policy, the gradient of the Sharpe
ratio, and the forward-mode sensitivity recursion. Together they explain
why no reinforcement-learning machinery is needed: **a differentiable
simulator turns "learning to trade" into ordinary calculus.**

## 11.1 The policy, and a theorem about what it is

Per date $t$: blend signals into a score $s_t = \sum_k \theta_k z_{k,t}$
(vectors over stocks), demean and $L_1$-normalize into a dollar-neutral aim
$A_t$ with gross exposure 2, then **partially adjust**:
$$
w_t = (1-\gamma)\,w_{t-1} + \gamma\,A_t, \qquad \gamma \in (0,1).
$$

**Theorem (the policy is an EWMA of aims).** With $w_0 = 0$,
$$
w_t \;=\; \gamma \sum_{j=0}^{t-1} (1-\gamma)^{\,j} A_{t-j}.
$$

**Proof.** Induction. Base: $w_1 = \gamma A_1$. Step: substitute the claim
for $w_{t-1}$ into the recursion:
$$
w_t = (1-\gamma)\,\gamma\!\sum_{j=0}^{t-2}(1-\gamma)^j A_{t-1-j} + \gamma A_t
= \gamma\!\sum_{j=1}^{t-1}(1-\gamma)^{j} A_{t-j} + \gamma A_t. \;\blacksquare
$$

So the agent has exactly two economic dials: *what to aim at* ($\theta$,
which shapes $A_t$) and *how fast to chase it* ($\gamma$, the memory of the
EWMA — half-life $\ln 2/|\ln(1-\gamma)|$). Compare §10.4: PULSE's filter is
an EWMA over *coefficient observations* with a noise-chosen weight; the
agent is an EWMA over *portfolios* with a cost-chosen weight. Same
mathematical object, two different economic forces selecting the memory —
a unification worth saying out loud in an interview.

Gârleanu–Pedersen (2013) prove that with quadratic trading costs the
*optimal* dynamic policy has precisely this partial-adjustment form (Stated
— their derivation is a linear-quadratic control problem beyond this plan's
scope). We therefore use the structure as an inductive bias and let the
data choose the dials.

## 11.2 The gradient of the Sharpe ratio (derived)

Training maximizes $J = m/s$ where $m$ and $s$ are the mean and (population)
standard deviation of the net return series $(r_1,\dots,r_T)$. We need
$\partial J/\partial r_t$ — how the objective responds to each month's P&L.

From $m = \tfrac1T\sum r_t$: $\;\partial m/\partial r_t = 1/T$.
From $s^2 = \tfrac1T\sum (r_t - m)^2$, differentiate:
$2s\,\tfrac{\partial s}{\partial r_t} = \tfrac{2}{T}(r_t - m)$ (the inner
$-\partial m/\partial r_t$ terms cancel because $\sum(r_u - m) = 0$), so
$\partial s/\partial r_t = (r_t - m)/(Ts)$. Quotient rule:
$$
\boxed{\;\frac{\partial J}{\partial r_t} = \frac{1}{T\,s} \;-\; \frac{m\,(r_t - m)}{T\,s^{3}}\;}
$$
Read it: every month's marginal value has a *baseline* $1/(Ts)$ (more return
is good) minus a *risk charge* proportional to how far that month already
sits from the mean — the objective actively dislikes months that add
variance faster than they add mean. Maximizing Sharpe is not maximizing
return; this formula is the precise statement of the difference, and it is
three lines of `_sharpe_and_grad`.

## 11.3 Forward-mode sensitivities: backprop through time, made simple

Each $r_t$ depends on the parameters $p = (\theta, g)$ through *every*
weight back to $w_1$ (inventory has memory). Rather than reverse-mode
backpropagation through time, the code uses **forward-mode accumulation**:
carry the Jacobian $D_t = \partial w_t/\partial p$ (a small $P\times N$
matrix, $P = K+1$) alongside the simulation. Differentiating the policy
recursion directly:
$$
D_t = (1-\gamma)\,D_{t-1} \;+\; \gamma\,\frac{\partial A_t}{\partial \theta}
\;+\; (A_t - w_{t-1})\,\frac{\partial \gamma}{\partial g},
\qquad \frac{\partial\gamma}{\partial g} = \gamma(1-\gamma)
$$
(the last factor from $\gamma = \sigma(g)$; note $D_{t-1}$ already contains
$g$'s influence on *past* weights, so no term is double-counted). The aim's
own gradient is a quotient-rule exercise on
$A = 2\tilde s/\!\sum_i\sqrt{\tilde s_i^2+\varepsilon}$, and the smoothed
absolute value contributes the derivative
$\tfrac{d}{dx}\sqrt{x^2+\varepsilon} = x/\sqrt{x^2+\varepsilon}$ — a
"soft sign" that equals $\pm1$ away from zero and rolls smoothly through
it, which is the entire reason the cost term $c\sum_i|w_{t,i}-w_{t-1,i}|$
becomes differentiable. Each month then contributes
$\partial r_t/\partial p = D_t\, y_t - c\,(D_t - D_{t-1})\,\mathrm{softsign}(\Delta_t)$,
and the chain rule assembles the full gradient as
$\sum_t (\partial J/\partial r_t)(\partial r_t/\partial p)$.

Forward mode costs $O(P)$ times the simulation — cheap when $P = 7$, which
is exactly why the economically-constrained policy class is also the
*computationally* right one. (A thousand-parameter network would want
reverse mode; the honest trade is documented in docs/06 §7.)

## 11.4 Why exact gradients beat reinforcement learning here

Model-free RL (the ML4T ch. 22 approach) *estimates*
$\nabla_p E[\text{reward}]$ from sampled episodes, with variance that must
be tamed by millions of interactions — available in Atari, absurd with 300
monthly observations. Here the "environment" (historical returns + a known
cost function) is a fixed, differentiable dataset, so the same gradient is
*computed* to machine precision in one pass. Every sample RL would spend
discovering the shape of the cost function, we spend on the only genuinely
scarce resource: independent months of data. The general principle:
**reach for RL when the environment is unknown or non-differentiable;
reach for calculus when you own the simulator.**

## 11.5 Checks (repo)

The planted-truth test verifies the learned speed is *monotone in cost*
($\gamma = 0.64 \to 0.30 \to 0.16$ at 0/40/100 bps — seed-stable, since the
7-parameter objective is smooth enough for Adam to find the same optimum),
and that the learned-speed policy beats the myopic $\gamma=1$ control on
the identical aim, net, out of sample. It also *fails to find* GP's
slow-signal aim tilt — and §10.4-style reasoning explains why: with one
shared $\gamma$, the EWMA attenuates each signal's mean and risk
contributions nearly proportionally, so the tilt cancels; recovering it
requires per-signal speeds (the documented extension). A theory prediction
confirmed, a control passed, and a null result located and explained: that
triple is what "validated" means in this repo.


---
---

# 12. MSRR and the Mathematics of Execution

Chapter 11 built the policy and its gradients. This chapter proves the two
closed forms that surround it: the **optimal aim** (Maximum Sharpe Ratio
Regression — solvable exactly, no optimizer needed) and the **capture
ratio** (exactly how much alpha a given trading speed harvests from a
signal of given persistence). Together they turn the cost sweep's measured
curve into derived structure — and they let the numerical agent be checked
against analytic truth, the same discipline every other chapter uses.

## 12.1 The factor collapse: a portfolio of stocks becomes a blend of factors

**Lemma.** If each stock's weight is linear in its (per-date demeaned)
signals, $w_{i,t} = \sum_k \theta_k \tilde z_{k,i,t}$, then the portfolio
return collapses onto $K$ numbers per date:
$$
r_p(t) \;=\; \sum_i w_{i,t}\, y_{i,t} \;=\; \sum_k \theta_k \underbrace{\Big(\sum_i \tilde z_{k,i,t}\, y_{i,t}\Big)}_{f_{k,t}} \;=\; \theta^\top f_t .
$$

**Proof.** Swap the two finite sums (linearity). ∎

The $f_{k,t}$ are **characteristic-managed portfolio returns**: what you
earn holding each stock in proportion to its signal-$k$ value. A universe of
500 stocks has just become a $K$-asset problem — the single most useful
dimension reduction in cross-sectional finance, and the reason the next
theorem has a closed form.

## 12.2 Theorem (MSRR): the max-Sharpe blend is $\Sigma^{-1}\mu$

Let $\mu = E[f_t]$ and $\Sigma = \mathrm{Cov}(f_t)$. Choose $\theta$ to
maximize the Sharpe ratio $S(\theta) = \theta^\top\mu / \sqrt{\theta^\top\Sigma\,\theta}$.

**Theorem.** $\theta^\* \propto \Sigma^{-1}\mu$, achieving
$S(\theta^\*) = \sqrt{\mu^\top\Sigma^{-1}\mu}$.

**Proof.** Substitute $x = \Sigma^{1/2}\theta$ and $b = \Sigma^{-1/2}\mu$
(the symmetric square root exists because $\Sigma$ is positive definite).
Then
$$
S = \frac{b^\top x}{\|x\|} \;\le\; \|b\|
$$
by Cauchy–Schwarz (chapter 1's theorem, reused verbatim), with equality iff
$x \propto b$, i.e. $\Sigma^{1/2}\theta \propto \Sigma^{-1/2}\mu$, i.e.
$\theta \propto \Sigma^{-1}\mu$; and $\|b\| = \sqrt{\mu^\top\Sigma^{-1}\mu}$. ∎

Read it: the optimal blend is the mean return of each factor, *deflated by
the risk it shares with the others* — high-mean factors get weight, but
redundant ones get discounted through $\Sigma^{-1}$.

## 12.3 Why "Regression": the Sherman–Morrison identity

MSRR's name comes from a lovely equivalence: **regressing the constant 1 on
the factor returns points at the same portfolio.** The OLS coefficient of
$1$ on $F$ (the $T\times K$ matrix of factor returns) is
$\hat\theta = (F^\top F)^{-1} F^\top \mathbf 1$, and
$F^\top\mathbf 1/T = \mu$, $F^\top F/T = \Sigma + \mu\mu^\top$ (the moment
identity $E[ff^\top] = \Sigma + \mu\mu^\top$). So we need
$(\Sigma + \mu\mu^\top)^{-1}\mu$.

**Lemma (Sherman–Morrison, rank-one update).**
$(A + uv^\top)^{-1} = A^{-1} - \dfrac{A^{-1}uv^\top A^{-1}}{1 + v^\top A^{-1}u}$.

**Proof.** Multiply the claimed inverse by $(A + uv^\top)$; with
$q = v^\top A^{-1} u$ the cross terms give
$uv^\top A^{-1}\big[1 - \tfrac{1}{1+q} - \tfrac{q}{1+q}\big] = 0$, leaving $I$. ∎

**Corollary.** $(\Sigma + \mu\mu^\top)^{-1}\mu = \dfrac{\Sigma^{-1}\mu}{1 + \mu^\top\Sigma^{-1}\mu} \;\propto\; \Sigma^{-1}\mu$.

**Proof.** Apply the lemma with $A=\Sigma$, $u=v=\mu$, then multiply by
$\mu$ and factor: $\Sigma^{-1}\mu\big[1 - \tfrac{q}{1+q}\big]$ with
$q = \mu^\top\Sigma^{-1}\mu$. ∎

So a *regression* — the humblest tool in the box, chapter 5's normal
equations — computes the maximum-Sharpe portfolio's direction exactly. In
practice $\mu$ and $\Sigma$ are noisy estimates, so the repo's
implementation solves $(\Sigma + \lambda I)^{-1}\mu$: ridge shrinkage,
chapter 5's bias–variance trade transplanted into portfolio space. (When
$K$ grows into the hundreds, how $\lambda$ interacts with $K > T$ is
precisely the virtue-of-complexity debate of the research-findings doc —
this corollary is that literature's front door.)

**Check (repo).** At zero cost the numerically-trained agent — Adam through
the full trading recursion, no knowledge of this chapter — recovers the
closed form: cosine similarity between learned $\theta$ and
$\Sigma^{-1}\mu$ direction $= \mathbf{1.000}$, gross Sharpe 1.204 vs the
closed form's 1.205, and both assign the placebo $\approx 0$ weight
(`test_msrr_closed_form_recovered_by_zero_cost_agent`). Optimizer and
theorem validate each other; the agreement is not exact because the agent's
per-date L1 normalization reweights dates slightly — a stated, understood
gap of 0.1%.

## 12.4 Theorem (capture ratio): what a trading speed costs in alpha

The partial-adjustment policy holds an EMA of past aims (ch. 11's theorem).
How much of a signal's *fresh* alpha does a stale EMA capture? Let the aim
track a single AR(1) signal $z_t$ with persistence $\rho$ (ch. 3), so
next-month alpha is $\beta z_t$, and let $w_t = \gamma\sum_{j\ge0}(1-\gamma)^j z_{t-j}$.

**Theorem.** Expected captured alpha is $\beta \cdot \kappa(\gamma,\rho)$ with
$$
\kappa(\gamma, \rho) \;=\; \gamma \sum_{j\ge0} (1-\gamma)^j \rho^j \;=\; \frac{\gamma}{1 - (1-\gamma)\rho}.
$$

**Proof.** $E[w_t\,\beta z_t] = \beta\gamma\sum_j (1-\gamma)^j E[z_{t-j}z_t]
= \beta\gamma\sum_j (1-\gamma)^j \rho^j$ by chapter 3's Claim 2
($\mathrm{Corr}(z_t, z_{t-j}) = \rho^j$); sum the geometric series. ∎

Sanity: $\kappa(1,\rho) = 1$ (myopic captures everything);
$\kappa \to 0$ as $\gamma \to 0$ (a frozen book captures nothing);
$\kappa$ increases in $\rho$ (slow signals forgive slow trading). Turnover,
meanwhile, *increases* in $\gamma$ (more chasing = more trading), so the
net objective is a tug-of-war
$\beta\,\kappa(\gamma,\rho) - c \cdot \text{turnover}(\gamma)$ whose
interior optimum moves **down** as $c$ rises — the derived skeleton of the
measured cost sweep ($\gamma: 0.92 \to 0.83 \to 0.60 \to 0.39 \to 0.21$ at
$c = 0/10/25/50/100$ bps). Put a number on the sweet spot: at the 25 bps
point the learned $\gamma = 0.60$ on a $\rho \approx 0.9$ blend gives
$\kappa = 0.60/(1 - 0.36) = 0.94$ — the agent kept **94% of the alpha while
cutting turnover from 0.42 to 0.27.** That ratio *is* the business case for
execution-aware trading, in one fraction.

## 12.5 Square-root impact: the cost of size, made differentiable

Linear cost ($c\cdot|\Delta w|$) says the thousandth dollar trades as
cheaply as the first. Markets disagree: the empirical **square-root law**
(Stated — one of the most replicated results in market microstructure) has
price impact growing like the square root of trade size, so total cost —
impact × quantity — grows like
$$
|\Delta w| \cdot |\Delta w|^{1/2} \;=\; |\Delta w|^{3/2},
$$
a **convex** cost. Convexity has one big behavioral consequence: marginal
cost rises with size, so optimal trades shrink and spread out — exactly the
verified comparative static (`test_sqrt_impact_slows_trading`: adding
impact lowered the learned speed 0.49 → 0.44 and turnover 0.22 → 0.20).
Differentiability is preserved by the same smoothing trick as chapter 11:
implement $|x|^{3/2}$ as $(x^2+\varepsilon)^{3/4}$, whose derivative
$\tfrac{3}{2}\,x\,(x^2+\varepsilon)^{-1/4}$ is the line in the code, and
which rolls smoothly through zero instead of kinking. One honest caveat
travels with the feature: the *coefficient* of impact is far harder to
estimate from data than a commission schedule, so on real data it is a
sensitivity parameter to sweep, not a constant to trust.


---
---

# 13. Symmetries, Identifiability, and Composition
## The mathematics of the bug we hit — and of the composition that worked

This chapter is different in kind: its centerpiece theorem *predicts a
failure that actually happened* during development (the sign-flipped
composition, docs/06 §8), and its final derivation turns the composed
agent's headline number into a quantity you can predict from a correlation.
Debugging, done properly, is applied mathematics.

## 13.1 The null-direction theorem: symmetry creates a flat direction

The agent's aim is scale-invariant by construction:
$A(\theta) = 2\,\widetilde{s}/\|\widetilde{s}\|_1$ with
$s = \sum_k \theta_k z_k$, so replacing $\theta \to c\theta$ (any $c>0$)
multiplies $\widetilde s$ and $\|\widetilde s\|_1$ by the same $c$, which
cancels.

**Theorem.** The training objective satisfies $J(c\theta, g) = J(\theta, g)$
for all $c > 0$, and consequently the gradient is everywhere orthogonal to
$\theta$:
$$
\theta^\top \nabla_\theta J \;=\; 0 .
$$

**Proof.** Every quantity downstream of the aim (weights, returns, costs,
Sharpe) depends on $\theta$ only through $A(\theta)$, which we just showed
is invariant, so $J(c\theta) = J(\theta)$. Differentiate this identity with
respect to $c$ at $c = 1$ (chain rule):
$\frac{d}{dc}J(c\theta)\big|_{c=1} = \theta^\top\nabla_\theta J = 0$. ∎

(This is Euler's homogeneous-function argument for degree-zero functions —
a symmetry always manufactures a direction the gradient cannot see, the
optimization cousin of Noether's principle.)

**Corollary (the K = 1 trap).** With a single feature, every $\theta > 0$
is a positive rescaling of every other, so $J$ is *constant* on each
half-line: it depends only on $\mathrm{sign}(\theta)$. The gradient in
$\theta$ is exactly zero everywhere except the kink at 0 (the softabs
$\varepsilon$ turns "exactly zero" into "numerical dust").

## 13.2 Why Adam random-walks a flat direction (the bug, derived)

A flat direction sounds harmless — no gradient, no movement. Not under
Adam. Its update is $\Delta p = \mathrm{lr}\cdot \hat m/(\sqrt{\hat v} + \epsilon)$,
where $\hat m$ and $\hat v$ are moving averages of the gradient and its
square. Feed it the flat direction's gradient — tiny numerical noise
$\delta_t$ with no consistent sign — and both $\hat m \sim O(\delta)$ and
$\sqrt{\hat v} \sim O(\delta)$, so their **ratio is $O(1)$ with a random
sign**: the parameter takes steps of size $\approx$ lr *regardless of how
small the noise is*. Scale-freeness, Adam's virtue on real gradients, turns
"zero gradient" into "unit-speed diffusion." Over 150 epochs, a $\theta$
that started at $+0.2$ can drift across zero — flipping the sign of every
weight in the book. The agent then observes a *negative-alpha* book in
training and does the rational thing: learns $\gamma \to 0$ and freezes.
That is precisely the observed failure: a forecast with IC $+0.074$
producing OOS Sharpe $-6.19$ under $\gamma = 1$ (the exact mirror image of
the true book's $+6.49$) and a "learned" $\gamma$ of 0.003.

**The fix, and a lemma that it works.** After each Adam step, project
$\theta$ back to the unit L1 sphere, $\theta \leftarrow \theta/\|\theta\|_1$
— optimization on the quotient of the symmetry, which deletes the flat
direction while leaving all perpendicular (real) gradients untouched.

**Lemma (no sign flip).** With $K = 1$, projection keeps
$\theta \in \{-1, +1\}$, and a flip would require a single step of
magnitude $> 1$; Adam's step is bounded near lr $= 0.05 \ll 1$, so the sign
is stable. For $K \ge 2$ the projection is inert where it should be:
genuine gradients are orthogonal to $\theta$ (the Theorem), hence tangent
to the sphere already. ∎ *(The step bound is the standard heuristic
$|\hat m|/\sqrt{\hat v} \lesssim 1$ — labeled honestly: an argument, not a
worst-case proof.)*

**Check (repo).** Post-fix, the same composition runs at net Sharpe
$+5.35$; the $K = 6$ baselines are unchanged to three decimals (perpendicular
gradients dominated all along, as the Theorem says they should); all 17
tests pass. The transferable lesson: **every invariance you build into a
model is a direction your optimizer can wander; either quotient it out or
expect it to be explored.**

## 13.3 Identifiability: the multi-speed cousin of the same disease

The multi-speed agent (per-signal EMAs $u_k$ with speeds $\gamma_k$) hides
a second flat-ish direction. How large is $u_k$?

**Lemma.** For an i.i.d. unit-variance input, the EMA
$u_t = (1-\gamma)u_{t-1} + \gamma A_t$ has stationary variance
$$
\mathrm{Var}(u) \;=\; \gamma^2 \sum_{j\ge0} (1-\gamma)^{2j} \;=\; \frac{\gamma^2}{1-(1-\gamma)^2} \;=\; \frac{\gamma}{2-\gamma},
$$
which $\to 0$ as $\gamma \to 0$. (Persistent inputs change the constant,
not the limit — Stated.) ∎

So a signal whose speed learns its way to $\gamma_k \approx 0$ contributes
a component $u_k$ of vanishing size — and **its blend weight $\theta_k$
multiplies nothing**: unidentified, free to wander, exactly like the null
direction of §13.1 but signal-by-signal. This is why the probe found the
placebo "holding" 20–25% of raw $\theta$ at $\gamma \approx 0.007$: an
economically empty parameter parked at a meaningless value. The corrected
metric is **effective exposure** $\theta_k \cdot \mathrm{sd}(u_k)$, which
the Lemma shows zeroes any frozen signal automatically — and which is the
pre-registered lens for the slow-signal-tilt test on real data (docs/06
§5), replacing the raw-$\theta$ version this analysis refuted.

## 13.4 Composition is point-in-time by induction

**Proposition.** If a forecast satisfies $x_t \in \mathcal F_t$ (chapter
10 §10.5 proved this for PULSE: its state feeding date $t$ uses
observations through $t-1$), and the policy is any map
$w_t = \pi(w_{t-1}, x_t)$, then $w_t \in \mathcal F_t$ for all $t$.

**Proof.** Induction. $w_0 = 0 \in \mathcal F_0$. If
$w_{t-1} \in \mathcal F_{t-1} \subseteq \mathcal F_t$ and
$x_t \in \mathcal F_t$, then $w_t$, a function of two
$\mathcal F_t$-measurable inputs, is $\mathcal F_t$-measurable. ∎

Chains of point-in-time components are point-in-time — leakage cannot be
created by composition, only by a leaky component. (That is exactly why
IC-Net is *not* composed the same way: its in-sample fitted values are not
$\mathcal F_t$-measurable objects during training, so it needs purged
stacking first — the documented requirement, now with its reason proved.)

## 13.5 What a proportional book earns: the Pearson formula (with a check)

The agent's aim holds each stock in proportion to its demeaned forecast:
$w = 2\tilde f/\|\tilde f\|_1$. Its one-date gross return has a closed
form. Using $\tilde f^\top \mathbf 1 = 0$ (so the return level drops out,
ch. 3's dollar-neutrality) and the definition of correlation:
$$
w^\top y \;=\; \frac{2\,\tilde f^\top \tilde y}{\|\tilde f\|_1}
\;=\; \frac{2\,n\, s_f\, s_y\, \rho_t}{\|\tilde f\|_1}
\;\approx\; 2\sqrt{\tfrac{\pi}{2}}\; s_y\,\rho_t \;\approx\; 2.507\, s_y\, \rho_t,
$$
where the last step uses $\|\tilde f\|_1 \approx n\,s_f\sqrt{2/\pi}$ for a
roughly Gaussian cross-section ($E|Z| = \sigma\sqrt{2/\pi}$ — one integral,
same trick as ch. 3's truncated-mean lemma). Two readings:

1. **Proportional weights monetize the per-date *Pearson* correlation** —
   the exact quantity IC-Net trains on (ch. 7) — just as quantile books
   monetize ordering (ch. 3). The forecast's *levels* matter here, which is
   why a sign flip is catastrophic while rank-IC stays serenely positive:
   the diagnostic signature that cracked the bug (rank-IC $+0.074$, book
   $-6.19$ ⇒ the pattern is right and the *sign/plumbing* is wrong).
2. **It predicts the composition's headline.** Measured mean Pearson
   $\bar\rho = 0.0834$ and cross-sectional return spread
   $s_y \approx 0.081$ (ch. 3) give expected gross
   $\approx 2.507 \times 0.0834 \times 0.081 = 0.0169$/month $= 20.3\%$/yr.
   The composed agent measured **19.0% net** at turnover 0.395 — adding
   back its cost drag ($0.395 \times 2 \times 10\,\text{bps} \times 12
   \approx 0.9\%$) implies gross $\approx 19.9\%$. Predicted 20.3, implied
   19.9 — agreement to within the Gaussian-$\|\cdot\|_1$ approximation,
   with zero fitted parameters. The lesson plan's oldest habit, applied to
   its newest artifact.

## 13.6 Coda: the shape of the whole argument

Chapters 12–13 close a loop the repo has been drawing since chapter 1:
Cauchy–Schwarz proved correlation bounded (ch. 1), became the max-Sharpe
theorem (§12.2); the AR(1) lemma (ch. 3) became the capture ratio (§12.4);
the information-set bookkeeping (ch. 8, 10) became composition closure
(§13.4); and the per-date correlation — the IC that started as an
evaluation statistic — ended as the exact quantity a proportional book
earns (§13.5). One toolbox, reused until forecasting, filtering, and
trading are visibly the same mathematics wearing three jackets.


---
---
