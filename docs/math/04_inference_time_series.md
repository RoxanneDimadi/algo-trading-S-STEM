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
