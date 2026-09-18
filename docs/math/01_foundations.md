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
