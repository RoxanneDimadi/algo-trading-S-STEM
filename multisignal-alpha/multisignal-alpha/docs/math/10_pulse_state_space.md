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
