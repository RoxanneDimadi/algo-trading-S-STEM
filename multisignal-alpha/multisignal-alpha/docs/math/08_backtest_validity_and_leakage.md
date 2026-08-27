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
