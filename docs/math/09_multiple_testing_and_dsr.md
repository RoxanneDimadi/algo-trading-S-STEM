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
