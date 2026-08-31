# PULSE: Per-date Update of Latent Signal Efficacy
## A decay-aware alpha model, with its claims made falsifiable

*Companion math: `docs/math/10_pulse_state_space.md`. Implementation:
`src/models/pulse.py` (pure NumPy, ~150 lines). Validation:
`tests/test_backtest.py::test_pulse_tracks_planted_decay_and_predicts_oos`.*

---

## 1. The honest paradox, resolved

"Build something no one can get by querying an LLM" contains its own trap:
any *idea* emitted on request is, by construction, available to everyone.
The resolvable version of the demand is different: an idea is differentiated
by (a) being a genuine synthesis of current research rather than a recycled
tutorial, and (b) — decisively — by the **validation infrastructure around
it**. This repo can do something almost no one else's can: test whether a
time-varying-efficacy model actually *tracks* time-varying efficacy, because
the synthetic generator plants the true efficacy path. The model below is
interesting; the planted-truth test of its central mechanism is the part
nobody gets from a chat window.

## 2. The hypothesis

Every standard ML asset-pricing model — including this repo's elastic net,
LightGBM, and IC-Net — is trained as if signal efficacy were **stationary**:
one fit over decades, implicitly averaging momentum's 1999 strength with its
2020 strength. The current literature says that assumption is wrong in a
*structured*, exploitable way:

- **Decay is real and directional.** McLean & Pontiff (2016): anomaly
  returns fall ~26% post-sample and ~58% post-publication. Efficacy has a
  systematic downward pull.
- **Efficacy is persistent month to month.** Factor momentum (Gupta & Kelly
  2019; Ehsani & Linnainmaa, *JF* 2022): factors' own returns are positively
  autocorrelated — *recent* efficacy predicts *near-future* efficacy.
- **Efficacy is timing-sensitive.** "Anomaly Time" (*JF* 2024): predictive
  power concentrates when information is fresh.

Jointly these say: **a signal's true coefficient is a slowly-moving latent
state with a decaying resting point** — not a constant. PULSE is the minimal
model that takes that sentence literally.

## 3. The model

For each feature $k$ of an interaction-expanded basis (all signals plus all
pairwise products, so drifting *interaction* strength is trackable too):

$$
\text{state: } \beta_{k,t} = a\,\beta_{k,t-1} + w_t,\; w\sim N(0, q_k)
\qquad
\text{observation: } \lambda_{k,t} = \beta_{k,t} + v_t,\; v \sim N(0, r_{k,t})
$$

- $\lambda_{k,t}$ is the date-$t$ cross-sectional OLS coefficient (the
  Fama–MacBeth first pass) — a noisy monthly *measurement* of live efficacy.
- $r_{k,t}$ is that coefficient's sampling variance, **which the regression
  itself supplies**. Known, per-date, heteroskedastic observation noise is
  what makes a Kalman filter the principled estimator here instead of an
  ad-hoc moving average (the math chapter proves the filter's steady state
  *is* an exponentially weighted average — with the weight chosen by the
  data, per signal, rather than by hand).
- $a \le 1$ encodes the **decay prior**: with no evidence, estimated efficacy
  relaxes toward zero. "Alpha dies by default" as a model assumption, not a
  slogan. $(a, q)$ are selected by one-step-ahead predictive log-likelihood
  **on training data only**.
- The forecast at formation date $t$ is
  $p_{i,t} = \sum_k \hat\beta_{k,t|t-1}\, z_{k,i,t}$ — today's signals,
  weighted by *today's filtered efficacy*. Point-in-time is provable: the
  observation $\lambda_{k,s}$ requires the return realized at $s{+}1$, so
  the filter feeding formation date $t$ has consumed data through $t{-}1$
  only, and inside a test fold the state evolves by the prior dynamics alone
  (no test-fold updating) — the walk-forward's purge guarantees carry over
  intact.

## 4. What was verified (and exactly what it does and does not mean)

**Mechanism validation (the planted-truth test, all passing):** the filtered
efficacy paths correlate with the *true* planted beta paths, detect each
signal's post-publication step-down, and pin the placebo's efficacy near
zero. The pipeline additionally exports the full filtered paths
(`results/tables/pulse_efficacy_path.csv`, figure
`pulse_efficacy_paths.png`) so you can watch the filter walk down the
planted staircase.

**Performance on the shipped config** (identical inputs, harness, and
portfolio constructor; 168 OOS months):

| model | OOS IC | ICIR | net Sharpe | one-way turnover |
|---|---|---|---|---|
| elasticnet | 0.043 | 0.93 | 2.66 | 0.47 |
| lightgbm | 0.060 | 1.38 | 4.05 | 0.91 |
| icnet | 0.070 | 1.50 | 4.57 | 0.48 |
| **pulse** | **0.072** | **1.55** | **5.05** | 0.51 |

Chosen dynamics: $a = 0.99$, smallest state-noise scale — i.e., the data
itself selected "persistent efficacy with a gentle decay pull," which is the
hypothesis in one line.

**The caveat that must travel with the table:** the synthetic world *is*
PULSE's model class — piecewise-constant betas plus one interaction is
exactly "time-varying linear on an interaction basis." Winning here
validates that the filter, the likelihood selection, and the point-in-time
plumbing work; it is **not** evidence of real-world alpha. The models it
beat were not built to exploit nonstationarity; on real data, where efficacy
dynamics are messier than a two-step staircase, the ranking is an open
question. That is what makes the next section the actual content.

## 5. Falsifiable real-data predictions (write these down before running)

1. **PULSE beats the static elastic net** on the OSAP panel out of sample,
   with the margin concentrated in the **post-2003 era** (post-publication
   for most classic anomalies, where stationarity is most wrong).
2. **The likelihood selects $a < 1$** on real data — the decay prior earns
   its place — and per-signal filtered paths trend downward after each
   signal's SignalDoc publication year *without being told those dates*.
3. **Efficacy paths mark known events**: momentum's filtered state should
   crater around the 2009 momentum crash, and short-term reversal's around
   the 2007 quant quake (qualitative, chart-inspectable).
4. If (1)–(2) fail, the honest conclusion is that monthly Fama–MacBeth
   observations are too noisy for online efficacy tracking at this
   frequency — itself a publishable-grade negative result for the writeup.

## 6. Failure modes and limits (known before anyone asks)

Abrupt regime breaks defeat a smooth filter (a crash is not a random-walk
step); with ~200 real signals the interaction expansion explodes — screen to
a shortlist first (the funnel discipline already in the roadmap); weak
signals give the filter mostly noise to track (the placebo test bounds, but
does not eliminate, this); per-coefficient independent filters ignore
cross-signal efficacy correlation (a full-covariance filter is the upgrade);
and the trial count for the deflated Sharpe grows again — logged
($n_{\text{trials}} = 10$).

## 7. Extension ladder

Online within-fold updating (the live-trading variant — kept out here so
every model faces the *identical* frozen-per-fold harness); efficacy-gated
IC-Net (tier 1 nonlinear scores, tier 2 PULSE gates — the two custom models
composed); full-covariance filtering; hidden-Markov regime alternative to
smooth decay; crowding covariates (short interest, factor flows) as
*exogenous inputs to the state equation*, which is where this line of
research genuinely ends up.
