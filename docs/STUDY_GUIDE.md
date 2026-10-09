# Multi-Signal Alpha — Complete Study Guide

**A single self-contained document covering the project, every model, every
formula, what the math means, all measured results, the analysis, and the
conclusions.**

Written in layers. Part 0 assumes you know nothing about finance or
statistics. Each later part adds precision, and by Part 5 you are reading
theorems. You can stop at any layer and still have a correct (if coarser)
picture — nothing in a later part contradicts an earlier one, it only
sharpens it.

| If you have… | Read |
|---|---|
| 10 minutes | Part 0 |
| 1 hour | Parts 0–2, then Part 6 and Part 9 |
| An afternoon | Parts 0–4, 6, 7, 8, 9 |
| A week, and you want to own it | Everything, with Appendix E as the exam |

**Companion documents in this repo** (this guide summarizes and connects
them; they hold the long-form proofs and the operational detail):

- `docs/math/00_index.md` … `13_*.md` — the formula-by-formula proof chapters
- `docs/USER_GUIDE.md` — setup, commands, per-column output reference
- `docs/figure_explanation.md` — narrative walkthrough of the synthetic run
- `multisignal-alpha/multisignal-alpha/docs/05_pulse_model.md` — PULSE design doc
- `multisignal-alpha/multisignal-alpha/docs/06_trading_agent.md` — agent design doc

**Contents**

- [Part 0 — The project in plain English](#part-0--the-project-in-plain-english)
- [Part 1 — Foundations: the vocabulary and the first formulas](#part-1--foundations-the-vocabulary-and-the-first-formulas)
- [Part 2 — The pipeline: what actually happens when you press run](#part-2--the-pipeline-what-actually-happens-when-you-press-run)
- [Part 3 — The four models, in full](#part-3--the-four-models-in-full)
- [Part 4 — The trading agent, in full](#part-4--the-trading-agent-in-full)
- [Part 5 — The statistical immune system](#part-5--the-statistical-immune-system)
- [Part 6 — Results I: the synthetic (planted-truth) run](#part-6--results-i-the-synthetic-planted-truth-run)
- [Part 7 — Results II: the real 212-factor run](#part-7--results-ii-the-real-212-factor-run)
- [Part 8 — Analysis: what the results actually mean](#part-8--analysis-what-the-results-actually-mean)
- [Part 9 — Conclusions](#part-9--conclusions)
- [Appendix A — Master formula sheet](#appendix-a--master-formula-sheet)
- [Appendix B — Glossary](#appendix-b--glossary)
- [Appendix C — Notation table](#appendix-c--notation-table)
- [Appendix D — Concept-to-code map](#appendix-d--concept-to-code-map)
- [Appendix E — Self-test questions (with answers)](#appendix-e--self-test-questions-with-answers)
- [Appendix F — Reading list](#appendix-f--reading-list)

---

# Part 0 — The project in plain English

## 0.1 One sentence

This project tries to predict which stocks (or which published trading
strategies) will do better than which over the next month, turns those
predictions into a portfolio, charges realistic trading costs, and then
applies every statistical correction known to the field so the final number
is honest rather than flattering.

## 0.2 The claim, stated precisely

The project does **not** claim "I found a way to make money." It claims
something narrower and much more defensible:

> Here is a measurement harness. It was first proved on data where I planted
> the truth myself, so I can check that it recovers exactly what I planted
> and reports zero for the thing I planted as worthless. Then I pointed the
> same harness at real data with every correction switched on, and I am
> reporting what survived — including the places where my own predictions
> were wrong.

That distinction is the whole intellectual content. A backtest that says
"Sharpe 5" tells you nothing unless you know the ruler is straight. Most of
this repository is about straightening and then re-checking the ruler.

## 0.3 The problem, as a game

Every month, you look at a list of candidates — say 500 stocks. For each you
compute some numbers from information you genuinely have *today*: how it has
performed over the last year, how volatile it has been, how cheap it looks
relative to its accounting value. Those numbers are called **signals**.

You then have to rank the 500 names. Next month the actual returns arrive,
and you ask: did my ranking line up with reality?

Two things make this hard and interesting:

1. **You are not predicting the market.** You are predicting *relative*
   performance. If every stock rises 8%, you earn nothing, because you buy
   the top of your ranking and simultaneously sell the bottom. This is
   deliberate: it removes the question "will the market go up?", which nobody
   can answer, and leaves the question "which names are relatively better?",
   which is faintly answerable.
2. **The signal is minuscule.** The correlation between a good signal and
   next month's returns is around **0.02 to 0.05**. That means being right
   barely more than half the time. Money comes from being slightly right,
   across hundreds of names, for hundreds of months. If you ever measure a
   correlation of 0.4, you have not found a goldmine — you have found a bug,
   and Part 5 shows you exactly what that bug looks like.

## 0.4 The two jobs: the models and the agent

There are two distinct questions, answered by two distinct pieces of
machinery. Confusing them is the single most common mistake in this field.

**Job 1 — the models: "which names look good next month?"**
Four different models take the same signals and produce a score per name.
They are a penalized linear regression (**elastic net**), gradient-boosted
decision trees (**LightGBM**), a small neural network whose loss function is
the correlation itself (**IC-Net**), and a Kalman filter that lets each
signal's strength drift through time (**PULSE**). A forecast is
*memoryless*: it does not know or care what you held last month.

**Job 2 — the agent: "given those scores, what do I hold *this* month,
knowing what I held last month and that trading costs money?"**
Positions have memory. Buying today creates the cost of selling tomorrow.
The agent learns a **trading speed** — how much of the gap toward its ideal
portfolio to close each month — by directly maximizing risk-adjusted return
*after* costs.

The slogan: **the models draw the map; the agent decides how to drive on
that map when every turn has a toll.**

## 0.5 The three data worlds

| World | What the data is | What a result there proves |
|---|---|---|
| **Synthetic** | A simulator where *we* wrote the return equation, planting known signal strengths, a known decay pattern, a known interaction, and one signal with exactly zero power | The harness works. We know the answer in advance, so we can check the machinery recovers it |
| **Real factor panel** | 212 real published anomaly strategies from the Open Source Asset Pricing project, each treated as a tradable asset, 1926–2024 | How every model and the agent behave on genuinely noisy real data, with no special data licence needed |
| **Firm-level OSAP** | Individual stocks with 200+ real signals and CRSP returns | The full academic setting — **not run here**, because CRSP returns require a paid WRDS licence |

Evidential weight runs in that order, left to right. A win in the synthetic
world is a *plumbing certificate*. A win on the real factor panel is
*evidence*. The third column is the honest frontier the project has not
reached.

## 0.6 The honesty machine: five defenses, in one paragraph each

**Leakage detection.** The most common way a backtest lies is by letting
tomorrow's information into today's features. The pipeline deliberately
builds a feature that secretly contains next month's return, measures it, and
shows you what "obviously fake" looks like in your own numbers. On the real
panel that fake feature scored 0.318 while honest noise scored −0.001.

**Purged walk-forward.** Models are only ever trained on the past and tested
on the future, with a one-month gap deleted between them so the last training
label (which is realized *over* the following month) cannot overlap the first
test month.

**Transaction costs, charged consistently.** Every reported "net" number pays
10 basis points (0.10%) per side on every dollar traded. This single
correction kills the fake signal in the synthetic run outright and reorders
the model ranking on real data.

**Factor controls.** A strategy's returns are regressed on the well-known
risk factors anyone can buy cheaply. What is left over — the intercept, called
**alpha** — is the claim to genuine information. If alpha vanishes, your
"signal" was a famous factor in a new hat.

**Multiple-testing deflation.** If you try 11 things and report the best, the
best looks good even when nothing works. The **Deflated Sharpe Ratio**
computes how good the luckiest of 11 tries would look by pure chance, and
makes you clear that bar instead of zero.

## 0.7 The headline conclusions, up front

1. **The harness is validated.** On planted data, the pipeline recovered the
   planted signal strengths in the right order and magnitude, scored the
   planted-zero signal as insignificant, flagged the deliberate leak, and
   recovered the planted decay pattern. Several results were *predicted
   numerically from the config before being measured* and matched to within a
   few percent — see the scorecard in §6.8.
2. **On real data, the simple model won.** Over 1044 out-of-sample months on
   the 212-factor panel, the elastic net delivered the best net Sharpe
   (0.59), ahead of IC-Net (0.53), LightGBM (0.41) and PULSE (0.30). This is
   the *exact reverse* of the synthetic ranking, and §8.1 explains why that
   reversal is the single most informative result in the project.
3. **Real signal survives costs, but modestly.** Trailing factor momentum
   predicts next month's factor returns with IC ≈ 0.09–0.12 at t > 8, and the
   combined models keep a factor-controlled alpha of 4–9% per year with
   t-statistics of 2.4–3.9. The Deflated Sharpe for the winner is 0.97 —
   above the conventional 0.95 bar, but on a strategy whose raw Sharpe is
   0.59, not 5.
4. **Published anomalies decay, and the decay was reproduced on real data.**
   Across all 212 factors, mean retention of the original return is **0.71
   after the sample ends** and **0.54 after publication** (median 0.42). The
   canonical literature benchmark is ~74% and ~42%. That is a close,
   independent replication of McLean–Pontiff (2016) computed from scratch in
   this repository.
5. **The agent's own pre-registered prediction failed on real data, and that
   is reported rather than buried.** On synthetic data the agent behaved
   exactly as theory says — trading speed fell monotonically as costs rose
   (γ: 0.92 → 0.21 across 0 → 100 bps), and at 100 bps it turned a losing
   strategy into a viable one. On the real factor panel at 10 bps, its learned
   speed collapsed to ≈0.0005 (a frozen book) and it *lost* to its own
   always-rebalance control, 0.31 net Sharpe versus 0.55. §8.3 diagnoses
   this in detail.

Those five, with their caveats, are the project. Everything below is the
detail behind them.

---

# Part 1 — Foundations: the vocabulary and the first formulas

This part builds every term used later. If you already know what a Sharpe
ratio and a t-statistic are, skim to §1.7.

## 1.1 Returns, cross-sections, signals

A stock worth 100 at the end of one month and 103 at the end of the next has
a **return** of $r = 103/100 - 1 = 0.03$. Returns are the only thing we
ultimately care about.

- $r_{i,t}$ — the return of name $i$ during month $t$.
- The **cross-section** at date $t$ is the whole list of names alive on that
  date; $n$ is how many there are.
- A **signal** $z_{i,t}$ is any number computable from information available
  at the *end* of month $t$.
- The **forward return** `fwd_ret` is $r_{i,t+1}$ — what happens *next*. The
  pairing (signal at $t$, return over $t \to t{+}1$) is the sacred alignment
  of the whole project. Pair a signal with the *same* month's return and you
  are measuring description, not prediction.
- $T$ — number of dates. The synthetic panel has $T = 299$, $n = 500$. The
  real factor panel has $T = 1176$ and up to $n = 212$.

## 1.2 Average, spread, and the two rules

For a random quantity $X$:

- **Expectation** $E[X]$ — its long-run average.
- **Variance** $\mathrm{Var}(X) = E[(X - E[X])^2]$ — the average *squared*
  distance from the average.
- **Standard deviation** $\sigma = \sqrt{\mathrm{Var}(X)}$ — spread in the
  original units.

Two rules get used constantly:

$$
E[aX + bY] = a\thinspace E[X] + b\thinspace E[Y] \quad \text{(always)}
$$

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) \quad \text{(if } X, Y \text{ independent)}
$$

**Proof of the second.** Write $\tilde X = X - E[X]$ and $\tilde Y = Y-E[Y]$
(this "put a tilde on it to mean subtract its average" notation is used
everywhere below). Then

$$
\mathrm{Var}(X+Y) = E[(\tilde X + \tilde Y)^2] = E[\tilde X^2] + 2E[\tilde X\tilde Y] + E[\tilde Y^2].
$$

The middle term is the **covariance** $\mathrm{Cov}(X,Y) = E[\tilde X\tilde Y]$,
and for independent variables the average of a product is the product of the
averages, so it is $E[\tilde X]\thinspace E[\tilde Y] = 0$. ∎

Covariance is positive when $X$ and $Y$ tend to be above their own averages
at the same time.

## 1.3 Correlation, and why it can never exceed 1

Covariance has awkward units. Dividing by both standard deviations gives the
unit-free **correlation**:

$$
\rho(X,Y) \mkern5mu=\mkern5mu \frac{\mathrm{Cov}(X,Y)}{\sigma_X \thinspace \sigma_Y}.
$$

**Theorem (Cauchy–Schwarz).** $|\rho| \le 1$, always.

**Proof.** For any real number $t$, $(\tilde X + t\tilde Y)^2$ is a square,
so its expectation cannot be negative:

$$
0 \mkern5mu\le\mkern5mu E[(\tilde X + t\tilde Y)^2] \mkern5mu=\mkern5mu \mathrm{Var}(Y)\thinspace t^2 + 2\thinspace\mathrm{Cov}(X,Y)\thinspace t + \mathrm{Var}(X).
$$

A quadratic that is never negative cannot cross zero twice, so its
discriminant is $\le 0$:
$4\thinspace\mathrm{Cov}(X,Y)^2 \le 4\thinspace\mathrm{Var}(X)\mathrm{Var}(Y)$.
Divide by $4\sigma_X^2\sigma_Y^2$ and take roots. ∎

Remember the trick — *"a square has non-negative expectation"*. It reappears
in Part 4 as the theorem that gives the optimal portfolio a closed form.

**What the number means.** $\rho = 1$ is a perfect increasing straight line,
$-1$ perfect decreasing, $0$ no *linear* relationship. **Calibration you
should memorize for this field: an honest monthly signal-to-return
correlation of 0.05 is strong. 0.3 is a bug.**

## 1.4 Correlation is a cosine (the geometric picture)

Take one date. Stack the model's predictions into a vector
$p = (p_1,\dots,p_n)$ and the forward returns into $y = (y_1,\dots,y_n)$,
one entry per name. Demean both. Then

$$
\rho(p,y) \mkern5mu=\mkern5mu \frac{\langle \tilde p, \tilde y\rangle}{\Vert\tilde p\Vert \thinspace \Vert\tilde y\Vert}
$$

where $\langle a,b\rangle = \sum_i a_i b_i$ is the dot product and
$\Vert a \Vert = \sqrt{\langle a,a\rangle}$ is the length. That expression
*is* the cosine of the angle between the two vectors in $n$-dimensional
space. Aligned: $\rho = 1$. Perpendicular: 0. Opposite: $-1$.

Two consequences follow in one line each, and they turn out to be the
economic heart of the project.

**Theorem A (translation invariance).** $\rho(p + c\mathbf 1, y) = \rho(p,y)$
for any constant $c$.
*Proof:* demeaning kills constants, and the formula only ever sees $\tilde p$. ∎

**Theorem B (positive-scale invariance).** $\rho(p, \lambda y) = \rho(p,y)$
for any $\lambda > 0$.
*Proof:* the numerator gains a factor $\lambda$ and so does the denominator
through $\Vert \lambda \tilde y \Vert = \lambda \Vert \tilde y\Vert$. ∎

**Why you should care.** Theorem A says correlation cannot be rewarded for
predicting whether next month is an up month — only for ordering the names.
Theorem B says a wild month and a calm month contribute equally. §1.8 shows
the long-short portfolio has *exactly the same two indifferences*, and Part 3
shows that this match is the entire argument for IC-Net's loss function.

## 1.5 Rank normalization

A signal measured in dollars and one in percent cannot be added raw. The
**z-score** $x \mapsto (x - \bar x)/\sigma_x$ recenters and rescales. This
project goes further and uses **ranks**: within each date, sort the values and
replace them by evenly spaced points in $[-1, 1]$. Ranks also kill outliers —
the largest value becomes "the largest rank" no matter *how* large it is.

Since ranks are an order-preserving transform, Pearson correlation applied to
ranks is **Spearman correlation**, which therefore inherits Theorems A and B
and adds outlier immunity. (The textbook shortcut
$\rho_s = 1 - 6\sum d_i^2 / (n(n^2-1))$ is derived in
`docs/math/02`; the code uses `scipy.stats.spearmanr` directly and never
needs it.)

## 1.6 The Sharpe ratio, and the √12 rule

A strategy's monthly returns have average $\mu$ and standard deviation
$\sigma$. The **Sharpe ratio** is reward per unit of risk:

$$
\mathrm{SR}_{\text{monthly}} = \mu / \sigma .
$$

It is the right *ratio* because a strategy can always be scaled: bet twice as
much and both $\mu$ and $\sigma$ double, leaving SR unchanged. SR measures
quality independent of how big you bet.

Convention reports it annualized, by multiplying by $\sqrt{12}$.

**Proof.** Assume monthly returns are independent with the same $\mu$ and
$\sigma$, and approximate the annual return as the sum of 12 monthly ones.
By linearity the annual mean is $12\mu$; by independence the annual variance
is $12\sigma^2$, so the annual standard deviation is $\sqrt{12}\thinspace\sigma$.
Hence

$$
\mathrm{SR}_{\text{annual}} = \frac{12\mu}{\sqrt{12}\thinspace\sigma} = \sqrt{12}\mkern5mu \frac{\mu}{\sigma}. \qquad \blacksquare
$$

Both assumptions are approximations — compounding is not a sum, and months
are not independent. Part 1.9 is the repair for the second one.

**Calibration.** Annualized Sharpe below 0.5 is weak; 0.5–1.0 is a real
strategy; above 2 sustained on real data after costs is rare; above 4 means
either synthetic data or a bug. Keep that scale in mind when Part 6 reports
5.05 and Part 7 reports 0.59.

## 1.7 The Information Coefficient (IC) and ICIR

$$
\mathrm{IC}_t \mkern5mu=\mkern5mu \text{Spearman correlation across names, between } z_{i,t} \text{ and } r_{i,t+1}.
$$

One number per date. The headline is the time-series mean
$\overline{\mathrm{IC}}$, and the stability ratio

$$
\mathrm{ICIR} = \overline{\mathrm{IC}} \thinspace / \thinspace \sigma_{\mathrm{IC}} .
$$

IC says *how strong*; ICIR says *how consistent*. A signal with IC 0.03 that
works almost every month can be more valuable than one with IC 0.05 that
works half the time and is catastrophic the other half.

**Two design choices that carry real theory:**

**(a) Forward, never contemporaneous.** Covered in §1.1.

**(b) Per date, then average — never pooled.** If you threw every
$(z, r)$ pair from every date into one giant correlation, the result would mix
a *between-dates* effect (do months with a high average signal have high
average returns?) with the *within-date* effect (which is the only thing a
market-neutral portfolio can harvest). Computing the correlation inside each
date and then averaging isolates the within-date part by construction,
because each date's demeaning deletes that date's market level (Theorem A).

**How noisy is an IC when the signal is worthless?** A correlation computed
from $n$ independent pairs has standard deviation $\approx 1/\sqrt{n-1}$. With
$n = 500$ names that is $0.0448$ per date. Averaging $T = 298$ dates divides
the noise by $\sqrt T$ (§1.9), giving a standard error of
$0.0448/\sqrt{298} \approx 0.0026$ on the mean IC. **Hold onto that number:
it is the yardstick for every IC in Part 6.**

## 1.8 From a signal to a portfolio

A **portfolio** assigns each name a **weight** $w_i$ — the fraction of capital
in it, negative meaning **short** (borrow the share, sell it, profit if it
falls). The portfolio's return is a dot product:

$$
r_p \mkern5mu=\mkern5mu \sum_i w_i\thinspace y_i \mkern5mu=\mkern5mu \langle w, y\rangle .
$$

This project's constructor (`score_to_weights`) sorts each date's names into
5 **quintiles** by score and sets

$$
w_i = \begin{cases} +1/n_{\text{top}} & i \in \text{top quintile}\cr
-1/n_{\text{bot}} & i \in \text{bottom quintile}\cr
0 & \text{otherwise,}\end{cases}
$$

i.e. **\$1 long the best fifth, \$1 short the worst fifth**, equal-weighted.
Gross exposure is \$2; net exposure is \$0.

**Theorem (dollar neutrality).** If $\sum_i w_i = 0$, then adding any
constant $c$ to every name's return leaves the portfolio return unchanged.

**Proof.** $\langle w, y + c\mathbf 1\rangle = \langle w,y\rangle + c\sum_i w_i = \langle w,y\rangle$. ∎

So a month in which *everything* rises 8% contributes exactly nothing: the
long leg's gain is the short leg's loss. **This is Theorem A of §1.4 wearing
an economic jacket.** The quintile sort uses only the ordering of scores, so
the portfolio is also indifferent to the scale of the scores — Theorem B,
likewise. The statistical test and the economic test are indifferent to
exactly the same two things, which is why they agree so often, and why
Part 3's IC-Net chooses correlation as its loss.

## 1.9 Turnover, costs, and the arithmetic of friction

Weights change each month, and changing them costs money.

$$
\text{traded}_t = \sum_i |w_{t,i} - w_{t-1,i}|
$$

— every dollar bought or sold. **One-way turnover** is $\text{traded}_t / 2$
(a \$1 sale funding a \$1 purchase is \$2 traded, one repositioning). The net
return is

$$
r^{\text{net}}_t \mkern5mu=\mkern5mu r^{\text{gross}}_t \mkern5mu-\mkern5mu \text{traded}_t \times \frac{\text{cost}_{\text{bps}}}{10000}.
$$

With the project's configured 10 bps per side, the **annual cost drag** of a
strategy with one-way turnover $\tau$ is

$$
\text{drag} \mkern5mu=\mkern5mu \tau \times 2 \times 0.001 \times 12 \mkern5mu=\mkern5mu 0.024\thinspace\tau .
$$

That one line explains most of the gross-to-net gaps in Parts 6 and 7, and
you can verify it yourself on any row of any results table. Worked example
from the real run: the elastic net has $\tau = 0.742$, so drag $= 0.0178$;
gross annual return 0.120 minus 0.0178 gives 0.102, which is exactly the
reported net. Turnover is therefore not a footnote — it is a *tax rate*, and
Part 8 shows it reordering the model ranking.

This is deliberately the simplest defensible cost model. Its two honest
refinements — drift-adjusting the previous weights between rebalances, and
making costs depend on trade size — are listed in the repo's backlog, and the
size-dependent version (square-root impact) is implemented inside the agent
(§4.8).

## 1.10 Persistence, turnover, and the staleness law

The synthetic signals are built as an **AR(1)** process:

$$
z_t = \rho\thinspace z_{t-1} + \sqrt{1-\rho^2}\thinspace \epsilon_t, \qquad \epsilon_t \text{ fresh noise of variance } 1 .
$$

**Claim 1 (the variance stays at 1).** If $\mathrm{Var}(z_{t-1}) = 1$ then
$\mathrm{Var}(z_t) = \rho^2 \cdot 1 + (1-\rho^2)\cdot 1 = 1$. ∎

**Claim 2 (correlation across $k$ months is $\rho^k$).**
$\mathrm{Cov}(z_t, z_{t-1}) = \rho\thinspace\mathrm{Var}(z_{t-1}) = \rho$
because the fresh noise is uncorrelated with the past; iterate $k$ times. ∎

**Consequence — the staleness law.** A $k$-month-old signal is a
$\rho^k$-strength copy of today's signal plus unrelated noise, so

$$
\mathrm{IC}(k\text{ months stale}) \mkern5mu\approx\mkern5mu \rho^k \thinspace \mathrm{IC}(\text{fresh}).
$$

This is a *rate*, and it has a practical twin: **turnover is persistence seen
from the portfolio's side.** A signal with $\rho = 0.98$ (slow, sticky) barely
reorders the cross-section, so its portfolio barely trades. A signal with
$\rho = 0.90$ reorders names constantly and trades a lot. Verified in Part 6:
`sig_value` ($\rho = 0.98$) turned over 0.232 per month, `sig_momentum`
($\rho = 0.90$) turned over 0.509.

## 1.11 t-statistics and the Newey–West correction

Every headline in this project is a **time-series average** — mean IC, mean
long-short return, alpha. Averages from finite noisy data are themselves
noisy.

**Theorem.** If $x_1,\dots,x_T$ are uncorrelated with common mean $\mu$ and
variance $\sigma^2$, then $\mathrm{Var}(\bar x) = \sigma^2/T$.

**Proof.** $\mathrm{Var}(\bar x) = \frac{1}{T^2}\sum_t \mathrm{Var}(x_t) = \frac{T\sigma^2}{T^2}$,
the cross terms vanishing because the $x_t$ are uncorrelated. ∎

So the **standard error** is $\sigma/\sqrt T$, and the **t-statistic**
$t = \bar x / \mathrm{se}(\bar x)$ counts how many standard errors the average
sits from zero. Under the null "true mean is zero", $t$ is approximately
standard normal, so $|t| > 2$ happens by luck about 5% of the time. *That is
the entire content of "significant at t > 2".*

**But strategy returns are autocorrelated** — a good month is more likely
after a good month — so those cross terms do *not* vanish. Redoing the
computation with $\rho_k$ the correlation between observations $k$ apart:

$$
\mathrm{Var}(\bar x) = \frac{\sigma^2}{T}\Big[1 + 2\sum_{k=1}^{T-1}\Big(1 - \frac{k}{T}\Big)\rho_k\Big],
$$

since there are exactly $T-k$ pairs at distance $k$. **If the $\rho_k$ are
positive, the true variance of your average is larger than $\sigma^2/T$, so
the naive t-statistic divides by too small a number and overstates
significance.** Intuitively, $T$ correlated months contain fewer than $T$
independent pieces of information.

**Newey–West (1987)** plugs estimated autocovariances into that bracket,
truncated at lag $L$ and damped by triangular (Bartlett) weights:

$$
\widehat{\mathrm{Var}}_{\text{NW}}(\bar x) = \frac{1}{T}\Big[\hat\gamma_0 + 2\sum_{k=1}^{L}\Big(1-\frac{k}{L+1}\Big)\hat\gamma_k\Big].
$$

The triangular weights are not cosmetic: they guarantee the estimate can
never come out negative. This project uses $L = 6$ for monthly data
everywhere, fixed in config, never tuned to make a result significant (which
would be a Part 5 sin).

**Reading rule for every table in this repo: a mean without its Newey–West
t-statistic is an anecdote.** Mean, NW-t, and the number of periods always
travel together.

## 1.12 Fama–MacBeth: does a signal add anything the others don't?

Standalone IC answers "does this signal work?". Often you need "does this
signal work *beyond* what the others already tell me?" — marginal, not
standalone, power. The 1973 Fama–MacBeth two-pass procedure:

**Pass 1.** On each date, run *one* cross-sectional regression across the $n$
names:

$$
r_{i,t+1} = c_t + \lambda_{a,t}\thinspace z_{a,i,t} + \lambda_{b,t}\thinspace z_{b,i,t} + \cdots + e_{i,t}.
$$

This produces a *time series* of slopes $\lambda_{a,t}$.

**Pass 2.** Report the time-series mean $\bar\lambda_a$ with a Newey–West
t-statistic.

That's it: **Fama–MacBeth is "the average of per-date regression slopes,
tested honestly."**

**Why the construction is clever.** Names are heavily correlated with each
other *within* a month — they share the market. A pooled regression over all
$n \times T$ stock-months would pretend it had $n \times T$ independent
observations and produce absurdly small standard errors. Fama–MacBeth
compresses each date into *one* observation per coefficient, so cross-name
correlation is absorbed inside each $\lambda_{a,t}$ and never contaminates
the inference, which then only has to handle time dependence — and
Newey–West handles that.

A side benefit used by PULSE in Part 3: pass 1 also hands you each
coefficient's *sampling variance*, $\mathrm{Var}(\hat\lambda) = s^2 (Z^\top Z)^{-1}$
on the diagonal. Knowing how noisy each monthly measurement is turns out to
be exactly what a Kalman filter needs.

---

# Part 2 — The pipeline: what actually happens when you press run

One command (`python -m src.pipeline --config configs/config.yaml`) runs ten
stages in order. Understanding the order is understanding the project's
methodology, because the order encodes a discipline: **evaluation precedes
modeling**, and **validation precedes evaluation**.

## 2.1 The ten stages

| # | Stage | Code | Output |
|---|---|---|---|
| 1 | Build the panel | `src/data/synthetic.py` or `src/data/loaders.py`, then `src/data/panel.py` | long DataFrame `[date, ticker, signals…, ret, fwd_ret]` |
| 2 | Leak checks | `src/pipeline.py` + `src/evaluation/ic.py` | `leak_report.csv`, `lookahead_demonstration.csv` |
| 3 | Per-signal evaluation | `src/evaluation/ic.py`, `portfolio.py` | `signal_evaluation.csv`, `staleness_profile.csv` |
| 4 | Fama–MacBeth | `src/evaluation/fama_macbeth.py` | `fama_macbeth.csv` |
| 5 | Decay analysis | `src/evaluation/decay.py` | `decay_analysis.csv` |
| 6 | Purged walk-forward, four models | `src/backtest/walkforward.py`, `engine.py` | `model_comparison.csv`, `*_feature_importance.csv`, `pulse_efficacy_path.csv` |
| 7 | Trading agent + myopic control | `src/agent/policy.py`, `backtest.py` | `agent_vs_myopic.csv`, `agent_params_by_fold.csv` |
| 8 | Factor controls | `src/evaluation/factor_controls.py` | `factor_controls.csv` |
| 9 | Deflated Sharpe | `src/evaluation/deflated_sharpe.py` | `deflated_sharpe.csv` |
| 10 | Figures + summary | `src/utils/plotting.py` | `figures/*.png`, `summary.md` |

Two extra scripts are run separately because they are experiments rather than
pipeline stages: `scripts/agent_cost_sweep.py` (retrain the agent at several
cost levels) and `scripts/compose_pulse_agent.py` (feed PULSE forecasts to the
agent as its aim portfolio).

## 2.2 The panel: the one data structure everything shares

Every stage reads the same long table, one row per (date, name):

```
date        ticker  sig_momentum  sig_liquidity  …  ret      fwd_ret
2010-01-31  AAPL    0.42          -0.17          …  0.031    0.018
```

Two columns deserve emphasis:

- `ret` is the return realized *during* month $t$. It is known at the end of
  $t$, so it may be used as a signal.
- `fwd_ret` is the return over $(t, t{+}1]$. It is the **label**. It is *not*
  known at $t$.

Signals are rank-normalized within each date (§1.5) before anything else
happens, so every model sees comparably scaled features and no model can be
helped or hurt by a signal's raw units.

## 2.3 Purged walk-forward: the testing protocol

Models are never evaluated on data they were fit on. The protocol:

```
fold 0:  train [-------- 120 months --------] X  [ test 12 ]
fold 1:  train [-------- 132 months ----------] X  [ test 12 ]
fold 2:  train [-------- 144 months ------------] X  [ test 12 ]
                                                 ^
                                                 the purge gap:
                                                 1 month deleted
```

Configured by `walkforward: {min_train: 120, test_size: 12, purge: 1,
embargo: 0, expanding: true}`. "Expanding" means each fold keeps all earlier
history rather than sliding a fixed window.

**Why the gap exists, and why it must be at least one month.**

**Theorem (purging).** Let the last training date be $t^\*$ and the first test
date be $\tau$, with a label horizon of $h$ periods. Training-label windows
and test-fold information are disjoint if and only if $t^\* + h < \tau$.

**Proof.** The label attached to training date $t$ is realized over
$(t, t+h]$, so the union of all training-label windows extends to
$t^\* + h$. The test fold begins at $\tau$. These overlap exactly when
$t^\* + h \ge \tau$. Deleting the $h$ dates between them is precisely the
condition $t^\* + h < \tau$. ∎

With $h = 1$ month, purge $= 1$. `walkforward.py` raises a hard `ValueError`
if anyone requests purge 0, and re-derives the realized gap with an assertion
on every fold. If you ever extend `fwd_ret` to a 3-month horizon, the theorem
tells you the config change that must travel with it: purge 3.

**One rule rides on the same logic: all tuning must consume training-window
data only.** That covers LightGBM's optional optuna search, IC-Net's early
stopping (which uses a chronological tail of the *training* dates, never a
random sample), and PULSE's $(a, q)$ grid search. A hyperparameter chosen
with test information is a leak wearing a suit.

**Fold counts.** Synthetic: 299 dates, `min_train` 120, purge 1 → first test
index 121, 14 folds, **168 out-of-sample months**. Real factor panel: 1176
dates → 87 folds, **1044 out-of-sample months**.

## 2.4 The agent's extra wrinkle: inventory crosses folds

The forecasting engine can treat folds independently, because a forecast has
no memory. The agent cannot: a portfolio is one continuous book. So
`src/agent/backtest.py` refits the policy on each purged training window but
**carries the final weight vector forward into the next fold's roll**. If it
liquidated at every fold boundary, 14 (or 87) rebalances' worth of cost would
silently escape the accounting. The purge theorem is unaffected: the policy's
*parameters* are still functions of training data only.

## 2.5 Handling a real, unbalanced panel

The synthetic panel is complete: every name exists on every date. The real
factor panel is not — factors start and stop at different dates, and the mean
cross-section is 145 of 212 names (3 names on the first date, 195 on the
last). Dropping any month with a missing name would have left 67 usable
months out of 1176 and produced *no* walk-forward folds at all.

The fix is **masking**. A NaN forward return marks a name untradable that
month: it is excluded from the demeaning and the L1 normalization, its weight
is forced to zero (an existing position is closed and the trade *is* charged),
and it contributes nothing to returns or gradients. On a complete panel the
mask is all-true and the code reduces exactly to the unmasked policy, which
is the property the tests pin down.

The quintile constructor has the analogous guard: a name must have a non-null
forward return *before* quantile formation, not just a non-null score —
otherwise the quantile cut is corrupted and the surviving leg quietly sums to
less than \$1 of exposure.

---

# Part 3 — The four models, in full

## 3.0 What the four models share

All four receive **identical** inputs and are judged on **identical** terms:

- the same rank-normalized signal matrix $X$ and the same label $y = $ `fwd_ret`
- the same purged walk-forward folds
- the same quintile portfolio constructor and the same 10 bps cost
- the same IC, ICIR, Sharpe, turnover and Newey–West statistics

So any difference between them is attributable to the model, not the harness.
That is the Gu–Kelly–Xiu comparison design in miniature: *does nonlinearity,
or a different loss, or time-variation, beat a linear benchmark on identical
inputs?*

Each model also embodies a different **inductive bias** — a different prior
about what structure returns have:

| Model | Prior about the world | Fails when |
|---|---|---|
| Elastic net | Effects are additive and constant | there are interactions, or effects drift |
| LightGBM | Effects are nonlinear and interacting, constant in time | the sample is too small/noisy to resolve interactions |
| IC-Net | Only the *ordering* matters; levels and scale are noise | the objective's non-convexity traps it, or magnitudes matter |
| PULSE | Effects are linear-on-an-interaction-basis but **drift**, decaying toward zero | regimes break abruptly, or monthly coefficient estimates are too noisy |

Part 8 shows the synthetic world rewards the last two and the real factor
world rewards the first. That is not a contradiction; it is the biases being
right about different worlds.

---

## 3.1 Elastic net — the benchmark that has to be beaten

### 3.1.1 Least squares from scratch

Given features $x_i$ (a vector including a leading 1 for the intercept) and
targets $y_i$, ordinary least squares picks $\beta$ minimizing

$$
L(\beta) = \sum_i (y_i - x_i^\top\beta)^2 = \Vert y - X\beta\Vert^2 .
$$

**Derivation.** $L$ is a smooth bowl; at the minimum its gradient vanishes.
Expanding $L = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta$ and
differentiating,

$$
\nabla L = -2X^\top y + 2X^\top X\beta = 0
\qquad\Longrightarrow\qquad
\hat\beta = (X^\top X)^{-1} X^\top y
$$

— the **normal equations**. With one feature this collapses to two formulas
worth memorizing:

$$
\hat\beta_1 = \frac{\mathrm{Cov}(x,y)}{\mathrm{Var}(x)}, \qquad \hat\beta_0 = \bar y - \hat\beta_1\bar x .
$$

**A regression slope is a rescaled covariance.** Regression, correlation and
the IC are one family of objects. And regressing on a constant *alone* gives
$\hat\beta_0 = \bar y$ — which is why the repo implements every "mean with a
Newey–West t-statistic" as a regression on a constant with HAC standard
errors. One tested code path serves every mean in the project.

### 3.1.2 Alpha: the intercept with a job title

The same machinery is the factor-control test. Regress a strategy's returns
on known factor returns $f_t$ (market, size, value, …):

$$
r^{\text{strat}}_t = \alpha + \beta^\top f_t + e_t
$$

with Newey–West errors. $\beta^\top f_t$ is the part explained by *rentable
exposures anyone can buy*. The intercept $\alpha$ is the average return left
over — **the claim to genuine information**. $R^2$ completes the picture:

> **Surviving alpha with low $R^2$ is the credible pattern. Vanished alpha
> means the "signal" was a known factor in disguise.**

Either outcome is worth reporting. One timing trap the code handles: returns
indexed by *formation* date must be paired with factor returns over the
*same holding window*, so `align="formation"` shifts the factor panel
accordingly — control variables can leak too.

### 3.1.3 Why shrink at all: the bias–variance decomposition

**Theorem.** Let the truth be $y = f(x) + \varepsilon$ with noise variance
$\sigma^2$, and let $\hat f$ be a model fit on a random training sample. For a
fixed test point $x$,

$$
E\big[(y - \hat f(x))^2\big] = \underbrace{\big(E[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}\big(\hat f(x)\big)}_{\text{variance}} + \sigma^2 .
$$

**Proof.** Write $y - \hat f = \varepsilon + (f - E[\hat f]) + (E[\hat f] - \hat f)$,
square, and take expectations. All three cross terms die: $\varepsilon$ is
independent with mean zero (killing two), and
$E[(f - E[\hat f])(E[\hat f] - \hat f)]$ has a constant first factor times a
mean-zero second factor. ∎

**Why finance sits at the extreme of this trade-off.** Return prediction has
brutal signal-to-noise: the *true* model's correlation ceiling here is about
0.05, i.e. $R^2$ well under 1%. When $\sigma^2$ dwarfs the signal, the
variance term dominates the bias term, so **deliberately biased, low-variance
estimators win**. That is the entire case for shrinkage, and it is the reason
Part 7's simple model beats the flexible ones on real data.

### 3.1.4 Ridge and lasso, solved exactly

The **elastic net** minimizes

$$
\Vert y - X\beta\Vert^2 + \lambda_2\Vert\beta\Vert^2 + \lambda_1\Vert\beta\Vert_1
$$

— squared error plus an L2 ("ridge") and an L1 ("lasso") penalty. In the
clean *orthonormal* case ($X^\top X = I$, i.e. uncorrelated standardized
features) both pieces solve in closed form, and the closed forms show exactly
what each penalty *does*. Let $\hat\beta = X^\top y$ be the OLS solution.

**Ridge.** Minimizing $\Vert y - X\beta\Vert^2 + \lambda\Vert\beta\Vert^2$ gives
$-2X^\top y + 2\beta + 2\lambda\beta = 0$, so

$$
\hat\beta^{\text{ridge}} = \frac{\hat\beta}{1+\lambda} .
$$

**Every coefficient shrinks toward zero by the same factor.** Pure variance
reduction, paid for with a little bias.

**Lasso.** The problem separates coordinate by coordinate: minimize
$(\beta_j - \hat\beta_j)^2 + \lambda|\beta_j|$. For $\beta_j > 0$ the
derivative $2(\beta_j - \hat\beta_j) + \lambda = 0$ gives
$\beta_j = \hat\beta_j - \lambda/2$, valid only while positive; symmetrically
for negative; otherwise the minimum sits at the kink $\beta_j = 0$. Compactly,

$$
\hat\beta_j^{\text{lasso}} = \mathrm{sign}(\hat\beta_j)\thinspace \max\negthinspace\big(|\hat\beta_j| - \tfrac{\lambda}{2},\thinspace 0\big)
$$

— **soft thresholding**: small coefficients are set *exactly* to zero
(automatic signal selection), large ones shrunk by a constant.

The repo's configuration keeps both penalties tiny (`alpha: 0.0001`,
`l1_ratio: 0.5`) because the features are few and already rank-normalized.
The benchmark's job is "best honest linear combination," not aggressive
selection.

### 3.1.5 What the linear model structurally cannot do

However shrunk, $\hat y = \beta^\top z$ is **additive**: momentum's
contribution is the same whatever the value signal's level. The synthetic
truth deliberately contains a term $\beta_{\text{int}}\thinspace z_{\text{mom}}\thinspace z_{\text{val}}$
whose whole point is non-additivity. §3.2 proves no additive model can
represent it.

---

## 3.2 LightGBM — gradient-boosted trees

### 3.2.1 Trees in one paragraph

A regression tree predicts by asking yes/no questions about features ("is
momentum rank above 0.3?"), routing each name down branches to a **leaf**, and
predicting that leaf's average target. Trees are step functions:
piecewise-constant surfaces over feature space. One tree is crude; the power
comes from adding many small ones.

### 3.2.2 Theorem: boosting with squared loss = repeatedly fitting residuals

**Setup.** Build the model in stages, $F_M(x) = \sum_{m=1}^M \eta\thinspace h_m(x)$,
where each $h_m$ is a small tree and $\eta$ a learning rate. Stage $m$ picks
$h_m$ to reduce $L(F) = \tfrac12\sum_i (y_i - F(x_i))^2$.

**Claim.** The steepest-descent direction at stage $m$ is the vector of
**residuals** $y_i - F_{m-1}(x_i)$.

**Proof.** Treat the predictions at the training points as free variables.
Then

$$
\frac{\partial L}{\partial F(x_i)} = -\big(y_i - F(x_i)\big),
$$

so the negative gradient — the fastest-decrease direction — is exactly the
residual vector. "Gradient descent in function space" therefore means: fit the
next tree to the residuals, then take a small step. ∎

That is the whole conceptual content of gradient boosting. LightGBM adds
industrial optimizations (histogram split search, leaf-wise growth, per-tree
row and column subsampling) which change speed and regularization, not the
mathematics. For other losses the recipe is identical with "residual"
replaced by "negative gradient of that loss" — a fact §3.3 exploits from the
opposite direction: *choose the loss so that its gradient is the thing you
care about.*

### 3.2.3 The interaction theorem — what additive models cannot say

The planted synthetic truth contains $\beta_{\text{int}}\thinspace z_1 z_2$
(momentum × value): *momentum works better among cheap stocks*. Linear models
— indeed anything of the **additive** form $f_1(z_1) + f_2(z_2)$, however
nonlinear each piece — cannot represent this.

**Theorem.** There exist no functions $f, g$ with $z_1 z_2 = f(z_1) + g(z_2)$
for all $z_1, z_2$.

**Proof.** Set $z_2 = 0$: then $0 = f(z_1) + g(0)$ for every $z_1$, so $f$ is
the constant $-g(0)$. By symmetry ($z_1 = 0$), $g$ is the constant $-f(0)$.
Then $f + g$ is constant while $z_1 z_2$ is not. Contradiction. ∎

(Calculus fingerprint: interactions are exactly where the mixed derivative
$\partial^2 F/\partial z_1\partial z_2 \ne 0$; additive functions have it
identically zero.)

**How trees escape.** A tree that splits on $z_1 > 0$ and then, *within that
branch*, on $z_2 > 0$ gives the four quadrants of $(z_1, z_2)$ four different
values — a staircase approximation to the saddle surface $z_1 z_2$, refined by
further boosting rounds. **Depth ≥ 2 plus sequential splits *is* interaction
capacity.** This theorem is the entire explanation of the synthetic model
ranking in Part 6.

### 3.2.4 Reading feature importances

LightGBM's **gain importance** sums, over every split made on a feature, the
reduction in squared error that split achieved — "how much work did this
feature do." The repo sets `importance_type="gain"` deliberately; the default
(counting splits) measures how *often* a feature was used, which is diffuse
and nearly uninformative here.

**The fine print, which matters.** In the synthetic run, gain importances put
the interaction pair on top (`sig_value` 0.199, `sig_momentum` 0.189) — but
the placebo still shows **0.153**, not 0. Trees will happily split on noise a
little, and gain accounting credits those splits. The general lesson travels
beyond this repo: **importances describe what the model *used*, which is
evidence about — but not identical to — what is *true*.**

---

## 3.3 IC-Net — when the loss function *is* the economics

This is the first of the two custom models. Implementation:
`src/models/icnet.py`, pure NumPy, forward pass, backpropagation and Adam all
written out by hand.

### 3.3.1 The motivating theorem: what MSE actually optimizes

Fix one date, with predictions $p$, forward returns $y$, $n$ names. Write
$\bar p, \bar y$ for the means, $s_p, s_y$ for the population standard
deviations, $\rho$ for the Pearson correlation.

**Theorem (exact per-date MSE decomposition).**

$$
\frac1n\sum_{i=1}^n (p_i - y_i)^2 \mkern5mu=\mkern5mu \underbrace{(\bar p - \bar y)^2}_{\text{level error}} \mkern5mu+\mkern5mu \underbrace{(s_p - s_y)^2}_{\text{scale error}} \mkern5mu+\mkern5mu \underbrace{2\thinspace s_p\thinspace s_y\thinspace(1-\rho)}_{\text{ordering error}} .
$$

**Proof.** Split each error into a mean part and a demeaned part:
$p_i - y_i = (\bar p - \bar y) + (\tilde p_i - \tilde y_i)$. Squaring and
averaging, the cross term vanishes because demeaned quantities average to
zero:

$$
\tfrac1n\sum (p_i - y_i)^2 = (\bar p - \bar y)^2 + \tfrac1n\sum(\tilde p_i - \tilde y_i)^2 .
$$

Expand the second piece as
$s_p^2 - 2 s_p s_y \rho + s_y^2$ (the middle term is the covariance), then
regroup: $s_p^2 + s_y^2 - 2s_p s_y\rho = (s_p - s_y)^2 + 2 s_p s_y (1-\rho)$. ∎

**Now read the three terms against Part 1.** The portfolio is dollar-neutral,
so it is *provably indifferent* to the level term (§1.8). The quintile sort
uses only ordering, so it is indifferent to the scale term too. **Only the
third term maps to money.** Yet an MSE-trained model spends capacity on all
three — and worse, the third term is weighted by $s_y$, the month's return
dispersion, so **volatile months dominate MSE training** even though (Theorem
B, §1.4) they carry no extra information about ordering skill.

That is the precise statement of what off-the-shelf regression wastes.

### 3.3.2 The objective

Delete the waste. Train the network $f(\cdot\thinspace; W)$ to directly
**maximize**

$$
J(W) \mkern5mu=\mkern5mu \frac{1}{T}\sum_{t=1}^{T}\rho_t\big(f(X_t; W),\thinspace y_t\big) \mkern5mu-\mkern5mu \lambda\Vert W\Vert^2 ,
$$

the average per-date cross-sectional correlation between predictions and
forward returns, minus a ridge penalty. On rank-normalized inputs this is a
differentiable stand-in for the Spearman IC — *the exact statistic every
signal in the repo is judged by.*

The invariances of §1.4 now read as **designed-in economics**: the objective
*cannot* reward market timing (translation invariance = dollar neutrality) and
*cannot* be bribed by volatile months (scale invariance). And the training
signal is **listwise**, not pointwise: "did you order this date's cross-section
correctly," which is literally the job.

### 3.3.3 The gradient of a correlation (full derivation)

The only nontrivial calculus is $\partial\rho/\partial p$ for one date. With
$M = I - \tfrac1n\mathbf 1\mathbf 1^\top$ the demeaning matrix (note
$M^\top = M$ and $MM = M$), write $\rho = A/B$ with
$A = \langle\tilde p,\tilde y\rangle$ and
$B = \Vert\tilde p\Vert\thinspace\Vert\tilde y\Vert$.

**Piece 1.** $A = (Mp)^\top\tilde y = p^\top(M\tilde y) = p^\top\tilde y$
(demeaning an already-demeaned vector changes nothing), so
$\nabla_p A = \tilde y$.

**Piece 2.** $\Vert\tilde p\Vert^2 = p^\top M p$, so
$\nabla_p\Vert\tilde p\Vert^2 = 2Mp = 2\tilde p$, and by the chain rule
$\nabla_p\Vert\tilde p\Vert = \tilde p/\Vert\tilde p\Vert$.

**Quotient rule.**

$$
\nabla_p\rho \mkern5mu=\mkern5mu \frac{\tilde y}{\Vert\tilde p\Vert\thinspace\Vert\tilde y\Vert} \mkern5mu-\mkern5mu \rho\thinspace\frac{\tilde p}{\Vert\tilde p\Vert^2}
$$

— exactly the two-term expression in `_objective_and_grad`. A pleasing sanity
property falls out free: both $\tilde y$ and $\tilde p$ have zero mean, so
**the gradient has zero mean** — nudging all predictions up together cannot
improve the objective. The calculus rediscovers Theorem A on its own.

### 3.3.4 Backpropagation, matched to the code line by line

The network is deliberately small: $Z = XW_1 + b_1$, $H = \tanh(Z)$,
$p = Hw_2 + b_2$, with 16 hidden units on ~6 features. *The point is the
objective, not the capacity.*

**Lemma.** $\tanh'(z) = 1 - \tanh^2(z)$.
*Proof:* with $u = e^z - e^{-z}$ and $v = e^z + e^{-z}$ (so $u' = v$,
$v' = u$), the quotient rule gives $(v^2 - u^2)/v^2 = 1 - (u/v)^2$. ∎

Let $g = \partial J/\partial p$ be the §3.3.3 vector stacked over dates. Each
line below is a line of `icnet.py`:

| Math | Code |
|---|---|
| $\partial J/\partial w_2 = H^\top g$ | `dw2 = H.T @ g_p` |
| $\partial J/\partial b_2 = \mathbf 1^\top g$ | `db2 = g_p.sum()` |
| $\partial J/\partial H = g\thinspace w_2^\top$ | `dH = np.outer(g_p, self.w2)` |
| $\partial J/\partial Z = (g\thinspace w_2^\top)\odot(1 - H^2)$ | `dZ = dH * (1 - H*H)` |
| $\partial J/\partial W_1 = X^\top\thinspace\partial J/\partial Z$ | `dW1 = X.T @ dZ` |
| $\partial J/\partial b_1 = \mathbf 1^\top\partial J/\partial Z$ | `db1 = dZ.sum(axis=0)` |

each with $-2\lambda W$ appended for the ridge term. **"Backpropagation" is
nothing more mysterious than this table**: the chain rule, organized so every
intermediate is reused.

### 3.3.5 Adam, and why the bias correction exists

Plain gradient ascent $W \leftarrow W + \eta g$ is fragile when gradient
scales differ across parameters. Adam keeps two exponential moving averages —
$m \leftarrow \beta_1 m + (1-\beta_1)g$ (smoothed direction) and
$v \leftarrow \beta_2 v + (1-\beta_2)g^2$ (per-parameter magnitude) — and steps
$W \leftarrow W + \eta\thinspace\hat m/(\sqrt{\hat v} + \epsilon)$.

**The correction, derived.** Both averages start at 0, so early values are
biased low. Unroll: $m_k = (1-\beta_1)\sum_{j=1}^{k}\beta_1^{\thinspace k-j}g_j$.
If gradients hover near a constant $g$, the weights form a geometric series
and $E[m_k] \approx g\thinspace(1 - \beta_1^k)$. Dividing by $(1-\beta_1^k)$ —
and likewise $v_k$ by $(1-\beta_2^k)$ — removes the startup bias exactly. ∎
That division is the otherwise-cryptic pair of lines in the training loop.

**Keep Adam's scale-freeness in mind.** It is a virtue on real gradients and
§4.9 shows it caused a genuine, documented bug when it met a *zero* gradient.

### 3.3.6 Early stopping, and the discipline that makes it legal

Neural training eventually memorizes noise (§3.1.3's variance term,
unleashed). The fix is **early stopping**: monitor the objective on held-out
data, keep the weights from the best epoch. This is implicit regularization —
training paths start near zero weights and grow, so stopping early caps
effective complexity, a cousin of ridge's explicit cap.

The non-negotiable detail is *which* held-out data: a **chronological tail of
the training dates** (`val_fraction: 0.15`), never anything from the test
fold. Randomly sampled validation rows would leak, because a random month's
neighbours sit in training with overlapping information.

### 3.3.7 Honest limits the math itself predicts

- The objective is **non-convex** — multiple local optima, hence seed
  dependence. Logged as a backlog item, not hidden.
- Correlation ignores **magnitude**. That is exactly right for quantile
  sorts, and incomplete for weight-proportional construction (where §4.13's
  Pearson formula shows levels *do* matter).
- The documented extension path: add a turnover penalty, then move to
  MSRR-style direct weight learning.

---

## 3.4 PULSE — tracking a moving truth with a Kalman filter

The second custom model, and the project's most distinctive idea.
Implementation: `src/models/pulse.py`, ~150 lines of NumPy including the
per-date regressions, the filter, the likelihood and the grid search.

**PULSE** = **P**er-date **U**pdate of **L**atent **S**ignal **E**fficacy.

### 3.4.1 The hypothesis

Every standard model — including the three above — is trained as if signal
efficacy were **stationary**: one fit over decades, implicitly averaging
momentum's 1999 strength with its 2020 strength. Three strands of current
literature say that assumption is wrong in a *structured, exploitable* way:

- **Decay is real and directional.** McLean & Pontiff (2016): anomaly returns
  fall ~26% after the original sample ends and ~58% after publication.
  Efficacy has a systematic downward pull.
- **Efficacy is persistent month to month.** Factor momentum (Gupta & Kelly
  2019; Ehsani & Linnainmaa 2022): factors' own returns are positively
  autocorrelated, so *recent* efficacy predicts *near-future* efficacy.
- **Efficacy is timing-sensitive.** "Anomaly Time" (2024): predictive power
  concentrates when information is fresh.

Jointly: **a signal's true coefficient is a slowly-moving latent state with a
decaying resting point** — not a constant. PULSE is the minimal model that
takes that sentence literally.

### 3.4.2 The state-space model

For each feature $k$ (of an interaction-expanded basis — see §3.4.6):

$$
\textbf{state:}\quad \beta_{k,t} = a\thinspace\beta_{k,t-1} + w_t, \qquad w_t \sim N(0, q_k)
$$

$$
\textbf{observation:}\quad \lambda_{k,t} = \beta_{k,t} + v_t, \qquad v_t \sim N(0, r_{k,t})
$$

Reading each piece:

- $\beta_{k,t}$ is the **true, unobservable** efficacy this month.
- The state equation says it drifts slowly ($q$ small) and relaxes toward zero
  at rate $a \le 1$ — **the decay prior. "Alpha dies by default" as a model
  assumption, not a slogan.**
- $\lambda_{k,t}$ is what we *can* see: the date-$t$ cross-sectional OLS
  coefficient — **the Fama–MacBeth first pass of §1.12** — a noisy monthly
  *measurement* of live efficacy.
- $r_{k,t}$ is that coefficient's sampling variance, **which the regression
  itself supplies** ($s^2(Z^\top Z)^{-1}$ on the diagonal). Known,
  per-date, heteroskedastic observation noise is the luxury that makes a
  Kalman filter the *principled* estimator here instead of an ad-hoc moving
  average.

### 3.4.3 The one lemma everything rests on

**Lemma (Bayesian update for a Gaussian).** If the prior is
$\beta \sim N(m, P)$ and we observe $\lambda \mid \beta \sim N(\beta, r)$, then

$$
\beta\mid\lambda \mkern5mu\sim\mkern5mu N\negthinspace\Big(m + K(\lambda - m),\mkern5mu (1-K)\thinspace P\Big), \qquad K = \frac{P}{P+r}.
$$

**Proof.** Bayes' rule multiplies densities; work with log densities, dropping
constants:

$$
-\tfrac{(\beta-m)^2}{2P} - \tfrac{(\lambda-\beta)^2}{2r} = -\tfrac12\Big[\beta^2\big(\tfrac1P + \tfrac1r\big) - 2\beta\big(\tfrac{m}{P} + \tfrac{\lambda}{r}\big)\Big] + \text{const}.
$$

A quadratic in $\beta$ is the exponent of a Gaussian whose precision
(= 1/variance) is the $\beta^2$ coefficient and whose mean is the linear
coefficient divided by that precision:

$$
P_{\text{post}} = \Big(\tfrac1P + \tfrac1r\Big)^{-1} = \frac{Pr}{P+r} = (1-K)P, \qquad m_{\text{post}} = P_{\text{post}}\Big(\tfrac{m}{P} + \tfrac{\lambda}{r}\Big) = m + K(\lambda - m). \mkern5mu\blacksquare
$$

**Read the mean: posterior = prior + gain × surprise.** The gain
$K = P/(P+r)$ is a precision-weighted compromise — trust the observation more
when your prior is uncertain ($P$ large) or the measurement is clean ($r$
small). **Every "learning rate" you have ever met is this formula with the
uncertainties hidden.**

### 3.4.4 The filter is just predict, then update

Between observations the state moves, so belief must move too. If
$\beta_{t-1} \sim N(m_{t-1}, P_{t-1})$ given data, then by linearity and the
variance rules of §1.2:

$$
\textbf{Predict:}\quad \beta_t \mid \text{data}_{t-1} \sim N\big(a\thinspace m_{t-1},\mkern5mu a^2 P_{t-1} + q\big)
$$

$$
\textbf{Update:}\quad m_t = a\thinspace m_{t-1} + K_t\big(\lambda_t - a\thinspace m_{t-1}\big), \qquad K_t = \frac{a^2 P_{t-1} + q}{a^2 P_{t-1} + q + r_t}
$$

Those two lines are, verbatim, the loop in `pulse.py::_filter_1d`. **That is
the entire Kalman filter** — there is no further mystery in it.

The forecast PULSE actually trades is the *predicted* mean $a\thinspace m_{t-1}$:
belief about *this* month's efficacy given data through last month.

**A free model-selection criterion.** The one-step **innovation**
$\lambda_t - a\thinspace m_{t-1}$ has distribution
$N(0,\thinspace a^2 P_{t-1} + q + r_t)$, so the summed innovation log-likelihood

$$
\ell(a,q) = \sum_t -\tfrac12\Big[\log\big(2\pi S_t\big) + \frac{(\lambda_t - a\thinspace m_{t-1})^2}{S_t}\Big], \qquad S_t = a^2 P_{t-1} + q + r_t
$$

scores how well the model's *predictions* explain the observed coefficient
series. Maximizing it over a small grid ($a \in \lbrace 0.97, 0.99, 1.0\rbrace$,
$q$-scale $\in \lbrace 0.002, 0.01, 0.05\rbrace$) picks the dynamics — and
because it only uses one-step-ahead predictions **inside the training
window**, it is an out-of-sample criterion that leaks nothing.

### 3.4.5 Theorem: the steady-state filter is an EWMA with a data-chosen weight

Practitioners smooth factor returns with exponentially weighted averages by
instinct. The filter explains *why that works and what the weight should be*.

**Theorem.** With constant $r_t = r$ and $a = 1$ (pure random walk), the
filter variance converges to a fixed point

$$
P^\* = \frac{-q + \sqrt{q^2 + 4qr}}{2},
$$

the gain to a constant $K^\* = (P^\* + q)/(P^\* + q + r)$, and the state
estimate becomes exactly

$$
m_t = (1-K^\*)\thinspace m_{t-1} + K^\*\thinspace\lambda_t \mkern5mu=\mkern5mu K^\*\sum_{j\ge 0}(1-K^\*)^j\thinspace\lambda_{t-j},
$$

an exponentially weighted moving average with half-life
$\ln 2 / |\ln(1-K^\*)|$.

**Proof.** At a fixed point the post-update variance reproduces itself through
one predict–update cycle: $P = (P+q)r/(P+q+r)$. Cross-multiplying gives
$P^2 + qP - qr = 0$, whose positive root is $P^\*$. Constant $P^\*$ gives
constant $K^\*$; substituting into the update recursion and unrolling the
geometric recursion gives the EWMA form. ∎

**The moral.** EWMA smoothing of factor efficacy is not ad hoc — it is the
*optimal* tracker under random-walk dynamics — **and** the right smoothing
constant is set by the signal-to-noise ratio $q/r$. Noisy monthly
coefficients (large $r$) ⇒ small $K^\*$ ⇒ long memory. Genuinely mobile
efficacy (large $q$) ⇒ fast adaptation. PULSE estimates that ratio per signal
by likelihood instead of guessing one half-life for everything.

### 3.4.6 Nonlinearity via basis expansion

PULSE's state equation is linear, but it is linear *on an expanded basis*:
`expand_interactions` appends every pairwise product $z_i z_j$, each with its
own filtered efficacy. So PULSE can track, for example, a momentum × value
interaction *whose strength itself drifts*. With $K$ base signals this gives
$K(K+1)/2$ coefficients — 21 for $K = 6$, which is why §3.4.8 lists the
explosion at $K = 200$ as a known limit.

### 3.4.7 Proposition: the PULSE forecast is point-in-time

**Claim.** The forecast for formation date $t$ uses only information available
at $t$.

**Proof.** The observation $\lambda_s$ is computed from signals at $s$ and
returns realized over $(s, s+1]$, so it only becomes known at $s+1$. At
formation date $t$ the available observations are therefore exactly
$\lambda_1,\dots,\lambda_{t-1}$. The filter state $m_{t-1}$ is a function of
those alone, and the traded forecast
$\sum_k a\thinspace m_{k,t-1}\thinspace z_{k,i,t}$ additionally uses only signals
dated $t$. Inside a walk-forward test fold **no update steps run at all** —
states propagate by the prior dynamics as $a^h m$ — so no test-fold return
ever touches the weights. ∎

A subtle implementation point worth noticing, because it is exactly the kind
of thing that silently breaks a backtest: $h$ must be the actual *calendar*
distance in months from the last fitted date, not the position of the date
within whatever array was passed. A fold's test dates begin
`purge + embargo` months after the last training date, so enumerating from 0
would under-decay every state (using $h=1$ where $h=2$ is correct). The code
computes calendar months explicitly for this reason.

### 3.4.8 Known failure modes, stated in advance

- **Abrupt regime breaks defeat a smooth filter** — a crash is not a
  random-walk step. A hidden-Markov regime model is the documented
  alternative.
- **The interaction expansion explodes** with ~200 real signals. Screen to a
  shortlist first.
- **Weak signals give the filter mostly noise to track.** The placebo test
  bounds this but does not eliminate it.
- **Per-coefficient independent filters ignore cross-signal efficacy
  correlation.** A full-covariance filter is the upgrade.
- **The trial count for the Deflated Sharpe grows again** — logged honestly as
  $n_{\text{trials}} = 10$ or 11.

---

# Part 4 — The trading agent, in full

Implementation: `src/agent/policy.py` and `src/agent/backtest.py`, pure
NumPy. Closed-form companion: `src/agent/msrr.py`. Live path:
`src/execution/`.

## 4.1 Why forecasting and trading are different problems

Everything in Part 3 answers *which names look good*. An agent answers the
question that actually generates profit and loss: **what do I hold this month,
given what I held last month and what trading costs?**

The difference is one word: **inventory**.

- A forecaster is **memoryless**. Ask it twice, get the same answer.
- An agent's decision today **constrains its costs tomorrow**. Holding a name
  you want to sell is cheap; holding a name you want to keep is free.

Everything upstream in this project charges costs *after the fact* — compute
weights, then subtract the bill. The agent is the first component that
reasons about costs *before acting*.

## 4.2 Why not reinforcement learning

The obvious textbook answer (Jansen's *ML4T* ch. 22) is model-free deep RL —
DQN on price series. That is the wrong tool here, and the reason is
quantitative rather than aesthetic.

Model-free RL **estimates** $\nabla_p E[\text{reward}]$ from sampled
episodes. The estimator's variance has to be tamed by millions of
interactions. That is available in Atari; it is absurd with ~300 monthly
observations.

But look at what the "environment" actually is here: **historical returns plus
a known cost function** — a fixed, *differentiable* dataset. So the same
gradient can be **computed** to machine precision in a single pass. Every
sample RL would spend discovering the shape of the cost function, we spend on
the only genuinely scarce resource: independent months of data.

> **The general principle: reach for RL when the environment is unknown or
> non-differentiable; reach for calculus when you own the simulator.**

The lineage of that choice: Moody & Saffell (2001), who optimized the
performance measure directly through the trading recurrence two decades
before it was rediscovered; Gârleanu & Pedersen (2013), who proved the
*optimal* policy shape; Zhang, Zohren & Roberts (2020), deep nets trained
end-to-end on net Sharpe; and Kelly & Malamud's MSRR, the closed-form version
(§4.6).

## 4.3 The policy: three lines of algebra

Rather than a free-form network, the policy **is** the Gârleanu–Pedersen
structure — theory used as an inductive bias:

$$
\textbf{1. blend:}\quad \text{score}_{t,i} = \sum_k \theta_k\thinspace z_{k,i,t}
$$

$$
\textbf{2. aim:}\quad A_t = \frac{2\thinspace\widetilde{\text{score}}_t}{\Vert\widetilde{\text{score}}_t\Vert_1}
$$

$$
\textbf{3. partial adjustment:}\quad w_t = (1-\gamma)\thinspace w_{t-1} + \gamma\thinspace A_t
$$

with net return

$$
\text{net}_t = \langle w_t, y_t\rangle \mkern5mu-\mkern5mu \frac{c}{10000}\sum_i |w_{t,i} - w_{t-1,i}| .
$$

Step 2 demeans (making the book dollar-neutral, §1.8) and divides by the L1
norm, then scales by 2 — so the book is always exactly \$1 long and \$1 short,
whatever the scores happen to be. Step 3 says: don't jump to the ideal
portfolio, walk a fraction $\gamma$ of the way there.

**Trainable parameters: $\theta$ (one per signal) and $\gamma = \sigma(g)$,
the trading speed.** That is $K+1 = 7$ numbers for 6 signals. $\gamma$ is
parameterized through a sigmoid so it stays in $(0,1)$ with no constraint
handling.

- $\gamma = 1$: rebalance fully every month (the **myopic** control).
- $\gamma = 0.5$: move halfway toward the aim.
- $\gamma \to 0$: freeze the book.

**Theorem (the policy is an EWMA of aims).** With $w_0 = 0$,

$$
w_t \mkern5mu=\mkern5mu \gamma\sum_{j=0}^{t-1}(1-\gamma)^{\thinspace j}\thinspace A_{t-j}.
$$

**Proof.** Induction. Base: $w_1 = \gamma A_1$. Step: substitute the claim for
$w_{t-1}$ into the recursion,

$$
w_t = (1-\gamma)\thinspace\gamma\negthinspace\sum_{j=0}^{t-2}(1-\gamma)^j A_{t-1-j} + \gamma A_t = \gamma\negthinspace\sum_{j=1}^{t-1}(1-\gamma)^{j}A_{t-j} + \gamma A_t. \mkern5mu\blacksquare
$$

So the agent has exactly **two economic dials**: *what to aim at* ($\theta$,
which shapes $A_t$) and *how fast to chase it* ($\gamma$, the memory of the
EWMA, with half-life $\ln 2/|\ln(1-\gamma)|$).

**A unification worth saying out loud.** Compare §3.4.5: PULSE's filter is an
EWMA over *coefficient observations* with a weight chosen by **noise**. The
agent is an EWMA over *portfolios* with a weight chosen by **cost**. The same
mathematical object, two different economic forces selecting the memory.

Gârleanu–Pedersen prove that with quadratic trading costs the *optimal*
dynamic policy has precisely this partial-adjustment form. Their derivation
is a linear-quadratic control problem; this project takes the structure as
given and lets the data choose the dials.

## 4.4 The objective and its gradient

Training maximizes the **Sharpe ratio of net returns over the training
window** — not gross return, not forecast accuracy. Let $J = m/s$ where $m$
and $s$ are the mean and population standard deviation of the net return
series. We need $\partial J/\partial r_t$: how the objective responds to each
month's profit.

From $m = \tfrac1T\sum r_t$: $\partial m/\partial r_t = 1/T$.
From $s^2 = \tfrac1T\sum(r_t - m)^2$, differentiating gives
$2s\thinspace\partial s/\partial r_t = \tfrac{2}{T}(r_t - m)$ — the inner
$-\partial m/\partial r_t$ terms cancel because $\sum(r_u - m) = 0$ — so
$\partial s/\partial r_t = (r_t - m)/(Ts)$. The quotient rule then gives

$$
\frac{\partial J}{\partial r_t} \mkern5mu=\mkern5mu \frac{1}{T\thinspace s} \mkern5mu-\mkern5mu \frac{m\thinspace(r_t - m)}{T\thinspace s^{3}} .
$$

**Read it.** Every month's marginal value has a *baseline* $1/(Ts)$ — more
return is good — minus a **risk charge** proportional to how far that month
already sits from the mean. The objective actively dislikes months that add
variance faster than they add mean. **Maximizing Sharpe is not maximizing
return, and this formula is the precise statement of the difference.** It is
three lines of `_sharpe_and_grad`.

## 4.5 Forward-mode sensitivities: differentiating through time

Each $r_t$ depends on the parameters through *every* weight back to $w_1$,
because inventory has memory. Rather than reverse-mode backpropagation
through time, the code uses **forward-mode accumulation**: carry the Jacobian
$D_t = \partial w_t/\partial p$ (a small $P \times N$ matrix, $P = K+1$)
alongside the simulation. Differentiating the policy recursion directly:

$$
D_t = (1-\gamma)\thinspace D_{t-1} \mkern5mu+\mkern5mu \gamma\thinspace\frac{\partial A_t}{\partial\theta} \mkern5mu+\mkern5mu (A_t - w_{t-1})\thinspace\frac{\partial\gamma}{\partial g}, \qquad \frac{\partial\gamma}{\partial g} = \gamma(1-\gamma)
$$

(the last factor because $\gamma = \sigma(g)$; note $D_{t-1}$ already carries
$g$'s influence on *past* weights, so nothing is double-counted).

Each month then contributes

$$
\frac{\partial r_t}{\partial p} = D_t\thinspace y_t \mkern5mu-\mkern5mu c\thinspace(D_t - D_{t-1})\thinspace\mathrm{softsign}(\Delta_t),
$$

and the chain rule assembles the full gradient as
$\sum_t (\partial J/\partial r_t)(\partial r_t/\partial p)$.

**The smoothing trick that makes costs differentiable.** The cost term
$c\sum_i|w_{t,i} - w_{t-1,i}|$ has a kink at zero, where the derivative does
not exist. Replace $|x|$ with $\sqrt{x^2 + \varepsilon}$ (`_softabs`), whose
derivative

$$
\frac{d}{dx}\sqrt{x^2+\varepsilon} = \frac{x}{\sqrt{x^2+\varepsilon}}
$$

is a **soft sign**: equal to $\pm 1$ away from zero, rolling smoothly through
it. That single substitution is the entire reason the cost term can be
differentiated at all. The same trick appears in the L1 normalization of the
aim.

**The cost of forward mode** is $O(P)$ times the simulation — cheap when
$P = 7$, which is exactly why the economically-constrained policy class is
also the *computationally* right one. A thousand-parameter network would want
reverse mode; that trade is documented, not hidden.

## 4.6 MSRR: the optimal aim, in closed form

The agent *learns* its blend $\theta$ numerically. There is also a closed form
for the best possible linear blend, and having both lets each validate the
other.

### 4.6.1 The factor collapse

**Lemma.** If each name's weight is linear in its (per-date demeaned)
signals, $w_{i,t} = \sum_k\theta_k\tilde z_{k,i,t}$, then the portfolio return
collapses onto $K$ numbers per date:

$$
r_p(t) = \sum_i w_{i,t}\thinspace y_{i,t} = \sum_k\theta_k\underbrace{\Big(\sum_i\tilde z_{k,i,t}\thinspace y_{i,t}\Big)}_{f_{k,t}} = \theta^\top f_t .
$$

**Proof.** Swap the two finite sums. ∎

The $f_{k,t}$ are **characteristic-managed portfolio returns**: what you earn
holding each name in proportion to its value of signal $k$. **A universe of
500 names has just become a $K$-asset problem** — the single most useful
dimension reduction in cross-sectional finance, and the reason the next
theorem has a closed form.

### 4.6.2 Theorem (MSRR): the max-Sharpe blend is $\Sigma^{-1}\mu$

Let $\mu = E[f_t]$ and $\Sigma = \mathrm{Cov}(f_t)$. Maximize
$S(\theta) = \theta^\top\mu/\sqrt{\theta^\top\Sigma\thinspace\theta}$.

**Theorem.** $\theta^\* \propto \Sigma^{-1}\mu$, achieving
$S(\theta^\*) = \sqrt{\mu^\top\Sigma^{-1}\mu}$.

**Proof.** Substitute $x = \Sigma^{1/2}\theta$ and $b = \Sigma^{-1/2}\mu$
(the symmetric square root exists because $\Sigma$ is positive definite).
Then $S = b^\top x/\Vert x\Vert \le \Vert b\Vert$ by **Cauchy–Schwarz —
§1.3's theorem, reused verbatim** — with equality iff $x \propto b$, i.e.
$\theta \propto \Sigma^{-1}\mu$; and $\Vert b\Vert = \sqrt{\mu^\top\Sigma^{-1}\mu}$. ∎

**Read it.** The optimal blend is each factor's mean return, *deflated by the
risk it shares with the others*. High-mean factors get weight, but redundant
ones get discounted through $\Sigma^{-1}$.

### 4.6.3 Why it is called a "Regression"

**Regressing the constant 1 on the factor returns points at the same
portfolio.** The OLS coefficient of $\mathbf 1$ on $F$ (the $T\times K$ matrix
of factor returns) is $(F^\top F)^{-1}F^\top\mathbf 1$, and
$F^\top\mathbf 1/T = \mu$ while $F^\top F/T = \Sigma + \mu\mu^\top$ (the
moment identity $E[ff^\top] = \Sigma + \mu\mu^\top$). So we need
$(\Sigma + \mu\mu^\top)^{-1}\mu$.

**Lemma (Sherman–Morrison).**
$(A + uv^\top)^{-1} = A^{-1} - \dfrac{A^{-1}uv^\top A^{-1}}{1 + v^\top A^{-1}u}$.

**Proof.** Multiply the claimed inverse by $(A + uv^\top)$; with
$q = v^\top A^{-1}u$ the cross terms give
$uv^\top A^{-1}[1 - \tfrac{1}{1+q} - \tfrac{q}{1+q}] = 0$, leaving $I$. ∎

**Corollary.** With $A = \Sigma$ and $u = v = \mu$,

$$
(\Sigma + \mu\mu^\top)^{-1}\mu = \frac{\Sigma^{-1}\mu}{1 + \mu^\top\Sigma^{-1}\mu} \mkern5mu\propto\mkern5mu \Sigma^{-1}\mu . \qquad\blacksquare
$$

So **a regression — the humblest tool in the box — computes the
maximum-Sharpe portfolio's direction exactly.** In practice $\mu$ and
$\Sigma$ are noisy estimates, so the implementation solves
$(\Sigma + \lambda I)^{-1}\mu$: §3.1.4's ridge shrinkage transplanted into
portfolio space. When $K$ grows into the hundreds, how $\lambda$ interacts
with $K > T$ *is* the "virtue of complexity" debate (Kelly–Malamud–Zhou 2024
versus Nagel 2025 and Buncic 2025) — this corollary is that literature's
front door.

### 4.6.4 The validation this buys

At zero cost the numerically-trained agent — Adam through the full trading
recursion, with no knowledge whatsoever of §4.6.2 — **recovers the closed
form**: cosine similarity between learned $\theta$ and the $\Sigma^{-1}\mu$
direction is **1.000**, gross Sharpe 1.204 versus the closed form's 1.205,
and both assign the placebo approximately zero weight. Optimizer and theorem
validate each other. The 0.1% gap is understood and stated: the agent's
per-date L1 normalization reweights dates slightly.

This makes the Gârleanu–Pedersen decomposition explicit and exact:
**aim = MSRR, execution = learned partial adjustment.**

## 4.7 The capture-ratio theorem — the business case in one fraction

How much of a signal's *fresh* alpha does a slow, stale EWMA book capture?

Let the aim track a single AR(1) signal $z_t$ with persistence $\rho$
(§1.10), so next month's alpha is $\beta z_t$, and let the book be
$w_t = \gamma\sum_{j\ge0}(1-\gamma)^j z_{t-j}$ (§4.3's theorem).

**Theorem.** Expected captured alpha is $\beta\cdot\kappa(\gamma,\rho)$ with

$$
\kappa(\gamma,\rho) \mkern5mu=\mkern5mu \gamma\sum_{j\ge0}(1-\gamma)^j\rho^j \mkern5mu=\mkern5mu \frac{\gamma}{1 - (1-\gamma)\rho} .
$$

**Proof.** $E[w_t\thinspace\beta z_t] = \beta\gamma\sum_j(1-\gamma)^j E[z_{t-j}z_t] = \beta\gamma\sum_j(1-\gamma)^j\rho^j$
by §1.10's Claim 2; sum the geometric series. ∎

**Sanity checks.** $\kappa(1,\rho) = 1$ — myopic captures everything.
$\kappa \to 0$ as $\gamma \to 0$ — a frozen book captures nothing.
$\kappa$ increases in $\rho$ — **slow signals forgive slow trading.**

Turnover, meanwhile, *increases* in $\gamma$, so the net objective is a
tug-of-war

$$
\beta\thinspace\kappa(\gamma,\rho) \mkern5mu-\mkern5mu c\cdot\text{turnover}(\gamma)
$$

whose interior optimum moves **down** as $c$ rises. That is the derived
skeleton of the measured cost sweep in Part 6.

**And here is the business case as a single fraction.** At the 25 bps point of
the synthetic sweep the agent learned $\gamma = 0.60$ on a $\rho \approx 0.9$
blend, giving

$$
\kappa = \frac{0.60}{1 - 0.40\times 0.9} = \frac{0.60}{0.64} = 0.94 .
$$

**The agent kept 94% of the alpha while cutting turnover from 0.42 to 0.27 —
a 36% cost reduction for a 6% alpha sacrifice.** That ratio *is* the argument
for execution-aware trading.

## 4.8 Square-root market impact: the cost of size

Linear cost ($c\cdot|\Delta w|$) says the thousandth dollar trades as cheaply
as the first. Markets disagree. The empirical **square-root law** — one of
the most replicated results in market microstructure — has price impact
growing like the square root of trade size, so total cost (impact × quantity)
grows like

$$
|\Delta w|\cdot|\Delta w|^{1/2} = |\Delta w|^{3/2},
$$

a **convex** cost. Convexity has one big behavioral consequence: marginal
cost rises with size, so optimal trades shrink and spread out. Verified as a
comparative static: adding impact lowered the learned speed 0.49 → 0.44 and
turnover 0.22 → 0.20.

Differentiability is preserved by the same smoothing idea as §4.5 —
implement $|x|^{3/2}$ as $(x^2+\varepsilon)^{3/4}$, derivative
$\tfrac32 x(x^2+\varepsilon)^{-1/4}$.

**One honest caveat travels with the feature:** the *coefficient* of impact is
far harder to estimate from data than a commission schedule, so on real data
it is a sensitivity parameter to sweep, not a constant to trust.

## 4.9 Case study: a real bug, predicted by a theorem

This section is the most instructive page in the project, because the
mathematics *predicted a failure that actually happened* during development.

### 4.9.1 The null-direction theorem

The aim is scale-invariant by construction: $A(\theta) = 2\tilde s/\Vert\tilde s\Vert_1$
with $s = \sum_k\theta_k z_k$, so replacing $\theta \to c\theta$ for any
$c>0$ multiplies numerator and denominator by the same $c$.

**Theorem.** $J(c\theta, g) = J(\theta, g)$ for all $c>0$, and consequently
the gradient is everywhere **orthogonal** to $\theta$:

$$
\theta^\top\nabla_\theta J \mkern5mu=\mkern5mu 0 .
$$

**Proof.** Everything downstream of the aim — weights, returns, costs, Sharpe
— depends on $\theta$ only through $A(\theta)$, which is invariant. So
$J(c\theta) = J(\theta)$. Differentiate this identity with respect to $c$ at
$c=1$: $\theta^\top\nabla_\theta J = 0$. ∎

(This is Euler's homogeneous-function argument for degree-zero functions. **A
symmetry always manufactures a direction the gradient cannot see** — the
optimization cousin of Noether's principle.)

**Corollary (the $K=1$ trap).** With a single feature, every $\theta>0$ is a
positive rescaling of every other, so $J$ depends only on
$\mathrm{sign}(\theta)$. The gradient is exactly zero everywhere except at
the kink, which the softabs $\varepsilon$ turns into numerical dust.

### 4.9.2 Why Adam random-walks a flat direction

A flat direction sounds harmless — no gradient, no movement. **Not under
Adam.** Its update is $\Delta p = \mathrm{lr}\cdot\hat m/(\sqrt{\hat v}+\epsilon)$.
Feed it tiny numerical noise $\delta_t$ with no consistent sign: then
$\hat m \sim O(\delta)$ and $\sqrt{\hat v}\sim O(\delta)$, so **their ratio is
$O(1)$ with a random sign.** The parameter takes steps of size ≈ lr
*regardless of how small the noise is*. Adam's scale-freeness, a virtue on
real gradients, turns "zero gradient" into **unit-speed diffusion**.

Over 150 epochs, a $\theta$ that started at $+0.2$ drifted across zero —
flipping the sign of every weight in the book. The agent then observed a
*negative-alpha* book in training and did the rational thing: it learned
$\gamma \to 0$ and froze.

**The observed symptom:** a forecast with IC $+0.074$ producing an
out-of-sample Sharpe of $-6.19$ under $\gamma = 1$ (the exact mirror image of
the true book's $+6.49$), and a "learned" $\gamma$ of 0.003.

**The diagnostic signature that cracked it:** rank-IC was serenely positive
while the book was catastrophically negative. §4.13 explains why those two
can disagree, and that disagreement is what localized the fault to the
sign/plumbing rather than the model.

### 4.9.3 The fix, and a lemma that it works

After each Adam step, project $\theta$ back to the unit L1 sphere:
$\theta \leftarrow \theta/\Vert\theta\Vert_1$. This is optimization on the
*quotient* of the symmetry — it deletes the flat direction while leaving all
perpendicular (real) gradients untouched.

**Lemma (no sign flip).** With $K=1$, projection keeps
$\theta \in \lbrace -1, +1\rbrace$, and a flip would need a single step of
magnitude $>1$; Adam's step is bounded near lr $= 0.05 \ll 1$, so the sign is
stable. For $K \ge 2$ the projection is inert where it should be: genuine
gradients are orthogonal to $\theta$ by the Theorem, hence already tangent to
the sphere. ∎ *(The step bound is the standard heuristic
$|\hat m|/\sqrt{\hat v}\lesssim 1$ — labeled honestly: an argument, not a
worst-case proof.)*

**Result after the fix:** the same composition runs at net Sharpe $+5.35$; the
$K=6$ baselines are unchanged to three decimals (perpendicular gradients
dominated all along, as the theorem says they should); all tests pass.

> **The transferable lesson: every invariance you build into a model is a
> direction your optimizer can wander. Either quotient it out, or expect it
> to be explored.**

## 4.10 Identifiability: the multi-speed cousin of the same disease

The implemented multi-speed agent (`MultiSpeedPolicyAgent`) gives each signal
its own EMA $u_k$ with its own speed $\gamma_k$. That hides a second
flat-ish direction.

**Lemma.** For an i.i.d. unit-variance input, the EMA
$u_t = (1-\gamma)u_{t-1} + \gamma A_t$ has stationary variance

$$
\mathrm{Var}(u) = \gamma^2\sum_{j\ge0}(1-\gamma)^{2j} = \frac{\gamma^2}{1-(1-\gamma)^2} = \frac{\gamma}{2-\gamma},
$$

which $\to 0$ as $\gamma \to 0$. ∎

So a signal whose speed learns its way to $\gamma_k \approx 0$ contributes a
component $u_k$ of vanishing size — and **its blend weight $\theta_k$
multiplies nothing.** Unidentified, free to wander, exactly like §4.9's null
direction but signal-by-signal.

This explains a real probe result: the placebo appeared to "hold" 20–25% of
raw $\theta$ at $\gamma \approx 0.007$ — an economically empty parameter
parked at a meaningless value. **The corrected metric is effective exposure**

$$
\text{eff}_k = \theta_k\cdot\mathrm{sd}(u_k),
$$

which the Lemma shows zeroes any frozen signal automatically. That is the
pre-registered lens for the real-data tilt test, *replacing* the raw-$\theta$
version which this analysis refuted. Noting that a metric was wrong, and
fixing the metric rather than the conclusion, is the methodological point.

## 4.11 A documented null result

Gârleanu–Pedersen theory predicts not only that speed falls with cost but
that **the aim should tilt toward persistent signals** as costs rise ("aim in
front of the target"). In this policy class **it does not**: the
value/momentum loading ratio was flat in cost (0.410 → 0.391 across the full
sweep).

On inspection, the theory agrees with the data. The tilt result requires
**signal-specific trading speeds**; with one shared $\gamma$, the EWMA
attenuates each signal's return contribution and its risk contribution nearly
proportionally, so the tilt cancels.

**What that is worth:** a planted-truth harness adjudicated a theoretical
prediction and located its precise policy-class dependence. The test suite
encodes it as a *documented null*, and it motivated the per-signal-speed
extension, which is implemented. At 60 bps the multi-speed agent beats the
myopic control decisively out of sample (monthly Sharpe 0.43 versus 0.22) and
the dominant fast signal's speed falls hard with cost
($\gamma_{\text{mom}}$: 0.70 → 0.27).

## 4.12 Composition: PULSE's forecast as the agent's aim

The agent blends *raw* signals linearly, so it structurally cannot see the
planted interaction (§3.2.3) — which is why its synthetic net Sharpe of 2.96
sits far below PULSE's 5.05. They are solving different problems on
different inputs.

So compose them: let **PULSE's forecast be the agent's aim**, and let the
agent decide the trading speed. Result on identical walk-forward and costs:
**composed net Sharpe 5.34 versus 2.96 for the static-blend baseline** — the
largest measured gain in the project's agent line, at similar turnover
(0.395 versus 0.354).

**Why PULSE and not IC-Net?** Leakage safety, and it is provable.

**Proposition (composition is point-in-time by induction).** If a forecast
satisfies $x_t \in \mathcal F_t$ (§3.4.7 proved this for PULSE) and the policy
is any map $w_t = \pi(w_{t-1}, x_t)$, then $w_t \in \mathcal F_t$ for all $t$.

**Proof.** Induction. $w_0 = 0 \in \mathcal F_0$. If
$w_{t-1}\in\mathcal F_{t-1}\subseteq\mathcal F_t$ and $x_t\in\mathcal F_t$,
then $w_t$ — a function of two $\mathcal F_t$-measurable inputs — is
$\mathcal F_t$-measurable. ∎

**Chains of point-in-time components are point-in-time: leakage cannot be
created by composition, only by a leaky component.** IC-Net is deliberately
*not* bolted on, because its in-sample fitted values are not
$\mathcal F_t$-measurable objects during training, so composing it would
require purged stacking first. That requirement is documented rather than
ignored — the discipline being that you do not take a +80% Sharpe improvement
if you cannot prove where the information came from.

## 4.13 What a proportional book earns: the Pearson formula

The composed agent's aim holds each name in proportion to its demeaned
forecast, $w = 2\tilde f/\Vert\tilde f\Vert_1$. Its one-date gross return has a
closed form. Using $\tilde f^\top\mathbf 1 = 0$ (dollar neutrality) and the
definition of correlation:

$$
w^\top y = \frac{2\thinspace\tilde f^\top\tilde y}{\Vert\tilde f\Vert_1} = \frac{2\thinspace n\thinspace s_f\thinspace s_y\thinspace\rho_t}{\Vert\tilde f\Vert_1} \mkern5mu\approx\mkern5mu 2\sqrt{\tfrac{\pi}{2}}\mkern5mu s_y\thinspace\rho_t \mkern5mu\approx\mkern5mu 2.507\thinspace s_y\thinspace\rho_t,
$$

the last step using $\Vert\tilde f\Vert_1 \approx n\thinspace s_f\sqrt{2/\pi}$
for a roughly Gaussian cross-section (since $E|Z| = \sigma\sqrt{2/\pi}$).

**Two readings, both important.**

1. **Proportional weights monetize the per-date *Pearson* correlation** — the
   exact quantity IC-Net trains on — just as quantile books monetize
   *ordering*. The forecast's *levels* matter here, which is precisely why a
   sign flip is catastrophic while rank-IC stays serenely positive. That is
   the §4.9.2 diagnostic, explained.
2. **It predicts the composition's headline number.** Measured mean Pearson
   $\bar\rho = 0.0834$ and cross-sectional return spread $s_y \approx 0.081$
   give expected gross $\approx 2.507\times0.0834\times0.081 = 0.0169$ per
   month $= 20.3$% per year. The composed agent measured **19.0% net** at
   turnover 0.395; adding back its cost drag
   ($0.395\times2\times10\thinspace\text{bps}\times12 \approx 0.9$%) implies
   gross $\approx 19.9$%. **Predicted 20.3, implied 19.9, with zero fitted
   parameters.**

## 4.14 From backtest to broker

Everything above is evaluated inside a purged walk-forward backtest; it is
never deployed there. The deployment path is separate and deliberately
different in two ways.

`src/execution/signal_generator.py` fits $\theta$ on every month with a
realized forward return, scores the latest cross-section into a
dollar-neutral aim via `aim_weights()` — the same score → demean → normalize
map as step 2 of §4.3, minus the recursion, since no forward return exists
yet for a live date — and hands that aim to
`src/execution/agent_evaluator.py`'s `CostAwareAgentEvaluator`, which performs
the partial-adjustment step against the **live broker position** rather than a
backtest's running inventory.

**Deliberately not reused: the backtest's fitted $\gamma$.** It is whatever
was Sharpe-optimal in-sample over the fit window. Live execution speed is the
evaluator's own `gamma_speed` from `configs/execution_config.yaml` — an
operator-set risk knob, reviewed independently of the research fit. This
preserves the same aim/execution split §4.6.4 established: **aim = model,
speed = execution.** Given §8.3's finding that the fitted $\gamma$ did not
generalize on real data, this separation looks like good judgment rather than
mere caution.

**Honest limits carried over unchanged:** OSAP signals are monthly with a real
publication lag, so "live" means "the most recently complete month," not
intraday; and OSAP/CRSP's `permno` identifier is not a tradable symbol
(licence restriction), so a `permno → ticker` map is a **required input**, not
something this repo can derive.

## 4.15 The agent's own honest limits

- **Linear costs by default** (square-root impact implemented but off in the
  pipeline config). Under real impact the agent would *over*-trade large
  positions.
- **Previous weights are not drift-adjusted** between rebalances — a
  simplification shared with the whole evaluation stack.
- **A single risk view**: Sharpe on net returns, with no explicit factor-risk
  or drawdown term.
- **Monthly frequency**, where the speed dial has the least room to matter.
  Daily is where this component would come alive.
- And the standing caveat of the whole project: **the synthetic world is a
  kind laboratory. Every synthetic number is a validation of mechanism, not a
  promise about markets.**

---

# Part 5 — The statistical immune system

Backtests lie in exactly two ways: by **using information from the future**
(§5.1–5.3) and by **selection among many tries** (§5.4–5.6). Both failure
modes are provable, predictable, and in this repository deliberately
demonstrated rather than merely avoided.

## 5.1 Information sets: the bookkeeping everything hangs on

Let $\mathcal F_t$ denote everything knowable at the end of month $t$. Three
rules, stated once:

1. A signal dated $t$ must be a function of $\mathcal F_t$ only.
2. The label paired with it, $r_{t\to t+1}$, is realized during $(t, t+1]$ —
   it belongs to $\mathcal F_{t+1}$, **not** $\mathcal F_t$.
3. Any statistic that lets fitting see test-period information is **leaked**.

The purging theorem (§2.3) is rule 2 made operational.

## 5.2 Predicting a leak before measuring it

The pipeline ships a deliberate leak (`demonstrate_lookahead`): a fake feature

$$
\ell = a\thinspace y + e, \qquad a = 0.5, \quad \sigma_e = \sigma_y
$$

— a half-strength copy of the very return it claims to predict, plus
independent noise. This simulates a timestamp error that lets next month's
information into today's feature.

**What IC *should* it produce? Compute, don't guess.**

$$
\rho(\ell, y) = \frac{\mathrm{Cov}(ay+e,\thinspace y)}{\sigma_\ell\thinspace\sigma_y} = \frac{a\thinspace\sigma_y^2}{\sigma_y\sqrt{a^2\sigma_y^2 + \sigma_e^2}} = \frac{a}{\sqrt{a^2+1}} = \frac{0.5}{\sqrt{1.25}} = 0.447 .
$$

Converting Pearson to Spearman for near-Gaussian data
($\rho_s = \tfrac{6}{\pi}\arcsin(\rho/2)$, a classical identity): predicted IC
$\approx \mathbf{0.431}$.

**Measured on the synthetic panel: 0.395 ($t = 179$).** Prediction and
measurement agree within the stated approximations (returns are not exactly
Gaussian; the noise is calibrated on pooled rather than per-date spread).

**The pedagogical point survives any of that slack.** An honest single signal
in this universe tops out near 0.04 — §6.2 *derives* that ceiling. So **a 0.4
IC is not a discovery, it is a diagnosis.** Ten times too good is not
"great," it is "broken." The leak table exists so you know the signature on
sight, in your own numbers rather than in a textbook.

## 5.3 Staleness is the mirror-image leak

Forward leakage uses information **too early**. Staleness uses it **too late**
and *destroys* real signal: §1.10 proved IC decays like $\rho^k$ with signal
age.

The practical moral — this is the "Anomaly Time" (2024) result — is that **the
formation timestamp is part of the strategy.** An evaluation on stale
features can "fail to replicate" a perfectly real anomaly. The academic
convention of forming portfolios once a year in June underestimates
predictability because the typical rebalance arrives ~80 trading days after
the information was released.

**Point-in-time correctness is therefore two-sided.** The repo's leak report
checks one side; the staleness profile checks the other.

## 5.4 Theorem: the best of $N$ tries is biased

Run $N$ strategy variants that are all, in truth, worthless — each backtest
Sharpe is a mean-zero random draw. You report the **best** one. The best of
$N$ mean-zero draws is not mean-zero.

**Theorem.** If $X_1,\dots,X_N$ are standard normal (any dependence allowed),
then $E[\max_i X_i] \le \sqrt{2\ln N}$.

**Proof.** For any $s>0$, Jensen's inequality ($e^{sx}$ is convex) gives

$$
e^{\thinspace s\thinspace E[\max X_i]} \mkern5mu\le\mkern5mu E\big[e^{\thinspace s\max X_i}\big] = E\big[\max_i e^{sX_i}\big] \mkern5mu\le\mkern5mu \sum_{i=1}^N E[e^{sX_i}] = N e^{s^2/2},
$$

using $\max \le \text{sum}$ for non-negative terms and the normal
moment-generating function. Take logs and divide by $s$:
$E[\max X_i] \le \frac{\ln N}{s} + \frac{s}{2}$. Minimize over $s$ (calculus
gives $s = \sqrt{2\ln N}$) to get the bound. ∎

**Read it as a price list.** Testing $N = 20$ worthless variants buys an
expected best t-statistic of up to $\sqrt{2\ln 20}\approx 2.45$ — past the
"significant" line — **for free. Selection manufactures significance.** The
only defenses are to (a) count your trials honestly and (b) raise the hurdle
accordingly.

## 5.5 The Probabilistic and Deflated Sharpe Ratios

**Step 1 — how noisy is a Sharpe estimate?** From $T$ observations its
approximate variance (Lo 2002; Bailey & López de Prado 2014, via the delta
method) is

$$
\mathrm{Var}(\widehat{SR}) \mkern5mu\approx\mkern5mu \frac{1 - \gamma_3\thinspace SR + \frac{\gamma_4-1}{4}SR^2}{T-1},
$$

where $\gamma_3$ is skewness and $\gamma_4$ *full* kurtosis (normal = 3). The
intuitions to keep: more months tightens it ($1/(T-1)$); **left-skewed
strategies — steady gains punctuated by crashes — have noisier Sharpes than
the normal formula suggests**; fat tails likewise. This is why
`annualized_stats` records skew and kurtosis for every series: they are
inputs here, not decorations.

**Step 2 — the Probabilistic Sharpe Ratio.** Standardize the estimate against
a benchmark $SR^\*$:

$$
\mathrm{PSR}(SR^\*) \mkern5mu=\mkern5mu \Phi\negthinspace\left(\frac{(\widehat{SR} - SR^\*)\thinspace\sqrt{T-1}}{\sqrt{\thinspace 1 - \gamma_3\widehat{SR} + \frac{\gamma_4-1}{4}\widehat{SR}^2\thinspace}}\right)
$$

— the probability the *true* Sharpe exceeds $SR^\*$, given the estimate, its
sample size and its non-normality. With $SR^\* = 0$ this is a
moment-corrected one-sided test. **Everything is per-period**; mixing a
monthly $\widehat{SR}$ with an annual benchmark is the classic
implementation bug the module docstring warns about.

**Step 3 — deflation.** The Deflated Sharpe Ratio makes one substitution: set
the benchmark to the Sharpe that *pure selection luck* would hand the best of
your $N$ trials,

$$
SR^\* \mkern5mu=\mkern5mu \sqrt{\mathrm{Var}(\widehat{SR}_n)}\mkern5mu\Big[(1-\gamma_E)\thinspace\Phi^{-1}\negthinspace\big(1 - \tfrac1N\big) + \gamma_E\thinspace\Phi^{-1}\negthinspace\big(1 - \tfrac{1}{Ne}\big)\Big],
$$

where $\mathrm{Var}(\widehat{SR}_n)$ is the variance of Sharpe estimates
*across the trials you ran* and $\gamma_E \approx 0.5772$ is the
Euler–Mascheroni constant. This is the extreme-value refinement of §5.4's
crude bound: same $\sqrt{\ln N}$ growth, correct constants. Then

$$
\mathrm{DSR} = \mathrm{PSR}(SR^\*)
$$

— **the probability the true Sharpe is positive, after charging for the search
that found it.**

**Note the two dials.** Deflation grows with the **number** of trials *and*
with their **dispersion**: trying many wildly different things costs more
than trying near-duplicates, exactly as intuition wants.

**Where DSR gets gamed.** $N$ must count *every* configuration examined — each
single-signal portfolio, every model, every tuning study — not just the
finalists. The repo wires this in (`count_single_signal_trials: true`, giving
$N = 10$ or 11) and the backlog's "trial register" makes the count an audit
trail.

## 5.6 Decay as estimation, not disappointment

McLean–Pontiff's finding — anomaly returns fall ~26% after the sample ends
and ~58% after publication — is, mathematically, just a **difference of
segment means**, with all of §1.11's caveats attached. Short segments have
huge standard errors.

Concretely: a 24-month post-sample window gives a Sharpe with a standard
error of roughly $\sqrt{12/24} \approx 0.7$ annualized. So any
`post_sample_retention` above 1 in the output tables is **noise, not a
finding**. §6.5 shows exactly that happening, and §7.7 shows what the same
quantity looks like when averaged over 212 real factors instead of 5
synthetic ones.

**The mature reading of any decay table — this repo's or a published paper's —
is point estimates *with* their error bars, and the placebo row as the
reminder of what pure noise looks like in the same format.**

## 5.7 The placebo that crossed t = 2, and why that is a gift

The synthetic leak table also evaluates `honest_noise`, a feature of pure
random numbers. It measured IC 0.0059 with $t = 2.12$ — **nominally
significant.**

Nothing is broken. §1.7 computed the null standard error of a mean IC here as
≈ 0.0026; a draw 2.3 standard errors out happens by chance a couple of
percent of the time, and this run drew one. Three lessons, cheaply bought:

1. **$t = 2$ is a probability statement, not a certificate.** One in twenty
   pure-noise features crosses it.
2. This is why the repo's placebo *test* uses the planted `sig_dead` under a
   fixed seed with a threshold that has margin — and why no conclusion should
   ever hang on a single borderline t-statistic.
3. **Scale the thought.** Evaluate 50 candidate signals and a couple *will*
   look significant by luck alone. Quantifying that inflation is §5.4–5.5's
   entire subject.

---

# Part 6 — Results I: the synthetic (planted-truth) run

Source: `multisignal-alpha/multisignal-alpha/results/`. Reproducible exactly
with `seed: 42` via `make demo`.

## 6.0 The caveat that governs this entire part

**Read before any number below.** `configs/config.yaml` sets
`data.mode: synthetic`. Every number here was computed on a simulated
500-name panel in which the signals were planted by hand. The generator is

$$
r_{i,t+1} \mkern5mu=\mkern5mu \sum_k \beta_k(t)\thinspace z_{k,i,t} \mkern5mu+\mkern5mu \beta_{\text{int}}\thinspace z_{a,i,t}z_{b,i,t} \mkern5mu+\mkern5mu b_i\thinspace m_{t+1} \mkern5mu+\mkern5mu \varepsilon_{i,t+1},
$$

with Gaussian AR(1) signals $z$, market factor $m \sim N(0.006, 0.045^2)$,
market betas $b_i \sim N(1, 0.3^2)$, idiosyncratic noise
$\varepsilon \sim N(0, 0.08^2)$, and $\beta_k(t)$ stepping down by $\times0.70$
after each signal's "sample end" and to $\times0.45$ after its "publication."

The planted parameters:

| signal | planted $\beta$ | AR persistence $\rho$ | sample end | publication |
|---|---|---|---|---|
| `sig_momentum` | 0.0040 | 0.90 | 2012-12 | 2014-12 |
| `sig_liquidity` | 0.0025 | 0.95 | 2014-12 | 2016-12 |
| `sig_volatility` | 0.0020 | 0.90 | 2013-12 | 2015-12 |
| `sig_value` | 0.0015 | 0.98 | 2015-12 | 2017-12 |
| `sig_quality` | 0.0012 | 0.97 | 2016-12 | 2018-12 |
| **`sig_dead`** | **0.0000** | 0.90 | 2014-12 | 2016-12 |

plus an interaction $\beta_{\text{int}} = 0.0060$ on momentum × value.

**So this run is not evidence of alpha. It is evidence that the measurement
machinery works.** A Sharpe of 5 is a property of the simulator, not of
markets. Read the whole thing as a diagnostic self-test — and the interesting
question is not "how high is the Sharpe" but "does every number land where
the planted truth says it should."

Panel: 299 months × 500 names ≈ 150,000 name-months, 6 predictors.

## 6.1 Leak detection

| feature | IC | IC t-stat |
|---|---|---|
| `leaky_feature` | **0.395** | 178.9 |
| `honest_noise` | 0.006 | 2.12 |

Predicted 0.431 (§5.2), measured 0.395. The leak detector fires, and now you
know what fake looks like. The `honest_noise` row is the gift discussed in
§5.7.

## 6.2 Per-signal evaluation

| signal | IC | IC_t | ICIR | ann_ret gross | SR gross | NW_t gross | ann_ret net | SR net | NW_t net | turnover | months |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `sig_momentum` | 0.037 | 10.95 | 0.807 | 0.102 | 2.541 | 10.59 | 0.089 | 2.234 | 9.32 | 0.509 | 298 |
| `sig_liquidity` | 0.023 | 7.91 | 0.486 | 0.059 | 1.464 | 6.40 | 0.051 | 1.256 | 5.49 | 0.354 | 298 |
| `sig_volatility` | 0.023 | 9.09 | 0.494 | 0.065 | 1.575 | 7.96 | 0.053 | 1.280 | 6.47 | 0.505 | 298 |
| `sig_value` | 0.010 | 4.17 | 0.231 | 0.028 | 0.682 | 3.53 | 0.023 | 0.549 | 2.84 | 0.232 | 298 |
| `sig_quality` | 0.013 | 4.98 | 0.292 | 0.037 | 0.917 | 4.53 | 0.031 | 0.753 | 3.71 | 0.277 | 298 |
| **`sig_dead`** | 0.005 | **1.90** | 0.113 | 0.015 | 0.424 | 1.97 | **0.003** | **0.095** | **0.44** | 0.500 | 298 |

**Four things to read off this table.**

**(a) The rank order is recovered.** Momentum, then volatility and liquidity,
then quality, then value, then dead — against planted betas 0.0040, 0.0020,
0.0025, 0.0012, 0.0015, 0.0000. The small value/quality flip is explained by
their different turnover and persistence, not by an error.

**(b) The placebo correctly dies — but only once costs are charged.** Gross,
`sig_dead` appears to earn 1.5% per year at Sharpe 0.42 with NW-t of 1.97.
That is pure noise dressed as a strategy, and it is *within a whisker of the
conventional significance bar*. Net of its 0.50 turnover it collapses to
Sharpe 0.095 and t = 0.44. **The control worked, and the thing that made it
work was the cost model.** This is the single most instructive row in the
run.

**(c) The staleness law is visible as turnover.** `sig_value` has $\rho = 0.98$
and turnover 0.232; `sig_momentum` has $\rho = 0.90$ and turnover 0.509 —
§1.10's claim that turnover *is* persistence seen from the portfolio side.
The measured staleness profile for momentum (IC at lags 1–4 as a ratio of
fresh IC: **0.93, 0.77, 0.65, 0.52**) versus predicted $\rho^k$ = 0.90, 0.81,
0.73, 0.66, matches at short lags and decays slightly faster at long lags
(the rank transform and the time-varying decay multipliers both shave
long-lag persistence; each ratio carries a standard error of roughly ±0.09).

**(d) Cost drag is exactly §1.9's arithmetic.** Momentum: $0.024\times0.509 = 0.0122$,
and $0.102 - 0.0122 = 0.0898 \approx 0.089$ reported. Check any row you like.

### The flagship prediction: 10.14% versus measured 10.17%

This is the most satisfying check in the project, and it uses only four
numbers from the config.

**Step 1 — the mean of the top slice of a bell curve.**

**Lemma.** For standard normal $Z$: $E[Z\mid Z>a] = \phi(a)/(1-\Phi(a))$.
*Proof:* since $\phi'(z) = -z\phi(z)$, we get
$\int_a^\infty z\phi(z)dz = [-\phi(z)]_a^\infty = \phi(a)$; divide by
$P(Z>a) = 1-\Phi(a)$. ∎

Top quintile: $a = \Phi^{-1}(0.8) = 0.8416$, so
$E[Z\mid\text{top}] = \phi(0.8416)/0.2 = 0.2800/0.2 = 1.400$. By symmetry the
bottom quintile averages $-1.400$. **The long-short spread in signal units is
2.80.**

**Step 2 — everything else in the generator has mean zero in the spread.** The
interaction has mean zero given the sort (the other signal is independent),
market exposure $b_i$ is independent of $z$ so both legs average $b\approx1$
and cancel by dollar neutrality, and noise averages out. So

$$
E[r_p] = \beta\cdot\big(E[z\mid\text{top}] - E[z\mid\text{bot}]\big) = 2.80\thinspace\beta .
$$

**Step 3 — the time-averaged decay multiplier.** The panel spans 299 months
of which about 155 are in-sample, 24 post-sample and 120 post-publication, so

$$
\frac{155(1.0) + 24(0.70) + 120(0.45)}{299} = 0.755 .
$$

**Prediction:** $12\times2.80\times0.0040\times0.755 = \mathbf{0.1014}$ —
10.14% per year gross.

**Measured: 0.1017.** Agreement to 0.3%, with **zero fitted parameters.**

That one check simultaneously exercises the generator, the rank sort, the
weight constructor, the forward-return alignment and the annualization.

**And the IC too.** The per-date correlation under the planted model is
$\beta$ divided by the cross-sectional return spread:
$0.0040/0.0813 = 0.049$ in-sample (Pearson). Two adjustments: Spearman on
near-Gaussian data is slightly smaller (factor ≈ 0.955 for small $\rho$), and
the 0.755 decay multiplier applies. Prediction
$0.049\times0.955\times0.755 = \mathbf{0.0355}$. **Measured 0.0374** — within
the ±0.0026 sampling error of §1.7.

## 6.3 Fama–MacBeth (marginal power)

| variable | mean coef | NW t-stat | periods |
|---|---|---|---|
| const | 0.0066 | 2.54 | 298 |
| `sig_momentum` | 0.0054 | 11.21 | 298 |
| `sig_liquidity` | 0.0033 | 8.31 | 298 |
| `sig_volatility` | 0.0033 | 8.95 | 298 |
| `sig_value` | 0.0016 | 4.51 | 298 |
| `sig_quality` | 0.0019 | 5.03 | 298 |
| **`sig_dead`** | 0.0007 | **1.95** | 298 |

**Three independent checks pass here.**

1. **The intercept is the market.** $c_t$ estimates the average name return
   on date $t$ (signals average zero), so $\bar c$ should equal the planted
   market drift of 0.006. **Measured 0.0066.** ✓
2. **The placebo fails jointly**, not just standalone: $t = 1.95 < 2$ even in
   the multivariate regression. ✓
3. **The coefficient magnitudes are computable.** The regression runs on
   rank-normalized signals $\tilde z \approx$ uniform on $[-1,1]$ (variance
   $\tfrac13$) while returns were generated from the Gaussian $z$. The slope
   of $r$ on $\tilde z$ is
   $\beta\thinspace\mathrm{Cov}(z,\tilde z)/\mathrm{Var}(\tilde z)$, and with
   $\tilde z = 2\Phi(z)-1$ we get $\mathrm{Cov} = 2E[z\Phi(z)] = 1/\sqrt\pi$
   (one line via Stein's lemma). Predicted momentum coefficient:
   $0.004\times\frac{1/\sqrt\pi}{1/3}\times0.755 = \mathbf{0.0051}$.
   **Measured 0.0054** — 6% apart, inside its own standard error. ✓

## 6.4 Model comparison, out of sample

168 OOS months, purged walk-forward, identical inputs and portfolio
constructor.

| model | OOS IC | OOS ICIR | ann_ret gross | SR gross | ann_ret net | **SR net** | NW_t net | turnover |
|---|---|---|---|---|---|---|---|---|
| elasticnet | 0.043 | 0.925 | 0.124 | 2.929 | 0.113 | 2.663 | 8.24 | 0.474 |
| lightgbm | 0.060 | 1.379 | 0.174 | 4.625 | 0.152 | 4.047 | 17.79 | **0.907** |
| icnet | 0.070 | 1.500 | 0.207 | 4.842 | 0.195 | 4.572 | 19.95 | 0.484 |
| **pulse** | **0.072** | **1.553** | 0.215 | 5.357 | 0.203 | **5.052** | 20.41 | 0.509 |

**Reading this table properly means reading three separate stories.**

**Story 1 — the interaction theorem, measured.** Elastic net 0.043 versus
LightGBM 0.060 and IC-Net 0.070. The elastic net harvests the additive part
of the planted truth *perfectly* and the interaction term *not at all*,
because §3.2.3 proves it cannot. The gap between 0.043 and 0.060 is the
planted interaction, made visible. The test suite enforces the sign of this
ordering, and it is robust precisely because the non-additive term was
*planted*.

**Story 2 — combination beats singles.** Every combined model beats every
individual signal (best single was momentum at net Sharpe 2.23). That is the
expected diversification result and the reason models exist at all. Note the
discipline implied: **the model's job is combination, not rescuing an empty
signal** — which is why evaluation precedes modeling in the pipeline order.

**Story 3 — and this is the interesting one — the turnover column.**
LightGBM's one-way turnover of **0.907** means it nearly rebuilds the
portfolio every month, because tree predictions jump discontinuously as
features cross split thresholds. That costs it 0.58 of Sharpe (4.625 gross →
4.047 net). PULSE reaches a *higher* gross Sharpe at turnover 0.509 and gives
up only 0.30. IC-Net likewise sits at 0.484 — predicted in advance by a
smoothness argument: the network is a smooth function of slowly-moving AR(1)
features, so its predictions (and therefore its sorts) evolve gradually,
whereas boosted trees are staircases.

> **PULSE and IC-Net win on *net* partly by being smoother, not only by being
> more accurate. That is a genuine structural finding about the methods, not
> an artifact of the data — and it is the kind of finding that only appears if
> you charge costs.**

**Attribution quality, as a bonus.** IC-Net's first-layer path importances
($\mathrm{imp}_k = \sum_h |W_{1,kh}||w_{2,h}|$) put the planted interaction
pair on top (value 0.32, momentum 0.28) with the placebo last at **0.07** —
compare LightGBM's gain importances, which credited the placebo **0.153**
(§3.2.4). Cleaner attribution is part of the custom model's appeal, not just
a higher IC.

## 6.5 PULSE chosen dynamics and decay recovery

| parameter | chosen | grid | at grid edge |
|---|---|---|---|
| $a$ | 0.99 | 0.97, 0.99, 1.0 | No |
| $q$-scale | 0.002 | 0.002, 0.01, 0.05 | **Yes** |

$a = 0.99$ means efficacy mean-reverts slowly — nearly but not quite
permanent. $q$-scale 0.002 is the smallest option offered, so the filter
concluded coefficients drift slowly. **The data selected "persistent efficacy
with a gentle decay pull," which is the hypothesis of §3.4.1 in one line.**

**An honest flag:** $a$ is interior to its grid, but $q$-scale sits at the
boundary, so there is no way to tell whether the true optimum is smaller
still. Extending `q_scale_grid` downward would settle it. That is listed as
the first next step rather than glossed over.

**Decay table** (retention = segment annualized return as a fraction of
in-sample):

| signal | in-sample SR | post-sample SR | post-pub SR | post-sample retention | **post-pub retention** |
|---|---|---|---|---|---|
| `sig_momentum` | 3.643 | 2.161 | 1.451 | 0.684 | **0.396** |
| `sig_liquidity` | 1.668 | 3.555 | 0.709 | *2.038* | **0.456** |
| `sig_volatility` | 1.639 | 2.067 | 1.357 | *1.105* | **0.683** |
| `sig_value` | 0.877 | 1.146 | 0.110 | 0.885 | **0.115** |
| `sig_quality` | 1.066 | 1.508 | 0.270 | *1.445* | **0.242** |
| `sig_dead` | 0.586 | 1.146 | −0.015 | *1.967* | **−0.028** |

Planted post-publication retention: **0.45**. Measured: 0.40, 0.46, 0.68,
0.12, 0.24 — noisy around the truth, tightest for the strongest signal
(momentum, 0.396), with `sig_dead` at −0.03 of a premium that never existed.

**The `post_sample_retention` entries above 1 are noise, not findings.**
Liquidity shows 2.04 and dead shows 1.97, which would mean they got *twice as
good* after their sample ended. They did not. Those windows are ~24 months,
and §5.6 put the standard error of a 24-month Sharpe at roughly 0.7
annualized. **Do not read anything into that column for any signal.** The
post-publication windows are 120+ months and correspondingly trustworthy.

## 6.6 The trading agent

**At the configured 10 bps:**

| policy | ann_ret net | SR net | NW_t net | turnover | mean $\gamma$ |
|---|---|---|---|---|---|
| agent (learned $\gamma$) | 0.0948 | **2.958** | 9.34 | 0.354 | 0.842 |
| myopic ($\gamma = 1$) | 0.0956 | **2.962** | 9.27 | 0.418 | 1.000 |

**The agent did not win at this cost level.** It learned to trade at 84%
speed, cut turnover by 15%, and landed a net Sharpe 0.004 *below* the myopic
control. That is a dead heat, well inside noise.

**That is not a bug; it is the predicted result.** The falsifiable claim was
always "*at high costs* the agent beats myopic." At 10 bps costs simply are
not high enough for trading speed to matter, and the learned policy *nests*
the myopic one, correctly recovering it when smoothing does not pay. The 10
bps row is the leftmost and least interesting column of the real experiment,
which is the sweep:

| cost (bps) | learned $\gamma$ | agent SR net | myopic SR net | advantage | agent turnover | myopic turnover |
|---|---|---|---|---|---|---|
| 0 | 0.922 | 2.607 | 2.629 | −0.022 | 0.390 | 0.424 |
| 10 | 0.829 | 2.295 | 2.293 | +0.002 | 0.352 | 0.422 |
| 25 | 0.600 | 1.927 | 1.795 | +0.132 | 0.267 | 0.420 |
| 50 | 0.393 | 1.501 | 0.966 | +0.535 | 0.195 | 0.418 |
| 100 | 0.213 | **0.952** | **−0.692** | **+1.644** | 0.128 | 0.418 |

**Three results in one table.**

1. **Gârleanu–Pedersen's core comparative static is confirmed, monotonically.**
   Learned $\gamma$ falls 0.92 → 0.83 → 0.60 → 0.39 → 0.21 as cost rises.
   The agent *discovers* "trade slower when trading is expensive" from data —
   nothing in the code tells it to. Across seeds the values are identical to
   three decimals, because the 7-parameter objective is smooth enough for
   Adam to find the same optimum.
2. **The advantage grows in cost, and at 100 bps it is the difference between
   a strategy and no strategy.** The myopic policy goes *negative* (−0.69)
   while the agent holds +0.95. Myopic turnover meanwhile barely moves
   (0.424 → 0.418), because it has no dial to turn.
3. **The capture-ratio theorem fits the curve.** §4.7 computed
   $\kappa = 0.94$ at the 25 bps point: 94% of the alpha kept for a 36%
   turnover cut.

**The composition result.** The agent's 2.96 is far below PULSE's 5.05, but
they solve different problems: the agent blends six *raw* signals linearly
and therefore cannot see the planted interaction. Composing them —
PULSE's forecast as the agent's aim — gives:

| agent | ann_ret net | SR net | NW_t net | turnover | mean $\gamma$ |
|---|---|---|---|---|---|
| composed (PULSE aim) | 0.1897 | **5.345** | 21.04 | 0.395 | 0.806 |
| baseline (static $\theta$) | 0.0948 | 2.958 | 9.34 | 0.354 | 0.842 |

Net Sharpe nearly doubles, at similar turnover, and it beats PULSE alone
(5.05). And §4.13 *predicted* that headline from a correlation: 20.3% expected
gross versus 19.9% implied.

## 6.7 Factor controls and Deflated Sharpe

| model | alpha (ann) | alpha t-stat | $R^2$ | n |
|---|---|---|---|---|
| elasticnet | 0.114 | 8.27 | 0.032 | 168 |
| lightgbm | 0.154 | 18.67 | 0.061 | 168 |
| icnet | 0.196 | 19.63 | 0.051 | 168 |
| pulse | 0.203 | 20.06 | 0.038 | 168 |

Low $R^2$ with large surviving alpha is the strong outcome (§3.1.2). But note
honestly: PULSE's alpha of 0.203 is *essentially identical to its raw net
return of 0.203*, so the factors explain almost nothing. **In synthetic data
that is unsurprising, because the planted signals are orthogonal to the market
by construction.** This test mostly confirms plumbing here. §7.6 is where it
becomes informative.

**Deflated Sharpe:**

| model | DSR | $SR^\*$ (monthly) | n_trials | PSR vs zero |
|---|---|---|---|---|
| pulse | 1.000 | 0.795 | 11 | 1.000 |

Reading it: 11 configurations were examined and the best was reported. The
hurdle $SR^\*$ of 0.795 monthly is about 2.75 annualized — **so the bar is not
zero, it is "beat a Sharpe of 2.75 that pure luck would have produced."**
PULSE's 5.05 clears it, and the probability rounds to 1.000 (stored value
0.9999999862).

> **Treat a DSR of 1.000 as a realism warning, not a trophy.** Nothing in
> finance is certain to that many decimal places. It reads 1.000 because the
> simulator planted a large, stable, clean signal with no regime shifts, no
> missing data, no crowding and no capacity constraints. The statistic earns
> its keep on real data, where Sharpes of 0.1–0.3 monthly meet trial counts
> in the dozens and deflation routinely moves conclusions from "significant"
> to "maybe."

## 6.8 The prediction scorecard

The habit that organizes the whole project: **predict the number from theory,
*then* measure it.** Everything below was derived from config parameters
before being compared to output.

| Quantity | Predicted | Measured | Agreement |
|---|---|---|---|
| Momentum gross annual return | 10.14% | 10.17% | 0.3% |
| Momentum mean IC | 0.0355 | 0.0374 | within ±0.0026 s.e. |
| Momentum in-sample Sharpe | ≈3.4 | 3.64 | ~7% |
| Fama–MacBeth intercept (market drift) | 0.0060 | 0.0066 | ✓ |
| Fama–MacBeth momentum coefficient | 0.0051 | 0.0054 | 6% |
| Leak-demo IC | 0.431 | 0.395 | within stated approximations |
| Null standard error of mean IC | 0.0026 | `sig_dead` at 1.9 s.e. | ✓ |
| Staleness ratios, lags 1–4 | 0.90, 0.81, 0.73, 0.66 | 0.93, 0.77, 0.65, 0.52 | ✓ short lags, faster at long |
| Planted post-pub retention | 0.45 | 0.40, 0.46, 0.68, 0.12, 0.24 | noisy around truth |
| MSRR closed form vs trained agent | cosine 1.0 | **1.000**, SR 1.204 vs 1.205 | 0.1% |
| Capture ratio at 25 bps | $\kappa = 0.94$ | turnover 0.42 → 0.27 | consistent |
| Composed-agent gross return | 20.3%/yr | 19.9% implied | ✓ |
| $\gamma$ monotone decreasing in cost | yes | 0.92→0.83→0.60→0.39→0.21 | ✓ |
| Aim tilts toward slow signals as cost rises | yes | **0.410 → 0.391: flat** | **✗ refuted** |

**Fourteen confirmations and one clean refutation, which was then explained
(§4.11) and which motivated an implemented extension.** That last row is
worth more than the fourteen above it, because it is the one that proves the
harness can say no.

## 6.9 What this run does and does not support

**Supported.** The pipeline is sound. Leak detection fires. The placebo dies
once costs are charged. Decay is recovered in direction and rough magnitude.
Walk-forward is purged with assertions. Costs are charged consistently.
Multiple testing is penalized with honest trial counting. Combination beats
singles. Nonlinear beats linear where nonlinearity was planted. PULSE's and
IC-Net's smoothness advantages show up in turnover — a real structural finding
about the methods. The agent recovers the MSRR closed form and the GP
comparative static, and its one failed prediction is reported and diagnosed.

**Not supported.** Any of these returns. A Sharpe of 5.05, a 20% alpha and a
DSR of 1.000 are artifacts of a generator written for this repository.

---

# Part 7 — Results II: the real 212-factor run

Source: `multisignal-alpha/multisignal-alpha/results_factor/`, generated by
`python -m src.pipeline --config configs/config_factor.yaml`. **This is real
data.** Everything in this part is evidence rather than a plumbing
certificate.

> **Documentation note.** `docs/USER_GUIDE.md` §8 and §12 still quote an
> earlier *5-factor smoke run* (IC-Net leading, DSR 0.79). The committed
> results — and everything in this part — are from the later **full
> 212-factor universe** run, where the ranking is different. Where the two
> disagree, this part is current.

## 7.1 What the data is, and why it is shaped this way

Firm-level CRSP returns require a paid WRDS licence, which puts the full
Gu–Kelly–Xiu setting out of reach. But the Open Source Asset Pricing project
(Chen & Zimmermann) *freely* publishes each predictor's long-short portfolio
return series. So the design pivots:

> **Each published anomaly becomes the tradable asset.** `ticker` is the
> signal's name; `fwd_ret` is that factor's long-short return next month;
> the features are built from the factor's own history plus its publication
> status.

This is the **factor-timing** setting of Ehsani–Linnainmaa and Gupta–Kelly,
and it is a genuinely interesting problem in its own right: *which of 212
known anomalies will work best next month?*

| Property | Value |
|---|---|
| Dates | 1926-12 to 2024-11, **1176 months** |
| Assets | **212** OSAP factors |
| Rows | 170,758 factor-months |
| Cross-section | mean 145 names/date (3 on the first date, 195 on the last) — **unbalanced** |
| Min names per date | 20 (configured) |
| Out-of-sample months | **1044** (87 folds) |
| Cost | 10 bps per side |
| Forward-return spread | $s_y \approx 0.044$ per month (versus 0.081 for synthetic stocks) |

**The six features:**

| feature | meaning | point-in-time care taken |
|---|---|---|
| `fmom_1m` | last month's realized factor return | *is* the `ret` column; known at $t$, predicts $t{+}1$ |
| `fmom_12_2` | factor return over months $t{-}12$ to $t{-}2$ | skips the most recent month |
| `fmom_12m` | factor return over the last 12 months | includes month $t$ |
| `fvol_12m` | trailing 12-month volatility of factor returns | |
| `post_pub` | 1 strictly *after* the publication date | 0 before — you could not have known the paper was coming |
| `years_since_pub` | years elapsed since publication | clamped at 0 pre-publication, so a negative value cannot leak the future date backwards |

That last pair is the **McLean–Pontiff angle turned into a tradable
feature**: if published factors decay, "has this been published, and for how
long?" should itself predict returns.

**One known limitation, stated in the source:** a factor with no SignalDoc
entry gets `post_pub = 0` and `years_since_pub = 0` for its whole history,
which looks identical to a published factor *before* its publication date. A
missing-date indicator would separate them but would be constant (and thus
useless as a signal) whenever every factor has dates, which is the usual
case. The builder logs how many factors are affected.

## 7.2 Leak checks on real data

| feature | IC | IC_t |
|---|---|---|
| `leaky_feature` | **0.318** | 35.8 |
| `honest_noise` | −0.001 | −0.40 |

The deliberate leak fires at 0.318 against an honest-signal ceiling around
0.12 — the same calibration lesson as §6.1, on real data.

**The leak report also has an instructive non-alarm:**

| signal | predictive IC | contemporaneous corr | note |
|---|---|---|---|
| `fmom_1m` | 0.092 | **1.000** | *near-perfect contemporaneous corr — CHECK ALIGNMENT* |
| `fmom_12m` | 0.116 | 0.304 | |
| `fmom_12_2` | 0.106 | 0.105 | |

`fmom_1m` is *perfectly* correlated with the contemporaneous return because it
**is** the contemporaneous return. That is legitimate — $\text{ret}_t$ is
known at the end of month $t$ and is being used to predict
$\text{ret}_ {t+1}$ — and the leak report **annotates** the case rather than
raising a false alarm. This is exactly the right design: the checker is a
*flag*, and the human reads the flag.

## 7.3 Per-signal evaluation

| signal | IC | IC_t | ICIR | ann_ret gross | SR gross | NW_t gross | ann_ret net | SR net | NW_t net | turnover | months |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `fmom_1m` | 0.092 | 8.84 | 0.256 | 0.101 | 0.464 | 4.29 | 0.068 | 0.310 | 2.84 | **1.406** | 1170 |
| `fmom_12_2` | 0.106 | 9.87 | 0.313 | 0.118 | 0.591 | 5.54 | 0.109 | 0.545 | 5.11 | 0.379 | 1170 |
| **`fmom_12m`** | **0.116** | **10.62** | **0.338** | 0.129 | 0.630 | 5.89 | **0.121** | **0.588** | 5.49 | 0.357 | 1170 |
| `fvol_12m` | 0.039 | 7.71 | 0.248 | 0.048 | 0.583 | 5.18 | 0.043 | 0.527 | 4.68 | 0.192 | 1170 |
| `post_pub` | −0.010 | −1.82 | −0.103 | — | — | — | — | — | — | — | — |
| `years_since_pub` | −0.013 | −2.77 | −0.142 | −0.000 | −0.010 | −0.04 | −0.001 | −0.016 | −0.07 | 0.008 | **311** |

**What to take from this.**

**(a) Factor momentum is real and it survives costs.** Trailing 12-month
factor return predicts next month's factor return with IC 0.116 at
$t = 10.6$, and nets 12.1% a year at Sharpe 0.588 with NW-t of 5.49. That is
consistent with the factor-momentum literature and it is the strongest honest
result in the project. **Note the scale: IC 0.116 and Sharpe 0.59, not IC
0.07 and Sharpe 5.** This is what real data looks like.

**(b) Turnover separates `fmom_1m` from the rest, decisively.** One-month
factor momentum has the *lowest* IC of the three momentum variants (0.092)
*and* a turnover of **1.406** — it reshuffles the whole book nearly one and a
half times a month, because a single month's return is a noisy, fast-moving
ranking variable. Cost drag: $0.024\times1.406 = 0.0337$, taking 10.1% gross
to 6.8% net and Sharpe 0.464 to 0.310. Compare `fmom_12m`: turnover 0.357,
drag 0.0086, Sharpe 0.630 → 0.588. **Same family of signal, one-third the
Sharpe after costs, entirely because of turnover.**

**(c) Two rows have missing portfolio columns, for a structural reason worth
understanding.** `post_pub` is a binary 0/1 variable, so a quintile sort on
it is impossible — `pd.qcut` cannot cut five quantiles from two distinct
values, and the constructor correctly skips those dates rather than inventing
a portfolio. `years_since_pub` has only **311** usable months out of 1176
because it is identically 0 for every factor until publications start
appearing, and a constant cross-section cannot be sorted either. These are
not errors; they are the constructor refusing to fabricate.

**(d) The publication features are weakly *negative*, as theory predicts.**
Both `post_pub` (IC −0.010) and `years_since_pub` (IC −0.013, $t = -2.77$)
point the right way: published factors, and longer-published factors, do
*worse* next month. The effect is small and the portfolio version is
non-tradable, but the sign is McLean–Pontiff's and the `years_since_pub`
t-statistic clears 2 in the right direction.

## 7.4 Fama–MacBeth, and a multicollinearity lesson

| variable | mean coef | NW t-stat | periods |
|---|---|---|---|
| const | 0.0051 | **14.27** | 1146 |
| `fmom_1m` | 0.0037 | 3.14 | 1146 |
| **`fmom_12_2`** | **0.0061** | **2.35** | 1146 |
| **`fmom_12m`** | **−0.0002** | **−0.06** | 1146 |
| `fvol_12m` | 0.0018 | 5.66 | 1146 |
| `post_pub` | 0.0112 | 0.87 | 1146 |
| `years_since_pub` | −0.0116 | −0.89 | 1146 |

**The headline anomaly in this table is `fmom_12m`.** Standalone it is the
*best* signal in the project (IC 0.116, $t = 10.6$). Multivariately it is
*exactly zero* ($-0.0002$, $t = -0.06$). Is something broken?

No — and the explanation is measurable. The pooled correlation between
`fmom_12m` and `fmom_12_2` is **0.968**. They are nearly the same variable:
one is the 12-month factor return, the other is the same 12-month window with
the most recent month removed. When two regressors are 97% collinear, the
regression cannot tell which one carries the information and arbitrarily
loads one; here it loaded `fmom_12_2` (0.0061, $t = 2.35$) and zeroed
`fmom_12m`. The *joint* information is intact; the *attribution* between the
two is not identified.

**The same pattern explains the publication pair.** `post_pub` and
`years_since_pub` correlate at **0.747** and come out with large,
opposite-signed, insignificant coefficients ($+0.0112$ and $-0.0116$, both
$|t| < 1$) — the classic signature of two collinear regressors cancelling
each other.

> **The transferable reading rule: a Fama–MacBeth coefficient near zero does
> not mean "this signal has no power." It means "this signal has no power
> *that its neighbours do not already carry*." Always check the correlation
> matrix before telling that story the wrong way round.**

**And one check passes cleanly.** The intercept $\bar c = 0.0051$ at
$t = 14.27$ estimates the average factor long-short return across the whole
universe — i.e. **the average published anomaly earned about 0.51% a month,
6.1% a year, over this sample, highly significantly.** That is itself a
replication-crisis-relevant number, and it is computed here from scratch.

## 7.5 Model comparison, out of sample — the ranking reverses

1044 OOS months, identical inputs, folds, constructor and costs.

| model | OOS IC | OOS ICIR | ann_ret gross | SR gross | ann_ret net | **SR net** | NW_t net | turnover |
|---|---|---|---|---|---|---|---|---|
| **elasticnet** | **0.129** | **0.392** | 0.120 | 0.695 | **0.102** | **0.592** | 4.89 | 0.742 |
| icnet | 0.124 | 0.381 | 0.114 | 0.654 | 0.092 | 0.526 | 4.56 | 0.927 |
| lightgbm | 0.082 | 0.363 | 0.073 | 0.640 | 0.047 | 0.407 | 3.81 | **1.105** |
| pulse | 0.081 | 0.311 | 0.061 | 0.451 | 0.040 | 0.297 | 2.26 | 0.867 |

**Compare directly against the synthetic ranking:**

| | synthetic net SR | **real net SR** |
|---|---|---|
| elasticnet | 2.66 (**last**) | **0.59 (first)** |
| lightgbm | 4.05 | 0.41 (third) |
| icnet | 4.57 | 0.53 (second) |
| pulse | **5.05 (first)** | **0.30 (last)** |

**The order is almost exactly inverted.** §8.1 is devoted to why, because it
is the most informative single result the project produced.

Two further observations from the table itself:

- **Combination bought almost nothing on real data.** Only two of the four
  models even match the best single signal on OOS IC (elastic net 0.129 and
  IC-Net 0.124 versus `fmom_12m`'s 0.116; LightGBM 0.082 and PULSE 0.081 are
  *below* it), and on net Sharpe only the elastic net (0.592) edges past
  `fmom_12m` alone (0.588) — by 0.004. Contrast the synthetic run, where every
  combined model beat every single signal comfortably. The reason is in the
  turnover column: every model turns over more (0.742–1.105) than the best
  single signal (0.357), so whatever diversification the blend buys is handed
  straight back at the tollbooth.
- **PULSE's turnover advantage evaporated.** On synthetic data PULSE was the
  smooth model (0.509 versus LightGBM's 0.907). Here it is 0.867 versus
  LightGBM's 1.105 — still smoother, but not smooth, and the gross Sharpe it
  starts from is the lowest of the four, so smoothness cannot rescue it.

**Chosen PULSE dynamics on real data:**

| parameter | chosen | grid | at grid edge |
|---|---|---|---|
| $a$ | **0.97** | 0.97, 0.99, 1.0 | **Yes** |
| $q$-scale | 0.002 | 0.002, 0.01, 0.05 | **Yes** |

This is a genuine, if partial, confirmation of PULSE prediction 2: **the
likelihood selected $a < 1$ on real data — the decay prior earned its
place.** On synthetic data it chose $a = 0.99$; on real data it chose the
strongest decay pull on offer. But both parameters now sit at grid
boundaries, so the honest statement is "the data wanted *more* decay and
*less* state noise than the grid allows," and the grid should be widened
before the result is quoted as a point estimate.

## 7.6 Factor-controlled alpha — now the test bites

| model | alpha (ann) | alpha t-stat | $R^2$ | n |
|---|---|---|---|---|
| **elasticnet** | **0.084** | **3.86** | 0.427 | 726 |
| icnet | 0.089 | 3.89 | 0.308 | 726 |
| lightgbm | 0.042 | 2.44 | 0.398 | 726 |
| pulse | 0.040 | 3.04 | 0.351 | 726 |

**This is the table where the synthetic run's prediction came true.** §6.7
warned: *"On real data, expect $R^2$ to be far higher and alpha to shrink
considerably."* It did, on both counts. $R^2$ went from 0.03–0.06 to
**0.31–0.43**, and alpha from 11–20% to **4–9%**.

**But alpha survived.** Every model keeps a statistically significant
factor-controlled alpha, with t-statistics of 2.4 to 3.9 over 726 months
(the sample shrinks from 1044 because the Fama–French factor series starts in
1963). The credible pattern of §3.1.2 — surviving alpha with substantial but
not overwhelming $R^2$ — is what this table shows.

**Interpretation, carefully.** The factors explain 31–43% of the variance of
a factor-timing strategy's returns, which is unsurprising: a book of
long-short anomaly portfolios shares plenty of variation with size, value and
momentum. What is left is 4–9% a year that the standard factor set does not
explain, at conventional significance. That is **promising, not proven** —
and the right next test is the one the project cannot run without a WRDS
licence.

## 7.7 The decay replication — the project's strongest real-data result

`real-data/data/processed/factor_decay.csv` splits **all 212 factors** into
in-sample, post-sample and post-publication segments using their SignalDoc
dates, and computes retention per factor. Aggregating across the full
universe:

| Quantity | Mean | Median |
|---|---|---|
| In-sample Sharpe | 0.683 | 0.543 |
| Post-sample Sharpe | 0.546 | 0.506 |
| Post-publication Sharpe | 0.290 | 0.258 |
| In-sample annual return | 7.37% | 6.15% |
| Post-sample annual return | 4.91% | 4.43% |
| Post-publication annual return | 3.54% | 2.71% |
| **Post-sample retention** | **0.709** | **0.727** |
| **Post-publication retention** | **0.541** | **0.419** |

Also: **99.1%** of the 212 factors had a positive in-sample Sharpe (as they
must — they were published for that reason), and **83.5%** still had a
positive post-publication Sharpe.

**Now compare to the literature.** McLean & Pontiff (2016) report anomaly
returns declining roughly **26% out-of-sample** (retention ≈ 0.74) and
roughly **58% post-publication** (retention ≈ 0.42).

| | Literature | This repo, 212 factors |
|---|---|---|
| Post-sample retention | ≈ 0.74 | **0.709 mean / 0.727 median** |
| Post-publication retention | ≈ 0.42 | 0.541 mean / **0.419 median** |

**That is a close, independent replication of one of the most important
results in empirical asset pricing, computed from scratch in this
repository.** The medians land almost exactly on the published figures. The
post-publication *mean* (0.54) sits above its median (0.42) because retention
ratios have a fat right tail — a handful of factors whose post-publication
returns exceeded their in-sample returns drag the mean up, which is precisely
why the median is the number to quote.

**And the nuance that justifies building on these signals at all:
predictability persists, it does not vanish.** 83.5% of published anomalies
still had a positive Sharpe after publication, and the median one kept 42% of
its original return. The honest framing is therefore not "anomalies are
fake" and not "anomalies are free money," but: *a real, partially-arbitraged
premium remains, and the decay splits into a data-mining component (~27%,
visible before publication) and an arbitrage/learning component (~31%, which
arrives with publication).*

## 7.8 The trading agent on real data — the prediction fails

| policy | ann_ret net | SR net | NW_t net | turnover | mean $\gamma$ |
|---|---|---|---|---|---|
| agent (learned $\gamma$) | 0.0219 | **0.312** | 2.88 | **0.038** | **0.033** |
| myopic ($\gamma = 1$) | 0.0741 | **0.547** | 4.61 | 0.618 | 1.000 |

**The agent lost to its own control, by a wide margin, and the mechanism is
visible in the learned speeds.** Across the 87 folds:

| statistic | learned $\gamma$ |
|---|---|
| fold 0 | 0.957 |
| fold 1 | 0.946 |
| fold 2 | 0.683 |
| folds 3–11 | 0.002 – 0.052 |
| folds 12–86 | ≈ 0.0005 |
| **median** | **0.00049** |
| mean | 0.0331 |

So the mean of 0.033 is a mirage created by the first three folds. **In 83 of
87 folds the agent learned $\gamma < 0.05$, and from fold 12 onward it
effectively froze the book** ($\gamma \approx 0.0005$, one-way turnover 0.038
per month — a near-static portfolio).

Pre-registered prediction 2 from `docs/06_trading_agent.md` §5 was: *"the
agent beats the myopic control net of costs at realistic (≥10 bps) cost
levels, with the margin growing in cost."* **On this panel at 10 bps, that is
refuted.** §8.3 diagnoses it.

**Two experiments are missing from this run and should be flagged as gaps,
not as absences of effect:** `results_factor/` contains **no
`agent_cost_sweep.csv` and no `composed_pulse_agent.csv`**. The cost sweep is
the experiment that actually tests the GP comparative static (the 10 bps row
is its least interesting column — exactly the point made in §6.6), and the
composition was the largest gain in the synthetic run. Neither has been run
on real data. Until they are, the honest statement is "the agent's speed dial
did not help at 10 bps on the factor panel," not "the agent does not work on
real data."

## 7.9 Deflated Sharpe on real data

| model | DSR | $SR^\*$ (monthly) | n_trials | PSR vs zero |
|---|---|---|---|---|
| elasticnet | **0.972** | 0.086 | 10 | 1.000 |

**Now read this one properly, because unlike §6.7 it is informative.**

The elastic net's annualized net Sharpe is 0.592, i.e. **0.171 monthly**. The
luck hurdle from 10 trials is $SR^\* = 0.086$ monthly — about **0.30
annualized**. So the question the statistic answers is: *given 1044 months of
data, the estimate's skew and kurtosis, and the fact that 10 configurations
were examined, what is the probability the true Sharpe exceeds the 0.30 that
the luckiest of 10 random strategies would have shown?*

**Answer: 0.972.** Above the conventional 0.95 "probably real" bar, and
*meaningfully* below 1. The `psr_vs_zero` of 1.000 says the Sharpe is almost
certainly positive; the gap between that and 0.972 is the price of the search.

**Note how much work the long sample is doing.** 1044 months is an unusually
large $T$; §5.5's variance formula scales as $1/(T-1)$, so the same Sharpe
over 168 months would be far less certain. **The DSR of 0.972 is a statement
about a 87-year factor panel, not a promise about the next decade.**

---

# Part 8 — Analysis: what the results actually mean

Parts 6 and 7 reported. This part interprets, and it is where the studying
pays off.

## 8.1 The ranking reversal — the project's most informative result

| | synthetic net SR | real net SR | rank change |
|---|---|---|---|
| elasticnet | 2.66 | **0.59** | 4th → **1st** |
| icnet | 4.57 | 0.53 | 2nd → 2nd |
| lightgbm | 4.05 | 0.41 | 3rd → 3rd |
| pulse | **5.05** | 0.30 | 1st → **4th** |

A naive reading is "the synthetic result was wrong." That is not what
happened, and the correct reading is more useful.

### 8.1.1 Why PULSE won on synthetic data, and why that was always a caveat

**The synthetic world *is* PULSE's model class.** The generator produces
piecewise-constant betas plus one interaction, which is *exactly*
"time-varying linear on an interaction basis." PULSE did not out-predict the
competition so much as *match the data-generating process*. The design doc
said so in advance, before the real run:

> *"The synthetic world is PULSE's model class… Winning here validates that
> the filter, the likelihood selection, and the point-in-time plumbing work;
> it is **not** evidence of real-world alpha… on real data, where efficacy
> dynamics are messier than a two-step staircase, the ranking is an open
> question."*

**That caveat is now cashed.** The open question got an answer, and the answer
was no. **A pre-stated caveat that later turns out to have been load-bearing
is worth more than a win**, because it demonstrates the author knew which
part of the evidence was weak at the time they published it.

### 8.1.2 Why the elastic net won on real data — four converging reasons

**(1) Bias–variance, in the regime where variance dominates.** §3.1.3 proved
$\text{error} = \text{bias}^2 + \text{variance} + \sigma^2$, and argued that
when $\sigma^2$ dwarfs the signal the variance term dominates, so
**deliberately biased, low-variance estimators win**. The factor panel has
monthly return spread $s_y \approx 0.044$ against signal content worth an IC
of ~0.12, i.e. $R^2$ on the order of 1%. This is the textbook regime, and the
textbook answer won. **Theory predicted this before the data did.**

**(2) There is no planted interaction to find.** LightGBM's and IC-Net's
synthetic edge came from the momentum × value product that §3.2.3 proves a
linear model cannot represent. The real factor panel has no such known term —
and whatever genuine nonlinearity exists is buried under far more noise than
168 synthetic months of clean data contained. LightGBM's OOS IC (0.082) is
*below* the best single signal's (0.116), which is what overfitting looks
like from the outside.

**(3) Collinear features punish flexibility.** §7.4 measured
a correlation of **0.968** between `fmom_12m` and `fmom_12_2`, and **0.747**
between `post_pub` and `years_since_pub`. Of six
features, four are two nearly-duplicated pairs. Shrinkage handles that
gracefully — ridge's $1/(1+\lambda)$ divides the shared signal between the
twins instead of fighting over it. Trees and networks will happily split on
either twin at random, which adds variance and (per the next point) turnover.

**(4) Turnover, again.** LightGBM turns over 1.105, IC-Net 0.927, PULSE
0.867, elastic net 0.742. Cost drag at 10 bps:

| model | turnover | drag = $0.024\tau$ | gross SR | net SR | SR lost |
|---|---|---|---|---|---|
| elasticnet | 0.742 | 1.78% | 0.695 | 0.592 | 0.103 |
| icnet | 0.927 | 2.22% | 0.654 | 0.526 | 0.128 |
| lightgbm | 1.105 | 2.65% | 0.640 | 0.407 | 0.233 |
| pulse | 0.867 | 2.08% | 0.451 | 0.297 | 0.154 |

Even on *gross* Sharpe the elastic net already leads (0.695), so costs are
not the whole story — but they widen the gap, most brutally for LightGBM,
which loses 0.233 of 0.640, more than a third of its risk-adjusted return,
to trading.

### 8.1.3 Why PULSE in particular struggled

Three of PULSE's own pre-stated failure modes (§3.4.8) are active on this
panel simultaneously:

- **"Weak signals give the filter mostly noise to track."** PULSE's
  observations are per-date cross-sectional regression coefficients
  $\lambda_t$. With a mean cross-section of 145 names and $P = 21$ expanded
  features, those monthly coefficients are *noisy*, and the filter's gain
  $K = P/(P+r)$ shrinks toward zero when $r$ is large. A filter that trusts
  its observations very little is close to a constant — in which case it is a
  worse-estimated elastic net.
- **"The interaction expansion explodes."** With 6 features, PULSE filters 21
  coefficients. Each extra filtered state is an extra thing to estimate from
  the same noisy monthly observations — and four of the six base features are
  collinear twins, so many of the 21 products are near-duplicates too.
- **"Abrupt regime breaks defeat a smooth filter."** The sample spans
  1926–2024, including 1929, 1987, 2000, 2008 and 2009's momentum crash.
  A random-walk-with-decay state equation cannot represent a crash.

And one structural point worth quantifying: **the panel is unbalanced**,
with 3 names on the first date rising to 195. PULSE's per-date regression
requires at least `max(min_names, P + obs_margin)` = 60 observations to be
well-identified, and **142 of the 1176 dates have fewer than 60 factors —
every month from 1926-12 to 1937-11.** PULSE therefore discards the first
decade outright and then builds its earliest states from the thinnest
cross-sections in the sample. The filter is being fed its least reliable
observations first, which is the worst case for a recursive estimator that
carries state forward. The elastic net, fitting pooled rows with no
per-date identification requirement, is untroubled by the same geometry.

### 8.1.4 The methodological conclusion

> **The synthetic result and the real result are both correct, and they are
> answers to different questions.** Synthetic asks: *given that the world has
> drifting coefficients and an interaction, can these models find them?* Real
> asks: *does the world have enough of that structure, cleanly enough, to pay
> for the variance of looking for it?* The first answer is yes. The second,
> on this panel, is no.
>
> **This is exactly why the project ran both.** A paper with only the
> synthetic result would be a plumbing demo claiming to be research. A paper
> with only the real result would not know whether a null was a true null or a
> broken pipeline. **Together they distinguish "the method cannot work" from
> "the method works and this world does not reward it" — and that distinction
> is the whole value of planted-truth validation.**

## 8.2 Turnover is the hidden variable in every result

If you take one practical lesson from the whole project, take this one.

Turnover appears as the deciding factor in **six separate places**:

1. **The placebo's death.** `sig_dead` had NW-t 1.97 gross — borderline
   significant — and 0.44 net. Its 0.50 turnover was what killed it (§6.2).
2. **LightGBM's synthetic shortfall.** Highest gross Sharpe of the tree/net
   group, third on net, because 0.907 turnover cost it 0.58 of Sharpe
   (§6.4).
3. **PULSE's synthetic win.** Partly *smoothness*, not only accuracy — the
   Kalman prior makes coefficients drift rather than jump (§6.4).
4. **`fmom_1m` on real data.** Lowest IC of the momentum trio *and* turnover
   1.406; nets 6.8% against `fmom_12m`'s 12.1% (§7.3).
5. **The real-data model ranking.** The turnover ordering and the net-Sharpe
   ordering are the same ordering (§8.1.2).
6. **The agent's entire reason for existing.** The capture-ratio theorem
   $\kappa = \gamma/(1-(1-\gamma)\rho)$ is a formula for trading off alpha
   against turnover (§4.7).

**The arithmetic is trivial and worth internalizing:** at 10 bps per side,
annual drag $= 0.024 \times$ one-way turnover. A turnover of 1.0 costs
2.4% a year. At 50 bps it costs 12% a year. **Most backtests that "work"
gross and fail net fail exactly here**, and a project that reports only gross
numbers is not reporting a strategy.

## 8.3 Diagnosing the agent's real-data failure

This deserves careful treatment because it is the project's main negative
result, and a sloppy post-mortem would waste it.

**The facts.** On the 212-factor panel at 10 bps, the agent learned
$\gamma \approx 0.0005$ in 75 of 87 folds (median 0.00049), froze its book
(turnover 0.038), and delivered net Sharpe 0.312 against the myopic control's
0.547. The first three folds learned $\gamma \approx 0.68$–0.96; the collapse
begins at fold 3 and is complete by fold 12.

**What is *not* the explanation.**

- **Not a cost-level problem.** At 10 bps the drag on the myopic book
  ($\tau = 0.618$) is only 1.48% a year. Freezing to save 1.48% while
  sacrificing most of the signal is not a sensible trade, and the
  out-of-sample result confirms it was not.
- **Not the §4.9 sign-flip bug.** That bug produced a *negative* Sharpe from a
  positive-IC forecast. Here the frozen book earns a positive 0.312 — it is
  underperforming, not inverted. And the L1-sphere projection fix is in
  place.

**Three candidate explanations, in order of how much I would bet on them.**

**(a) The agent is overfitting the speed dial in-sample, and the objective is
*right* to freeze in-sample.** This is the strongest hypothesis. Consider
what a frozen dollar-neutral book across ~145 factor long-short portfolios
actually *is*: a static, maximally diversified tilt across almost every
published anomaly. Its return series is a near-constant portfolio of 145
weakly-correlated streams — **very low variance, positive mean**. Sharpe is
mean over standard deviation, so a low-variance frozen book can post an
excellent *in-sample* Sharpe while a chasing book pays variance for
time-variation it cannot reliably predict. The agent maximizes in-sample net
Sharpe, finds that optimum, and it does not generalize. **Note that this is
not a bug in the gradient — it is the objective being optimized correctly on
a sample where the optimum is a bad out-of-sample choice.**

**(b) The policy class is too poor for this panel.** The agent blends raw
signals with *one* $\theta$ and *one* $\gamma$. Four of the six features are
collinear twins (§7.4). With a single shared speed, §4.11's argument applies:
the EWMA attenuates each signal's return and risk contributions nearly
proportionally, so the agent cannot express "track `fmom_12m` slowly but
`fmom_1m` not at all" — which is precisely the policy this panel calls for,
given that `fmom_1m` is the high-turnover signal and `fmom_12m` the
persistent one. **The per-signal-speed agent exists (§4.11) and has not been
run here.** That is the first experiment I would run.

**(c) The expanding window is working against the fit.** Folds 0–2 (120–144
months of 1930s–1940s data, with thin cross-sections) learned fast speeds;
every later fold, with more history, learned to freeze. That is consistent
with (a) — more history means a better-estimated static tilt, which makes
freezing look better in-sample — but it could also mean the objective
landscape flattens as $T$ grows, letting $\gamma$ drift down a nearly-flat
direction in a manner reminiscent of §4.10's identifiability problem.

**What would settle it — four concrete diagnostics.**

1. **Run `agent_cost_sweep.py` on the factor panel.** If learned $\gamma$ is
   ≈0.0005 at *every* cost level including 0 bps, the collapse is not about
   costs at all and hypothesis (a) or (c) is right. If $\gamma$ falls
   monotonically from some higher value, the GP static holds and 10 bps is
   simply the wrong operating point.
2. **Log in-sample versus out-of-sample net Sharpe per fold.** If in-sample
   Sharpe at $\gamma \approx 0$ beats in-sample Sharpe at $\gamma = 1$, the
   objective really is choosing the freeze, confirming (a).
3. **Run the multi-speed agent**, and report **effective exposure**
   $\theta_k\cdot\mathrm{sd}(u_k)$ rather than raw $\theta$ (§4.10).
4. **Add the documented ridge penalty on $\theta$** and check whether the
   collapse persists.

**And what to say in the meantime.** Not "the agent does not work." The
accurate statement is: **on a 212-factor timing panel at 10 bps, with a
single shared trading speed fitted on expanding windows, the learned speed
collapsed to a frozen book and underperformed full rebalancing
out-of-sample. The pre-registered prediction is refuted in that
configuration. The experiments that would distinguish "wrong operating
point" from "wrong policy class" have not been run.**

**One structural detail that looks like good judgment in hindsight.** §4.14
notes the live execution path deliberately does *not* reuse the backtest's
fitted $\gamma$, using an operator-set risk knob instead, on the grounds that
the fitted value "is whatever was Sharpe-optimal in-sample over the fit
window." This result is precisely the failure mode that separation protects
against.

## 8.4 The pre-registered prediction scorecard

Both custom-component design documents wrote down falsifiable real-data
predictions *before* the real run. Grading them is the most honest thing this
guide can do.

### PULSE (`docs/05_pulse_model.md` §5)

| # | Prediction | Verdict | Evidence |
|---|---|---|---|
| 1 | PULSE beats the static elastic net out of sample, margin concentrated post-2003 | **Refuted on this panel** | net SR 0.30 vs 0.59; OOS IC 0.081 vs 0.129. (The prediction named the *firm-level OSAP* panel, which remains unrun — so this is refutation on the available proxy, not on the stated target) |
| 2 | The likelihood selects $a < 1$ on real data | **Confirmed** | $a = 0.97$ chosen, the strongest decay on the grid (though at the grid edge) |
| 2b | Per-signal filtered paths trend down after each signal's publication year, unprompted | **Untested** | `pulse_efficacy_path.csv` is exported but no publication-alignment analysis was run |
| 3 | Efficacy paths mark known events (2009 momentum crash, 2007 quant quake) | **Untested** | chart-inspectable from `pulse_efficacy_paths.png`; not analyzed |
| 4 | If 1–2 fail, conclude monthly Fama–MacBeth observations are too noisy for online efficacy tracking at this frequency | **This is now the live reading** | §8.1.3 gives the mechanism |

**Prediction 4 is the one that matters**, because the design doc pre-committed
to the interpretation of its own failure, and called that interpretation "a
publishable-grade negative result." It should be read as exactly that: *at
monthly frequency, on a 145-name average cross-section with 21 expanded
features, cross-sectional regression coefficients are too noisy to support
online efficacy tracking that pays for itself.*

### The agent (`docs/06_trading_agent.md` §5)

| # | Prediction | Verdict | Evidence |
|---|---|---|---|
| 1 | Learned $\gamma$ decreases in the configured cost level on real data, tracing a smooth curve | **Untested** | no cost sweep was run on the factor panel |
| 2 | The agent beats myopic net of costs at ≥10 bps, margin growing in cost | **Refuted at 10 bps** | 0.312 vs 0.547 net Sharpe |
| 3 | With per-signal speeds, the **effective-exposure** tilt shifts toward persistent signals as cost rises | **Untested** | multi-speed agent not run on real data |
| 4 | Failure of (2) would indicate monthly rebalancing is too coarse for the speed dial to matter at realistic costs | **Plausible but not established** | §8.3 offers two competing explanations; prediction 4's reading is one of them |

**Note prediction 4's wording, which was pre-committed and is now partly
vindicated and partly too generous to itself.** It said failure would mean
*"monthly rebalancing is too coarse for the speed dial to matter."* But the
observed failure is stronger than "did not matter" — the dial actively hurt,
by freezing. §8.3's hypothesis (a) — in-sample overfitting of a single shared
speed — is a different and less flattering diagnosis than the pre-registered
one, and it is the one the evidence currently favors. **Reporting that the
pre-registered explanation may itself be wrong is the correct move.**

### Overall

| Category | Count |
|---|---|
| Synthetic predictions confirmed | 14 |
| Synthetic predictions refuted (then explained, extension built) | 1 |
| Real-data predictions confirmed | 1 |
| Real-data predictions refuted | 2 |
| Real-data predictions untested (experiment not run) | 4 |

**The four "untested" rows are the project's clearest actionable gap**, and
three of the four are a single afternoon's compute: run the cost sweep, the
composition, and the multi-speed agent on the factor panel.

## 8.5 What is genuinely established, and what is not

### Established by this project's own evidence

1. **The harness measures what it claims to measure.** Fourteen quantitative
   predictions from first principles matched measured output, several to
   within 1% with zero fitted parameters (§6.8). This is a stronger form of
   validation than passing unit tests.
2. **Leakage is detected, and the detector is calibrated.** A planted leak
   reads 0.395/0.318 against an honest ceiling of 0.04/0.12, and the
   prediction of *what the leak would read* was computed in advance (§5.2).
3. **The placebo dies, and costs are what kill it** (§6.2).
4. **Published anomalies decay, at close to the published rate.** Independent
   replication across 212 factors: post-sample retention 0.71, post-publication
   median retention 0.42, against literature values of ~0.74 and ~0.42
   (§7.7). **This is the project's strongest real-data result, and it is a
   replication rather than a discovery — which is exactly what should be
   trusted most.**
5. **Factor momentum carries real, cost-surviving cross-sectional
   information.** `fmom_12m`: IC 0.116, $t = 10.6$, net Sharpe 0.588 over
   1170 months (§7.3).
6. **A combined model retains significant factor-controlled alpha on real
   data**, 4–9% a year at $t = 2.4$–3.9, with $R^2$ of 0.31–0.43 (§7.6).
7. **In low signal-to-noise cross-sectional prediction, shrinkage beats
   flexibility.** Demonstrated, not asserted, by an identical-input
   comparison in which the simple benchmark won on both gross and net
   (§8.1.2) — and predicted in advance by the bias–variance theorem.
8. **Turnover is a first-order determinant of realized performance**, not a
   footnote (§8.2).
9. **Two negative results, correctly identified and diagnosed**: the GP aim
   tilt does not appear with a single shared speed (§4.11), and the agent's
   speed dial hurt on the real factor panel (§8.3).
10. **Optimizers exploit symmetries you forgot you created** — with a theorem
    that predicted the bug and a projection that fixed it (§4.9).

### Not established

1. **Any claim of deployable alpha.** The best real-data net Sharpe is 0.59
    on a factor-timing book, before any capacity, borrow-cost, shorting-
    feasibility or slippage realism. OSAP long-short portfolio returns are
    *research series*, not tradable instruments.
2. **The Gu–Kelly–Xiu result itself.** That ML beats linear models on
    *firm-level* characteristics is the paper's claim; this project tested
    something different (factor timing) and found the reverse. **Those are
    not in conflict** — different cross-section, different features,
    different $n$ — and the firm-level test remains unrun.
3. **That PULSE does not work.** It was beaten on one panel that activates
    three of its own documented failure modes at once. The firm-level
    setting, a screened feature shortlist, and a wider hyperparameter grid are
    all untried.
4. **That the agent does not work.** Three of its four real-data predictions
    were never tested (§8.4).
5. **Anything about the post-2003 era specifically.** PULSE prediction 1
    located its expected margin there; no era-split analysis was run.
6. **Out-of-sample in the strict sense.** The 212-factor panel is 1926–2024
    and has been examined. The deflation accounting ($N = 10$) covers the
    configurations the pipeline ran; it does not and cannot cover the
    research decisions that shaped the pipeline itself.

## 8.6 Threats to validity, ranked

**1. The factor panel's assets are not tradable instruments.** Each "name" is
an OSAP long-short portfolio — itself a book of hundreds of stocks with its
own internal turnover, borrow costs and capacity limits, none of which the
10 bps per side charge represents. A strategy that rebalances *across* 145
such books is charging a retail commission on an institutional operation.
**This is the single largest gap between the reported numbers and reality.**

**2. Survivorship and selection in the factor universe itself.** OSAP
contains anomalies that were *published*, which means they passed a
significance filter in their original papers. 99.1% have positive in-sample
Sharpes by construction (§7.7). Any strategy trading this universe inherits
that selection, and no deflation inside this project can correct for a
selection event that happened in the literature.

**3. The deflation count is a lower bound.** $N = 10$ counts the pipeline's
configurations. It does not count the choice of six features, the choice to
treat factors as assets, the hyperparameter grids, or the development history.
The repo acknowledges this and proposes a "trial register"; until that exists,
read DSR 0.972 as optimistic.

**4. Simplifications in the cost model.** No drift adjustment between
rebalances, no size dependence in the pipeline runs (square-root impact is
implemented but not enabled), no bid-ask asymmetry, no short-borrow cost. All
push realized performance *down*.

**5. The missing `osap` mode.** The project's own hierarchy of evidential
weight puts firm-level OSAP + CRSP at the top, and it is gated on a licence.
Every conclusion here is from the middle tier.

**6. Unbalanced-panel artifacts.** The cross-section grows from 3 to 195
names. Early folds train on data that barely has a cross-section. The masking
machinery handles this correctly and the tests pin it down — but "handled
correctly" is not the same as "informative," and early-fold parameter
estimates should probably be discarded rather than used.

**7. Multiple-hypothesis exposure in the *analysis*.** Several readings in
this Part (notably §8.3's three hypotheses) are post-hoc. They are labelled as
hypotheses with named diagnostics precisely so they are not mistaken for
findings.

---

# Part 9 — Conclusions

## 9.1 What the project set out to do, and whether it did it

**The stated goal was not to find alpha.** It was: build a measurement
harness, prove it on data where the truth is planted, point it at real data
with every correction applied, and report honestly what survives.

**Judged against that goal, the project succeeded**, and the evidence for
"succeeded" is unusual in its form: fourteen quantities were predicted from
first principles and then measured, several matching to within 1% with no
fitted parameters. That is a stronger claim than "the tests pass," because it
checks the mathematics and the code *against each other* rather than against
themselves.

**Judged against the goal of finding real, exploitable predictability, the
result is a qualified, modest yes with a large asterisk.** Factor momentum
predicts next-month factor returns at IC 0.116 and net Sharpe 0.588 over
1170 months; a combined model keeps 4–9% annual factor-controlled alpha at
conventional significance; the Deflated Sharpe of the best model is 0.972.
And the asterisk is §8.6's threat list, headed by the fact that the traded
assets are research portfolios rather than instruments.

## 9.2 The five conclusions, with their confidence levels

**1. Anomaly decay is real, directional, and quantitatively close to the
published estimates.** *(High confidence — it is an independent replication
across the full 212-factor universe.)* Post-sample retention 0.71 mean / 0.73
median; post-publication retention 0.54 mean / **0.42 median**, against
literature values of ~0.74 and ~0.42. And the nuance matters as much as the
headline: **83.5% of published anomalies still had a positive Sharpe after
publication.** Predictability is partially arbitraged, not destroyed.

**2. In the low signal-to-noise regime of cross-sectional return prediction,
shrinkage beats flexibility.** *(High confidence on this panel; moderate as a
general claim.)* The elastic net led on OOS IC, gross Sharpe, net Sharpe and
alpha, against boosted trees, a custom neural network and a Kalman-filtered
time-varying model, on identical inputs and folds, over 1044 out-of-sample
months. The bias–variance theorem (§3.1.3) predicted this *in advance*, from
the observation that $R^2$ here is on the order of 1%. **The flexible models
were not beaten by bad luck; they were beaten by variance.**

**3. Costs are not a correction applied at the end — they are a structural
determinant of which model wins.** *(High confidence.)* Turnover decided the
placebo's fate, LightGBM's ranking in both worlds, PULSE's synthetic win,
`fmom_1m`'s real-data collapse, and the real-data ordering. At 10 bps per
side the annual tax is $0.024\times$ turnover, and at turnover above 1 that is
more than most honest gross edges.

**4. Planted-truth validation earns its keep by distinguishing two kinds of
null result.** *(High confidence — this is the project's main methodological
contribution.)* When PULSE lost on real data, the synthetic run had already
established that the filter tracks a known efficacy path, detects
publication step-downs, holds the placebo at zero, and is point-in-time by
proof. So the real-data loss could be attributed to *the world* rather than
to *the code* — and §8.1.3 could then name which three of PULSE's own
documented failure modes were active. **Without the planted-truth run, the
same number would have been uninterpretable.**

**5. Both custom components produced honest negative results, and the
negatives are more informative than the positives.** *(High confidence.)*
The GP aim-tilt prediction failed with a single shared trading speed, and the
*reason* was derived (§4.11) and used to motivate an implemented extension.
The agent's speed dial actively hurt on the real factor panel, and the most
likely diagnosis (§8.3) is less flattering than the pre-registered one.
**Reporting that your own pre-registered explanation of your own failure may
itself be wrong is the behaviour that makes the positive results credible.**

## 9.3 The methodological lessons, in portable form

Strip away the finance and these transfer to any empirical modeling work:

1. **Validate the ruler before measuring anything.** Build a world where you
   know the answer; check that your pipeline recovers it; *then* look at real
   data.
2. **Predict the number, then measure it.** A claim you cannot turn into a
   falsifiable number is not yet research. A number you did not deflate for
   search, protect from leakage and equip with honest error bars is not yet a
   result.
3. **Plant a placebo and a leak, deliberately.** You need to know what broken
   looks like *in your own numbers*, not in a textbook.
4. **Choose the loss function to be the thing you are judged on.** §3.3.1
   proves exactly what MSE wastes when you are judged on ordering. This is
   the most portable single idea in the repository.
5. **Write down your predictions before you run the experiment, including
   what failure would mean.** Then grade yourself, in public.
6. **Every invariance you build in is a direction your optimizer can
   wander.** Quotient it out or expect it to be explored (§4.9).
7. **Reach for RL when the environment is unknown or non-differentiable;
   reach for calculus when you own the simulator** (§4.2).
8. **A near-zero multivariate coefficient means "adds nothing beyond its
   neighbours," not "has no power."** Check the correlation matrix first
   (§7.4).
9. **$t = 2$ is a probability statement, not a certificate** (§5.7).
10. **Report the gaps as gaps.** "The cost sweep was not run on real data" is
    a sentence that costs nothing to write and makes everything else more
    believable.

## 9.4 What to do next, in priority order

**Tier 1 — cheap, and closes stated gaps (an afternoon of compute each).**

1. **Run `agent_cost_sweep.py` on the factor panel.** This is the single
   highest-value missing experiment: it is the test of the agent's
   prediction 1, and §8.3's diagnostic 1 distinguishes "wrong operating
   point" from "wrong policy class."
2. **Run `compose_pulse_agent.py` on the factor panel.** The composition was
   the largest gain in the synthetic run and has never been tried on real
   data.
3. **Run the multi-speed agent on the factor panel, reporting effective
   exposure** $\theta_k\cdot\mathrm{sd}(u_k)$ (agent prediction 3).
4. **Widen both PULSE grids.** $a$ and $q$-scale both selected grid
   boundaries on real data (and $q$-scale did on synthetic too), so neither
   chosen value is a point estimate yet.
5. **Log in-sample versus out-of-sample Sharpe per agent fold**, to confirm
   or kill §8.3's hypothesis (a).

**Tier 2 — moderate effort, materially strengthens the claims.**

6. **Add the documented ridge penalty on the agent's $\theta$.**
7. **Era-split the model comparison** (pre/post-2003), which is where PULSE
   prediction 1 located its expected margin.
8. **Align PULSE's filtered efficacy paths to publication dates** and check
   whether they trend down unprompted (PULSE prediction 2b) and whether they
   mark 2007 and 2009 (prediction 3). Both are already exported; only the
   analysis is missing.
9. **Build the trial register**, so the deflation count becomes an audit
   trail rather than a configuration flag.
10. **Update `docs/USER_GUIDE.md` §8 and §12**, which still quote the
    superseded 5-factor run.

**Tier 3 — the real frontier.**

11. **Obtain WRDS/CRSP access and run `data.mode: osap`.** This is the test
    the whole project is pointed at, the one where the Gu–Kelly–Xiu
    comparison is actually on its home ground, and the one where PULSE's and
    the agent's refuted predictions deserve a second, fair hearing with a
    screened feature shortlist.
12. **Move to daily frequency**, where the agent's speed dial has far more
    room to matter.
13. **Model costs properly for the factor panel** — or stop treating OSAP
    long-short series as tradable, and say so loudly in every table caption.

## 9.5 The closing thought

The project's own index page states its organizing habit: *every chapter ends
by predicting a number and checking it.* That habit is what makes the
difference between this and a backtest.

> **A claim you cannot turn into a falsifiable number is not yet research.
> A number you did not deflate for search, protect from leakage, and equip
> with honest error bars is not yet a result. And a result you did not try to
> break — with a placebo, a planted leak, a myopic control, and a written-down
> prediction you might have to grade as wrong — is not yet evidence.**

By that standard: the harness is evidence, the decay replication is evidence,
the shrinkage-beats-flexibility finding is evidence, and the two negative
results are the best evidence of all — because they are the ones that prove
the machine can say no.

---

# Appendix A — Master formula sheet

Everything in one place, grouped by what it is for. Section references point
back into this guide.

## A.1 Descriptive statistics

| Name | Formula | §|
|---|---|---|
| Return | $r = P_t/P_{t-1} - 1$ | 1.1 |
| Variance | $\mathrm{Var}(X) = E[(X - E[X])^2]$ | 1.2 |
| Covariance | $\mathrm{Cov}(X,Y) = E[\tilde X\tilde Y]$ | 1.2 |
| Correlation | $\rho = \mathrm{Cov}(X,Y)/(\sigma_X\sigma_Y)$, and $|\rho|\le1$ | 1.3 |
| Correlation as cosine | $\rho(p,y) = \langle\tilde p,\tilde y\rangle/(\Vert\tilde p\Vert\Vert\tilde y\Vert)$ | 1.4 |
| z-score | $(x - \bar x)/\sigma_x$ | 1.5 |
| Spearman shortcut | $\rho_s = 1 - 6\sum d_i^2/(n(n^2-1))$ | 1.5 |
| Pearson → Spearman (near-Gaussian) | $\rho_s \approx \tfrac{6}{\pi}\arcsin(\rho/2) \approx 0.955\rho$ for small $\rho$ | 5.2 |

## A.2 Performance and signal quality

| Name | Formula | §|
|---|---|---|
| Sharpe (monthly) | $\mathrm{SR} = \mu/\sigma$ | 1.6 |
| Annualization | $\mathrm{SR}_ {\text{ann}} = \sqrt{12}\thinspace\mu/\sigma$; $\mu_ {\text{ann}} = 12\mu$; $\sigma_ {\text{ann}} = \sqrt{12}\sigma$ | 1.6 |
| Information Coefficient | $\mathrm{IC}_t = \mathrm{Spearman}_i(z_{i,t},\thinspace r_{i,t+1})$ | 1.7 |
| ICIR | $\overline{\mathrm{IC}}/\sigma_{\mathrm{IC}}$ | 1.7 |
| Null s.e. of a per-date IC | $\approx 1/\sqrt{n-1}$ | 1.7 |
| Null s.e. of a mean IC | $\approx 1/\sqrt{(n-1)T}$ | 1.7 |

## A.3 Portfolios and costs

| Name | Formula | §|
|---|---|---|
| Portfolio return | $r_p = \langle w, y\rangle$ | 1.8 |
| Quintile long-short weights | $w_i = +1/n_{\text{top}}$, $-1/n_{\text{bot}}$, else 0 | 1.8 |
| Dollar neutrality | $\sum_i w_i = 0 \Rightarrow \langle w, y + c\mathbf 1\rangle = \langle w,y\rangle$ | 1.8 |
| Traded notional | $\text{traded}_t = \sum_i|w_{t,i} - w_{t-1,i}|$ | 1.9 |
| One-way turnover | $\tau_t = \text{traded}_t/2$ | 1.9 |
| Net return | $r^{\text{net}}_t = r^{\text{gross}}_t - \text{traded}_t\cdot\text{bps}/10000$ | 1.9 |
| **Annual cost drag at 10 bps** | $0.024\times\tau$ | 1.9 |
| Mean of the top slice | $E[Z\mid Z>a] = \phi(a)/(1-\Phi(a))$; top quintile = 1.400 | 6.2 |
| Long-short spread in signal units | $2\times1.400 = 2.80$ | 6.2 |
| Expected LS monthly return (planted) | $E[r_p] = 2.80\thinspace\beta$ | 6.2 |
| Proportional book return | $w^\top y \approx 2\sqrt{\pi/2}\thinspace s_y\thinspace\rho_t \approx 2.507\thinspace s_y\thinspace\rho_t$ | 4.13 |

## A.4 Inference

| Name | Formula | §|
|---|---|---|
| S.e. of a mean (i.i.d.) | $\sigma/\sqrt T$ | 1.11 |
| Variance of a mean with autocorrelation | $\frac{\sigma^2}{T}\big[1 + 2\sum_k (1-k/T)\rho_k\big]$ | 1.11 |
| Newey–West | $\frac{1}{T}\big[\hat\gamma_0 + 2\sum_{k=1}^{L}(1-\tfrac{k}{L+1})\hat\gamma_k\big]$, $L=6$ here | 1.11 |
| t-statistic | $t = \bar x/\mathrm{se}(\bar x)$ | 1.11 |
| Fama–MacBeth pass 1 | $r_{i,t+1} = c_t + \sum_k\lambda_{k,t}z_{k,i,t} + e_{i,t}$ | 1.12 |
| Fama–MacBeth pass 2 | report $\bar\lambda_k$ with NW t-stat | 1.12 |
| Coefficient sampling variance | $\mathrm{Var}(\hat\lambda) = s^2(Z^\top Z)^{-1}$ (diagonal) | 1.12 |
| Factor-controlled alpha | $r^{\text{strat}}_t = \alpha + \beta^\top f_t + e_t$, NW errors | 3.1.2 |

## A.5 Linear models

| Name | Formula | §|
|---|---|---|
| Normal equations | $\hat\beta = (X^\top X)^{-1}X^\top y$ | 3.1.1 |
| One-feature slope | $\hat\beta_1 = \mathrm{Cov}(x,y)/\mathrm{Var}(x)$ | 3.1.1 |
| Bias–variance | $E[(y-\hat f)^2] = \text{bias}^2 + \text{variance} + \sigma^2$ | 3.1.3 |
| Elastic net objective | $\Vert y - X\beta\Vert^2 + \lambda_2\Vert\beta\Vert^2 + \lambda_1\Vert\beta\Vert_1$ | 3.1.4 |
| Ridge (orthonormal) | $\hat\beta^{\text{ridge}} = \hat\beta/(1+\lambda)$ | 3.1.4 |
| Lasso (orthonormal) | $\mathrm{sign}(\hat\beta_j)\max(|\hat\beta_j| - \lambda/2,\thinspace 0)$ | 3.1.4 |

## A.6 Trees and boosting

| Name | Formula | §|
|---|---|---|
| Boosted model | $F_M(x) = \sum_{m=1}^M \eta\thinspace h_m(x)$ | 3.2.2 |
| Functional gradient (squared loss) | $\partial L/\partial F(x_i) = -(y_i - F(x_i))$ — the residual | 3.2.2 |
| Interaction theorem | no $f,g$ satisfy $z_1z_2 = f(z_1)+g(z_2)$ | 3.2.3 |
| Interaction fingerprint | $\partial^2 F/\partial z_1\partial z_2 \ne 0$ | 3.2.3 |

## A.7 IC-Net

| Name | Formula | §|
|---|---|---|
| **MSE decomposition** | $\tfrac1n\sum(p_i-y_i)^2 = (\bar p-\bar y)^2 + (s_p-s_y)^2 + 2s_ps_y(1-\rho)$ | 3.3.1 |
| Objective | $J(W) = \tfrac1T\sum_t\rho_t(f(X_t;W), y_t) - \lambda\Vert W\Vert^2$ | 3.3.2 |
| **Gradient of a correlation** | $\nabla_p\rho = \dfrac{\tilde y}{\Vert\tilde p\Vert\Vert\tilde y\Vert} - \rho\dfrac{\tilde p}{\Vert\tilde p\Vert^2}$ | 3.3.3 |
| Network | $Z = XW_1+b_1$; $H = \tanh Z$; $p = Hw_2+b_2$ | 3.3.4 |
| tanh derivative | $\tanh'(z) = 1-\tanh^2 z$ | 3.3.4 |
| Adam | $m\leftarrow\beta_1 m + (1-\beta_1)g$; $v\leftarrow\beta_2 v + (1-\beta_2)g^2$; step $\eta\hat m/(\sqrt{\hat v}+\epsilon)$ | 3.3.5 |
| Adam bias correction | $\hat m = m/(1-\beta_1^k)$, $\hat v = v/(1-\beta_2^k)$ | 3.3.5 |
| Path importance | $\mathrm{imp}_k = \sum_h|W_{1,kh}||w_{2,h}|$ | 6.4 |

## A.8 PULSE / Kalman

| Name | Formula | §|
|---|---|---|
| State equation | $\beta_t = a\beta_{t-1} + w_t$, $w\sim N(0,q)$ | 3.4.2 |
| Observation equation | $\lambda_t = \beta_t + v_t$, $v\sim N(0,r_t)$ | 3.4.2 |
| **Gaussian update lemma** | posterior $N(m + K(\lambda-m),\thinspace(1-K)P)$, $K = P/(P+r)$ | 3.4.3 |
| Predict step | $N(a\thinspace m_{t-1},\thinspace a^2P_{t-1}+q)$ | 3.4.4 |
| Update step | $m_t = a m_{t-1} + K_t(\lambda_t - a m_{t-1})$, $K_t = \frac{a^2P_{t-1}+q}{a^2P_{t-1}+q+r_t}$ | 3.4.4 |
| Innovation log-likelihood | $-\tfrac12\sum_t[\log(2\pi S_t) + (\lambda_t - am_{t-1})^2/S_t]$ | 3.4.4 |
| Steady-state variance | $P^\* = \tfrac{-q+\sqrt{q^2+4qr}}{2}$ | 3.4.5 |
| Steady-state EWMA | $m_t = (1-K^\*)m_{t-1} + K^\*\lambda_t$ | 3.4.5 |
| EWMA half-life | $\ln 2/|\ln(1-K^\*)|$ | 3.4.5 |
| Forecast | $p_{i,t} = \sum_k a^h\thinspace m_{k,t-1}\thinspace z_{k,i,t}$, $h$ = calendar months past training end | 3.4.7 |

## A.9 The agent

| Name | Formula | §|
|---|---|---|
| Score | $s_{t,i} = \sum_k\theta_k z_{k,i,t}$ | 4.3 |
| Aim | $A_t = 2\tilde s_t/\Vert\tilde s_t\Vert_1$ | 4.3 |
| Partial adjustment | $w_t = (1-\gamma)w_{t-1} + \gamma A_t$, $\gamma = \sigma(g)$ | 4.3 |
| Net return | $\langle w_t,y_t\rangle - \tfrac{c}{10^4}\sum_i|w_{t,i}-w_{t-1,i}|$ | 4.3 |
| **Policy as EWMA** | $w_t = \gamma\sum_{j=0}^{t-1}(1-\gamma)^jA_{t-j}$ | 4.3 |
| Policy half-life | $\ln 2/|\ln(1-\gamma)|$ | 4.3 |
| **Gradient of Sharpe** | $\dfrac{\partial J}{\partial r_t} = \dfrac{1}{Ts} - \dfrac{m(r_t-m)}{Ts^3}$ | 4.4 |
| Forward sensitivity | $D_t = (1-\gamma)D_{t-1} + \gamma\frac{\partial A_t}{\partial\theta} + (A_t-w_{t-1})\gamma(1-\gamma)$ | 4.5 |
| Softabs | $\sqrt{x^2+\varepsilon}$, derivative $x/\sqrt{x^2+\varepsilon}$ | 4.5 |
| Factor collapse | $r_p(t) = \theta^\top f_t$, $f_{k,t} = \sum_i\tilde z_{k,i,t}y_{i,t}$ | 4.6.1 |
| **MSRR** | $\theta^\* \propto (\Sigma+\lambda I)^{-1}\mu$; $S(\theta^\*) = \sqrt{\mu^\top\Sigma^{-1}\mu}$ | 4.6.2 |
| Sherman–Morrison | $(A+uv^\top)^{-1} = A^{-1} - \frac{A^{-1}uv^\top A^{-1}}{1+v^\top A^{-1}u}$ | 4.6.3 |
| **Capture ratio** | $\kappa(\gamma,\rho) = \dfrac{\gamma}{1-(1-\gamma)\rho}$ | 4.7 |
| Square-root impact | cost $\propto|\Delta w|^{3/2}$, smoothed as $(x^2+\varepsilon)^{3/4}$ | 4.8 |
| Null-direction theorem | $J(c\theta) = J(\theta) \Rightarrow \theta^\top\nabla_\theta J = 0$ | 4.9.1 |
| EMA stationary variance | $\mathrm{Var}(u) = \gamma/(2-\gamma)$ | 4.10 |
| Effective exposure | $\text{eff}_k = \theta_k\cdot\mathrm{sd}(u_k)$ | 4.10 |

## A.10 Validity and multiple testing

| Name | Formula | §|
|---|---|---|
| AR(1) signal | $z_t = \rho z_{t-1} + \sqrt{1-\rho^2}\thinspace\epsilon_t$ | 1.10 |
| **Staleness law** | $\mathrm{IC}(k) \approx \rho^k\thinspace\mathrm{IC}(0)$ | 1.10 |
| **Purging theorem** | disjoint iff $t^\* + h < \tau$, i.e. purge $\ge h$ | 2.3 |
| Leak-demo IC | $\rho(ay+e, y) = a/\sqrt{a^2+1}$ | 5.2 |
| **Best of N bound** | $E[\max_i X_i] \le \sqrt{2\ln N}$ | 5.4 |
| Sharpe estimate variance | $\frac{1 - \gamma_3 SR + \frac{\gamma_4-1}{4}SR^2}{T-1}$ | 5.5 |
| **PSR** | $\Phi\big(\frac{(\widehat{SR}-SR^\*)\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{SR}+\frac{\gamma_4-1}{4}\widehat{SR}^2}}\big)$ | 5.5 |
| **DSR hurdle** | $SR^\* = \sqrt{\mathrm{Var}(\widehat{SR}_n)}[(1-\gamma_E)\Phi^{-1}(1-\tfrac1N) + \gamma_E\Phi^{-1}(1-\tfrac{1}{Ne})]$ | 5.5 |
| DSR | $\mathrm{PSR}(SR^\*)$ | 5.5 |
| Retention | segment annualized return / in-sample annualized return | 5.6 |

---

# Appendix B — Glossary

**Alpha** — the intercept of a regression of strategy returns on known factor
returns; the average return *not* explained by rentable risk exposures.

**AR(1)** — a process where today equals $\rho$ times yesterday plus fresh
noise. Governs how persistent a signal is, and therefore how much a
portfolio built on it trades.

**Backtest** — simulating a strategy on historical data. Lies in two ways:
lookahead and selection.

**Basis point (bps)** — one hundredth of a percent. 10 bps = 0.10%.

**Characteristic-managed portfolio** — the portfolio that holds each name in
proportion to its value of one signal. Its return series is the "factor
return" $f_{k,t}$ that makes MSRR tractable.

**Cross-section** — all the names alive on one date. "Cross-sectional"
prediction means ranking names against each other, not forecasting the
market.

**Deflated Sharpe Ratio (DSR)** — the probability a strategy's true Sharpe is
positive, after charging for the number and dispersion of configurations
tried.

**Dollar-neutral** — weights summing to zero, so a uniform move in all
returns contributes nothing.

**Effective exposure** — $\theta_k\cdot\mathrm{sd}(u_k)$; a signal's real
influence in a multi-speed agent, which is zero if its EMA is frozen even
when its raw weight is not.

**Elastic net** — linear regression with both L1 and L2 penalties.

**Embargo** — extra periods dropped from the end of a training window, beyond
the purge, as insurance against dependence past the label horizon.

**Fama–MacBeth** — one cross-sectional regression per date, then the
time-series mean of the slopes with a Newey–West t-statistic.

**Forward return (`fwd_ret`)** — the return over $(t, t{+}1]$; the label.

**Gain importance** — a tree model's accounting of how much squared-error
reduction each feature's splits achieved.

**IC (Information Coefficient)** — the per-date rank correlation between a
signal and forward returns, averaged over dates. Real values: 0.02–0.12.

**ICIR** — mean IC divided by its standard deviation across dates; stability
rather than strength.

**Information set $\mathcal F_t$** — everything knowable at the end of month
$t$. The bookkeeping device that makes "leakage" precise.

**Innovation** — the one-step prediction error $\lambda_t - a m_{t-1}$ in a
Kalman filter. Its likelihood is the filter's own model-selection criterion.

**Kalman filter** — the optimal tracker of a drifting Gaussian state:
predict, then update by "prior + gain × surprise."

**Kalman gain** $K = P/(P+r)$ — how much to trust a new observation versus
your prior.

**Leakage** — letting information from outside $\mathcal F_t$ into a feature
or a fitting decision dated $t$.

**LightGBM** — a fast gradient-boosted-tree implementation.

**Masking** — excluding untradable names (NaN forward return) from the
demeaning, normalization, returns and gradients, while still charging the
cost of closing any open position.

**McLean–Pontiff** — the 2016 paper establishing ~26% post-sample and ~58%
post-publication anomaly decay.

**MSRR (Maximum Sharpe Ratio Regression)** — the closed-form max-Sharpe blend
$\Sigma^{-1}\mu$ over characteristic-managed portfolio returns; equivalently
the coefficient of regressing the constant 1 on factor returns.

**Myopic policy** — $\gamma = 1$: rebalance fully to the aim every period.
The agent's control.

**Newey–West** — a standard-error estimator robust to autocorrelation and
heteroskedasticity.

**Null direction** — a direction in parameter space along which the objective
is constant, so the gradient is zero. Created by symmetries; explored by
scale-free optimizers like Adam.

**OSAP (Open Source Asset Pricing)** — Chen & Zimmermann's open dataset of
200+ replicated cross-sectional predictors, with published long-short return
series.

**Partial adjustment** — moving a fraction $\gamma$ of the way from current
holdings to the aim. Gârleanu–Pedersen prove it is optimal under quadratic
costs.

**Placebo** — a feature planted with exactly zero true power, used to confirm
the pipeline reports zero. Here: `sig_dead`.

**Point-in-time** — using only information actually available at the decision
date. Two-sided: too early is leakage, too late is staleness.

**PSR (Probabilistic Sharpe Ratio)** — the probability a true Sharpe exceeds a
benchmark, corrected for sample size, skew and kurtosis.

**PULSE** — Per-date Update of Latent Signal Efficacy: a Kalman filter over
each signal's time-varying predictive coefficient, with a decay prior.

**Purging** — deleting periods between a training window and a test window so
training labels cannot overlap test information. Must be at least the label
horizon.

**Rank normalization** — mapping each date's values to evenly spaced points in
$[-1,1]$; scale-free and outlier-immune.

**Retention** — a segment's return as a fraction of the in-sample return; the
decay measure.

**Sharpe ratio** — mean over standard deviation; annualized by $\sqrt{12}$ for
monthly data.

**Spearman correlation** — Pearson correlation applied to ranks.

**Staleness** — using information later than it was available, which destroys
real signal at rate $\rho^k$.

**Turnover (one-way)** — half the traded notional; the fraction of the book
replaced per period. The project's hidden deciding variable.

**Walk-forward** — train on the past, test on the future, roll forward.

---

# Appendix C — Notation table

| Symbol | Meaning |
|---|---|
| $r_{i,t}$ | return of name $i$ during month $t$ |
| $y_{i,t}$ or `fwd_ret` | forward return, $r_{i,t+1}$ — the label |
| $z_{k,i,t}$ | value of signal $k$ for name $i$, known at end of month $t$ |
| $n$ | number of names in one cross-section |
| $T$ | number of dates |
| $K$ | number of signals |
| $P$ | number of parameters (agent: $K+1$; PULSE: expanded feature count) |
| $\tilde x$ | $x$ with its (per-date) mean subtracted |
| $\mathbf 1$ | the all-ones vector |
| $\langle a,b\rangle$ | dot product $\sum_i a_ib_i$ |
| $\Vert a\Vert$, $\Vert a\Vert_1$ | Euclidean and L1 norms |
| $\phi$, $\Phi$ | standard normal density and CDF |
| $\rho$ | correlation; also AR(1) persistence (context disambiguates) |
| $\rho_t$ | per-date cross-sectional correlation |
| $s_p$, $s_y$ | cross-sectional standard deviations of predictions and returns |
| $\mu$, $\sigma$ | mean and standard deviation of a return *series* |
| $\mathrm{SR}$ | Sharpe ratio |
| $\beta_k$ | planted or true coefficient of signal $k$ |
| $\lambda_{k,t}$ | date-$t$ cross-sectional regression coefficient (PULSE's observation) |
| $r_{k,t}$ | sampling variance of $\lambda_{k,t}$ (PULSE's observation noise) |
| $a$, $q$ | PULSE's mean-reversion rate and state-noise variance |
| $m_t$, $P_t$ | Kalman filtered mean and variance |
| $K_t$ | Kalman gain |
| $\theta$ | agent's signal blend |
| $\gamma = \sigma(g)$ | agent's trading speed |
| $A_t$ | aim portfolio |
| $w_t$ | held weights |
| $D_t$ | $\partial w_t/\partial p$, the forward-mode Jacobian |
| $f_{k,t}$ | characteristic-managed portfolio return |
| $\Sigma$, $\mu$ | covariance and mean of factor returns (MSRR) |
| $\kappa$ | capture ratio |
| $c$ | cost in basis points per side |
| $\tau$ | one-way turnover |
| $\alpha$ | factor-controlled intercept |
| $\gamma_3$, $\gamma_4$ | skewness and full kurtosis |
| $\gamma_E$ | Euler–Mascheroni constant, ≈0.5772 |
| $N$ | number of trials, for deflation |
| $h$ | label horizon in periods (1 month here) |
| $\mathcal F_t$ | information set at end of month $t$ |

**One notation clash to watch.** $\rho$ means correlation in Parts 1–3 and
AR(1) persistence in §1.10, §4.7 and §12 of the math docs; $\gamma$ means
trading speed in Part 4 but $\gamma_3,\gamma_4,\gamma_E$ are moments and a
constant in §5.5. Both overloads are standard in the literature and the
context always disambiguates.

---

# Appendix D — Concept-to-code map

Paths relative to `multisignal-alpha/multisignal-alpha/` unless noted.

| Concept | File | Key function |
|---|---|---|
| Synthetic generator (planted truth) | `src/data/synthetic.py` | — |
| Real OSAP / panel loading | `src/data/loaders.py`, `src/data/panel.py` | — |
| Factor panel construction (no WRDS) | `real-data/src/factor_panel.py` | `build_factor_panel` |
| Rank normalization | `src/data/panel.py` | `rank_normalize_cross_section` |
| IC, ICIR, rolling IC | `src/evaluation/ic.py` | — |
| Leak report, lookahead demo | `src/evaluation/ic.py`, `src/pipeline.py` | `leak_report`, `demonstrate_lookahead` |
| Quintile long-short portfolios | `src/evaluation/portfolio.py` | `score_to_weights`, `portfolio_returns` |
| Annualization, skew, kurtosis | `src/utils/stats.py` | `annualized_stats` |
| Newey–West mean test | `src/utils/stats.py` | `nw_mean_test` |
| Fama–MacBeth | `src/evaluation/fama_macbeth.py` | — |
| Factor-controlled alpha | `src/evaluation/factor_controls.py` | — |
| Decay / retention | `src/evaluation/decay.py` | — |
| PSR / DSR | `src/evaluation/deflated_sharpe.py` | — |
| Purged walk-forward splits | `src/backtest/walkforward.py` | `walkforward_splits` |
| Model backtest engine | `src/backtest/engine.py` | — |
| Elastic net, LightGBM, optuna | `src/models/models.py` | `make_linear`, `make_lgbm`, `tune_lgbm` |
| **IC-Net** | `src/models/icnet.py` | `_objective_and_grad`, `fit` |
| **PULSE** | `src/models/pulse.py` | `_per_date_coefs`, `_filter_1d`, `predict` |
| Interaction expansion | `src/models/pulse.py` | `expand_interactions` |
| **Agent policy + gradients** | `src/agent/policy.py` | `_roll`, `_sharpe_and_grad`, `fit` |
| Agent walk-forward (inventory carried) | `src/agent/backtest.py` | `backtest_agent` |
| MSRR closed form | `src/agent/msrr.py` | — |
| Cost sweep experiment | `scripts/agent_cost_sweep.py` | — |
| PULSE + agent composition | `scripts/compose_pulse_agent.py` | — |
| Live signal generation | `src/execution/signal_generator.py` | `aim_weights` |
| Live partial adjustment | `src/execution/agent_evaluator.py` | `CostAwareAgentEvaluator` |
| Broker bridge | `src/execution/alpaca_bridge.py` | — |
| Orchestration | `src/pipeline.py` | — |
| Test suite | `tests/test_backtest.py`, `test_evaluation.py`, … | — |

**The tests worth reading as documentation**, because each one pins a theorem
to a measurement:

- `test_models_backtest_runs_and_lgbm_beats_linear_on_interaction` — §3.2.3
- `test_pulse_tracks_planted_decay_and_predicts_oos` — §3.4
- `test_agent_learns_garleanu_pedersen_comparative_statics` — §4.3, §6.6
- `test_msrr_closed_form_recovered_by_zero_cost_agent` — §4.6.4
- `test_sqrt_impact_slows_trading` — §4.8
- the placebo test — §5.7

---

# Appendix E — Self-test questions (with answers)

Work through these without looking. If you can answer all of them you own the
material.

## Tier 1 — concepts

**1. Why is the project "cross-sectional," and what does that buy?**
<details>
<summary>Answer</summary>

It ranks names against each other on each date rather than forecasting the
market. It buys dollar neutrality: a uniform move in all returns cancels
between the long and short legs (§1.8), so the strategy does not need to
answer "will the market go up?" — a question nobody can answer.

</details>

**2. A colleague reports a signal with IC 0.35. What is your first move?**
<details>
<summary>Answer</summary>

Hunt for the timestamp error. Honest single-signal ICs live at 0.02–0.12; the
project *derives* a ceiling near 0.04 on its synthetic panel (§6.2) and the
deliberate leak reads 0.395 (§6.1). Ten times too good is a diagnosis, not a
discovery.

</details>

**3. Why is the mean IC computed per date and then averaged, rather than
pooled over all name-months?**
<details>
<summary>Answer</summary>

Pooling mixes a between-dates effect (do high-signal months have high-return
months?) with the within-date effect, which is the only part a market-neutral
book can harvest. Per-date demeaning deletes each date's market level, so
market-direction effects contribute exactly zero (§1.7).

</details>

**4. Why does the pipeline evaluate signals *before* fitting any model?**
<details>
<summary>Answer</summary>

Because the model's job is *combination*, not rescuing an empty signal. If
you fit first you cannot tell whether a good combined result came from real
signal or from a flexible model memorizing noise (§6.4).

</details>

## Tier 2 — mechanics

**5. A strategy has one-way turnover 1.2 and gross annual return 9%. What is
its net return at 10 bps per side, and at 50 bps?**
<details>
<summary>Answer</summary>

Drag $= \tau\times2\times\text{bps}/10^4\times12$. At 10 bps:
$1.2\times0.024 = 0.0288$, so net $= 6.1$%. At 50 bps:
$1.2\times0.12 = 0.144$, so net $= -5.4$% — the strategy does not exist
(§1.9).

</details>

**6. Why must the purge be at least the label horizon, and what breaks if it
is zero?**
<details>
<summary>Answer</summary>

The label at training date $t$ is realized over $(t, t+h]$. If the test fold
starts at $t+1$ with $h = 1$, the last training label's realization window is
the first test period — the model was fitted on information from the test
window. The purging theorem (§2.3) states the condition $t^\*+h<\tau$ exactly.
`walkforward.py` raises a `ValueError` on purge 0.

</details>

**7. Why is random-sample validation illegal for IC-Net's early stopping,
when it is standard practice elsewhere in ML?**
<details>
<summary>Answer</summary>

A randomly chosen month's temporal neighbours remain in training and carry
overlapping information, so the "held-out" score is contaminated. The repo
uses a chronological tail of the *training* dates (§3.3.6) — the same rule
purging imposes on everything else.

</details>

**8. The pipeline reports a pure-noise feature at t = 2.12. Is something
broken?**
<details>
<summary>Answer</summary>

No. $t=2$ is a 5% probability statement, not a certificate. The null standard
error of the mean IC here is ≈0.0026, so a 2.3-s.e. draw happens a couple of
percent of the time, and this run drew one (§5.7).

</details>

**9. `fmom_12m` has the project's best standalone IC (0.116, t = 10.6) but a
Fama–MacBeth coefficient of −0.0002 (t = −0.06). Explain.**
<details>
<summary>Answer</summary>

It correlates 0.968 with `fmom_12_2`. The two are nearly the same variable, so
the multivariate regression cannot attribute the shared information and
loaded the other twin. The joint information is intact; only the attribution
is unidentified (§7.4).

</details>

## Tier 3 — the mathematics

**10. State and prove the MSE decomposition, and say which term the long-short
portfolio actually cares about.**
<details>
<summary>Answer</summary>

$\tfrac1n\sum(p_i-y_i)^2 = (\bar p-\bar y)^2 + (s_p-s_y)^2 + 2s_ps_y(1-\rho)$.
Proof in §3.3.1. The portfolio is dollar-neutral so it is indifferent to the
level term, and quantile-sorted so indifferent to the scale term. **Only the
correlation term maps to money** — and MSE additionally weights it by $s_y$,
so volatile months dominate training for no informational reason.

</details>

**11. Derive the gradient of a per-date correlation with respect to the
predictions, and state the sanity property that falls out.**
<details>
<summary>Answer</summary>

$\nabla_p\rho = \tilde y/(\Vert\tilde p\Vert\Vert\tilde y\Vert) - \rho\thinspace\tilde p/\Vert\tilde p\Vert^2$
(§3.3.3). Both $\tilde y$ and $\tilde p$ are demeaned, so the gradient has
zero mean: nudging all predictions up together cannot help. The calculus
rediscovers translation invariance on its own.

</details>

**12. Prove that no additive function can represent $z_1z_2$.**
<details>
<summary>Answer</summary>

Set $z_2=0$: $0 = f(z_1)+g(0)$ for all $z_1$, so $f$ is constant. By symmetry
$g$ is constant. Then $f+g$ is constant while $z_1z_2$ is not (§3.2.3).

</details>

**13. Derive the Kalman update from Bayes' rule, and interpret the gain.**
<details>
<summary>Answer</summary>

Multiply the prior $N(m,P)$ by the likelihood $N(\beta, r)$, work in log
densities, complete the square: precision adds, giving
$P_{\text{post}} = (1/P + 1/r)^{-1} = (1-K)P$ and
$m_{\text{post}} = m + K(\lambda-m)$ with $K = P/(P+r)$ (§3.4.3).
**Posterior = prior + gain × surprise.** The gain is a precision-weighted
compromise: trust the observation when your prior is vague or the measurement
is clean. Every learning rate is this formula with the uncertainties hidden.

</details>

**14. Show the steady-state Kalman filter is an EWMA, and say what sets its
half-life.**
<details>
<summary>Answer</summary>

At a fixed point $P = (P+q)r/(P+q+r)$, so $P^2+qP-qr=0$ and
$P^\* = (-q+\sqrt{q^2+4qr})/2$. Constant $P^\*$ gives constant $K^\*$;
unrolling the update gives $m_t = K^\*\sum_j(1-K^\*)^j\lambda_{t-j}$, half-life
$\ln2/|\ln(1-K^\*)|$ (§3.4.5). **The signal-to-noise ratio $q/r$ sets it** —
noisy observations give long memory, genuinely mobile states give fast
adaptation.

</details>

**15. Derive the gradient of the Sharpe ratio with respect to one month's
return, and read off what it says about risk.**
<details>
<summary>Answer</summary>

$\partial J/\partial r_t = 1/(Ts) - m(r_t-m)/(Ts^3)$ (§4.4). A baseline reward
for return, minus a risk charge proportional to how far that month already
sits from the mean. **Maximizing Sharpe is not maximizing return**, and this
is the precise statement of the difference.

</details>

**16. Prove the agent's policy is an EWMA of past aims, and name the two
economic dials.**
<details>
<summary>Answer</summary>

Induction on $w_t = (1-\gamma)w_{t-1}+\gamma A_t$ gives
$w_t = \gamma\sum_{j=0}^{t-1}(1-\gamma)^jA_{t-j}$ (§4.3). The dials are
$\theta$ (what to aim at) and $\gamma$ (how fast to chase), which are exactly
the two Gârleanu–Pedersen theory says matter.

</details>

**17. Prove the max-Sharpe blend over factor returns is $\Sigma^{-1}\mu$.**
<details>
<summary>Answer</summary>

Substitute $x = \Sigma^{1/2}\theta$, $b = \Sigma^{-1/2}\mu$; then
$S = b^\top x/\Vert x\Vert \le \Vert b\Vert$ by Cauchy–Schwarz, with equality
iff $x\propto b$, i.e. $\theta\propto\Sigma^{-1}\mu$ (§4.6.2). Note this is
the *same* theorem as $|\rho|\le1$ from §1.3.

</details>

**18. Compute the capture ratio at $\gamma = 0.6$, $\rho = 0.9$, and say what
it means commercially.**
<details>
<summary>Answer</summary>

$\kappa = 0.6/(1-0.4\times0.9) = 0.6/0.64 = 0.94$ (§4.7). The agent keeps 94%
of the alpha while cutting turnover from 0.42 to 0.27 — a 36% cost reduction
for a 6% alpha sacrifice. That fraction *is* the business case for
execution-aware trading.

</details>

**19. Prove the best of $N$ worthless strategies is biased upward, and price
$N = 20$.**
<details>
<summary>Answer</summary>

Jensen plus $\max \le \text{sum}$ plus the normal MGF gives
$e^{sE[\max]} \le Ne^{s^2/2}$; take logs, divide by $s$, minimize at
$s=\sqrt{2\ln N}$ to get $E[\max] \le \sqrt{2\ln N}$ (§5.4). At $N = 20$ that
is ≈2.45 — past the significance line, for free.

</details>

**20. Why does Adam move a parameter whose gradient is exactly zero?**
<details>
<summary>Answer</summary>

Its step is $\mathrm{lr}\cdot\hat m/(\sqrt{\hat v}+\epsilon)$. Feed it numerical
noise $\delta$: both $\hat m$ and $\sqrt{\hat v}$ are $O(\delta)$, so the ratio
is $O(1)$ with random sign and the step is ≈lr *regardless of how small the
noise is*. Scale-freeness turns "zero gradient" into unit-speed diffusion
(§4.9.2). On this project it silently flipped a book's sign; the fix is
projecting $\theta$ onto the unit L1 sphere, which quotients out the symmetry
that created the flat direction.

</details>

## Tier 4 — judgment

**21. PULSE won on synthetic data and came last on real data. Was the
synthetic result wrong?**
<details>
<summary>Answer</summary>

No. The synthetic generator *is* PULSE's model class — piecewise-constant
betas plus an interaction is literally "time-varying linear on an interaction
basis" — and the design doc said so *before* the real run. The two results
answer different questions: can the method find drifting coefficients
(yes), and does the real world have enough of that structure to pay for the
variance of looking (on this panel, no). §8.1.

</details>

**22. Why did the elastic net win on real data? Give four reasons.**
<details>
<summary>Answer</summary>

(i) Bias–variance: with $R^2$ ~1%, variance dominates, so biased
low-variance estimators win — predicted in advance by §3.1.3. (ii) No planted
interaction to find, and real nonlinearity is buried in noise. (iii) Four of
six features are two collinear twin-pairs (0.968 and 0.747), which shrinkage
handles gracefully and flexible models do not. (iv) Turnover: 0.742 versus
0.927–1.105, so the flexible models also pay more tax. §8.1.2.

</details>

**23. The agent froze its book on real data and lost to always-rebalancing.
What are the competing explanations, and which experiment settles it?**
<details>
<summary>Answer</summary>

(a) It overfits the speed dial: a frozen dollar-neutral book across ~145
weakly-correlated factor portfolios has very low variance and so a high
*in-sample* Sharpe, which does not generalize. (b) The policy class is too
poor — one shared $\gamma$ cannot track `fmom_12m` slowly while ignoring
`fmom_1m`. (c) The expanding window flattens the objective as $T$ grows.
**The settling experiment is the cost sweep**: if $\gamma \approx 0.0005$ even
at 0 bps, the collapse is not about costs at all. §8.3.

</details>

**24. A DSR of 1.000 and a DSR of 0.972 are reported in this project. Which
is the better news?**
<details>
<summary>Answer</summary>

0.972. The 1.000 is synthetic: a planted, stable, clean signal with no regime
shifts saturates the statistic, and nothing in finance is certain to that
many decimals — read it as a realism warning. The 0.972 is real: 1044 months,
10 counted trials, a monthly Sharpe of 0.171 against a 0.086 luck hurdle,
above the 0.95 bar and meaningfully below 1. §6.7, §7.9.

</details>

**25. What is the largest single gap between this project's reported numbers
and reality?**
<details>
<summary>Answer</summary>

The factor panel's "assets" are OSAP long-short research portfolios, each
itself a book of hundreds of stocks with its own turnover, borrow costs and
capacity limits. Charging 10 bps per side to rebalance *across* 145 such
books applies a retail commission to an institutional operation. §8.6, threat 1.

</details>

---

# Appendix F — Reading list

Ordered by how much it will help you with *this* project.

**Read first — the anchor results.**

- **Gu, Kelly & Xiu (2020)**, *Empirical Asset Pricing via Machine Learning*,
  RFS. The paper this project's model comparison is modeled on: ML beats
  linear on firm characteristics, gains come from nonlinear interactions, and
  the dominant signals are momentum, liquidity and volatility. (Public
  replication available via Tidy Finance.)
- **Chen & Zimmermann (2022)**, *Open Source Cross-Sectional Asset Pricing*,
  Critical Finance Review. The data behind Part 7: 200+ replicated
  predictors, open code, `pip install openassetpricing`. Note the practical
  caveat the project hits — 209 of 212 characteristics are in the package;
  Price, Size and short-term reversal need CRSP.
- **McLean & Pontiff (2016)**, *Does Academic Research Destroy Stock Return
  Predictability?*, JF. The ~26% / ~58% decay result that §7.7 replicates.

**Read for the methods in Parts 3–4.**

- **Gârleanu & Pedersen (2013)**, *Dynamic Trading with Predictable Returns
  and Transaction Costs*, JF. Proves partial adjustment is optimal under
  quadratic costs — the agent's policy class (§4.3).
- **Moody & Saffell (2001)**, *Learning to Trade via Direct Reinforcement*.
  The original differentiable-simulator argument, two decades early (§4.2).
- **Zhang, Zohren & Roberts (2020)**. Deep networks trained end-to-end on net
  Sharpe.
- **Kelly & Malamud**, MSRR. The closed-form aim of §4.6.
- **Bailey & López de Prado (2014)**, the Deflated Sharpe Ratio; and
  **Lo (2002)** on Sharpe-ratio statistics (§5.5).
- **López de Prado**, *Advances in Financial Machine Learning*. Purged
  cross-validation, embargoes, multiple testing — the source of §2.3 and
  Part 5's discipline.
- **Newey & West (1987)** for §1.11; **Fama & MacBeth (1973)** for §1.12.

**Read for context and to be current.**

- **Kelly & Xiu (2023)**, *Financial Machine Learning*. The canonical survey;
  the way practitioners now frame the whole area.
- **Kelly, Malamud & Zhou (2024)**, *The Virtue of Complexity in Return
  Prediction*, JF — plus the **Nagel (2025)** and **Buncic (2025)** critiques
  and the **Kelly–Malamud (2025)** response. An active, unresolved debate
  that lives exactly where §4.6.3's ridge parameter sits.
- **Bowles, Reed, Ringgenberg & Thornock (2024)**, *Anomaly Time*, JF.
  Formation timing and stale information — §5.3's staleness argument.
- **Ehsani & Linnainmaa (2022)**, factor momentum; **Gupta & Kelly (2019)**.
  The literature Part 7's panel design sits inside.
- **Jensen, Kelly & Pedersen (2023)**, *Is There a Replication Crisis in
  Finance?*, JF. A second major open factor dataset (150+ factors, 90+
  countries) and a natural robustness check on §7.7.
- **Chen & Zimmermann (2022)**, *Publication Bias in Asset Pricing Research*.
  Predictability persists out-of-sample (~74% remains in the first three
  years) — the nuance in §7.7.
- **Grinold & Kahn**, *Active Portfolio Management*, for IC and signal
  quality as practitioners use them.
- **Jansen**, *Machine Learning for Algorithmic Trading*. The workflow
  reference — and note §4.2's disagreement with its chapter 22. Use the
  maintained `-reloaded` package forks (`alphalens-reloaded`,
  `pyfolio-reloaded`, `zipline-reloaded`, `empyrical-reloaded`); the original
  Quantopian packages are unmaintained and break on modern Python.

---

*This guide was written against the repository state at commit `cc77d6e`
(the 212-factor results run). Every number quoted comes from a committed
output file under `multisignal-alpha/multisignal-alpha/results/`,
`results_factor/`, or `real-data/data/processed/`; the aggregate decay
statistics in §7.7 and the feature correlations in §7.4 and §8.1.2 were
computed directly from `factor_decay.csv` and `factor_panel.csv`. Where a
number here disagrees with `docs/USER_GUIDE.md`, that guide predates the
212-factor run and this one is current.*
