# Pipeline summary

Panel: 299 months x 500 names; signals: sig_momentum, sig_liquidity, sig_volatility, sig_value, sig_quality, sig_dead

## Lookahead demonstration (should look absurd)

| feature       |    IC |    IC_t |
|:--------------|------:|--------:|
| leaky_feature | 0.395 | 178.888 |
| honest_noise  | 0.006 |   2.124 |


## Per-signal evaluation

| signal         |    IC |   IC_t |   ICIR |   ann_ret_gross |   sharpe_gross |   nw_t_gross |   ann_ret_net |   sharpe_net |   nw_t_net |   one_way_turnover |   n_months |
|:---------------|------:|-------:|-------:|----------------:|---------------:|-------------:|--------------:|-------------:|-----------:|-------------------:|-----------:|
| sig_momentum   | 0.037 | 10.951 |  0.807 |           0.102 |          2.541 |       10.591 |         0.089 |        2.234 |      9.321 |              0.509 |        298 |
| sig_liquidity  | 0.023 |  7.907 |  0.486 |           0.059 |          1.464 |        6.4   |         0.051 |        1.256 |      5.492 |              0.354 |        298 |
| sig_volatility | 0.023 |  9.091 |  0.494 |           0.065 |          1.575 |        7.96  |         0.053 |        1.28  |      6.468 |              0.505 |        298 |
| sig_value      | 0.01  |  4.172 |  0.231 |           0.028 |          0.682 |        3.525 |         0.023 |        0.549 |      2.841 |              0.232 |        298 |
| sig_quality    | 0.013 |  4.981 |  0.292 |           0.037 |          0.917 |        4.528 |         0.031 |        0.753 |      3.713 |              0.277 |        298 |
| sig_dead       | 0.005 |  1.897 |  0.113 |           0.015 |          0.424 |        1.971 |         0.003 |        0.095 |      0.441 |              0.5   |        298 |


## Fama-MacBeth (multivariate marginal power)

| variable       |   mean_coef |   nw_tstat |   n_periods |
|:---------------|------------:|-----------:|------------:|
| const          |      0.0066 |     2.537  |         298 |
| sig_momentum   |      0.0054 |    11.2146 |         298 |
| sig_liquidity  |      0.0033 |     8.3075 |         298 |
| sig_volatility |      0.0033 |     8.9548 |         298 |
| sig_value      |      0.0016 |     4.5059 |         298 |
| sig_quality    |      0.0019 |     5.0338 |         298 |
| sig_dead       |      0.0007 |     1.9487 |         298 |


## Decay (McLean-Pontiff pattern)

| signal         |   in_sample_sharpe |   post_sample_sharpe |   post_pub_sharpe |   post_sample_retention |   post_pub_retention |
|:---------------|-------------------:|---------------------:|------------------:|------------------------:|---------------------:|
| sig_momentum   |              3.643 |                2.161 |             1.451 |                   0.684 |                0.396 |
| sig_liquidity  |              1.668 |                3.555 |             0.709 |                   2.038 |                0.456 |
| sig_volatility |              1.639 |                2.067 |             1.357 |                   1.105 |                0.683 |
| sig_value      |              0.877 |                1.146 |             0.11  |                   0.885 |                0.115 |
| sig_quality    |              1.066 |                1.508 |             0.27  |                   1.445 |                0.242 |
| sig_dead       |              0.586 |                1.146 |            -0.015 |                   1.967 |               -0.028 |


## Model comparison (purged walk-forward, out-of-sample)

| model      |   oos_IC |   oos_ICIR |   ann_ret_gross |   sharpe_gross |   ann_ret_net |   sharpe_net |   nw_t_net |   one_way_turnover |   n_oos_months |
|:-----------|---------:|-----------:|----------------:|---------------:|--------------:|-------------:|-----------:|-------------------:|---------------:|
| elasticnet |    0.043 |      0.925 |           0.124 |          2.929 |         0.113 |        2.663 |      8.244 |              0.474 |            168 |
| lightgbm   |    0.06  |      1.379 |           0.174 |          4.625 |         0.152 |        4.047 |     17.79  |              0.907 |            168 |
| icnet      |    0.07  |      1.5   |           0.207 |          4.842 |         0.195 |        4.572 |     19.946 |              0.484 |            168 |
| pulse      |    0.072 |      1.553 |           0.215 |          5.357 |         0.203 |        5.052 |     20.408 |              0.509 |            168 |


## PULSE chosen dynamics

| parameter   |   chosen | grid              | at_grid_edge   |
|:------------|---------:|:------------------|:---------------|
| a           |    0.99  | 0.97, 0.99, 1.0   | False          |
| q_scale     |    0.002 | 0.002, 0.01, 0.05 | True           |


## Factor-controlled alpha (net strategies)

| model      |   alpha_ann |   alpha_t |    r2 |   n |
|:-----------|------------:|----------:|------:|----:|
| elasticnet |       0.114 |     8.274 | 0.032 | 168 |
| lightgbm   |       0.154 |    18.671 | 0.061 | 168 |
| icnet      |       0.196 |    19.627 | 0.051 | 168 |
| pulse      |       0.203 |    20.061 | 0.038 | 168 |


## Deflated Sharpe (pulse)

| model   |   dsr |   sr_star_monthly |   n_trials |   psr_vs_zero |
|:--------|------:|------------------:|-----------:|--------------:|
| pulse   |     1 |             0.795 |         11 |             1 |


---

# Appendix: what these figures mean, in plain language

> Hand-written narrative for the synthetic (`data.mode: synthetic`) run above.
> The tables above are regenerated by `src/pipeline.py`; this appendix is not.
> If you rerun the pipeline, re-append this section or move it into the writer.

## The caveat that governs everything below

`configs/config.yaml` sets `data.mode: synthetic`. Every number above was
computed on a simulated 500-stock panel in which the signals were planted by
hand: `sig_momentum` was given a true beta of 0.0040 per month, `sig_dead` was
given exactly 0.0000, and a momentum-by-value interaction term plus
post-sample and post-publication decay factors of 0.70 and 0.45 were planted
on top.

So this run is not evidence of alpha. It is evidence that the measurement
machinery works: it recovers the effects that were planted, in roughly the
right rank order, with the right things showing up as zero. Validating the
ruler before measuring anything real is the correct first step, but a Sharpe
ratio of 5 is a property of the simulator, not of markets. Read the whole run
as a diagnostic self-test that passed.

## The panel

299 months by 500 names is about 25 years of monthly data, roughly 150,000
stock-months, with 6 predictors per stock. Monthly frequency means every
statistic is monthly unless the column name starts with `ann_`.

## Lookahead demonstration

This is deliberate sabotage. The pipeline builds a feature that secretly
contains next month's return and checks that the evaluation code screams. An
IC of 0.395 is absurd; no real equity signal predicts returns that well. Next
to it, pure noise scores 0.006.

The point is calibration. If a lookahead leak is ever introduced by accident,
the result will look obviously fake, and now you know what "obviously fake"
looks like in your own numbers. This check passed.

## Per-signal evaluation

Each row is one signal evaluated on its own over 298 months.

**IC (Information Coefficient).** Each month, the cross-sectional rank
correlation between the signal's ordering of the 500 stocks and their actual
next-month returns, averaged over all months. `sig_momentum` at 0.037 means
the ranking is right about 52% of the time in a pairwise sense. That sounds
pathetic, and it is, but it is also normal: real equity signals live at IC
0.02 to 0.05. Money is made by being slightly right 500 times a month for 300
months.

**IC_t.** The t-statistic on that average IC: reliably non-zero, or luck?
Above roughly 2 is the conventional bar. Momentum's 10.95 is overwhelming.
`sig_dead` at 1.90 is the only signal below 2, which is exactly right, because
its true beta is zero.

**ICIR.** Mean IC divided by its month-to-month standard deviation. IC says
how strong; ICIR says how consistent. Momentum's 0.807 monthly is about 2.8
annualized, strong and steady. Value at 0.231 is weak and erratic.

**ann_ret_gross and sharpe_gross.** The economic test rather than the
statistical one. Per `src/evaluation/portfolio.py`, each month the 500 names
are sorted into quintiles on the signal, the top 100 are bought and the bottom
100 sold, equal weighted, normalized to $1 long and $1 short. `ann_ret_gross`
is what that spread earns per year before trading costs, and `sharpe_gross` is
return per unit of volatility.

**nw_t_gross.** A Newey-West t-statistic. The correction matters because
monthly strategy returns are autocorrelated, which a naive t-statistic ignores
and thereby overstates. This is the conservative version.

**one_way_turnover.** The fraction of the book replaced each month. Momentum's
0.509 means half the portfolio turns over monthly. Value's 0.232 means it is
slow and sticky, because value ranks barely move (its AR coefficient in the
config is 0.98).

**Net versus gross.** The same numbers after paying the configured 10 basis
points per side. Turnover is what decides the size of the gap:

| signal | gross ann_ret | net ann_ret | cost drag |
|---|---|---|---|
| `sig_momentum` | 10.2% | 8.9% | -1.3pp (high turnover) |
| `sig_value` | 2.8% | 2.3% | -0.5pp (low turnover) |
| `sig_dead` | 1.5% | 0.3% | -1.2pp (eaten alive) |

The headline of this table is `sig_dead`. It was planted with zero true
predictive power. Gross, it appears to earn 1.5% per year at a Sharpe of 0.42,
which is pure noise dressed up as a strategy. Net of costs it collapses to a
Sharpe of 0.095 and a t-statistic of 0.44. The placebo correctly failed, which
is the control working.

The recovered rank order is momentum, then volatility and liquidity, then
quality, then value, then dead. That matches the planted betas (0.0040, 0.0020,
0.0025, 0.0012, 0.0015, 0.0000) closely. The small value and quality flip is
explained by their different turnover and persistence.

## Decay

This reproduces McLean and Pontiff (2016), who found that published anomalies
weaken by about 26% after the original study's sample ends and about 58% after
publication. Decay factors of 0.70 and 0.45 were planted, and this table checks
that the analysis finds them.

Retention is a segment's annualized return as a fraction of the in-sample one.
So a `post_pub_retention` of 0.396 for momentum means it kept about 40% of its
original strength after publication.

Reading the column that matters, `post_pub_retention`: momentum 0.40, liquidity
0.46, volatility 0.68, quality 0.24, value 0.12, and `sig_dead` at -0.028,
which never had anything to lose. Against a planted truth of 0.45 these cluster
in the right neighborhood, and the scatter around it is the honest amount of
estimation error from splitting 300 months into three short pieces.

**The `post_sample_retention` entries above 1 are noise, not findings.**
Liquidity shows 2.04 and dead shows 1.97, which would mean they got twice as
good after the sample ended. They did not. The post-sample window is only about
24 months for most signals, and a Sharpe estimated from 24 months carries a
standard error of roughly 0.7. Do not read anything into that column for any
signal. The post-publication window is longer and correspondingly more
trustworthy.

## Model comparison

Four models blend all six signals into a single score, evaluated walk-forward
over 168 out-of-sample months: train on at least 120 months, predict the next
12, roll forward, with a 1-month purge gap so training data cannot touch the
test label. Every number here is on data the model had not seen at prediction
time.

- **elasticnet**: penalized linear regression, the benchmark that has to be
  beaten.
- **lightgbm**: gradient-boosted trees, able to capture nonlinearity.
- **icnet**: the from-scratch NumPy network whose loss function is the IC
  itself rather than squared error.
- **pulse**: the Kalman-filtered model that lets each signal's strength drift
  over time instead of assuming it is constant.

On net Sharpe: elasticnet 2.66, lightgbm 4.05, icnet 4.57, pulse 5.05. Every
combination beats every individual signal (the best single was momentum at
2.23), which is the expected diversification result. Trees beat linear, as they
should here, because a momentum-by-value interaction was planted and a linear
model structurally cannot see it.

**The turnover column is the interesting one.** LightGBM's 0.907 one-way
turnover means it nearly rebuilds the portfolio every month, because tree
predictions jump discontinuously as features cross split thresholds. That costs
it 0.58 of Sharpe (4.625 gross to 4.047 net). PULSE reaches a higher gross
Sharpe at 0.509 turnover and gives up only 0.30. PULSE wins on net partly by
being smoother, not only by being more accurate: the Kalman prior makes its
coefficients drift rather than jump. That is a genuine structural advantage and
worth writing up.

## PULSE chosen dynamics

`a=0.99` means signal efficacy mean-reverts slowly, so the effect is nearly but
not quite permanent. `q_scale=0.002` is the smallest option offered, so the
filter concluded that coefficients drift slowly. Both sit at the conservative
end of their grids, which is the model reporting that this world is fairly
stable.

`a` is interior to its grid, but `q_scale` sits at the boundary, so there is no
way to tell whether the true optimum is smaller still. Extending
`q_scale_grid` downward would settle it.

## Trading agent

A separate experiment, in `src/agent/policy.py`. Instead of predicting returns
and then trading the prediction, this learns a signal blend and a trading speed
`gamma` at the same time, by backpropagating through a differentiable simulator
of the trading process that includes costs. The recursion is
`w_t = (1 - gamma) * w_{t-1} + gamma * aim_t`, so `gamma = 1` means jumping
straight to the target portfolio every month and `gamma = 0.5` means moving
halfway. This is Garleanu-Pedersen partial adjustment.

At the configured 10 basis points, the agent learned to trade at 84% speed, cut
turnover by 15%, and landed a net Sharpe 0.004 below the myopic policy. That is
a dead heat, well inside noise, so **the agent did not win at this cost level.**

That is not a bug; it is the predicted result. The module docstring states the
falsifiable claim precisely: at high costs the agent's net Sharpe should beat
the myopic `gamma = 1` policy. At 10 basis points, costs are simply not high
enough for trading speed to matter. `scripts/agent_cost_sweep.py` is where the
claim is actually tested, and its output
(`results/tables/agent_cost_sweep.csv`) confirms it: learned gamma falls
monotonically from 0.922 at 0 bps to 0.213 at 100 bps, and at 100 bps the agent
holds a net Sharpe of 0.952 while myopic rebalancing goes negative at -0.692.
The 10 bps row above is the leftmost and least interesting column of that
sweep.

One thing not to misread: the agent's 2.96 net Sharpe is far below PULSE's
5.05, but the two are solving different problems on different inputs. The agent
blends the six raw signals linearly and therefore cannot see the planted
interaction. Composing the two (`results/tables/composed_pulse_agent.csv`),
so that PULSE's forecast becomes the agent's aim, gives a net Sharpe of 5.345,
which beats PULSE alone.

## Factor-controlled alpha

Each model's return series is regressed on known factor returns, and the
intercept is what survives.

- **alpha_ann**: the annualized return left unexplained by the factors. PULSE:
  20.3%.
- **alpha_t**: the Newey-West t-statistic on that intercept. PULSE: 20.1.
- **r2**: how much of the strategy's variation the factors explain. PULSE:
  0.038, that is 3.8%.

Low R-squared with large surviving alpha is the strong outcome: the strategy is
not a known factor in disguise. Note that PULSE's `alpha_ann` of 0.203 is
essentially identical to its raw net return of 0.203, so the factors explain
almost nothing. In synthetic data that is unsurprising, because the planted
signals are orthogonal to the market by construction. **On real data, expect
R-squared to be far higher and alpha to shrink considerably.** This test will
be much more informative then; here it mostly confirms the plumbing.

## Deflated Sharpe

The problem being solved: 11 configurations were tried and the best one was
reported. The best of 11 tries looks good even when nothing works, purely by
selection. The Deflated Sharpe Ratio corrects for that.

- **n_trials = 11**: how many things were looked at. `config.yaml` sets
  `count_single_signal_trials: true`, which is honest accounting. Undercounting
  trials is the standard way this statistic gets gamed.
- **sr_star_monthly = 0.795**: the hurdle, meaning the Sharpe the luckiest of 11
  random strategies would be expected to show. Monthly 0.795 is about 2.75
  annualized, so the bar is not zero; it is "beat a Sharpe of 2.75 that pure
  luck would have produced".
- **dsr = 1.000**: the probability that the true Sharpe exceeds that hurdle,
  after correcting for skew and fat tails. PULSE's net Sharpe of 5.05 against a
  2.75 luck hurdle rounds the probability to 1.000. The stored value is
  0.9999999862.

**Treat a DSR of 1.000 as a realism warning, not a trophy.** Nothing in finance
is certain to that many decimal places. It reads 1.000 because the simulator
planted a large, stable, clean signal with no regime shifts, no missing data,
no crowding and no capacity constraints. On real data, expect 0.7 to 0.95 for
something genuinely good. The verified 5-factor real-data run scored 0.79,
which `docs/USER_GUIDE.md` correctly describes as "promising, not proven".

## What this run does and does not support

**Supported.** The pipeline is sound. Leak detection fires. The placebo dies
once costs are charged. Decay is recovered in the right direction and rough
magnitude. Walk-forward is purged. Costs are charged consistently. Multiple
testing is penalized with honest trial counting. Combination beats singles,
nonlinear beats linear where nonlinearity was planted, and PULSE's smoothness
advantage shows up in turnover, which is a real structural finding about the
method rather than an artifact of the data.

**Not supported.** Any of these returns. A Sharpe of 5.05, 20% alpha and a DSR
of 1.000 are artifacts of a generator written for this repository.

**Next steps, in order.**

1. Extend `q_scale_grid` below 0.002, since PULSE selected the boundary value
   and the optimum may lie outside the grid.
2. Promote the cost sweep and the composed PULSE-plus-agent result into the
   headline story. Both are stronger than the 10 bps null shown above, and
   neither currently appears in this summary.
3. Switch `data.mode` to `osap` and rerun. `configs/config.yaml` documents the
   three files needed in `data/raw/`. Every number above will get worse, and
   that run is the one worth showing people.
