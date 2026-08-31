# The Trading Agent: Differentiable, Cost-Aware Policy Learning
## Replacing ML4T's RL chapter with the modern consensus

*Implementation: `src/agent/policy.py` + `src/agent/backtest.py` (pure NumPy).
Math: `docs/math/11_differentiable_trading.md`. Validation:
`tests/test_backtest.py::test_agent_learns_garleanu_pedersen_comparative_statics`.*

---

## 1. What an agent is, and why the repo needed one

Everything upstream in this repo answers *which stocks look good*. An agent
answers the question that actually generates P&L: **what do I hold this
month, given what I held last month and what trading costs?** Forecasting
and trading are different problems because of one word — *inventory*. A
forecaster is memoryless; an agent's decision today constrains its costs
tomorrow. The repo's evaluation stack always charged costs *after the fact*;
the agent is the first component that reasons about them *before acting*.

## 2. Improving on the book, with the literature's blessing

Jansen's ML4T handles this with model-free deep RL (ch. 22: DQN on price
series) — the book's weakest chapter by common assent, because model-free RL
estimates policy gradients from sampled episodes, which is hopeless at
financial signal-to-noise with a few hundred monthly observations. The
modern line replaces estimation with calculation: **when the trading
simulation is differentiable — holdings, returns, and costs are smooth in
the policy parameters — the policy gradient can be computed *exactly* by
backpropagating through the simulator.** No replay buffers, no exploration
noise, no Bellman bootstrap; just calculus. The lineage:

- **Moody & Saffell (2001)** — *Learning to Trade via Direct Reinforcement*:
  optimize the performance measure directly through the trading recurrence.
  The original, two decades ahead of its rediscovery.
- **Gârleanu & Pedersen (2013, JF)** — with quadratic costs the *optimal*
  dynamic policy is closed-form **partial adjustment**: each period, trade a
  fraction of the way from current holdings toward an "aim" portfolio.
- **Zhang, Zohren & Roberts (2020)** — deep networks trained end-to-end on
  the Sharpe ratio of **net** returns.
- **Kelly & Malamud (MSRR)** — signals mapped directly to weights,
  maximizing Sharpe; the same philosophy in closed form, and the endpoint
  IC-Net's proposal document already pointed to.

## 3. The design: an economically-derived policy class

Rather than a free-form network, the policy *is* the GP structure — theory
as inductive bias:

$$
\text{score}_{t,i} = \textstyle\sum_k \theta_k z_{k,i,t}
\quad\to\quad
\text{aim}_t = \frac{2\,\widetilde{\text{score}}_t}{\|\widetilde{\text{score}}_t\|_1}
\quad\to\quad
w_t = (1-\gamma)\,w_{t-1} + \gamma\,\text{aim}_t
$$

with net return $\langle w_t, y_t\rangle - c\sum_i|w_{t,i}-w_{t-1,i}|$.
Seven parameters: a learned signal blend $\theta$ and a learned **trading
speed** $\gamma = \sigma(g)$. The recursion unrolls to an exponentially
weighted average of past aims, so the agent is provably a *learned-speed
EWMA of a learned aim* — the two dials GP theory says matter, with the data
setting both. Training maximizes the **net Sharpe over the training window**
by Adam, with exact gradients from forward-mode sensitivity propagation
through the recursion (derived line-by-line in the math chapter); the $|\cdot|$
in the cost is smoothed as $\sqrt{x^2+\varepsilon}$ so the objective is
differentiable. Walk-forward discipline is identical to everything else in
the repo — per-fold training on purged windows, inventory carried
*continuously across folds* so no cost ever escapes the accounting.

## 4. What was verified against planted truth

**Comparative static 1 — trading speed falls in cost (GP's core
prediction), monotonically and deterministically:** trained at per-side
costs of 0 / 40 / 100 / 150 bps, the learned speed came out
$\gamma = 0.64 / 0.30 / 0.16 / 0.11$ — identical across seeds to three
decimals. The agent *discovers* "trade slower when trading is expensive"
from data; nothing in the code tells it to.

**Comparative static 2 — the myopic control:** out of sample, at meaningful
cost, the learned-speed policy beats the $\gamma = 1$ full-rebalance policy
*on its own aim portfolio* — smoothing pays for itself net of the gross
return it sacrifices. (Pipeline table `agent_vs_myopic.csv`; the control
reuses the identical learned $\theta$, so the comparison isolates the speed
dial alone.)

**A null result, reported rather than buried:** GP theory also predicts the
aim should *tilt toward persistent signals* as costs rise ("aim in front of
the target"). In this policy class it does not — the value/momentum loading
ratio is flat in cost (0.410 → 0.391 across the full cost sweep). On
inspection the theory agrees with the data: the tilt result requires
**signal-specific trading speeds**, and with one shared $\gamma$ the EMA
attenuates each signal's return contribution and its risk contribution
nearly proportionally, so the tilt cancels. Our planted-truth harness
adjudicated a theoretical prediction and located its precise policy-class
dependence. The test suite encodes this as a *documented null*, and the
extension it motivates — per-signal speeds $\gamma_k$, GP's full structure —
is the top item on the agent's ladder.

## 4b. Measured results at pipeline scale

**Walk-forward at the config's 10 bps** (continuous inventory, 168 OOS
months): agent net Sharpe **2.96** at turnover 0.35 with mean learned
$\gamma = 0.84$, vs myopic **2.96** at turnover 0.42 — a tie, and a
theory-consistent one: at cheap costs the optimal speed is near-myopic, and
the learned policy *nests* the myopic one, recovering it when smoothing
doesn't pay. The value of the speed dial is a **curve**, mapped by
`scripts/agent_cost_sweep.py` (table `agent_cost_sweep.csv`):

| cost (bps) | learned γ | net-Sharpe advantage vs myopic |
|---|---|---|
| 0 | 0.92 | −0.02 |
| 10 | 0.83 | +0.00 |
| 25 | 0.60 | +0.13 |
| 50 | 0.39 | +0.54 |
| 100 | 0.21 | **+1.64** (myopic is *negative*, −0.69: smoothing is the difference between a viable strategy and none) |

**Per-signal speeds (`MultiSpeedPolicyAgent`), implemented and probed:** at
60 bps the multi-speed agent beats the myopic control decisively out of
sample (monthly Sharpe 0.43 vs 0.22), and the dominant fast signal's speed
falls hard with cost ($\gamma_{\text{mom}}: 0.70 \to 0.27$). Two honest
findings travel with it: (i) the raw-$\theta$ slow-signal tilt is *still*
null — and the probe exposed why the metric itself is wrong in a
multi-speed world: a signal with $\gamma_k \approx 0$ has a frozen,
near-zero EMA, so its $\theta_k$ is economically meaningless (the placebo
parked 0.2 of raw $\theta$ there, at zero effective exposure). The correct
tilt metric is **effective exposure** $\theta_k \cdot \mathrm{sd}(u_k)$,
which zeroes the placebo automatically and is the lens pre-registered for
the real-data test; (ii) per-signal speeds introduce identifiability slack
($\theta$ unpinned where $\gamma \to 0$) — a ridge penalty on $\theta$
is the documented fix.

## 5. Falsifiable real-data predictions (pre-registered, as with PULSE)

1. Learned $\gamma$ decreases in the configured cost level on the OSAP
   panel, tracing a smooth curve as cost is swept.
2. The agent beats the myopic control net of costs at realistic (≥10 bps)
   cost levels, with the margin growing in cost.
3. With per-signal speeds (now implemented), the **effective-exposure**
   tilt $\theta_k \cdot \mathrm{sd}(u_k)$ shifts toward persistent
   signals as cost rises on real data — the raw-$\theta$ version of this
   prediction is already refuted on synthetic data (§4b), so this is the
   surviving, sharper form.
4. Failure of (2) would indicate monthly rebalancing is too coarse for the
   speed dial to matter at realistic costs — reportable either way.

## 6. Honest limits

Linear costs only (no square-root market impact — the agent would *over*-
trade large positions under real impact); previous weights not
drift-adjusted between rebalances (shared simplification with the
evaluation stack; backlog item 11); complete-panel requirement (mask-based
handling of entering/leaving stocks is needed for real CRSP data); a single
risk view (Sharpe on net returns — no explicit factor-risk or drawdown
term); and the standing caveat of this whole repo: the synthetic world is a
kind laboratory, and every number above is a validation of *mechanism*, not
a promise about markets.

## 8. Three proposed upgrades, adjudicated and built

**(a) "Write the real trading fee into the loss" — was already the design.**
The trained objective has been net Sharpe from day one: the per-side fee
sits inside the differentiated loss via the smoothed absolute value, and
the cost sweep (§4b) is that fee reshaping the policy. The genuine upgrade
in this direction, now implemented: **square-root market impact**
(`impact_bps`) — a convex $|\Delta w|^{3/2}$ cost term, since real fees are
not linear in trade size. Verified: adding impact lowers both the learned
speed and realized turnover (tested against planted truth).

**(b) MSRR — implemented as closed form, and it validates the agent.**
`src/agent/msrr.py` computes the max-Sharpe blend analytically over
characteristic-managed portfolio returns ($\theta^* \propto \Sigma^{-1}\mu$,
equivalently a regression of the constant 1 on factor returns — hence the
name). The payoff is the validation: at zero cost the numerically-trained
agent recovers the closed form with **cosine similarity 1.000** and gross
Sharpe agreement to the third decimal (1.204 vs 1.205), and MSRR assigns
the placebo ≈0 weight. The GP decomposition is now explicit and exact:
**aim = MSRR, execution = learned partial adjustment.**

**(c) Composing the models — the largest measured gain in the repo's agent
line.** PULSE's forecasts are structurally point-in-time (filter state at
$t$ uses observations through $t-1$), making it the leakage-safe
composition; hyperparameters are selected on the first train window only
(`scripts/compose_pulse_agent.py`). Result, identical walk-forward and
costs: **composed agent net Sharpe 5.35 vs 2.96 for the static-blend
baseline** — the decay-aware, interaction-aware aim nearly doubles
risk-adjusted performance at similar turnover. IC-Net composition is
deliberately NOT bolted on: its in-sample predictions would leak into
agent training without purged stacking (documented as the requirement).

**A bug worth its documentation:** the first composition run produced a
*negative* Sharpe from a forecast with excellent IC. Diagnosis: with the
aim L1-normalized, $\theta$'s scale is a **null direction** of the
objective, and Adam's scale-free updates random-walk null directions — for
$K=1$ this silently flipped the book's sign, whereupon the agent rationally
learned $\gamma \to 0$ to freeze a losing book. Fix: project $\theta$ to
the unit L1 sphere after each step (kills the null direction; perpendicular
gradients unaffected; all 17 tests pass and the $K=6$ baselines are
unchanged to three decimals). Lesson worth telling in interviews:
optimizers exploit symmetries you forgot you created.

## 9. Extension ladder

Per-signal trading speeds — **implemented** (§4b); MSRR closed-form aim —
**implemented** (§8b); square-root impact — **implemented** (§8a); PULSE
composition — **implemented** (§8c); next: effective-exposure tilt on real
data; purged-stacking IC-Net composition; a ridge penalty on $\theta$;
remaining square-root impact in the differentiable cost term;
drift-adjusted inventory; plugging PULSE's filtered efficacies in as the
score (composing the two custom models: time-varying alpha *and*
cost-aware execution); a factor-risk penalty in the objective; and daily
frequency, where the speed dial has far more room to matter.
