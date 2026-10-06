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
