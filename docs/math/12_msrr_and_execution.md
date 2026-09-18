# 12. MSRR and the Mathematics of Execution

Chapter 11 built the policy and its gradients. This chapter proves the two
closed forms that surround it: the **optimal aim** (Maximum Sharpe Ratio
Regression — solvable exactly, no optimizer needed) and the **capture
ratio** (exactly how much alpha a given trading speed harvests from a
signal of given persistence). Together they turn the cost sweep's measured
curve into derived structure — and they let the numerical agent be checked
against analytic truth, the same discipline every other chapter uses.

## 12.1 The factor collapse: a portfolio of stocks becomes a blend of factors

**Lemma.** If each stock's weight is linear in its (per-date demeaned)
signals, $w_{i,t} = \sum_k \theta_k \tilde z_{k,i,t}$, then the portfolio
return collapses onto $K$ numbers per date:
$$
r_p(t) \;=\; \sum_i w_{i,t}\, y_{i,t} \;=\; \sum_k \theta_k \underbrace{\Big(\sum_i \tilde z_{k,i,t}\, y_{i,t}\Big)}_{f_{k,t}} \;=\; \theta^\top f_t .
$$

**Proof.** Swap the two finite sums (linearity). ∎

The $f_{k,t}$ are **characteristic-managed portfolio returns**: what you
earn holding each stock in proportion to its signal-$k$ value. A universe of
500 stocks has just become a $K$-asset problem — the single most useful
dimension reduction in cross-sectional finance, and the reason the next
theorem has a closed form.

## 12.2 Theorem (MSRR): the max-Sharpe blend is $\Sigma^{-1}\mu$

Let $\mu = E[f_t]$ and $\Sigma = \mathrm{Cov}(f_t)$. Choose $\theta$ to
maximize the Sharpe ratio $S(\theta) = \theta^\top\mu / \sqrt{\theta^\top\Sigma\,\theta}$.

**Theorem.** $\theta^\* \propto \Sigma^{-1}\mu$, achieving
$S(\theta^\*) = \sqrt{\mu^\top\Sigma^{-1}\mu}$.

**Proof.** Substitute $x = \Sigma^{1/2}\theta$ and $b = \Sigma^{-1/2}\mu$
(the symmetric square root exists because $\Sigma$ is positive definite).
Then
$$
S = \frac{b^\top x}{\|x\|} \;\le\; \|b\|
$$
by Cauchy–Schwarz (chapter 1's theorem, reused verbatim), with equality iff
$x \propto b$, i.e. $\Sigma^{1/2}\theta \propto \Sigma^{-1/2}\mu$, i.e.
$\theta \propto \Sigma^{-1}\mu$; and $\|b\| = \sqrt{\mu^\top\Sigma^{-1}\mu}$. ∎

Read it: the optimal blend is the mean return of each factor, *deflated by
the risk it shares with the others* — high-mean factors get weight, but
redundant ones get discounted through $\Sigma^{-1}$.

## 12.3 Why "Regression": the Sherman–Morrison identity

MSRR's name comes from a lovely equivalence: **regressing the constant 1 on
the factor returns points at the same portfolio.** The OLS coefficient of
$1$ on $F$ (the $T\times K$ matrix of factor returns) is
$\hat\theta = (F^\top F)^{-1} F^\top \mathbf 1$, and
$F^\top\mathbf 1/T = \mu$, $F^\top F/T = \Sigma + \mu\mu^\top$ (the moment
identity $E[ff^\top] = \Sigma + \mu\mu^\top$). So we need
$(\Sigma + \mu\mu^\top)^{-1}\mu$.

**Lemma (Sherman–Morrison, rank-one update).**
$(A + uv^\top)^{-1} = A^{-1} - \dfrac{A^{-1}uv^\top A^{-1}}{1 + v^\top A^{-1}u}$.

**Proof.** Multiply the claimed inverse by $(A + uv^\top)$; with
$q = v^\top A^{-1} u$ the cross terms give
$uv^\top A^{-1}\big[1 - \tfrac{1}{1+q} - \tfrac{q}{1+q}\big] = 0$, leaving $I$. ∎

**Corollary.** $(\Sigma + \mu\mu^\top)^{-1}\mu = \dfrac{\Sigma^{-1}\mu}{1 + \mu^\top\Sigma^{-1}\mu} \;\propto\; \Sigma^{-1}\mu$.

**Proof.** Apply the lemma with $A=\Sigma$, $u=v=\mu$, then multiply by
$\mu$ and factor: $\Sigma^{-1}\mu\big[1 - \tfrac{q}{1+q}\big]$ with
$q = \mu^\top\Sigma^{-1}\mu$. ∎

So a *regression* — the humblest tool in the box, chapter 5's normal
equations — computes the maximum-Sharpe portfolio's direction exactly. In
practice $\mu$ and $\Sigma$ are noisy estimates, so the repo's
implementation solves $(\Sigma + \lambda I)^{-1}\mu$: ridge shrinkage,
chapter 5's bias–variance trade transplanted into portfolio space. (When
$K$ grows into the hundreds, how $\lambda$ interacts with $K > T$ is
precisely the virtue-of-complexity debate of the research-findings doc —
this corollary is that literature's front door.)

**Check (repo).** At zero cost the numerically-trained agent — Adam through
the full trading recursion, no knowledge of this chapter — recovers the
closed form: cosine similarity between learned $\theta$ and
$\Sigma^{-1}\mu$ direction $= \mathbf{1.000}$, gross Sharpe 1.204 vs the
closed form's 1.205, and both assign the placebo $\approx 0$ weight
(`test_msrr_closed_form_recovered_by_zero_cost_agent`). Optimizer and
theorem validate each other; the agreement is not exact because the agent's
per-date L1 normalization reweights dates slightly — a stated, understood
gap of 0.1%.

## 12.4 Theorem (capture ratio): what a trading speed costs in alpha

The partial-adjustment policy holds an EMA of past aims (ch. 11's theorem).
How much of a signal's *fresh* alpha does a stale EMA capture? Let the aim
track a single AR(1) signal $z_t$ with persistence $\rho$ (ch. 3), so
next-month alpha is $\beta z_t$, and let $w_t = \gamma\sum_{j\ge0}(1-\gamma)^j z_{t-j}$.

**Theorem.** Expected captured alpha is $\beta \cdot \kappa(\gamma,\rho)$ with
$$
\kappa(\gamma, \rho) \;=\; \gamma \sum_{j\ge0} (1-\gamma)^j \rho^j \;=\; \frac{\gamma}{1 - (1-\gamma)\rho}.
$$

**Proof.** $E[w_t\,\beta z_t] = \beta\gamma\sum_j (1-\gamma)^j E[z_{t-j}z_t]
= \beta\gamma\sum_j (1-\gamma)^j \rho^j$ by chapter 3's Claim 2
($\mathrm{Corr}(z_t, z_{t-j}) = \rho^j$); sum the geometric series. ∎

Sanity: $\kappa(1,\rho) = 1$ (myopic captures everything);
$\kappa \to 0$ as $\gamma \to 0$ (a frozen book captures nothing);
$\kappa$ increases in $\rho$ (slow signals forgive slow trading). Turnover,
meanwhile, *increases* in $\gamma$ (more chasing = more trading), so the
net objective is a tug-of-war
$\beta\,\kappa(\gamma,\rho) - c \cdot \text{turnover}(\gamma)$ whose
interior optimum moves **down** as $c$ rises — the derived skeleton of the
measured cost sweep ($\gamma: 0.92 \to 0.83 \to 0.60 \to 0.39 \to 0.21$ at
$c = 0/10/25/50/100$ bps). Put a number on the sweet spot: at the 25 bps
point the learned $\gamma = 0.60$ on a $\rho \approx 0.9$ blend gives
$\kappa = 0.60/(1 - 0.36) = 0.94$ — the agent kept **94% of the alpha while
cutting turnover from 0.42 to 0.27.** That ratio *is* the business case for
execution-aware trading, in one fraction.

## 12.5 Square-root impact: the cost of size, made differentiable

Linear cost ($c\cdot|\Delta w|$) says the thousandth dollar trades as
cheaply as the first. Markets disagree: the empirical **square-root law**
(Stated — one of the most replicated results in market microstructure) has
price impact growing like the square root of trade size, so total cost —
impact × quantity — grows like
$$
|\Delta w| \cdot |\Delta w|^{1/2} \;=\; |\Delta w|^{3/2},
$$
a **convex** cost. Convexity has one big behavioral consequence: marginal
cost rises with size, so optimal trades shrink and spread out — exactly the
verified comparative static (`test_sqrt_impact_slows_trading`: adding
impact lowered the learned speed 0.49 → 0.44 and turnover 0.22 → 0.20).
Differentiability is preserved by the same smoothing trick as chapter 11:
implement $|x|^{3/2}$ as $(x^2+\varepsilon)^{3/4}$, whose derivative
$\tfrac{3}{2}\,x\,(x^2+\varepsilon)^{-1/4}$ is the line in the code, and
which rolls smoothly through zero instead of kinking. One honest caveat
travels with the feature: the *coefficient* of impact is far harder to
estimate from data than a commission schedule, so on real data it is a
sensitivity parameter to sweep, not a constant to trust.
