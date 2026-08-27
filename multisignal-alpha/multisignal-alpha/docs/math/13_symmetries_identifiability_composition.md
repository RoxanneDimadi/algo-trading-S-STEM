# 13. Symmetries, Identifiability, and Composition
## The mathematics of the bug we hit — and of the composition that worked

This chapter is different in kind: its centerpiece theorem *predicts a
failure that actually happened* during development (the sign-flipped
composition, docs/06 §8), and its final derivation turns the composed
agent's headline number into a quantity you can predict from a correlation.
Debugging, done properly, is applied mathematics.

## 13.1 The null-direction theorem: symmetry creates a flat direction

The agent's aim is scale-invariant by construction:
$A(\theta) = 2\,\widetilde{s}/\|\widetilde{s}\|_1$ with
$s = \sum_k \theta_k z_k$, so replacing $\theta \to c\theta$ (any $c>0$)
multiplies $\widetilde s$ and $\|\widetilde s\|_1$ by the same $c$, which
cancels.

**Theorem.** The training objective satisfies $J(c\theta, g) = J(\theta, g)$
for all $c > 0$, and consequently the gradient is everywhere orthogonal to
$\theta$:
$$
\theta^\top \nabla_\theta J \;=\; 0 .
$$

**Proof.** Every quantity downstream of the aim (weights, returns, costs,
Sharpe) depends on $\theta$ only through $A(\theta)$, which we just showed
is invariant, so $J(c\theta) = J(\theta)$. Differentiate this identity with
respect to $c$ at $c = 1$ (chain rule):
$\frac{d}{dc}J(c\theta)\big|_{c=1} = \theta^\top\nabla_\theta J = 0$. ∎

(This is Euler's homogeneous-function argument for degree-zero functions —
a symmetry always manufactures a direction the gradient cannot see, the
optimization cousin of Noether's principle.)

**Corollary (the K = 1 trap).** With a single feature, every $\theta > 0$
is a positive rescaling of every other, so $J$ is *constant* on each
half-line: it depends only on $\mathrm{sign}(\theta)$. The gradient in
$\theta$ is exactly zero everywhere except the kink at 0 (the softabs
$\varepsilon$ turns "exactly zero" into "numerical dust").

## 13.2 Why Adam random-walks a flat direction (the bug, derived)

A flat direction sounds harmless — no gradient, no movement. Not under
Adam. Its update is $\Delta p = \mathrm{lr}\cdot \hat m/(\sqrt{\hat v} + \epsilon)$,
where $\hat m$ and $\hat v$ are moving averages of the gradient and its
square. Feed it the flat direction's gradient — tiny numerical noise
$\delta_t$ with no consistent sign — and both $\hat m \sim O(\delta)$ and
$\sqrt{\hat v} \sim O(\delta)$, so their **ratio is $O(1)$ with a random
sign**: the parameter takes steps of size $\approx$ lr *regardless of how
small the noise is*. Scale-freeness, Adam's virtue on real gradients, turns
"zero gradient" into "unit-speed diffusion." Over 150 epochs, a $\theta$
that started at $+0.2$ can drift across zero — flipping the sign of every
weight in the book. The agent then observes a *negative-alpha* book in
training and does the rational thing: learns $\gamma \to 0$ and freezes.
That is precisely the observed failure: a forecast with IC $+0.074$
producing OOS Sharpe $-6.19$ under $\gamma = 1$ (the exact mirror image of
the true book's $+6.49$) and a "learned" $\gamma$ of 0.003.

**The fix, and a lemma that it works.** After each Adam step, project
$\theta$ back to the unit L1 sphere, $\theta \leftarrow \theta/\|\theta\|_1$
— optimization on the quotient of the symmetry, which deletes the flat
direction while leaving all perpendicular (real) gradients untouched.

**Lemma (no sign flip).** With $K = 1$, projection keeps
$\theta \in \{-1, +1\}$, and a flip would require a single step of
magnitude $> 1$; Adam's step is bounded near lr $= 0.05 \ll 1$, so the sign
is stable. For $K \ge 2$ the projection is inert where it should be:
genuine gradients are orthogonal to $\theta$ (the Theorem), hence tangent
to the sphere already. ∎ *(The step bound is the standard heuristic
$|\hat m|/\sqrt{\hat v} \lesssim 1$ — labeled honestly: an argument, not a
worst-case proof.)*

**Check (repo).** Post-fix, the same composition runs at net Sharpe
$+5.35$; the $K = 6$ baselines are unchanged to three decimals (perpendicular
gradients dominated all along, as the Theorem says they should); all 17
tests pass. The transferable lesson: **every invariance you build into a
model is a direction your optimizer can wander; either quotient it out or
expect it to be explored.**

## 13.3 Identifiability: the multi-speed cousin of the same disease

The multi-speed agent (per-signal EMAs $u_k$ with speeds $\gamma_k$) hides
a second flat-ish direction. How large is $u_k$?

**Lemma.** For an i.i.d. unit-variance input, the EMA
$u_t = (1-\gamma)u_{t-1} + \gamma A_t$ has stationary variance
$$
\mathrm{Var}(u) \;=\; \gamma^2 \sum_{j\ge0} (1-\gamma)^{2j} \;=\; \frac{\gamma^2}{1-(1-\gamma)^2} \;=\; \frac{\gamma}{2-\gamma},
$$
which $\to 0$ as $\gamma \to 0$. (Persistent inputs change the constant,
not the limit — Stated.) ∎

So a signal whose speed learns its way to $\gamma_k \approx 0$ contributes
a component $u_k$ of vanishing size — and **its blend weight $\theta_k$
multiplies nothing**: unidentified, free to wander, exactly like the null
direction of §13.1 but signal-by-signal. This is why the probe found the
placebo "holding" 20–25% of raw $\theta$ at $\gamma \approx 0.007$: an
economically empty parameter parked at a meaningless value. The corrected
metric is **effective exposure** $\theta_k \cdot \mathrm{sd}(u_k)$, which
the Lemma shows zeroes any frozen signal automatically — and which is the
pre-registered lens for the slow-signal-tilt test on real data (docs/06
§5), replacing the raw-$\theta$ version this analysis refuted.

## 13.4 Composition is point-in-time by induction

**Proposition.** If a forecast satisfies $x_t \in \mathcal F_t$ (chapter
10 §10.5 proved this for PULSE: its state feeding date $t$ uses
observations through $t-1$), and the policy is any map
$w_t = \pi(w_{t-1}, x_t)$, then $w_t \in \mathcal F_t$ for all $t$.

**Proof.** Induction. $w_0 = 0 \in \mathcal F_0$. If
$w_{t-1} \in \mathcal F_{t-1} \subseteq \mathcal F_t$ and
$x_t \in \mathcal F_t$, then $w_t$, a function of two
$\mathcal F_t$-measurable inputs, is $\mathcal F_t$-measurable. ∎

Chains of point-in-time components are point-in-time — leakage cannot be
created by composition, only by a leaky component. (That is exactly why
IC-Net is *not* composed the same way: its in-sample fitted values are not
$\mathcal F_t$-measurable objects during training, so it needs purged
stacking first — the documented requirement, now with its reason proved.)

## 13.5 What a proportional book earns: the Pearson formula (with a check)

The agent's aim holds each stock in proportion to its demeaned forecast:
$w = 2\tilde f/\|\tilde f\|_1$. Its one-date gross return has a closed
form. Using $\tilde f^\top \mathbf 1 = 0$ (so the return level drops out,
ch. 3's dollar-neutrality) and the definition of correlation:
$$
w^\top y \;=\; \frac{2\,\tilde f^\top \tilde y}{\|\tilde f\|_1}
\;=\; \frac{2\,n\, s_f\, s_y\, \rho_t}{\|\tilde f\|_1}
\;\approx\; 2\sqrt{\tfrac{\pi}{2}}\; s_y\,\rho_t \;\approx\; 2.507\, s_y\, \rho_t,
$$
where the last step uses $\|\tilde f\|_1 \approx n\,s_f\sqrt{2/\pi}$ for a
roughly Gaussian cross-section ($E|Z| = \sigma\sqrt{2/\pi}$ — one integral,
same trick as ch. 3's truncated-mean lemma). Two readings:

1. **Proportional weights monetize the per-date *Pearson* correlation** —
   the exact quantity IC-Net trains on (ch. 7) — just as quantile books
   monetize ordering (ch. 3). The forecast's *levels* matter here, which is
   why a sign flip is catastrophic while rank-IC stays serenely positive:
   the diagnostic signature that cracked the bug (rank-IC $+0.074$, book
   $-6.19$ ⇒ the pattern is right and the *sign/plumbing* is wrong).
2. **It predicts the composition's headline.** Measured mean Pearson
   $\bar\rho = 0.0834$ and cross-sectional return spread
   $s_y \approx 0.081$ (ch. 3) give expected gross
   $\approx 2.507 \times 0.0834 \times 0.081 = 0.0169$/month $= 20.3\%$/yr.
   The composed agent measured **19.0% net** at turnover 0.395 — adding
   back its cost drag ($0.395 \times 2 \times 10\,\text{bps} \times 12
   \approx 0.9\%$) implies gross $\approx 19.9\%$. Predicted 20.3, implied
   19.9 — agreement to within the Gaussian-$\|\cdot\|_1$ approximation,
   with zero fitted parameters. The lesson plan's oldest habit, applied to
   its newest artifact.

## 13.6 Coda: the shape of the whole argument

Chapters 12–13 close a loop the repo has been drawing since chapter 1:
Cauchy–Schwarz proved correlation bounded (ch. 1), became the max-Sharpe
theorem (§12.2); the AR(1) lemma (ch. 3) became the capture ratio (§12.4);
the information-set bookkeeping (ch. 8, 10) became composition closure
(§13.4); and the per-date correlation — the IC that started as an
evaluation statistic — ended as the exact quantity a proportional book
earns (§13.5). One toolbox, reused until forecasting, filtering, and
trading are visibly the same mathematics wearing three jackets.
