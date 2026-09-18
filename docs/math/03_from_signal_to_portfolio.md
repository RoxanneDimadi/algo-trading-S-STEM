# 3. From Signal to Portfolio: The Economics as Algebra

The IC is the statistical test. The **long-short portfolio** is the economic
one: can the signal's ordering be turned into money? This chapter derives the
portfolio's properties and then does the most satisfying thing in the whole
lesson plan: it **predicts the pipeline's measured 10.17% annual return from
four numbers in the config file.**

## 3.1 A portfolio is a dot product

Assign each stock a **weight** $w_i$ (fraction of capital; negative =
**short**, i.e. borrow the stock, sell it, profit if it falls). The
portfolio's return over the month is
$$
r_p \;=\; \sum_i w_i\, y_i \;=\; \langle w, y\rangle ,
$$
because each dollar of weight earns that stock's return. The repo's
constructor (`src/evaluation/portfolio.py::score_to_weights`) sorts each
date's stocks into 5 **quintiles** by score and sets
$$
w_i = \begin{cases} +1/n_{\text{top}} & i \in \text{top quintile}\\[2pt]
-1/n_{\text{bot}} & i \in \text{bottom quintile}\\[2pt] 0 & \text{else,}\end{cases}
$$
i.e. \$1 long the best fifth, \$1 short the worst fifth.

## 3.2 Theorem (dollar neutrality): the market drops out

**Theorem.** If $\sum_i w_i = 0$, then adding any constant $c$ to *every*
stock's return leaves the portfolio return unchanged:
$\langle w,\, y + c\mathbf 1\rangle = \langle w, y\rangle$.

**Proof.** $\langle w, y + c\mathbf 1\rangle = \langle w,y\rangle + c\sum_i w_i = \langle w,y\rangle$. ∎

The weights above sum to zero by construction ($+1$ and $-1$). So a month in
which *everything* rises 8% contributes nothing: the long leg's gain is the
short leg's loss. Compare Theorem A of chapter 2: the portfolio is
indifferent to the return *level* in exactly the way correlation is — the
first half of the argument that correlation is the natural training
objective (completed in chapter 7).

## 3.3 A tool: the mean of the top slice of a bell curve

To predict the strategy's return we need: *what is the average signal value
inside the top quintile?* The synthetic returns are generated from Gaussian
(bell-curve) signals, and sorting preserves order, so we need the mean of a
standard normal *conditional on being in its top 20%*.

**Lemma.** For standard normal $Z$ with density $\phi(z) = \tfrac{1}{\sqrt{2\pi}}e^{-z^2/2}$
and CDF $\Phi$: $\;E[Z \mid Z > a] = \dfrac{\phi(a)}{1 - \Phi(a)}$.

**Proof.** The key is that $\phi'(z) = -z\,\phi(z)$ (differentiate the
exponential). Therefore
$\int_a^\infty z\,\phi(z)\,dz = \big[-\phi(z)\big]_a^\infty = \phi(a)$.
Dividing by the probability of the event, $P(Z>a) = 1-\Phi(a)$, gives the
conditional mean. ∎

Top quintile: $a = \Phi^{-1}(0.8) = 0.8416$, so
$E[Z \mid \text{top } 20\%] = \phi(0.8416)/0.2 = 0.2800/0.2 = \mathbf{1.400}$.
By symmetry the bottom quintile averages $-1.400$: the **long–short spread in
signal units is 2.80**.

## 3.4 The flagship prediction: 10.14% vs measured 10.17%

The planted model (index page) says $r_{i,t+1} = \beta\, z_{i,t} + (\text{terms
with mean 0 in both legs})$: the interaction has mean zero given the sort
(the *other* signal is independent), market exposure $b_i$ is independent of
$z$ so both legs average $b \approx 1$ and cancel by §3.2, and noise averages
out. So the expected monthly long–short return is
$$
E[r_p] \;=\; \beta \cdot \big(E[z\mid\text{top}] - E[z\mid\text{bot}]\big) \;=\; 2.80\,\beta .
$$

Now the config numbers for `sig_momentum`: $\beta = 0.0040$ in-sample,
stepped to $0.70\beta$ after 2012-12 and $0.45\beta$ after 2014-12. The panel
spans 299 months of which about 155 are in-sample, 24 post-sample, 120
post-publication, so the *time-averaged* multiplier is
$$
\frac{155(1.0) + 24(0.70) + 120(0.45)}{299} \;=\; 0.755 .
$$
Prediction: $12 \times 2.80 \times 0.0040 \times 0.755 = \mathbf{0.1014}$,
i.e. **10.14% per year, gross**.

**Check (repo).** `results/tables/signal_evaluation.csv` measures
`sig_momentum` gross annual return $= \mathbf{0.1017}$. The derivation and
the code agree to 0.3% — with *zero* fitted parameters. This one check
exercises the generator, the rank sort, the weight constructor, the forward-
return alignment, and the annualization all at once. (Volatility can be
predicted the same way — noise averaging $\sigma/\sqrt{n_{\text{leg}}}$ per
leg plus a small market-beta mismatch term — giving monthly $\approx 0.0115$
and hence in-sample Sharpe $\approx \sqrt{12}\cdot(2.8\times0.004)/0.0115 \approx 3.4$;
measured in-sample Sharpe: 3.64. Same machinery, ~7% agreement; the gap is
sampling error plus the small terms we dropped.)

**Predicting the IC too.** The per-date correlation between $z$ and $r$ under
the planted model is $\beta$ divided by the cross-sectional return spread:
$\rho \approx \beta / \sqrt{\beta_{\text{all}}^2\text{-terms} + b\text{-dispersion}^2 E[m^2] + \sigma_\varepsilon^2}
= 0.0040/0.0813 = 0.049$ in-sample (Pearson). Two adjustments: Spearman on
near-Gaussian data is slightly smaller (factor $\tfrac{6}{\pi}\arcsin(\rho/2) \approx 0.955\rho$
for small $\rho$ — Stated, classical result for bivariate normals), and the
decay multiplier 0.755 applies. Prediction:
$0.049 \times 0.955 \times 0.755 = \mathbf{0.0355}$. Measured: **0.0374**.
Within sampling error (chapter 2 put the standard error at $\approx 0.0026$).

## 3.5 Turnover and costs: the algebra of friction

Weights change each month; trading costs money. With weight vectors $w_t$
(pivoted stock-by-date), the **traded notional** is
$\text{traded}_t = \sum_i |w_{t,i} - w_{t-1,i}|$ — every dollar bought or
sold — and the net return is
$$
r^{\text{net}}_t \;=\; r^{\text{gross}}_t \;-\; \text{traded}_t \times \frac{\text{cost}_{\text{bps}}}{10{,}000}.
$$
"One-way turnover" $= \text{traded}_t/2$ (a \$1 sale funding a \$1 purchase is
\$2 traded, one repositioning). This is deliberately the simplest defensible
cost model; `docs/03_improvement_backlog.md` lists its two honest
refinements (drift correction; size-dependent costs).

## 3.6 Persistence, staleness, and the law IC(k) ≈ ρᵏ · IC(0)

The synthetic signals follow an **AR(1)**:
$z_t = \rho z_{t-1} + \sqrt{1-\rho^2}\,\epsilon_t$ with fresh noise
$\epsilon_t$ (variance 1).

**Claim 1 (variance is stable at 1).** If $\mathrm{Var}(z_{t-1}) = 1$ then
$\mathrm{Var}(z_t) = \rho^2\cdot 1 + (1-\rho^2)\cdot 1 = 1$ (independence of
$\epsilon_t$, §1.2). ∎

**Claim 2 (correlation across k months is ρᵏ).**
$\mathrm{Cov}(z_t, z_{t-1}) = \rho\,\mathrm{Var}(z_{t-1}) = \rho$
(the noise term is uncorrelated with the past); iterate $k$ times to get
$\mathrm{Corr}(z_t, z_{t-k}) = \rho^k$. ∎

**Consequence.** A $k$-month-old signal is (in the correlation sense) a
$\rho^k$-strength copy of today's plus unrelated noise, so its predictive
correlation is scaled: $\mathrm{IC}(k\text{-stale}) \approx \rho^k\,\mathrm{IC}(\text{fresh})$.
This is the mathematical content of the "Anomaly Time" lesson (stale
formation hides real signal), turned into a *rate*.

**Check (repo).** `results/tables/staleness_profile.csv`, momentum
($\rho = 0.90$): measured IC ratios at lags 1–4 are **0.93, 0.77, 0.65, 0.52**
vs predicted $0.90, 0.81, 0.73, 0.66$ — matching at short lags and decaying
slightly faster at long lags (the rank transform and the time-varying decay
multipliers both shave long-lag persistence; each measured ratio carries a
standard error of roughly $\pm 0.09$). The qualitative law — geometric decay
at rate set by the signal's persistence — is exactly what the table shows,
and it is also why slow signals (`sig_value`, $\rho = 0.98$) trade little
(measured one-way turnover 0.23) while fast ones trade a lot (momentum 0.51):
**turnover is persistence, seen from the portfolio's side.**
