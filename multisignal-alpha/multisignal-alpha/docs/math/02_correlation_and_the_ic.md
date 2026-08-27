# 2. Correlation as Geometry, Ranks, and the Information Coefficient

## 2.1 Correlation is a cosine

Take one date's cross-section: predictions $p = (p_1,\dots,p_n)$ and forward
returns $y = (y_1,\dots,y_n)$, one entry per stock. Demean each
($\tilde p = p - \bar p\mathbf{1}$, $\tilde y = y - \bar y\mathbf{1}$, where
$\mathbf 1$ is the all-ones vector). The **sample correlation** is
$$
\rho(p, y) \;=\; \frac{\langle \tilde p, \tilde y\rangle}{\|\tilde p\|\,\|\tilde y\|},
$$
which is *literally the cosine of the angle* between the two demeaned vectors
in $n$-dimensional space (that is the definition of the angle between
vectors). Perfectly aligned: $\rho = 1$. Opposite: $-1$. Perpendicular: $0$.
This geometric picture makes the next two theorems one-line proofs, and it is
the exact expression IC-Net differentiates in chapter 7.

## 2.2 Two invariance theorems (small proofs, large consequences)

**Theorem A (translation invariance).** $\rho(p + c\mathbf 1,\, y) = \rho(p, y)$
for any constant $c$.

**Proof.** Demeaning kills constants: $(p + c\mathbf 1) - \overline{(p+c\mathbf 1)}\,\mathbf 1
= p + c\mathbf 1 - (\bar p + c)\mathbf 1 = \tilde p$. The formula only sees
$\tilde p$. ∎

**Theorem B (positive scale invariance).** $\rho(p,\, \lambda y) = \rho(p, y)$
for any $\lambda > 0$ (and likewise in $p$).

**Proof.** $\widetilde{\lambda y} = \lambda\tilde y$, so the numerator gains a
factor $\lambda$ and the denominator gains $\|\lambda \tilde y\| = \lambda\|\tilde y\|$;
they cancel. ∎

*Why these matter here:* Theorem A says correlation cannot reward predicting
the *level* of a date's returns ("is this an up month?") — only the ordering
across stocks. Theorem B says a violently volatile month, with all its
returns stretched by a common factor, contributes the same correlation as a
calm one. Chapter 3 proves the long-short portfolio has exactly the same two
indifferences — which is precisely why chapter 7 argues correlation is the
*right training objective* and mean-squared-error is the wrong one.

## 2.3 Ranks, and deriving Spearman's famous formula

**Spearman correlation** = Pearson correlation applied to the *ranks* of the
data (smallest value gets rank 1, …, largest gets rank $n$; assume no ties).
Robustness is the reason: replacing the largest return by something 10× as
large changes Pearson a lot but changes ranks not at all.

Textbooks state Spearman's shortcut $\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$
where $d_i$ is the difference between stock $i$'s two ranks. Here is where it
comes from — a pleasant exercise in the definitions.

**Step 1: mean and variance of ranks.** Ranks are the numbers $1,\dots,n$ in
some order, so $\bar R = \frac1n\sum_{k=1}^n k = \frac{n+1}{2}$ and, using
$\sum k^2 = \frac{n(n+1)(2n+1)}{6}$,
$$
\mathrm{Var}(R) = \frac1n\sum k^2 - \bar R^2
= \frac{(n+1)(2n+1)}{6} - \frac{(n+1)^2}{4}
= \frac{(n+1)\big(2(2n+1) - 3(n+1)\big)}{12}
= \frac{n^2-1}{12}.
$$

**Step 2: covariance via the $d_i$'s.** Let $R_i, S_i$ be the two rank lists
and $d_i = R_i - S_i$. Expanding $d_i^2 = R_i^2 - 2R_iS_i + S_i^2$ and summing,
$$
\sum R_iS_i = \tfrac12\Big(\sum R_i^2 + \sum S_i^2 - \sum d_i^2\Big)
= \frac{n(n+1)(2n+1)}{6} - \tfrac12\sum d_i^2 ,
$$
because both rank lists contain the same numbers $1..n$. Then
$$
\mathrm{Cov}(R,S) = \frac1n\sum R_iS_i - \bar R\,\bar S
= \frac{(n+1)(2n+1)}{6} - \frac{\sum d_i^2}{2n} - \frac{(n+1)^2}{4}
= \frac{n^2-1}{12} - \frac{\sum d_i^2}{2n}.
$$

**Step 3: divide.** $\rho_s = \mathrm{Cov}/\mathrm{Var} = 1 - \dfrac{6\sum d_i^2}{n(n^2-1)}$. ∎

The code never uses the shortcut (`scipy.stats.spearmanr` computes Pearson on
ranks directly, ties handled by average ranks), but the derivation teaches
the real content: **Spearman is nothing new — it is Pearson after the rank
transform**, inheriting §2.2's invariances and adding outlier immunity.

## 2.4 The Information Coefficient (IC): definition and design choices

$$
\mathrm{IC}_t \;=\; \text{Spearman correlation, across stocks } i \text{, between } z_{i,t} \text{ and } r_{i,t+1}.
$$

One number per date; the headline statistic is the time-series mean
$\overline{\mathrm{IC}}$, with dispersion $\sigma_{\mathrm{IC}}$ and the
stability ratio $\mathrm{ICIR} = \overline{\mathrm{IC}}/\sigma_{\mathrm{IC}}$
(implemented in `src/evaluation/ic.py`). Two design choices carry the theory:

**(a) Forward, never contemporaneous.** The pair is (signal known at $t$,
return over $t\!\to\!t{+}1$). Correlating a signal with the *same* month's
return measures description, not prediction — chapter 8 quantifies exactly
how spectacular that mistake looks.

**(b) Per date, then average — never pooled.** Suppose you instead pooled all
$(z, r)$ pairs across dates into one giant correlation. Write each variable
as (its date average) + (deviation from the date average). The pooled
covariance then splits into a *between-dates* part — do months with a
high average signal have high average returns? — and a *within-date* part,
which is the only part a market-neutral portfolio can harvest. Computing the
correlation within each date and averaging (the IC) isolates the within-date
part by construction: every date's demeaning removes that date's market
level (Theorem A), so market-direction effects contribute exactly zero.

**Sampling noise under "no skill" (used repeatedly later).** If a signal is
pure noise, its per-date IC is a correlation between $n$ independent pairs;
a standard result (Stated; see any mathematical-statistics text) is that its
standard deviation is $\approx 1/\sqrt{n-1}$. With $n = 500$ stocks:
$\sigma_{\mathrm{IC}} \approx 0.0448$ per date. Averaging $T = 298$
independent dates divides the noise by $\sqrt{T}$ (chapter 4 proves this):
the mean IC of a *useless* signal fluctuates with standard error
$0.0448/\sqrt{298} \approx 0.0026$.

**Check (repo).** The placebo `sig_dead` measured $\overline{\mathrm{IC}} = 0.0048$
— about 1.9 of those standard errors from zero, i.e. exactly the size of
fluctuation the null predicts, and correctly *not* significant. The planted
signals measured 0.010–0.037, i.e. 4–11 standard errors: cleanly detected.
Chapter 3 predicts the 0.037 itself from the config.
