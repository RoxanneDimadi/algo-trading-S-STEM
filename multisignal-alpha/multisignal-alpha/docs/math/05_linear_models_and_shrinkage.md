# 5. Linear Models: Least Squares, Alpha, and Why Shrinkage Wins in Noise

## 5.1 Least squares from scratch (the normal equations)

Given features $x_i$ (vector, includes a leading 1 for the intercept) and
targets $y_i$, ordinary least squares (OLS) picks coefficients $\beta$
minimizing the sum of squared errors
$L(\beta) = \sum_i (y_i - x_i^\top \beta)^2 = \|y - X\beta\|^2$.

**Derivation.** $L$ is a smooth bowl in $\beta$; at the minimum its gradient
is zero. Expanding $L = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X\beta$
and differentiating: $\nabla L = -2X^\top y + 2X^\top X\beta = 0$, so
$$
\boxed{\;\hat\beta = (X^\top X)^{-1} X^\top y\;}
$$
(the **normal equations**). In the one-feature case this collapses, after a
little algebra, to two formulas worth memorizing:
$$
\hat\beta_1 = \frac{\mathrm{Cov}(x,y)}{\mathrm{Var}(x)}, \qquad \hat\beta_0 = \bar y - \hat\beta_1 \bar x .
$$
So a regression slope *is* a rescaled covariance — regression, correlation,
and the IC are one family. And regressing on a constant alone gives
$\hat\beta_0 = \bar y$: §4.3's "mean test as regression" identity, proved.

## 5.2 Alpha: the intercept with a job title

Chapter 3's strategy returns might secretly be repackaged *known* risk
premia. The test (`src/evaluation/factor_controls.py`): regress strategy
returns on factor returns $f_t$ (market, size, value, …),
$$
r^{\text{strat}}_t = \alpha + \beta^\top f_t + e_t ,
$$
with Newey–West errors. $\beta^\top f_t$ is the part explained by *rentable
exposures anyone can buy*; the intercept $\alpha$ is the average return left
over — the claim to genuine information. $R^2$ (the fraction of variance the
factors explain) completes the picture: **surviving $\alpha$ with low $R^2$
is the credible pattern; vanished $\alpha$ means the "signal" was a known
factor in disguise** (worth reporting, honestly, either way). One timing trap
the code handles: returns indexed by *formation* date must be paired with
factor returns over the *same holding window* (`align="formation"` shifts the
factor panel accordingly).

## 5.3 The bias–variance decomposition (proved)

Why not always fit the most flexible model? Let the truth be
$y = f(x) + \varepsilon$ with noise variance $\sigma^2$, and let $\hat f$ be
a model fit on a random training sample. For a fixed test point $x$:

**Theorem.** $\;E\big[(y - \hat f(x))^2\big] = \underbrace{\big(E[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}\big(\hat f(x)\big)}_{\text{variance}} + \sigma^2 .$

**Proof.** Add and subtract $E[\hat f]$ inside the square:
$y - \hat f = \varepsilon + (f - E[\hat f]) + (E[\hat f] - \hat f)$.
Square and take expectations. The three cross terms die: $\varepsilon$ is
independent of everything with mean zero (kills two), and
$E\big[(f - E[\hat f])(E[\hat f] - \hat f)\big] = (f - E[\hat f])\cdot E[E[\hat f] - \hat f] = 0$
(the first factor is a constant, the second has mean zero). What survives is
the three claimed terms. ∎

**Why finance sits at the extreme of this tradeoff.** In return prediction
the signal-to-noise ratio is brutal — chapter 3 computed the *true* model's
correlation ceiling at about 0.05–0.10 monthly, i.e. $R^2$ well under 1%.
When $\sigma^2$ dwarfs the signal, the variance term dominates the bias term,
so **deliberately biased, low-variance estimators win**. That is the entire
case for shrinkage.

## 5.4 Ridge and lasso, solved exactly in the clean case

The **elastic net** (the repo's linear benchmark, `make_linear`) minimizes
$\|y - X\beta\|^2 + \lambda_2\|\beta\|^2 + \lambda_1\|\beta\|_1$ — squared
error plus an L2 ("ridge") and an L1 ("lasso") penalty. In the *orthonormal*
case ($X^\top X = I$; think uncorrelated standardized features) both pieces
solve in closed form, and the closed forms teach exactly what each penalty
does. Let $\hat\beta = X^\top y$ be the OLS solution.

**Ridge (L1 off).** Minimize $\|y - X\beta\|^2 + \lambda\|\beta\|^2$.
Gradient: $-2X^\top y + 2\beta + 2\lambda\beta = 0$, so
$$
\hat\beta^{\text{ridge}} = \frac{\hat\beta}{1+\lambda}:
$$
**every coefficient is shrunk toward zero by the same factor.** Pure
variance reduction, paid for with a little bias — §5.3's trade made explicit.

**Lasso (L2 off).** The problem separates coordinate by coordinate: minimize
$g(\beta_j) = (\beta_j - \hat\beta_j)^2 + \lambda|\beta_j|$ (up to constants).
For $\beta_j > 0$: $g' = 2(\beta_j - \hat\beta_j) + \lambda = 0 \Rightarrow
\beta_j = \hat\beta_j - \lambda/2$, valid only if positive; symmetric case
for negative; otherwise the minimum is at the kink $\beta_j = 0$. Compactly:
$$
\hat\beta_j^{\text{lasso}} = \mathrm{sign}(\hat\beta_j)\,\max\!\big(|\hat\beta_j| - \tfrac{\lambda}{2},\, 0\big)
$$
— **soft thresholding**: small coefficients are set *exactly* to zero
(automatic signal selection), large ones shrunk by a constant. The elastic
net blends both behaviors; the repo keeps its penalties tiny because the
features are already few and rank-normalized — the benchmark's job is "best
honest linear combination," not aggressive selection.

## 5.5 What the linear model *cannot* do — the cliffhanger

However shrunk, $\hat y = \beta^\top z$ is **additive**: momentum's
contribution is the same regardless of the value signal's level. The
synthetic truth contains a term $\beta_{\text{int}}\, z_{\text{mom}}\, z_{\text{val}}$
whose whole point is that it is *not* additive. Chapter 6 proves no additive
model can represent it — and shows why trees can. That gap is exactly the
measured OOS difference between the elastic net (IC 0.043) and the
tree/net models (0.060–0.070).
