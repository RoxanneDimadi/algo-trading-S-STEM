"""MSRR: Maximum Sharpe Ratio Regression (Kelly-Malamud), in closed form.

The idea
--------
Let each stock's weight be linear in its signals: w_{i,t} = theta' z_{i,t}
(demeaned per date, so the book is dollar-neutral). Then the portfolio
return collapses onto K "characteristic-managed portfolio" (factor) returns:

    r_p(t) = sum_i (theta' z_{i,t}) y_{i,t} = theta' f_t,
    f_{k,t} = sum_i z_{k,i,t} y_{i,t}.

Maximizing the Sharpe ratio of theta' f_t over theta is the classic
mean-variance problem on the factor returns, with closed-form solution

    theta* ∝ (Sigma_f + lambda I)^{-1} mu_f            (ridge-shrunk).

Why "Regression": regressing the constant 1 on f_t gives
(F'F)^{-1} F' 1 ∝ (Sigma + mu mu')^{-1} mu, which by Sherman-Morrison
points in the SAME direction as Sigma^{-1} mu -- max-Sharpe weights ARE a
regression coefficient. Hence the name.

Role in this repo
-----------------
1. A closed-form, optimization-free AIM portfolio (gross-Sharpe-optimal in
   the linear class) -- exactly the "Markowitz aim" of Garleanu-Pedersen,
   making the agent's decomposition explicit: aim = MSRR, execution =
   learned partial adjustment.
2. A validation target: at zero cost the differentiable agent's learned
   theta should agree with theta* (tested against planted truth).
3. A fast warm start / benchmark for real data.

Caveats (honest): closed form is exact for the UNNORMALIZED linear book;
the agent's per-date L1 normalization changes date weighting slightly, so
agreement is near, not exact. Ridge lambda matters when K grows (the
virtue-of-complexity literature lives exactly here); with K = 6 a small
lambda suffices.
"""
from __future__ import annotations

import numpy as np


def factor_returns(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """f_{t,k} = sum_i ztilde_{k,i,t} * y_{i,t}, with per-date demeaned z.

    Z: (K, T, N), Y: (T, N)  ->  F: (T, K)
    """
    Zt = Z - Z.mean(axis=2, keepdims=True)
    return np.einsum("ktn,tn->tk", Zt, Y)


def msrr_theta(Z: np.ndarray, Y: np.ndarray, ridge: float = 1e-4):
    """Closed-form max-Sharpe blend over the factor returns.

    Returns (theta, info) with theta scaled to unit L1 norm (only the
    direction matters downstream -- the agent's aim normalization removes
    scale). info carries the factor Sharpe achieved in-sample.
    """
    F = factor_returns(Z, Y)
    mu = F.mean(axis=0)
    Sig = np.cov(F, rowvar=False)
    lam = ridge * np.trace(Sig) / len(mu)
    theta = np.linalg.solve(Sig + lam * np.eye(len(mu)), mu)
    s = np.abs(theta).sum()
    theta = theta / (s if s > 0 else 1.0)
    port = F @ theta
    info = {"monthly_sharpe_factor_space": float(port.mean() / (port.std() + 1e-12)),
            "mu": mu, "n_periods": len(F)}
    return theta, info
