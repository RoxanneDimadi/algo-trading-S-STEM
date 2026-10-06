"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

The problem it solves: try enough strategy variants and the best backtest
Sharpe is inflated by selection. The DSR asks: what is the probability that
the TRUE Sharpe exceeds zero, given (a) how many trials were run, (b) the
variance of Sharpe across those trials, and (c) the non-normality of the
strategy's returns?

Mechanics: compute the expected maximum Sharpe under the null across N
trials (SR*), then evaluate the Probabilistic Sharpe Ratio of the candidate
against that benchmark, with skew/kurtosis corrections.

ALL Sharpe inputs here are PER-PERIOD (monthly), not annualized -- mixing
frequencies is the classic implementation bug. `kurtosis` is FULL kurtosis
(normal = 3), matching the paper's formula.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int,
                         skew: float, kurtosis: float) -> float:
    """P(true SR > sr_benchmark | observed sr, n_obs, higher moments)."""
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr +
                    (kurtosis - 1.0) / 4.0 * sr**2))
    z = (sr - sr_benchmark) * np.sqrt(max(n_obs - 1, 1)) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(var_sr_across_trials: float, n_trials: int) -> float:
    """E[max SR] under the null, across n_trials independent tries."""
    if n_trials <= 1 or var_sr_across_trials <= 0:
        return 0.0
    e = EULER_MASCHERONI
    return float(np.sqrt(var_sr_across_trials) * (
        (1 - e) * norm.ppf(1 - 1.0 / n_trials)
        + e * norm.ppf(1 - 1.0 / (n_trials * np.e))
    ))


def deflated_sharpe(sr_monthly: float, n_obs: int, skew: float,
                    kurtosis: float,
                    trial_sharpes_monthly: list[float]) -> dict:
    """DSR of a candidate given the full set of trial Sharpes examined.

    Honest usage note: `trial_sharpes_monthly` should include EVERY
    configuration you looked at before choosing the headline strategy --
    every single-signal portfolio, every model. Undercounting trials is how
    the deflation gets gamed.
    """
    trials = [s for s in trial_sharpes_monthly if np.isfinite(s)]
    n_trials = len(trials)
    var_sr = float(np.var(trials, ddof=1)) if n_trials > 1 else 0.0
    sr_star = expected_max_sharpe(var_sr, n_trials)
    dsr = probabilistic_sharpe(sr_monthly, sr_star, n_obs, skew, kurtosis)
    return {"dsr": dsr, "sr_star_monthly": sr_star, "n_trials": n_trials,
            "psr_vs_zero": probabilistic_sharpe(sr_monthly, 0.0, n_obs,
                                                skew, kurtosis)}
