"""Shared statistical utilities.

Every headline number in this project that is a *time-series mean* (mean IC,
mean long-short return, alpha) is reported with a Newey-West (HAC) t-statistic,
because monthly strategy returns and IC series are autocorrelated and
heteroskedastic. Plain OLS standard errors overstate significance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

PERIODS_PER_YEAR = 12  # monthly data throughout


def month_end_freq() -> str:
    """Return a pandas month-end offset alias compatible with the installed version.

    pandas < 2.2 uses ``M``; pandas >= 2.2 renamed it to ``ME``.
    """
    try:
        pd.tseries.frequencies.to_offset("ME")
        return "ME"
    except (ValueError, KeyError):
        return "M"


def nw_mean_test(series: pd.Series, lags: int = 6) -> dict:
    """Mean of a time series with a Newey-West t-statistic.

    Implemented as an OLS regression of the series on a constant with a
    HAC covariance (maxlags=lags): the intercept is the mean, and its
    t-stat is the autocorrelation-robust test that the mean is zero.
    """
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 12:
        return {"mean": np.nan, "tstat": np.nan, "n": len(s)}
    res = sm.OLS(s.values, np.ones((len(s), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": int(lags)}
    )
    return {"mean": float(res.params[0]), "tstat": float(res.tvalues[0]), "n": len(s)}


def annualized_stats(monthly_returns: pd.Series, nw_lags: int = 6) -> dict:
    """Annualized return / vol / Sharpe for a monthly return series, plus the
    Newey-West t-stat of the monthly mean and higher moments (needed later by
    the deflated Sharpe ratio)."""
    r = pd.Series(monthly_returns).dropna().astype(float)
    if len(r) < 12:
        return {k: np.nan for k in
                ["ann_return", "ann_vol", "sharpe", "nw_tstat", "n_months",
                 "monthly_sharpe", "skew", "kurtosis"]}
    mu, sd = r.mean(), r.std(ddof=1)
    nw = nw_mean_test(r, lags=nw_lags)
    return {
        "ann_return": mu * PERIODS_PER_YEAR,
        "ann_vol": sd * np.sqrt(PERIODS_PER_YEAR),
        "sharpe": (mu / sd) * np.sqrt(PERIODS_PER_YEAR) if sd > 0 else np.nan,
        "nw_tstat": nw["tstat"],
        "n_months": len(r),
        "monthly_sharpe": mu / sd if sd > 0 else np.nan,
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()) + 3.0,  # full (not excess) kurtosis
    }


def rank_normalize_cross_section(df: pd.DataFrame, cols: list[str],
                                 date_col: str = "date") -> pd.DataFrame:
    """GKX-style preprocessing: within each date, map each signal to its
    cross-sectional rank scaled into [-1, 1].

    This is the standard normalization of Gu-Kelly-Xiu (2020) / Kelly-Pruitt-Su
    (2019): robust to outliers, comparable across signals, and it makes linear
    coefficients interpretable as 'return per unit of cross-sectional rank'.
    """
    out = df.copy()
    g = out.groupby(date_col)
    for c in cols:
        ranks = g[c].rank(method="average")
        counts = g[c].transform("count")
        out[c] = 2.0 * (ranks - 0.5) / counts - 1.0
    return out
