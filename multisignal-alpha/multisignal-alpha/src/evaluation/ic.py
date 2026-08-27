"""Information Coefficient (IC) evaluation.

IC here is the per-date Spearman rank correlation between the signal observed
at t and the FORWARD return over (t, t+1]. Rank correlation (not Pearson)
because cross-sectional return relationships are noisy and outlier-heavy, and
because rank IC is the industry-standard quantity (Grinold & Kahn).

Calibration for reading the numbers: honest single-signal monthly ICs are
SMALL. The literature treats ~0.05 as strong (a published deep-learning factor
project celebrates 0.065). If you see 0.3, hunt for the leak -- see
src/data/panel.py::demonstrate_lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..utils.stats import nw_mean_test


def ic_series(panel: pd.DataFrame, signal_col: str, fwd_col: str = "fwd_ret",
              min_names: int = 30) -> pd.Series:
    """Per-date Spearman IC of signal vs forward return."""
    df = panel[["date", signal_col, fwd_col]].dropna()

    def _one(g: pd.DataFrame) -> float:
        if len(g) < min_names:
            return np.nan
        rho, _ = sps.spearmanr(g[signal_col], g[fwd_col])
        return rho

    out = df.groupby("date", sort=True).apply(_one, include_groups=False)
    out.name = f"IC[{signal_col}]"
    return out.dropna()


def mean_ic(panel: pd.DataFrame, signal_col: str, fwd_col: str = "fwd_ret",
            nw_lags: int = 6) -> dict:
    """Mean IC with a Newey-West t-stat, plus ICIR.

    ICIR = mean(IC) / std(IC): the STABILITY of predictive power, valued as
    highly as its magnitude (a large-but-erratic IC is hard to monetize).
    """
    s = ic_series(panel, signal_col, fwd_col)
    if len(s) < 12:
        return {"ic_mean": np.nan, "ic_tstat": np.nan, "icir": np.nan,
                "ic_std": np.nan, "n_periods": len(s)}
    nw = nw_mean_test(s, lags=nw_lags)
    sd = s.std(ddof=1)
    return {
        "ic_mean": nw["mean"],
        "ic_tstat": nw["tstat"],
        "ic_std": sd,
        "icir": nw["mean"] / sd if sd > 0 else np.nan,
        "n_periods": len(s),
    }


def rolling_ic(panel: pd.DataFrame, signal_col: str, window: int = 24,
               fwd_col: str = "fwd_ret") -> pd.Series:
    """Rolling-mean IC -- the stability picture behind ICIR."""
    return ic_series(panel, signal_col, fwd_col).rolling(window).mean()
