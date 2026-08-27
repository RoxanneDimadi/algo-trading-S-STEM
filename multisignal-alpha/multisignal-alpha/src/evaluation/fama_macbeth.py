"""Fama-MacBeth (1973) cross-sectional regressions.

The workhorse test of whether signals carry MARGINAL predictive power for
the cross-section of forward returns, holding the other signals fixed:

  pass 1: each date t, regress fwd_ret on the signal vector cross-sectionally
  pass 2: the time-series mean of each coefficient, with a Newey-West t-stat

Signals should be rank-normalized first (comparable scales); coefficients
then read as monthly return per unit of cross-sectional rank -- directly
comparable to the synthetic generator's planted betas, which is exactly how
the test suite validates this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.stats import nw_mean_test


def fama_macbeth(panel: pd.DataFrame, signal_cols: list[str],
                 fwd_col: str = "fwd_ret", nw_lags: int = 6,
                 min_names: int = 50) -> pd.DataFrame:
    df = panel[["date", *signal_cols, fwd_col]].dropna()
    coefs = []
    for dt, g in df.groupby("date", sort=True):
        if len(g) < min_names:
            continue
        X = np.column_stack([np.ones(len(g)), g[signal_cols].to_numpy()])
        y = g[fwd_col].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        coefs.append([dt, *beta])
    cdf = pd.DataFrame(coefs, columns=["date", "const", *signal_cols]).set_index("date")
    rows = []
    for c in ["const", *signal_cols]:
        nw = nw_mean_test(cdf[c], lags=nw_lags)
        rows.append({"variable": c, "mean_coef": nw["mean"],
                     "nw_tstat": nw["tstat"], "n_periods": nw["n"]})
    return pd.DataFrame(rows).set_index("variable")
