"""Factor-controlled alpha: the credibility multiplier.

A strategy return that merely loads on known factors is beta in disguise, not
alpha. The test: regress the long-short return series on the factor returns
and report the INTERCEPT (alpha) with a Newey-West t-statistic, plus R^2.
Low R^2 with surviving alpha = a claim to genuine information; alpha that
vanishes under controls = a known factor in disguise (also worth reporting).

Timing note (important, and easy to get wrong): the strategy return earned
over (t, t+1] must be regressed on factor returns over the SAME window.
Because this repo indexes the return earned over (t, t+1] by its formation
date t, factors indexed by their own period-END must be shifted back one
period to align. `align="formation"` (default) does that shift; use
align="none" if your factor series is already formation-dated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

PERIODS_PER_YEAR = 12


def alpha_regression(ls_returns: pd.Series, factors: pd.DataFrame,
                     nw_lags: int = 6, align: str = "formation") -> dict:
    f = factors.copy()
    if align == "formation":
        f = f.shift(-1)  # factor return over (t, t+1] onto formation date t
    df = pd.concat([ls_returns.rename("ls"), f], axis=1, join="inner").dropna()
    if len(df) < 24:
        return {"alpha_ann": np.nan, "alpha_t": np.nan, "r2": np.nan,
                "n": len(df), "betas": {}}
    X = sm.add_constant(df.drop(columns=["ls"]))
    res = sm.OLS(df["ls"], X).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": int(nw_lags)})
    return {
        "alpha_ann": float(res.params["const"]) * PERIODS_PER_YEAR,
        "alpha_t": float(res.tvalues["const"]),
        "r2": float(res.rsquared),
        "n": int(res.nobs),
        "betas": {k: float(v) for k, v in res.params.items() if k != "const"},
    }
