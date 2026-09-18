"""Point-in-time panel utilities and leak diagnostics.

The project's non-negotiable rule is point-in-time correctness in BOTH
directions (see docs/02_project_roadmap.md §0):

  * forward leakage  -- using information before it was public. The classic
    fake edge. `demonstrate_lookahead` shows exactly what it looks like in
    the metrics, so you can recognize it.
  * staleness        -- using information long after it was fresh ("Anomaly
    Time", Bowles-Reed-Ringgenberg-Thornock, JF 2024: returns concentrate in
    the first month after release). `staleness_experiment` measures it.

Conventions
-----------
A panel is a long DataFrame with columns:
    date, ticker, <signal columns...>, ret, fwd_ret
where signals dated t are KNOWN at t, `ret` is the return over month t, and
`fwd_ret` is the return over month t+1. Every predictive statistic in this
repo pairs signal_t with fwd_ret (never with ret).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.ic import mean_ic


def add_forward_returns(panel: pd.DataFrame, ret_col: str = "ret",
                        horizon: int = 1) -> pd.DataFrame:
    """fwd_ret[t] = return over (t, t+horizon], per ticker.

    For horizon > 1 the forward return is compounded; remember to purge at
    least `horizon` periods in the walk-forward if you change this.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    if horizon == 1:
        out["fwd_ret"] = out.groupby("ticker")[ret_col].shift(-1)
    else:
        def _compound(s: pd.Series) -> pd.Series:
            gross = (1.0 + s).shift(-1)
            roll = gross.rolling(horizon).apply(
                np.prod, raw=True).shift(-(horizon - 1))
            return roll - 1.0
        out["fwd_ret"] = out.groupby("ticker")[ret_col].transform(_compound)
    return out


def leak_report(panel: pd.DataFrame, signal_cols: list[str], *,
                min_names: int = 30,
                nw_lags: int = 6) -> pd.DataFrame:
    """Compare each signal's PREDICTIVE IC (vs fwd_ret -- the only number that
    is tradeable) against its CONTEMPORANEOUS correlation (vs ret).

    Interpretation: a signal whose contemporaneous correlation dwarfs its
    predictive IC is describing the present, not forecasting the future --
    the classic trap with news/sentiment data. Neither number should ever be
    computed with misaligned timestamps; this table makes the comparison
    explicit and auditable.

    The ``note`` column separates the two ways a correlation of 1.0 can
    arise. A feature that simply IS the contemporaneous return (factor
    mode's ``fmom_1m`` is ``ret_t``) correlates perfectly with it by
    construction -- that is not lookahead, because ``ret_t`` is known at the
    end of month t, and its t-statistic is meaningless, so it is dropped.
    Any OTHER signal sitting at |corr| > 0.999 is the alarm this table
    exists to raise.
    """
    rows = []
    for c in signal_cols:
        pred = mean_ic(panel, c, "fwd_ret",
                       nw_lags=nw_lags, min_names=min_names)
        cont = mean_ic(panel, c, "ret", nw_lags=nw_lags, min_names=min_names)
        duplicate = "ret" in panel.columns and panel[c].equals(panel["ret"])
        corr, tstat = cont["ic_mean"], cont["ic_tstat"]
        if duplicate:
            note = "IS ret_t by construction -- point-in-time, not a leak"
            tstat = np.nan
        elif np.isfinite(corr) and abs(corr) > 0.999:
            note = "near-perfect contemporaneous corr -- CHECK ALIGNMENT"
        else:
            note = ""
        rows.append({
            "signal": c,
            "predictive_IC": pred["ic_mean"],
            "predictive_IC_t": pred["ic_tstat"],
            "contemporaneous_corr": corr,
            "contemporaneous_t": tstat,
            "note": note,
        })
    return pd.DataFrame(rows).set_index("signal")


def demonstrate_lookahead(panel: pd.DataFrame, seed: int = 0, *,
                          min_names: int = 30,
                          nw_lags: int = 6) -> pd.DataFrame:
    """Deliberately construct a LEAKED feature and show what it does to the IC.

    leaky = 0.5 * fwd_ret + noise  -- i.e., a 'signal' contaminated with the
    very return it claims to predict (equivalent to a timestamp error that
    lets tomorrow's information into today's feature).

    The point of shipping this in the repo: the resulting IC is absurd
    (~0.3+ when honest single signals live near 0.02-0.06). If a real feature
    ever produces numbers like this, the correct response is not excitement --
    it is an audit of the timestamps. This is the 'break it on purpose' test
    from the roadmap, made permanent.
    """
    rng = np.random.default_rng(seed)
    df = panel.dropna(subset=["fwd_ret"]).copy()
    df["leaky_feature"] = 0.5 * df["fwd_ret"] + rng.normal(
        0, df["fwd_ret"].std(), size=len(df)
    )
    df["honest_noise"] = rng.standard_normal(len(df))
    rows = []
    for c in ["leaky_feature", "honest_noise"]:
        r = mean_ic(df, c, "fwd_ret", nw_lags=nw_lags, min_names=min_names)
        rows.append({"feature": c, "IC": r["ic_mean"], "IC_t": r["ic_tstat"]})
    return pd.DataFrame(rows).set_index("feature")


def staleness_experiment(panel: pd.DataFrame, signal_cols: list[str], *,
                         min_names: int = 30,
                         max_lag: int = 6, nw_lags: int = 6) -> pd.DataFrame:
    """Measure how each signal's IC decays as the signal gets STALE.

    Uses signal_{t-lag} against fwd_ret_t for lag = 0..max_lag. Motivated by
    'Anomaly Time' (JF 2024): forming portfolios on stale information (the
    old June-formation convention) understates predictability. Fresh
    information (lag 0) should dominate; a flat profile means the signal is
    slow-moving; a steep drop means formation timing matters a lot.
    """
    df = panel.sort_values(["ticker", "date"]).copy()
    rows = []
    for c in signal_cols:
        for lag in range(0, max_lag + 1):
            col = f"__stale_{c}_{lag}"
            df[col] = df.groupby("ticker")[c].shift(lag)
            r = mean_ic(df, col, "fwd_ret", nw_lags=nw_lags,
                        min_names=min_names)
            rows.append({"signal": c, "staleness_months": lag,
                         "IC": r["ic_mean"], "IC_t": r["ic_tstat"]})
            df.drop(columns=[col], inplace=True)
    return pd.DataFrame(rows)
