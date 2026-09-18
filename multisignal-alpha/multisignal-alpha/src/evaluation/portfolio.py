"""Quantile-sorted long-short portfolios: the ECONOMIC test of a signal.

Statistical significance (IC t-stats) is necessary but not sufficient; a
signal must also separate winners from losers by enough to survive costs.
The standard procedure, implemented here exactly as in the literature:

  1. Each date, sort eligible names into `n_q` quantiles on the score.
  2. Long the top quantile, short the bottom, equal-weighted within legs,
     normalized to $1 long / $1 short (sum |w| = 2).
  3. The spread's forward return is the strategy return -- reported GROSS
     and NET of a per-side transaction-cost assumption applied to traded
     notional.

Sign convention: scores are assumed oriented so that HIGHER score => HIGHER
expected return (the Chen-Zimmermann files are distributed pre-signed this
way). Flip a raw signal before evaluation if its documented relationship is
negative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.stats import annualized_stats


def score_to_weights(panel: pd.DataFrame, score_col: str, n_q: int = 5,
                     min_names: int = 50) -> pd.DataFrame:
    """Map a per-date cross-sectional score to long-short weights.

    Returns a long DataFrame [date, ticker, weight] with, per date,
    +1/n_top on the top quantile and -1/n_bottom on the bottom quantile.
    """
    df = panel[["date", "ticker", score_col]].dropna()
    frames = []
    for dt, g in df.groupby("date", sort=True):
        if len(g) < min_names:
            continue
        try:
            q = pd.qcut(g[score_col], n_q, labels=False, duplicates="drop")
        except ValueError:
            continue
        if q.nunique() < 2:
            continue
        top, bot = q.max(), q.min()
        w = pd.Series(0.0, index=g.index)
        w[q == top] = 1.0 / (q == top).sum()
        w[q == bot] = -1.0 / (q == bot).sum()
        frames.append(pd.DataFrame({"date": dt, "ticker": g["ticker"], "weight": w}))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "weight"])
    return pd.concat(frames, ignore_index=True)


def portfolio_returns(weights: pd.DataFrame, panel: pd.DataFrame,
                      fwd_col: str = "fwd_ret") -> pd.Series:
    """Gross long-short return series: sum_i w[t,i] * fwd_ret[t,i].

    Weights formed at t earn the (t, t+1] return -- the alignment is inherited
    from the panel's fwd_ret construction, so there is no way to accidentally
    trade on contemporaneous information here.
    """
    m = weights.merge(panel[["date", "ticker", fwd_col]], on=["date", "ticker"],
                      how="left")
    ls = (m["weight"] * m[fwd_col]).groupby(m["date"]).sum(min_count=1)
    ls.name = "ls_gross"
    return ls.dropna()


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    """Traded notional per $1-long/$1-short book, per rebalance.

    traded[t] = sum_i |w[t,i] - w[t-1,i]|   (weights pivoted date x ticker)

    One-way turnover = traded / 2. On the first date traded = sum|w| = 2
    (the book is established from cash).
    """
    wide = weights.pivot_table(index="date", columns="ticker", values="weight",
                               fill_value=0.0)
    traded = wide.diff().abs().sum(axis=1)
    if len(traded) > 0:
        traded.iloc[0] = wide.iloc[0].abs().sum()
    traded.name = "traded_notional"
    return traded


def apply_costs(gross: pd.Series, traded: pd.Series,
                cost_bps_per_side: float) -> pd.Series:
    """net[t] = gross[t] - traded[t] * cost_bps/1e4.

    Cost is charged on every dollar traded (each side of each trade pays the
    per-side cost once, and `traded` already counts both the buy and the sell
    legs of a rebalance as separate notional).
    """
    cost = traded.reindex(gross.index).fillna(0.0) * (cost_bps_per_side / 1e4)
    net = gross - cost
    net.name = "ls_net"
    return net


def evaluate_signal_portfolio(panel: pd.DataFrame, score_col: str,
                              n_q: int = 5, cost_bps_per_side: float = 10.0,
                              nw_lags: int = 6, min_names: int = 50) -> dict:
    """Full economic evaluation of one score: gross/net long-short stats plus
    turnover. Returns the series too, for plotting and decay analysis."""
    w = score_to_weights(panel, score_col, n_q=n_q, min_names=min_names)
    gross = portfolio_returns(w, panel)
    traded = turnover_series(w)
    net = apply_costs(gross, traded, cost_bps_per_side)

    g, n = annualized_stats(gross, nw_lags), annualized_stats(net, nw_lags)
    return {
        "signal": score_col,
        "gross": g,
        "net": n,
        "avg_one_way_turnover": float(traded.mean() / 2.0) if len(traded) else np.nan,
        "series": {"gross": gross, "net": net, "traded": traded, "weights": w},
    }


def summary_row(res: dict) -> dict:
    """Flatten evaluate_signal_portfolio output into one table row."""
    return {
        "signal": res["signal"],
        "ann_ret_gross": res["gross"]["ann_return"],
        "sharpe_gross": res["gross"]["sharpe"],
        "nw_t_gross": res["gross"]["nw_tstat"],
        "ann_ret_net": res["net"]["ann_return"],
        "sharpe_net": res["net"]["sharpe"],
        "nw_t_net": res["net"]["nw_tstat"],
        "one_way_turnover": res["avg_one_way_turnover"],
        "n_months": res["gross"]["n_months"],
    }
