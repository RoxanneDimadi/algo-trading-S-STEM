"""Walk-forward backtest for the trading agent.

Same discipline as the forecasting engine, adapted for a STATEFUL policy:
train the policy on each purged train window, then roll it through the test
window carrying inventory FORWARD ACROSS FOLDS (the book is one continuous
portfolio -- selling everything at fold boundaries would smuggle costs out
of the accounting). The purge theorem (docs/math/08) applies unchanged: the
policy's parameters are functions of training-window data only, and the
roll through the test window uses only formation-date signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.walkforward import walkforward_splits
from ..utils.stats import annualized_stats
from .policy import DiffPolicyAgent, panel_to_matrices


def backtest_agent(panel, signal_cols, wf_cfg: dict, eval_cfg: dict,
                   agent_cfg: dict, gamma_override: float | None = None,
                   params_by_fold=None):
    """Purged walk-forward for the policy agent.

    gamma_override=1.0 reruns the identical experiment with the myopic
    full-rebalance policy on each fold's learned aim -- the control.
    """
    Z, Y, dates, _tickers = panel_to_matrices(panel, signal_cols)
    folds = walkforward_splits(
        dates,
        min_train=int(wf_cfg.get("min_train", 120)),
        test_size=int(wf_cfg.get("test_size", 12)),
        purge=int(wf_cfg.get("purge", 1)),
        embargo=int(wf_cfg.get("embargo", 0)),
        expanding=bool(wf_cfg.get("expanding", True)),
    )
    pos = {d: i for i, d in enumerate(dates)}
    cost = float(eval_cfg.get("cost_bps_per_side", 10.0))

    net_parts, gross_parts, traded_parts = [], [], []
    idx_parts, fold_rows = [], []
    w_carry = None
    for f in folds:
        tr = [pos[d] for d in f.train_dates if d in pos]
        te = [pos[d] for d in f.test_dates if d in pos]
        if not tr or not te:
            continue
        agent = DiffPolicyAgent(
            cost_bps_per_side=cost,
            epochs=int(agent_cfg.get("epochs", 150)),
            lr=float(agent_cfg.get("lr", 0.05)),
            seed=int(agent_cfg.get("seed", 0)),
        )
        if params_by_fold is not None and f.fold_id in params_by_fold.index:
            # reuse previously learned parameters (control runs): no refit
            row = params_by_fold.loc[f.fold_id]
            agent.theta_ = row[[
                f"theta_{c}" for c in signal_cols]].to_numpy(float)
            agent.gamma_ = float(row["gamma"])
        else:
            agent.fit(Z[:, tr, :], Y[tr])
        out = agent.roll(Z[:, te, :], Y[te], w0=w_carry, gamma=gamma_override)
        w_carry = out["w_last"]
        net_parts.append(out["net"])
        gross_parts.append(out["gross"])
        traded_parts.append(out["traded"])
        idx_parts.append([dates[i] for i in te])
        fold_rows.append({"fold": f.fold_id, "gamma": agent.gamma_,
                          **{f"theta_{c}": t for c, t in
                             zip(signal_cols, agent.theta_)}})

    idx = pd.DatetimeIndex(np.concatenate(idx_parts))
    net = pd.Series(np.concatenate(net_parts), index=idx, name="agent_net")
    gross = pd.Series(np.concatenate(gross_parts),
                      index=idx, name="agent_gross")
    traded = pd.Series(np.concatenate(traded_parts), index=idx)
    nw = int(eval_cfg.get("nw_lags", 6))
    return {
        "series": {"net": net, "gross": gross, "traded": traded},
        "net": annualized_stats(net, nw),
        "gross": annualized_stats(gross, nw),
        "avg_one_way_turnover": float(traded.mean() / 2.0),
        "params_by_fold": pd.DataFrame(fold_rows).set_index("fold"),
    }
