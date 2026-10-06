"""Composition experiment: PULSE forecasts as the trading agent's aim.

The standard agent blends RAW signals with a static learned theta -- blind
to decay and to interactions. PULSE produces a decay-aware, interaction-
aware forecast whose point-in-time property is structural: the filter
state feeding date t uses observations through t-1 only. Composing them:

    aim_t  <-  PULSE one-step forecast      (time-varying, nonlinear basis)
    speed  <-  learned by the agent          (cost-aware execution)

Leakage audit: PULSE hyperparameters (a, q) are selected on the FIRST
min_train months only (before any test fold); the filter then runs forward
once, online; per-date coefficients lambda_t consume the month-t forward
return, so the forecast for date t uses states through t-1. The agent
trains per purged walk-forward fold as always.

    python3 scripts/compose_pulse_agent.py
"""
import sys
import pathlib

import pandas as pd
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The project package lives one level up; the bootstrap above
# has to run before these imports resolve.
# pylint: disable=wrong-import-position
from src.agent.backtest import backtest_agent
from src.models.pulse import make_pulse, expand_interactions
from src.utils.stats import rank_normalize_cross_section
from src.data.synthetic import make_synthetic_panel


def pulse_forecast_column(panel, feats, cfg):
    """Point-in-time PULSE forecasts for every (date, ticker)."""
    d = panel.dropna(subset=[*feats, "fwd_ret"]).sort_values(
        ["date", "ticker"], kind="stable")
    dates_all = np.sort(d["date"].unique())
    min_train = int(cfg["walkforward"]["min_train"])

    # hyperparameters from the first train window ONLY
    head = d[d["date"].isin(dates_all[:min_train])]
    m0 = make_pulse(cfg["models"]["pulse"]).fit(
        head[feats], head["fwd_ret"].to_numpy(), dates=head["date"].to_numpy())
    a, qs = m0.a_, m0.q_scale_

    # one online filtering pass over the full history
    Zl, _names = expand_interactions(d[feats].to_numpy(float), feats)
    # this diagnostic deliberately drives PULSE's internals
    # pylint: disable=protected-access
    dts, LAM, R = m0._per_date_coefs(Zl, d["fwd_ret"].to_numpy(float),
                                     d["date"].to_numpy())
    med_r = np.median(R, axis=0)
    M = np.empty_like(LAM)
    for k in range(LAM.shape[1]):
        M[:, k], _, _ = m0._filter_1d(LAM[:, k], R[:, k], a, qs * med_r[k])
    # predicted state for date index t uses filtered state through t-1
    pred = np.zeros_like(M)
    pred[1:] = a * M[:-1]

    pos = {dt: i for i, dt in enumerate(dts)}
    fc = np.full(len(d), np.nan)
    dvals = d["date"].to_numpy()
    for dt, i in pos.items():
        if i == 0:
            continue  # no prior information for the very first date
        mask = dvals == dt
        fc[mask] = Zl[mask] @ pred[i]
    out = d[["date", "ticker"]].copy()
    out["pulse_fcst"] = fc
    print(f"[pulse->agent] hyperparams from first {min_train} months: "
          f"a={a}, q_scale={qs}")
    return out.dropna()


def main():
    with open(ROOT / "configs/config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    panel, _, meta = make_synthetic_panel(cfg, seed=cfg["run"]["seed"])
    feats = list(meta.index)
    panel = rank_normalize_cross_section(panel, feats)

    fcst = pulse_forecast_column(panel, feats, cfg)
    composed_panel = panel.merge(fcst, on=["date", "ticker"], how="inner")

    wcfg, ecfg, acfg = cfg["walkforward"], cfg["evaluation"], cfg["agent"]
    composed = backtest_agent(composed_panel, ["pulse_fcst"], wcfg, ecfg, acfg)
    baseline = backtest_agent(panel, feats, wcfg, ecfg, acfg)

    tbl = pd.DataFrame([
        {"agent": "composed (PULSE aim)",
         "ann_ret_net": composed["net"]["ann_return"],
         "sharpe_net": composed["net"]["sharpe"],
         "nw_t_net": composed["net"]["nw_tstat"],
         "one_way_turnover": composed["avg_one_way_turnover"],
         "mean_gamma": composed["params_by_fold"]["gamma"].mean()},
        {"agent": "baseline (static theta)",
         "ann_ret_net": baseline["net"]["ann_return"],
         "sharpe_net": baseline["net"]["sharpe"],
         "nw_t_net": baseline["net"]["nw_tstat"],
         "one_way_turnover": baseline["avg_one_way_turnover"],
         "mean_gamma": baseline["params_by_fold"]["gamma"].mean()},
    ]).set_index("agent")
    tbl.to_csv(ROOT / "results/tables/composed_pulse_agent.csv")
    print(tbl.round(3).to_string())


if __name__ == "__main__":
    main()
