"""End-to-end research pipeline.

    python -m src.pipeline --config configs/config.yaml

Stages (mirroring docs/02_project_roadmap.md):
  1. data          -- build the point-in-time panel (synthetic by default)
  2. leak checks   -- predictive-vs-contemporaneous table + the deliberate
                      lookahead demonstration + staleness profile
  3. signal eval   -- per-signal IC/ICIR/turnover + quantile long-short,
                      gross and net; Fama-MacBeth multivariate check
  4. decay         -- in-sample vs post-sample vs post-publication
  5. models        -- purged walk-forward: elastic net benchmark vs LightGBM
  6. controls+DSR  -- factor-controlled alpha of the OOS strategies; deflated
                      Sharpe with honest trial accounting
Outputs land in results/tables, results/figures, and results/summary.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .backtest.engine import comparison_table, run_model_backtest
from .data.panel import demonstrate_lookahead, leak_report, staleness_experiment
from .data.synthetic import make_synthetic_panel
from .evaluation.decay import decay_table, segment_performance
from .evaluation.deflated_sharpe import deflated_sharpe
from .evaluation.fama_macbeth import fama_macbeth
from .evaluation.ic import mean_ic, rolling_ic
from .evaluation.portfolio import evaluate_signal_portfolio, summary_row
from .evaluation.factor_controls import alpha_regression
from .models.icnet import make_icnet
from .models.pulse import make_pulse
from .agent.backtest import backtest_agent
from .models.models import make_lgbm, make_linear
from .utils.plotting import (plot_cumulative_ls, plot_decay,
                             plot_model_comparison, plot_rolling_ic)
from .utils.stats import rank_normalize_cross_section


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_data(cfg: dict):
    mode = cfg["data"]["mode"]
    if mode == "synthetic":
        return make_synthetic_panel(cfg, seed=int(cfg["run"]["seed"]))
    if mode == "osap":
        from .data.loaders import (build_panel_from_osap, load_french_factors,
                                   load_signal_doc)
        ocfg = cfg["data"]["osap"]
        panel = build_panel_from_osap(ocfg["signals_csv"], ocfg["returns_csv"],
                                      ocfg["signals"])
        meta = load_signal_doc(ocfg["signal_doc_csv"]).reindex(ocfg["signals"])
        factors = load_french_factors(csv_path=ocfg.get("french_factors_csv"))
        return panel, factors, meta
    if mode == "panel_csv":
        # generic prebuilt panel (e.g. the real-data repo's factor-mode
        # output: OSAP long-short portfolios as tradable assets -- the
        # supported real-data path when WRDS/CRSP returns are unavailable)
        from .data.loaders import load_french_factors, load_prebuilt_panel
        pcfg = cfg["data"]["panel_csv"]
        panel, meta = load_prebuilt_panel(pcfg["panel"], pcfg["meta"])
        factors = load_french_factors(csv_path=pcfg.get("french_factors_csv"))
        return panel, factors, meta
    raise ValueError(f"unknown data.mode: {mode}")


def main(config_path: str = "configs/config.yaml") -> dict:
    cfg = load_config(config_path)
    out = Path(cfg["run"]["output_dir"])
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    ecfg, wcfg = cfg["evaluation"], cfg["walkforward"]
    nw = int(ecfg["nw_lags"])
    # optional min cross-section size; None keeps each module's own default
    # (30 for IC-type stats, 50 for quantile books, 60 for PULSE dates)
    mn = ecfg.get("min_names_per_date")
    mn_kw = {} if mn is None else {"min_names": int(mn)}
    if mn is not None:
        for m in ("icnet", "pulse"):
            cfg["models"].setdefault(m, {}).setdefault("min_names", int(mn))

    # ---- 1. data -------------------------------------------------------------
    panel, factors, meta = build_data(cfg)
    signal_cols = list(meta.index)
    panel = rank_normalize_cross_section(panel, signal_cols)
    print(f"[data] panel: {panel['date'].nunique()} months x "
          f"{panel['ticker'].nunique()} names, signals={signal_cols}")

    # ---- 2. leak checks --------------------------------------------------------
    leaks = leak_report(panel, signal_cols, nw_lags=nw, **mn_kw)
    leaks.to_csv(tables / "leak_report.csv")
    demo = demonstrate_lookahead(panel, seed=int(cfg["run"]["seed"]), nw_lags=nw,
                                 **mn_kw)
    demo.to_csv(tables / "lookahead_demonstration.csv")
    stale = staleness_experiment(panel, signal_cols, max_lag=6, nw_lags=nw,
                                 **mn_kw)
    stale.to_csv(tables / "staleness_profile.csv", index=False)
    print("[leak] lookahead demo -- leaky feature IC "
          f"{demo.loc['leaky_feature', 'IC']:.3f} vs honest noise "
          f"{demo.loc['honest_noise', 'IC']:.3f} (leaky must look absurd)")

    # ---- 3. per-signal evaluation ---------------------------------------------
    rows, ls_gross, roll = [], {}, {}
    for c in signal_cols:
        res = evaluate_signal_portfolio(
            panel, c, n_q=int(ecfg["n_quantiles"]),
            cost_bps_per_side=float(ecfg["cost_bps_per_side"]), nw_lags=nw,
            **mn_kw)
        icr = mean_ic(panel, c, nw_lags=nw, **mn_kw)
        row = summary_row(res)
        row.update({"IC": icr["ic_mean"], "IC_t": icr["ic_tstat"],
                    "ICIR": icr["icir"]})
        rows.append(row)
        ls_gross[c] = res["series"]["gross"]
        roll[c] = rolling_ic(panel, c, window=int(ecfg["rolling_ic_window"]),
                             **mn_kw)
    sig_table = pd.DataFrame(rows).set_index("signal")
    order = ["IC", "IC_t", "ICIR", "ann_ret_gross", "sharpe_gross", "nw_t_gross",
             "ann_ret_net", "sharpe_net", "nw_t_net", "one_way_turnover",
             "n_months"]
    sig_table = sig_table[order]
    sig_table.to_csv(tables / "signal_evaluation.csv")
    print("[signals]\n" + sig_table.round(3).to_string())

    fm = fama_macbeth(panel, signal_cols, nw_lags=nw)
    fm.to_csv(tables / "fama_macbeth.csv")

    plot_cumulative_ls(ls_gross, str(figures / "single_signal_cumulative.png"))
    plot_rolling_ic(roll, str(figures / "rolling_ic.png"),
                    window=int(ecfg["rolling_ic_window"]))

    # ---- 4. decay ----------------------------------------------------------------
    # Only meaningful when features have publication dates (synthetic planted
    # signals, or firm-level OSAP anomalies). Derived features -- e.g. the
    # factor-momentum columns of panel_csv mode -- carry no pub dates, and
    # the McLean-Pontiff split does not apply to them; the REAL per-factor
    # decay exhibit for that mode lives in real-data (factor_decay.csv).
    have_dates = meta[["sample_end", "pub_date"]].notna().all(axis=1)
    if have_dates.any():
        dec = decay_table({k: v for k, v in ls_gross.items()
                           if have_dates.get(k, False)},
                          meta.loc[have_dates], nw_lags=nw)
        dec.to_csv(tables / "decay_analysis.csv")
        dated = [c for c in signal_cols if have_dates.get(c, False)]
        lead = sig_table.loc[dated, "sharpe_gross"].idxmax()
        plot_decay(ls_gross[lead], meta.loc[lead, "sample_end"],
                   meta.loc[lead, "pub_date"],
                   str(figures / "decay_leading_signal.png"), lead)
        print("[decay]\n" + dec.round(3).to_string())
    else:
        dec = pd.DataFrame(
            {"note": ["skipped: no publication dates in feature meta "
                      "(see real-data factor_decay.csv for the per-factor "
                      "McLean-Pontiff exhibit)"]})
        print("[decay] skipped -- features carry no publication dates")

    # ---- 5. models: linear benchmark vs LightGBM ---------------------------------
    results = {
        "elasticnet": run_model_backtest(
            panel, signal_cols, lambda: make_linear(cfg["models"]["linear"]),
            wcfg, ecfg),
        "lightgbm": run_model_backtest(
            panel, signal_cols, lambda: make_lgbm(cfg["models"]["lgbm"]),
            wcfg, ecfg),
        "icnet": run_model_backtest(
            panel, signal_cols, lambda: make_icnet(cfg["models"]["icnet"]),
            wcfg, ecfg),
        "pulse": run_model_backtest(
            panel, signal_cols, lambda: make_pulse(cfg["models"]["pulse"]),
            wcfg, ecfg),
    }
    comp = comparison_table(results)
    comp.to_csv(tables / "model_comparison.csv")
    print("[models]\n" + comp.round(3).to_string())
    for mname, r in results.items():
        if "feature_importance" in r:
            r["feature_importance"].rename("importance").to_csv(
                tables / f"{mname}_feature_importance.csv")
    plot_model_comparison({k: v["series"]["net"] for k, v in results.items()},
                          str(figures / "model_comparison_net.png"))

    # ---- 5b. PULSE efficacy-path diagnostic (in-sample exhibit) -------------------
    # Full-panel filter fit, saved so filtered efficacy paths can be inspected
    # against known events (synthetic: the planted decay steps). This is a
    # diagnostic of STATE TRACKING, not a performance claim.
    import pandas as _pd
    import matplotlib.pyplot as _plt
    dfit = panel.dropna(subset=[*signal_cols, "fwd_ret"])
    pdiag = make_pulse(cfg["models"]["pulse"]).fit(
        dfit[signal_cols], dfit["fwd_ret"].to_numpy(), dates=dfit["date"].to_numpy())
    eff = _pd.DataFrame(pdiag.filter_history_,
                        index=_pd.DatetimeIndex(pdiag.filter_dates_),
                        columns=pdiag.feature_names_)
    eff.to_csv(tables / "pulse_efficacy_path.csv")
    fig, ax = _plt.subplots(figsize=(9, 5))
    for c in signal_cols:
        ax.plot(eff.index, eff[c], lw=1.3, label=c)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"PULSE filtered efficacy paths (chosen a={pdiag.a_:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "pulse_efficacy_paths.png", dpi=140)
    _plt.close(fig)
    print(f"[pulse] chosen dynamics: a={pdiag.a_}, q_scale={pdiag.q_scale_}")

    # ---- 5c. trading agent: differentiable cost-aware policy ----------------------
    agent_monthly_sharpe = None
    if not cfg["agent"].get("enabled", True):
        # stage skipped (e.g., partial runs); reuse existing table for DSR if present
        try:
            prev = pd.read_csv(tables / "agent_vs_myopic.csv", index_col=0)
            agent_monthly_sharpe = float(
                prev.loc["agent (learned gamma)", "sharpe_net"]) / np.sqrt(12)
            print("[agent] stage disabled; reusing existing agent_vs_myopic.csv")
        except Exception:
            print("[agent] stage disabled; no prior agent table found")
    else:
      agent_res = backtest_agent(panel, signal_cols, wcfg, ecfg, cfg["agent"])
      myopic_res = backtest_agent(panel, signal_cols, wcfg, ecfg, cfg["agent"],
                                  gamma_override=1.0,
                                  params_by_fold=agent_res["params_by_fold"])
      agent_tbl = pd.DataFrame([
          {"policy": "agent (learned gamma)",
           "ann_ret_net": agent_res["net"]["ann_return"],
           "sharpe_net": agent_res["net"]["sharpe"],
           "nw_t_net": agent_res["net"]["nw_tstat"],
           "one_way_turnover": agent_res["avg_one_way_turnover"],
           "mean_gamma": agent_res["params_by_fold"]["gamma"].mean()},
          {"policy": "myopic (gamma=1)",
           "ann_ret_net": myopic_res["net"]["ann_return"],
           "sharpe_net": myopic_res["net"]["sharpe"],
           "nw_t_net": myopic_res["net"]["nw_tstat"],
           "one_way_turnover": myopic_res["avg_one_way_turnover"],
           "mean_gamma": 1.0},
      ]).set_index("policy")
      agent_tbl.to_csv(tables / "agent_vs_myopic.csv")
      agent_res["params_by_fold"].to_csv(tables / "agent_params_by_fold.csv")
      plot_model_comparison(
          {"agent": agent_res["series"]["net"],
           "myopic": myopic_res["series"]["net"],
           "pulse (model)": results["pulse"]["series"]["net"]},
          str(figures / "agent_vs_myopic_net.png"))
      agent_monthly_sharpe = agent_res["net"]["monthly_sharpe"]
      print("[agent]\n" + agent_tbl.round(3).to_string())

    # ---- 6. factor controls + deflated Sharpe -------------------------------------
    ctrl_rows = []
    for name, r in results.items():
        a = alpha_regression(r["series"]["net"], factors, nw_lags=nw)
        ctrl_rows.append({"model": name, "alpha_ann": a["alpha_ann"],
                          "alpha_t": a["alpha_t"], "r2": a["r2"], "n": a["n"]})
    ctrl = pd.DataFrame(ctrl_rows).set_index("model")
    ctrl.to_csv(tables / "factor_controls.csv")
    print("[controls]\n" + ctrl.round(3).to_string())

    # trials for DSR: every single-signal book examined + both models (honest
    # accounting of the selection surface -- see config note). Monthly
    # Sharpe = annualized Sharpe / sqrt(12).
    trials = [results[m]["net"]["monthly_sharpe"] for m in results]
    if agent_monthly_sharpe is not None and np.isfinite(agent_monthly_sharpe):
        trials.append(agent_monthly_sharpe)
    if cfg["deflated_sharpe"]["count_single_signal_trials"]:
        trials += [sig_table.loc[c, "sharpe_net"] / np.sqrt(12)
                   for c in signal_cols]
    best = comp["sharpe_net"].idxmax()
    bstats = results[best]["net"]
    dsr = deflated_sharpe(bstats["monthly_sharpe"], bstats["n_months"],
                          bstats["skew"], bstats["kurtosis"], trials)
    dsr_row = pd.DataFrame([{"model": best, **dsr}]).set_index("model")
    dsr_row.to_csv(tables / "deflated_sharpe.csv")
    print(f"[dsr] best={best}: DSR={dsr['dsr']:.3f} "
          f"(SR*={dsr['sr_star_monthly']:.3f} monthly, "
          f"n_trials={dsr['n_trials']})")

    # ---- summary.md -----------------------------------------------------------------
    lines = [
        "# Pipeline summary\n",
        f"Panel: {panel['date'].nunique()} months x {panel['ticker'].nunique()} names; "
        f"signals: {', '.join(signal_cols)}\n",
        "## Lookahead demonstration (should look absurd)\n",
        demo.round(3).to_markdown(), "\n",
        "## Per-signal evaluation\n", sig_table.round(3).to_markdown(), "\n",
        "## Fama-MacBeth (multivariate marginal power)\n",
        fm.round(4).to_markdown(), "\n",
        "## Decay (McLean-Pontiff pattern)\n", dec.round(3).to_markdown(), "\n",
        "## Model comparison (purged walk-forward, out-of-sample)\n",
        comp.round(3).to_markdown(), "\n",
        "## Factor-controlled alpha (net strategies)\n",
        ctrl.round(3).to_markdown(), "\n",
        f"## Deflated Sharpe ({best})\n", dsr_row.round(3).to_markdown(), "\n",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    print(f"[done] tables -> {tables}, figures -> {figures}, "
          f"summary -> {out/'summary.md'}")
    return {"signal_table": sig_table, "decay": dec, "comparison": comp,
            "controls": ctrl, "dsr": dsr, "results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
