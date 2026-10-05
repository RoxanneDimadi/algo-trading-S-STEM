"""Run the trading agent + models against the latest real-data cross-section
and send the resulting buy/sell signals through the execution bridge.

This is the missing link between the research side of the repo (the agent
in src/agent/policy.py, fit on OSAP/CRSP data by src/pipeline.py) and the
execution side (src/execution/*, already wired to Alpaca paper trading).
See src/execution/signal_generator.py for the mechanism.

    python scripts/run_agent_signals.py --permno-ticker-map data/raw/permno_ticker_map.csv

By default this submits REAL (paper-account, unless --live) orders for
every approved signal. Use --dry-run to only fit, score, and print what the
risk gate would approve, without calling the broker's order endpoint.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.agent_evaluator import AgentPolicyConfig, CostAwareAgentEvaluator
from src.execution.alpaca_bridge import AlpacaConfig, AlpacaExecutionBridge
from src.execution.config import load_execution_config
from src.execution.ledger import ExecutionLedger
from src.execution.signal_generator import (
    SignalGeneratorConfig,
    load_permno_ticker_map,
    run_live_signals,
)
from src.pipeline import build_data, load_config as load_research_config
from src.utils.stats import rank_normalize_cross_section


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml",
                        help="research config (signals, data.mode, agent fit params)")
    parser.add_argument("--execution-config", default="configs/execution_config.yaml",
                        help="broker credentials + risk limits")
    parser.add_argument("--permno-ticker-map", default=None,
                        help="CSV mapping the panel's ticker column (permno for OSAP) "
                             "to a real tradable symbol; overrides the execution config")
    parser.add_argument("--min-abs-weight", type=float, default=None,
                        help="drop names whose |aim weight| is below this")
    parser.add_argument("--max-names", type=int, default=None,
                        help="cap how many names get sent to the risk gate")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="allow data.mode=synthetic (demo/dry-run only -- "
                             "synthetic tickers are not real symbols)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fit + score + show what would be approved; never submit orders")
    parser.add_argument("--live", action="store_true", default=False,
                        help="trade the live account instead of paper (requires --live "
                             "on top of a non-paper execution config)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("execution.run_agent_signals")

    research_cfg = load_research_config(args.config)
    data_mode = research_cfg["data"]["mode"]
    if data_mode != "osap" and not args.allow_synthetic:
        raise SystemExit(
            f"data.mode={data_mode!r}: live signals need real tickers, which only "
            "data.mode=osap provides a path to (via a permno->ticker map). Pass "
            "--allow-synthetic to run the wiring on synthetic data anyway (for "
            "testing the mechanism only -- synthetic tickers are not real symbols)."
        )

    panel, _factors, meta = build_data(research_cfg)
    signal_cols = list(meta.index)
    panel = rank_normalize_cross_section(panel, signal_cols)
    log.info("panel: %d months x %d names, signals=%s",
             panel["date"].nunique(), panel["ticker"].nunique(), signal_cols)

    exec_cfg = load_execution_config(args.execution_config)
    if args.live:
        exec_cfg["broker"]["paper"] = False
        log.warning("live trading mode enabled")

    broker = exec_cfg.get("broker", {})
    alpaca = AlpacaConfig(
        api_key=broker.get("api_key", ""),
        secret_key=broker.get("secret_key", ""),
        paper=broker.get("paper", True),
        base_url=broker.get("base_url"),
        timeout_seconds=broker.get("timeout_seconds", 10.0),
    )
    ledger_cfg = exec_cfg.get("ledger", {})
    ledger = ExecutionLedger(
        db_path=ledger_cfg.get("db_path", "data/ledger.db"),
        jsonl_path=ledger_cfg.get("jsonl_path", "data/audit_log.jsonl"),
    )
    bridge = AlpacaExecutionBridge(config=alpaca, ledger=ledger)

    ap = exec_cfg.get("agent_policy", {})
    policy = AgentPolicyConfig(
        gamma_speed=ap.get("gamma_speed", 0.35),
        cost_bps=ap.get("cost_bps", 10.0),
        impact_bps=ap.get("impact_bps", 2.0),
        alpha_hurdle_bps=ap.get("alpha_hurdle_bps", 5.0),
        max_position_pct=ap.get("max_position_pct", 0.20),
        max_order_notional=ap.get("max_order_notional", 50000.0),
        min_trade_notional=ap.get("min_trade_notional", 50.0),
        allow_short=ap.get("allow_short", False),
        default_base_alpha_bps=ap.get("default_base_alpha_bps", 30.0),
    )
    evaluator = CostAwareAgentEvaluator(bridge=bridge, config=policy, ledger=ledger)

    sg = exec_cfg.get("signal_generator", {})
    sg_config = SignalGeneratorConfig(
        min_abs_aim_weight=(args.min_abs_weight if args.min_abs_weight is not None
                           else sg.get("min_abs_aim_weight", 0.0)),
        max_names=(args.max_names if args.max_names is not None
                  else sg.get("max_names", 50)),
        agent_epochs=sg.get("agent_epochs", research_cfg.get("agent", {}).get("epochs", 150)),
        agent_lr=sg.get("agent_lr", research_cfg.get("agent", {}).get("lr", 0.05)),
        agent_seed=sg.get("agent_seed", research_cfg.get("agent", {}).get("seed", 0)),
        cost_bps_per_side=sg.get("cost_bps_per_side",
                                 research_cfg["evaluation"]["cost_bps_per_side"]),
        min_train_months=sg.get("min_train_months", 24),
        strategy_name=sg.get("strategy_name", "diffpolicy_agent"),
    )

    map_path = args.permno_ticker_map or sg.get("permno_ticker_map_csv")
    permno_ticker_map = None
    if map_path:
        permno_ticker_map = load_permno_ticker_map(map_path)
        log.info("loaded %d permno->ticker mappings from %s", len(permno_ticker_map), map_path)
    elif data_mode == "osap":
        raise SystemExit(
            "data.mode=osap but no permno->ticker map was given (--permno-ticker-map "
            "or signal_generator.permno_ticker_map_csv in the execution config). "
            "OSAP/CRSP files ship permno, not a tradable symbol (CRSP license) -- "
            "without a map every name would be skipped."
        )
    else:
        log.warning("no permno->ticker map: using the panel's raw ticker column as the "
                    "order symbol (fine for synthetic demo wiring, not for real orders)")

    mode = "paper" if alpaca.paper else "live"
    log.info("mode=%s dry_run=%s broker=%s", mode, args.dry_run, alpaca.get_effective_base_url())

    report = run_live_signals(
        panel, signal_cols, evaluator, bridge,
        config=sg_config, permno_ticker_map=permno_ticker_map, dry_run=args.dry_run,
    )

    print(f"\nas of {report['as_of']} | "
          f"{report['n_candidates']} candidates, {report['n_evaluated']} evaluated, "
          f"{report['n_skipped']} skipped")
    if report["fit_meta"]:
        print(f"fit: {report['fit_meta']}")

    approved = [r for r in report["results"] if r["approved"]]
    print(f"\n{len(approved)} approved / {len(report['results'])} evaluated:")
    header = f"{'symbol':<10}{'action':<7}{'aim_wt':>9}  {'order?':<8}{'reason'}"
    print(header)
    for r in report["results"]:
        order_flag = "SENT" if r.get("order_submitted") else ("DRY-RUN" if r["approved"] else "-")
        print(f"{str(r['symbol']):<10}{r['action']:<7}{r['aim_weight']:>9.4f}  "
              f"{order_flag:<8}{r['reason']}")

    if report["skipped"]:
        print(f"\n{len(report['skipped'])} skipped before reaching the risk gate "
              f"(see --debug for the full list)")
        for s in report["skipped"][:10]:
            print(f"  permno={s.get('permno')} symbol={s.get('symbol')} "
                  f"aim={s.get('aim_weight', 0):.4f} reason={s['reason']}")


if __name__ == "__main__":
    main()
