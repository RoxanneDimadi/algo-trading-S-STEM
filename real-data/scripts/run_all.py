"""Run the full real-data ingest.

OSAP -> French -> returns -> sync to agent.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The project package lives one level up; the bootstrap above
# has to run before these imports resolve.
# pylint: disable=wrong-import-position
from src.returns import resolve_returns
from src.panel_build import (
    augment_osap_with_streversal,
    build_panel,
    sync_to_agent,
    write_agent_config_snippet,
    write_agent_factor_config,
)
from src.osap_download import run_osap_download
from src.french_download import run_french_download
from src.factor_panel import build_factor_panel, factor_decay_table
from src import (agent_raw_dir, agent_root, load_config, processed_dir,
                 raw_dir, signal_list)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--skip-osap", action="store_true")
    p.add_argument("--skip-french", action="store_true")
    p.add_argument("--skip-returns", action="store_true",
                   help="skip returns resolution (OSAP+French only)")
    p.add_argument("--skip-sync", action="store_true")
    p.add_argument("--skip-factor-panel", action="store_true",
                   help="skip building the factor-mode panel")
    p.add_argument("--build-panel", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("real_data.run_all")
    cfg = load_config(args.config)
    raw = raw_dir(cfg)

    if not args.skip_osap:
        log.info("=== OSAP ===")
        run_osap_download(cfg, raw)

    if not args.skip_french:
        log.info("=== Kenneth French ===")
        run_french_download(cfg, raw)

    if not args.skip_returns:
        log.info("=== returns / STreversal ===")
        try:
            returns_path = resolve_returns(cfg, raw)
            signals_csv = raw / "signed_predictors_dl_wide.csv"
            if signals_csv.exists():
                # keep pristine OSAP dump
                pristine = raw / "signed_predictors_osap_only.csv"
                if not pristine.exists():
                    pristine.write_bytes(signals_csv.read_bytes())
                augment_osap_with_streversal(
                    pristine, returns_path, signals_csv)
        except FileNotFoundError as e:
            log.error("%s", e)
            log.error(
                "continuing without returns - sync will omit "
                "returns.csv / STreversal")

    if not args.skip_sync:
        log.info("=== sync to agent ===")
        signals = signal_list(cfg)
        if (raw / "returns.csv").exists() and "STreversal" not in signals:
            signals = signals + ["STreversal"]
        sync_to_agent(raw, agent_raw_dir(cfg))
        write_agent_config_snippet(agent_root(cfg), signals)

    # Factor-mode panel: the real-data path that needs NO returns.csv / WRDS.
    if not args.skip_factor_panel:
        form = cfg.get("osap", {}).get("portfolio_form", "op")
        ports = raw / f"osap_portfolios_{form}.csv"
        if ports.exists():
            log.info("=== factor panel (no-WRDS path) ===")
            fcfg = cfg.get("factor_panel", {})
            panel, meta = build_factor_panel(
                ports,
                raw / "SignalDoc.csv" if (raw /
                                          "SignalDoc.csv").exists() else None,
                lookback=int(fcfg.get("lookback_months", 12)),
                min_names_per_month=int(fcfg.get("min_names_per_month", 3)),
                units=str(fcfg.get("units", "auto")),
            )
            processed = processed_dir(cfg)
            panel.to_csv(processed / "factor_panel.csv", index=False)
            meta.to_csv(processed / "factor_meta.csv", index=False)
            if (raw / "SignalDoc.csv").exists():
                factor_decay_table(ports, raw / "SignalDoc.csv",
                                   units=str(fcfg.get("units", "auto"))
                                   ).to_csv(processed / "factor_decay.csv")
            if not args.skip_sync:
                sync_to_agent(processed, agent_raw_dir(cfg),
                              files=["factor_panel.csv", "factor_meta.csv"])
                write_agent_factor_config(agent_root(cfg),
                                          n_factors=panel["ticker"].nunique())
        else:
            log.warning(
                "no %s -- skipping factor panel (run download_osap first)",
                ports.name)

    if args.build_panel and (raw / "returns.csv").exists():
        log.info("=== panel ===")
        signals = signal_list(cfg)
        if "STreversal" not in signals:
            signals = signals + ["STreversal"]
        panel = build_panel(
            raw / "signed_predictors_dl_wide.csv",
            raw / "returns.csv",
            signals,
            min_names_per_month=int(
                cfg.get("panel", {}).get("min_names_per_month", 50)),
        )
        out = processed_dir(cfg) / "panel.csv"
        panel.to_csv(out, index=False)
        log.info("wrote %s", out)

    log.info("done")


if __name__ == "__main__":
    main()
