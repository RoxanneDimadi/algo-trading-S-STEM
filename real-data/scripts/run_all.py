"""Run the full real-data ingest: OSAP -> French -> returns -> sync to agent."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import agent_raw_dir, agent_root, load_config, processed_dir, raw_dir, signal_list
from src.french_download import run_french_download
from src.osap_download import run_osap_download
from src.panel_build import (
    augment_osap_with_streversal,
    build_panel,
    sync_to_agent,
    write_agent_config_snippet,
)
from src.returns import resolve_returns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--skip-osap", action="store_true")
    p.add_argument("--skip-french", action="store_true")
    p.add_argument("--skip-returns", action="store_true",
                   help="skip returns resolution (OSAP+French only)")
    p.add_argument("--skip-sync", action="store_true")
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
                augment_osap_with_streversal(pristine, returns_path, signals_csv)
        except FileNotFoundError as e:
            log.error("%s", e)
            log.error("continuing without returns — sync will omit returns.csv / STreversal")

    if not args.skip_sync:
        log.info("=== sync to agent ===")
        signals = signal_list(cfg)
        if (raw / "returns.csv").exists() and "STreversal" not in signals:
            signals = signals + ["STreversal"]
        sync_to_agent(raw, agent_raw_dir(cfg))
        write_agent_config_snippet(agent_root(cfg), signals)

    if args.build_panel and (raw / "returns.csv").exists():
        log.info("=== panel ===")
        signals = signal_list(cfg)
        if "STreversal" not in signals:
            signals = signals + ["STreversal"]
        panel = build_panel(
            raw / "signed_predictors_dl_wide.csv",
            raw / "returns.csv",
            signals,
            min_names_per_month=int(cfg.get("panel", {}).get("min_names_per_month", 50)),
        )
        out = processed_dir(cfg) / "panel.csv"
        panel.to_csv(out, index=False)
        log.info("wrote %s", out)

    log.info("done")


if __name__ == "__main__":
    main()
