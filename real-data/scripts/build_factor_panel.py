"""Build the factor-timing panel (the no-WRDS real-data path) and sync it.

Outputs:
  data/processed/factor_panel.csv
      -- [date, ticker, features..., ret, fwd_ret]
  data/processed/factor_meta.csv    -- feature meta for the agent's decay guard
  data/processed/factor_decay.csv   -- REAL McLean-Pontiff decay per factor
  <agent>/data/raw/factor_panel.csv, factor_meta.csv (synced)
  <agent>/configs/config_factor.yaml -- complete, runnable agent config

Then, from the agent repo:
  python -m src.pipeline --config configs/config_factor.yaml
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
from src.panel_build import sync_to_agent, write_agent_factor_config
from src.factor_panel import build_factor_panel, factor_decay_table
from src import agent_raw_dir, agent_root, load_config, processed_dir, raw_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--no-sync", action="store_true",
                   help="build locally only; don't copy into the agent repo")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("real_data.build_factor_panel")

    cfg = load_config(args.config)
    fcfg = cfg.get("factor_panel", {})
    raw, processed = raw_dir(cfg), processed_dir(cfg)

    form = cfg.get("osap", {}).get("portfolio_form", "op")
    portfolios_csv = raw / f"osap_portfolios_{form}.csv"
    signal_doc_csv = raw / "SignalDoc.csv"
    if not portfolios_csv.exists():
        sys.exit(
            f"missing {portfolios_csv} -- run scripts/download_osap.py first")

    panel, meta = build_factor_panel(
        portfolios_csv,
        signal_doc_csv if signal_doc_csv.exists() else None,
        lookback=int(fcfg.get("lookback_months", 12)),
        min_names_per_month=int(fcfg.get("min_names_per_month", 3)),
        units=str(fcfg.get("units", "auto")),
    )
    panel_path = processed / "factor_panel.csv"
    meta_path = processed / "factor_meta.csv"
    panel.to_csv(panel_path, index=False)
    meta.to_csv(meta_path, index=False)
    log.info("wrote %s (%d rows) and %s", panel_path, len(panel), meta_path)

    if signal_doc_csv.exists():
        decay = factor_decay_table(portfolios_csv, signal_doc_csv,
                                   units=str(fcfg.get("units", "auto")))
        decay_path = processed / "factor_decay.csv"
        decay.to_csv(decay_path)
        log.info("wrote real McLean-Pontiff decay exhibit -> %s", decay_path)
        cols = [c for c in ("in_sample_sharpe", "post_sample_sharpe",
                            "post_pub_sharpe", "post_pub_retention")
                if c in decay.columns]
        print("\n[real factor decay]\n" + decay[cols].round(3).to_string())

    if not args.no_sync:
        agent_raw = agent_raw_dir(cfg)
        sync_to_agent(processed, agent_raw,
                      files=["factor_panel.csv", "factor_meta.csv"])
        n_factors = panel["ticker"].nunique()
        out_cfg = write_agent_factor_config(
            agent_root(cfg), n_factors=n_factors)
        print(f"\nNext: cd {agent_root(cfg)} && "
              f"python -m src.pipeline --config configs/{out_cfg.name}")


if __name__ == "__main__":
    main()
