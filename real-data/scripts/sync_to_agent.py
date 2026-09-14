"""Copy raw artifacts into multisignal-alpha/data/raw and write an OSAP config overlay."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import agent_raw_dir, agent_root, load_config, processed_dir, raw_dir, signal_list
from src.panel_build import build_panel, sync_to_agent, write_agent_config_snippet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--build-panel", action="store_true", help="also write processed panel parquet/csv")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    raw = raw_dir(cfg)
    agent_raw = agent_raw_dir(cfg)
    signals = signal_list(cfg)
    if "STreversal" not in signals:
        signals = signals + ["STreversal"]

    copied = sync_to_agent(raw, agent_raw)
    overlay = write_agent_config_snippet(agent_root(cfg), signals)
    print(f"synced {len(copied)} files -> {agent_raw}")
    print(f"overlay: {overlay}")

    if args.build_panel:
        panel = build_panel(
            raw / "signed_predictors_dl_wide.csv",
            raw / "returns.csv",
            signals,
            add_streversal=True,
            min_names_per_month=int(cfg.get("panel", {}).get("min_names_per_month", 50)),
        )
        out = processed_dir(cfg) / "panel.parquet"
        try:
            panel.to_parquet(out, index=False)
        except Exception:
            out = processed_dir(cfg) / "panel.csv"
            panel.to_csv(out, index=False)
        print(f"panel: {out} ({len(panel)} rows)")


if __name__ == "__main__":
    main()
