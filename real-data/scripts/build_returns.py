"""Resolve monthly equity returns (CSV / WRDS) and build STreversal."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import load_config, processed_dir, raw_dir
from src.panel_build import augment_osap_with_streversal
from src.returns import resolve_returns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    raw = raw_dir(cfg)
    try:
        returns_path = resolve_returns(cfg, raw)
    except FileNotFoundError as e:
        logging.getLogger("real_data.returns").error("%s", e)
        raise SystemExit(1) from e
    print(f"returns: {returns_path}")

    signals_csv = raw / "signed_predictors_dl_wide.csv"
    if signals_csv.exists():
        pristine = raw / "signed_predictors_osap_only.csv"
        if not pristine.exists():
            pristine.write_bytes(signals_csv.read_bytes())
        out = processed_dir(cfg) / "signed_predictors_with_streversal.csv"
        augment_osap_with_streversal(pristine, returns_path, out)
        augment_osap_with_streversal(pristine, returns_path, signals_csv)
        print(f"streversal merged into {signals_csv}")
    else:
        print("OSAP signals CSV not found yet — run scripts/download_osap.py first")


if __name__ == "__main__":
    main()
