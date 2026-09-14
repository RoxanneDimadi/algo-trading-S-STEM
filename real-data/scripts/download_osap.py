"""Download OSAP firm signals + SignalDoc (+ optional portfolios)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import load_config, raw_dir
from src.osap_download import run_osap_download


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
    paths = run_osap_download(cfg, raw_dir(cfg))
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
