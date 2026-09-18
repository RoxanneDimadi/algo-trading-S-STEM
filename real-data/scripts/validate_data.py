"""Validate raw/processed data and report which agent modes are runnable.

    python scripts/validate_data.py

Checks every expected file for presence, schema, date coverage, NaN rates,
and unit sanity, then prints a readiness verdict per agent data mode:

  synthetic  -- always runnable (no data needed)
  factor     -- OSAP LS portfolios + SignalDoc      (no WRDS needed)
  osap       -- firm-level; needs returns.csv (CRSP/WRDS-licensed)

Exit code 0 if at least one real-data mode is runnable, 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import agent_raw_dir, load_config, processed_dir, raw_dir

OK, BAD, WARN = "[ok]  ", "[MISS]", "[warn]"


def _report(lines, status, msg):
    lines.append(f"  {status} {msg}")
    print(lines[-1])


def check_csv(path: Path, need_cols, lines, date_col=None, unit_col=None):
    """Presence + schema + coverage + NaN + units for one CSV. Returns bool."""
    if not path.exists():
        _report(lines, BAD, f"{path.name}: not found")
        return False
    try:
        df = pd.read_csv(path, nrows=200_000)
    except Exception as e:  # noqa: BLE001
        _report(lines, BAD, f"{path.name}: unreadable ({e})")
        return False
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        _report(lines, BAD, f"{path.name}: missing columns {missing}")
        return False
    bits = [f"{len(df):,}+ rows"]
    if date_col and date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        bits.append(f"{d.min().date()}..{d.max().date()}")
    nan_pct = df[list(need_cols)].isna().mean().mean() * 100
    bits.append(f"{nan_pct:.1f}% NaN in key cols")
    if unit_col and unit_col in df.columns:
        med = pd.to_numeric(df[unit_col], errors="coerce").abs().median()
        bits.append(f"units look like {'PERCENT' if med > 0.2 else 'decimal'} "
                    f"(median |{unit_col}|={med:.3f})")
    _report(lines, OK, f"{path.name}: " + ", ".join(bits))
    return True


def main() -> int:
    cfg = load_config(None)
    raw, processed = raw_dir(cfg), processed_dir(cfg)
    agent_raw = agent_raw_dir(cfg)
    form = cfg.get("osap", {}).get("portfolio_form", "op")
    lines: list[str] = []

    print(f"\n=== raw ingest files ({raw}) ===")
    have_wide = check_csv(raw / "signed_predictors_dl_wide.csv",
                          ["permno", "yyyymm"], lines)
    have_doc = check_csv(raw / "SignalDoc.csv",
                         ["Acronym", "Year", "SampleEndYear"], lines)
    have_ports = check_csv(raw / f"osap_portfolios_{form}.csv",
                           ["signalname", "port", "date", "ret"], lines,
                           date_col="date", unit_col="ret")
    have_french = check_csv(raw / "french_factors.csv",
                            ["date", "mkt_rf", "mom"], lines,
                            date_col="date", unit_col="mkt_rf")
    have_returns = check_csv(raw / "returns.csv",
                             ["permno", "yyyymm", "ret"], lines)

    print(f"\n=== processed ({processed}) ===")
    have_panel = check_csv(processed / "factor_panel.csv",
                           ["date", "ticker", "ret", "fwd_ret"], lines,
                           date_col="date", unit_col="fwd_ret")
    check_csv(processed / "factor_decay.csv", [], lines)

    print(f"\n=== synced to agent ({agent_raw}) ===")
    for name in ("factor_panel.csv", "factor_meta.csv", "french_factors.csv"):
        p = agent_raw / name
        _report(lines, OK if p.exists() else WARN,
                f"{name}: {'synced' if p.exists() else 'not synced yet'}")

    print("\n=== agent-mode readiness ===")
    print("  synthetic : RUNNABLE (needs no data; `make demo` in the agent repo)")
    factor_ready = have_ports and have_doc and have_french
    print(f"  factor    : {'RUNNABLE' if factor_ready else 'blocked'} "
          "(OSAP LS portfolios as tradable assets -- no WRDS needed)"
          + ("" if have_panel else
             " -- run scripts/build_factor_panel.py to (re)build the panel"))
    print(f"  osap      : {'RUNNABLE' if (have_wide and have_returns) else 'blocked'} "
          "(firm-level; returns.csv requires CRSP via WRDS or your own "
          "permno-keyed returns file)")
    if not have_returns:
        print("              -> without WRDS this stays blocked by design; "
              "the factor mode is the supported real-data path.")
    return 0 if factor_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
