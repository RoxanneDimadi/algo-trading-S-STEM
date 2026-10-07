"""Regression test for build_panel's point-in-time fwd_ret alignment.

This ingest repo ships with no test infrastructure otherwise; run with
`python -m pytest tests/` after `pip install pytest` (not in
requirements.txt -- it's a dev-only tool, not needed to run the ingest
scripts themselves).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.panel_build import build_panel


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_panel_nulls_fwd_ret_across_a_gap(tmp_path):
    """permno 1 has Jan/Feb/Apr (no Mar); permno 2 is contiguous Jan-Apr.
    shift(-1) is positional, so without a gap check Feb's fwd_ret would
    silently become Apr's return (a 2-month return mislabeled as 1-month)."""
    signals_csv = tmp_path / "signals.csv"
    returns_csv = tmp_path / "returns.csv"

    _write(signals_csv, [
        {"permno": 1, "yyyymm": 200001, "sig": 0.1},
        {"permno": 1, "yyyymm": 200002, "sig": 0.2},
        {"permno": 1, "yyyymm": 200004, "sig": 0.4},
        {"permno": 2, "yyyymm": 200001, "sig": 1.1},
        {"permno": 2, "yyyymm": 200002, "sig": 1.2},
        {"permno": 2, "yyyymm": 200003, "sig": 1.3},
        {"permno": 2, "yyyymm": 200004, "sig": 1.4},
    ])
    _write(returns_csv, [
        {"permno": 1, "yyyymm": 200001, "ret": 0.01},
        {"permno": 1, "yyyymm": 200002, "ret": 0.02},
        {"permno": 1, "yyyymm": 200004, "ret": 0.04},
        {"permno": 2, "yyyymm": 200001, "ret": 0.11},
        {"permno": 2, "yyyymm": 200002, "ret": 0.12},
        {"permno": 2, "yyyymm": 200003, "ret": 0.13},
        {"permno": 2, "yyyymm": 200004, "ret": 0.14},
    ])

    panel = build_panel(signals_csv, returns_csv, ["sig"],
                        add_streversal=False, min_names_per_month=1)

    p1 = panel[panel["ticker"] == "1"].set_index(panel[panel["ticker"] == "1"]["date"])
    assert p1.loc[pd.Timestamp("2000-01-31"), "fwd_ret"] == 0.02
    assert pd.isna(p1.loc[pd.Timestamp("2000-02-29"), "fwd_ret"]), \
        "Feb->Apr skips March: fwd_ret must be NaN, not Apr's return"

    p2 = panel[panel["ticker"] == "2"].set_index(panel[panel["ticker"] == "2"]["date"])
    assert p2.loc[pd.Timestamp("2000-02-29"), "fwd_ret"] == 0.13


def test_build_panel_dedupes_merge_keys(tmp_path):
    signals_csv = tmp_path / "signals.csv"
    returns_csv = tmp_path / "returns.csv"
    _write(signals_csv, [
        {"permno": 1, "yyyymm": 200001, "sig": 0.1},
        {"permno": 1, "yyyymm": 200001, "sig": 0.1},  # duplicate row
        {"permno": 1, "yyyymm": 200002, "sig": 0.2},
    ])
    _write(returns_csv, [
        {"permno": 1, "yyyymm": 200001, "ret": 0.01},
        {"permno": 1, "yyyymm": 200002, "ret": 0.02},
    ])
    panel = build_panel(signals_csv, returns_csv, ["sig"],
                        add_streversal=False, min_names_per_month=1)
    assert len(panel[(panel["ticker"] == "1") & (panel["date"] == pd.Timestamp("2000-01-31"))]) == 1
