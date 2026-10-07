"""Tests for src.data.loaders: the real-data (OSAP/CRSP) path.

Not exercised anywhere else offline (the repo's own docs say so), so these
construct minimal synthetic CSVs matching OSAP's wide-file / returns-file
schema rather than requiring network or WRDS access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loaders import build_panel_from_osap


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_panel_from_osap_nulls_fwd_ret_across_a_gap(tmp_path):
    """permno 1 has Jan/Feb/Apr (no Mar); permno 2 is contiguous Jan-Apr.
    shift(-1) is positional, so without a gap check Feb's fwd_ret would
    silently become Apr's return (a 2-month return mislabeled as 1-month).
    It must be NaN instead; permno 2's fwd_ret must be unaffected."""
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

    panel = build_panel_from_osap(str(signals_csv), str(returns_csv), ["sig"])

    p1 = panel[panel["ticker"] == "1"].set_index(panel[panel["ticker"] == "1"]["date"])
    jan1 = p1.loc[pd.Timestamp("2000-01-31"), "fwd_ret"]
    feb1 = p1.loc[pd.Timestamp("2000-02-29"), "fwd_ret"]
    assert jan1 == 0.02, "Jan->Feb is contiguous: fwd_ret must be Feb's return"
    assert pd.isna(feb1), "Feb->Apr skips March: fwd_ret must be NaN, not Apr's return"

    p2 = panel[panel["ticker"] == "2"].set_index(panel[panel["ticker"] == "2"]["date"])
    for d0, d1, expected in [
        ("2000-01-31", "2000-02-29", 0.12),
        ("2000-02-29", "2000-03-31", 0.13),
        ("2000-03-31", "2000-04-30", 0.14),
    ]:
        assert p2.loc[pd.Timestamp(d0), "fwd_ret"] == expected
    assert pd.isna(p2.loc[pd.Timestamp("2000-04-30"), "fwd_ret"]), "no data past the last row"


def test_build_panel_from_osap_dedupes_merge_keys(tmp_path):
    """A duplicate (permno, yyyymm) row in either source file must not
    silently multiply that ticker-month's weight in the merged panel."""
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

    panel = build_panel_from_osap(str(signals_csv), str(returns_csv), ["sig"])
    assert len(panel[(panel["ticker"] == "1") & (panel["date"] == pd.Timestamp("2000-01-31"))]) == 1
