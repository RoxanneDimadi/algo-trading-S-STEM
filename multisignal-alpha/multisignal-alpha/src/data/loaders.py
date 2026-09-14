"""Real-data loaders: Chen-Zimmermann OSAP + Kenneth French factors.

The synthetic mode needs none of this. When you graduate to real data, the
expected workflow (documented in README §Real data) is:

  1. OSAP signals -- either
       a) `pip install openassetpricing` and download programmatically, or
       b) download `signed_predictors_dl_wide.csv` (Oct-2025 release or
          later) + `SignalDoc.csv` from openassetpricing.com into data/raw/.
     The wide file is (permno, yyyymm, <one column per signal>), and the
     signals are PRE-SIGNED so that higher value => higher expected return.
  2. Returns -- a CSV with columns (permno, yyyymm, ret) from CRSP via WRDS.
     NOTE: Price, Size, and STreversal are NOT distributed in the OSAP file
     (CRSP license); STreversal is simply the prior 1-month return, so it can
     be computed from any price feed if you lack WRDS.
  3. Factors -- Fama-French 5 + momentum from Ken French's library, via
     pandas_datareader or manual CSV.

Point-in-time discipline for OSAP data: a signal stamped yyyymm is built
from information available at the END of that month, and OSAP's portfolio
convention pairs it with the NEXT month's return. `build_panel_from_osap`
reproduces exactly that: signal at t joined to fwd_ret over (t, t+1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _yyyymm_to_date(yyyymm: pd.Series) -> pd.Series:
    s = yyyymm.astype(int).astype(str)
    return pd.to_datetime(s, format="%Y%m") + pd.offsets.MonthEnd(0)


def load_osap_wide_csv(path: str, signals: list[str]) -> pd.DataFrame:
    """Load selected signals from the OSAP wide file into long panel form
    [date, ticker, <signals...>] (ticker = permno as string)."""
    usecols = ["permno", "yyyymm", *signals]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    missing = [s for s in signals if s not in df.columns]
    if missing:
        raise ValueError(
            f"Signals not in OSAP file: {missing}. If one of them is "
            "Price/Size/STreversal, those are CRSP-licensed and must be "
            "constructed from your returns file (STreversal = prior-month "
            "return) or obtained via WRDS."
        )
    df["date"] = _yyyymm_to_date(df["yyyymm"])
    df["ticker"] = df["permno"].astype(int).astype(str)
    return df[["date", "ticker", *signals]]


def load_returns_csv(path: str) -> pd.DataFrame:
    """CRSP-style returns file: (permno, yyyymm, ret) -> [date, ticker, ret]."""
    df = pd.read_csv(path)
    df["date"] = _yyyymm_to_date(df["yyyymm"])
    df["ticker"] = df["permno"].astype(int).astype(str)
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    return df[["date", "ticker", "ret"]]


def build_panel_from_osap(signals_csv: str, returns_csv: str,
                          signals: list[str]) -> pd.DataFrame:
    """Join OSAP signals with returns and construct fwd_ret with the correct
    (signal_t, return_{t->t+1}) alignment."""
    sig = load_osap_wide_csv(signals_csv, signals)
    ret = load_returns_csv(returns_csv)
    panel = sig.merge(ret, on=["date", "ticker"], how="inner")
    panel = panel.sort_values(["ticker", "date"])
    panel["fwd_ret"] = panel.groupby("ticker")["ret"].shift(-1)
    return panel.reset_index(drop=True)


def load_signal_doc(path: str) -> pd.DataFrame:
    """Signal metadata for the decay analysis, from OSAP's SignalDoc.csv.

    Maps SampleEndYear -> sample_end (Dec-31 of that year) and Year
    (publication year) -> pub_date. Column names occasionally shift across
    releases; this loader is defensive and tells you what it found.
    """
    doc = pd.read_csv(path)
    cols = {c.lower(): c for c in doc.columns}
    acr = cols.get("acronym")
    se = cols.get("sampleendyear")
    py = cols.get("year")
    if not (acr and se and py):
        raise ValueError(f"Unexpected SignalDoc schema; columns = {list(doc.columns)}")
    out = pd.DataFrame({
        "signal": doc[acr],
        "sample_end": pd.to_datetime(doc[se].astype("Int64").astype(str) + "-12-31",
                                     errors="coerce"),
        "pub_date": pd.to_datetime(doc[py].astype("Int64").astype(str) + "-12-31",
                                   errors="coerce"),
    }).dropna(subset=["signal"]).set_index("signal")
    return out


def load_french_factors(start: str = "1963-07-01",
                        csv_path: str | None = None) -> pd.DataFrame:
    """FF5 + momentum, monthly, as decimal returns indexed by month-end.

    Prefer a local CSV written by the real-data ingest repo
    (``data/raw/french_factors.csv``). Otherwise pull via pandas_datareader.
    """
    from pathlib import Path

    candidates = []
    if csv_path:
        candidates.append(Path(csv_path))
    candidates.append(Path("data/raw/french_factors.csv"))

    for path in candidates:
        if path.exists():
            f = pd.read_csv(path, index_col=0, parse_dates=True)
            f.index = pd.to_datetime(f.index) + pd.offsets.MonthEnd(0)
            f.index.name = "date"
            f.columns = [c.strip().lower().replace("-", "_") for c in f.columns]
            start_ts = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
            f = f.loc[f.index >= start_ts]
            return f.drop(columns=[c for c in ["rf"] if c in f.columns], errors="ignore")

    from pandas_datareader import data as pdr  # lazy optional import
    ff5 = pdr.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench",
                         start=start)[0] / 100.0
    mom = pdr.DataReader("F-F_Momentum_Factor", "famafrench", start=start)[0] / 100.0
    f = ff5.join(mom, how="inner")
    f.index = f.index.to_timestamp("M") + pd.offsets.MonthEnd(0)
    f.columns = [c.strip().lower().replace("-", "_") for c in f.columns]
    f = f.rename(columns={"mkt_rf": "mkt_rf", "mom": "mom"})
    f.index.name = "date"
    return f.drop(columns=[c for c in ["rf"] if c in f.columns])
