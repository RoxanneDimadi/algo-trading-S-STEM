"""Factor-timing panel from OSAP long-short portfolio returns.

This is the no-WRDS path.

Firm-level CRSP returns are license-blocked without WRDS, which makes
``data.mode: osap`` unreachable. But OSAP *does* freely publish each
predictor's long-short portfolio return series -- so the refined real-data
design treats **each factor as the tradable asset**:

    ticker  = signalname (one per OSAP predictor)
    fwd_ret = that factor's LS return next month (decimal)
    signals = point-in-time features from the factor's OWN history
              (factor momentum / vol) plus publication-status features
              (the McLean-Pontiff angle)

This is the factor-momentum / factor-timing setting (Ehsani-Linnainmaa;
Gupta-Kelly), and it is PULSE's home turf on *real* data: the Kalman filter
tracks each factor's time-varying efficacy directly.

Point-in-time discipline (why the features are shaped this way):
  * All return-derived features at month t use returns THROUGH month t only;
    fwd_ret is month t+1. Same (signal_t, ret_{t->t+1}) convention as the
    rest of the project.
  * ``post_pub`` is 1 only strictly AFTER the publication date -- before
    publication you could not have known the paper was coming.
  * ``years_since_pub`` is clamped at 0 pre-publication for the same reason:
    a negative value would leak the future publication date backwards.
  * A post-SAMPLE-end flag is deliberately NOT a feature: the sample end
    only becomes public knowledge at publication, so conditioning on it
    pre-publication is lookahead. (It remains fine for the retrospective
    decay table below, which is descriptive, not a trading signal.)

Everything numeric is config-driven (``factor_panel:`` in data_config.yaml).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("real_data.factor_panel")

# ---------------------------------------------------------------------------
# loading + units
# ---------------------------------------------------------------------------


def load_ls_returns(portfolios_csv: Path, units: str = "auto") -> pd.DataFrame:
    """Long-short series per signal -> tidy [date, ticker, ret], DECIMAL.

    OSAP ships portfolio returns in PERCENT while the agent (and the French
    factor file) works in decimals; ``units`` may be 'auto', 'percent', or
    'decimal'. Auto-detection: median |ret| > 0.2 is treated as percent
    (monthly LS factor returns are a few percent, i.e. a few *hundredths*
    in decimal).
    """
    df = pd.read_csv(portfolios_csv)
    need = {"signalname", "port", "date", "ret"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{portfolios_csv} missing columns {sorted(missing)}")

    ls = df.loc[df["port"].astype(str).str.upper() == "LS",
                ["signalname", "date", "ret"]].copy()
    if ls.empty:
        raise ValueError(
            f"No 'LS' rows in {portfolios_csv}. Re-download portfolios with "
            "a form that includes long-short series (e.g. 'op')."
        )
    ls["date"] = pd.to_datetime(ls["date"]) + pd.offsets.MonthEnd(0)
    ls["ret"] = pd.to_numeric(ls["ret"], errors="coerce")
    ls = ls.dropna(subset=["ret"])

    med = float(ls["ret"].abs().median())
    if units == "auto":
        units = "percent" if med > 0.2 else "decimal"
        logger.info(
            "units auto-detected as %s (median |ret| = %.3f)", units, med)
    if units == "percent":
        ls["ret"] = ls["ret"] / 100.0
    elif units != "decimal":
        raise ValueError(f"units must be auto|percent|decimal, got {units!r}")

    out = ls.rename(columns={"signalname": "ticker"})
    # collapse accidental duplicates (one row per factor-month)
    out = (out.groupby(["ticker", "date"], as_index=False)["ret"].mean()
              .sort_values(["ticker", "date"]).reset_index(drop=True))
    logger.info("LS returns: %d factors, %d rows, %s..%s",
                out["ticker"].nunique(), len(out),
                out["date"].min().date(), out["date"].max().date())
    return out


def load_pub_dates(signal_doc_csv: Path) -> pd.DataFrame:
    """[ticker, sample_end, pub_date] from SignalDoc (Dec-31 of the years)."""
    doc = pd.read_csv(signal_doc_csv)
    cols = {c.lower(): c for c in doc.columns}
    acr, se, py = cols.get("acronym"), cols.get(
        "sampleendyear"), cols.get("year")
    if not (acr and se and py):
        raise ValueError(f"Unexpected SignalDoc schema: {list(doc.columns)}")
    out = pd.DataFrame({
        "ticker": doc[acr].astype(str),
        "sample_end": pd.to_datetime(
            doc[se].astype("Int64").astype(str) + "-12-31", errors="coerce"),
        "pub_date": pd.to_datetime(
            doc[py].astype("Int64").astype(str) + "-12-31", errors="coerce"),
    }).dropna(subset=["ticker"]).drop_duplicates("ticker")
    return out

# ---------------------------------------------------------------------------
# feature construction
# ---------------------------------------------------------------------------


def _add_history_features(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Return-history features per factor. All windows end AT month t.

    Implemented with groupby-transform (not groupby-apply) so behavior is
    identical across pandas 2.x/3.x grouping-column semantics.
    """
    panel = panel.sort_values(["ticker", "date"]).copy()
    gb = panel.groupby("ticker")["ret"]
    panel["fmom_1m"] = panel["ret"]                    # last month's LS return
    panel["fmom_12_2"] = gb.transform(                 # 12-2 style momentum
        lambda s: s.shift(1).rolling(lookback - 1).mean())
    panel["fmom_12m"] = gb.transform(lambda s: s.rolling(lookback).mean())
    panel["fvol_12m"] = gb.transform(lambda s: s.rolling(lookback).std())
    panel["fwd_ret"] = gb.shift(-1)
    return panel


def build_factor_panel(
    portfolios_csv: Path,
    signal_doc_csv: Optional[Path] = None,
    lookback: int = 12,
    min_names_per_month: int = 3,
    units: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the factor-timing panel and its feature-meta table.

    Returns (panel, meta):
      panel: [date, ticker, <features>, ret, fwd_ret]
      meta:  one row per FEATURE column with (empty) sample_end / pub_date --
             features like fmom_12m are constructs, not published anomalies,
             so the McLean-Pontiff decay stage does not apply to them and the
             agent pipeline skips it when these dates are absent.
    """
    ls = load_ls_returns(portfolios_csv, units=units)
    panel = _add_history_features(ls, lookback=lookback)

    features = ["fmom_1m", "fmom_12_2", "fmom_12m", "fvol_12m"]

    if signal_doc_csv is not None and Path(signal_doc_csv).exists():
        pub = load_pub_dates(Path(signal_doc_csv))
        panel = panel.merge(pub, on="ticker", how="left")
        yrs = (panel["date"] - panel["pub_date"]).dt.days / 365.25
        # point-in-time: nothing is known before publication
        # (see the module docstring)
        panel["post_pub"] = ((panel["date"] > panel["pub_date"])
                             .fillna(False).astype(float))
        panel["years_since_pub"] = yrs.clip(lower=0.0).fillna(0.0)
        panel = panel.drop(columns=["sample_end", "pub_date"])
        features += ["post_pub", "years_since_pub"]
    else:
        logger.warning("SignalDoc not provided; publication features omitted")

    panel = panel.dropna(subset=features + ["fwd_ret"])
    counts = panel.groupby("date")["ticker"].transform("size")
    panel = (panel.loc[counts >= min_names_per_month]
                  .sort_values(["ticker", "date"])
                  .reset_index(drop=True))
    panel = panel[["date", "ticker", *features, "ret", "fwd_ret"]]

    meta = pd.DataFrame({"signal": features,
                         "sample_end": pd.NaT, "pub_date": pd.NaT})
    logger.info("factor panel: %d rows, %d months, %d factors, features=%s",
                len(panel), panel["date"].nunique(),
                panel["ticker"].nunique(), features)
    return panel, meta

# ---------------------------------------------------------------------------
# real McLean-Pontiff decay on the downloaded factors (descriptive exhibit)
# ---------------------------------------------------------------------------


def factor_decay_table(
    portfolios_csv: Path,
    signal_doc_csv: Path,
    units: str = "auto",
    ann: int = 12,
) -> pd.DataFrame:
    """In-sample vs post-sample vs post-publication stats for each REAL factor.

    This is the McLean-Pontiff (2016) exhibit measured on actual OSAP data --
    the retrospective counterpart to the synthetic decay module. Retention =
    segment annualized return / in-sample annualized return.
    """
    ls = load_ls_returns(portfolios_csv, units=units)
    pub = load_pub_dates(signal_doc_csv)
    rows = []
    for tkr, g in ls.groupby("ticker"):
        p = pub.loc[pub["ticker"] == tkr]
        if p.empty or p[["sample_end", "pub_date"]].isna().any(axis=None):
            continue
        se, pd_ = p["sample_end"].iloc[0], p["pub_date"].iloc[0]
        segs = {
            "in_sample": g.loc[g["date"] <= se, "ret"],
            "post_sample": g.loc[(g["date"] > se) & (g["date"] <= pd_), "ret"],
            "post_pub": g.loc[g["date"] > pd_, "ret"],
        }
        row: Dict[str, Any] = {"signal": tkr, "sample_end": se.date(),
                               "pub_date": pd_.date()}
        for name, s in segs.items():
            n = len(s)
            mu = s.mean() * ann if n else np.nan
            sd = s.std() * np.sqrt(ann) if n > 1 else np.nan
            row[f"{name}_ann_ret"] = mu
            row[f"{name}_sharpe"] = mu / sd if sd and sd > 0 else np.nan
            row[f"{name}_n"] = n
        base = row["in_sample_ann_ret"]
        for name in ("post_sample", "post_pub"):
            row[f"{name}_retention"] = (
                row[f"{name}_ann_ret"] / base
                if base and np.isfinite(base) and abs(base) > 1e-12
                else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("signal").sort_index()
    logger.info("factor decay table: %d signals", len(out))
    return out
