"""CRSP-style monthly returns: existing CSV, optional WRDS, STreversal construction."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("real_data.returns")


def _yyyymm_to_month_end(yyyymm: pd.Series) -> pd.Series:
    s = yyyymm.astype(int).astype(str)
    return pd.to_datetime(s, format="%Y%m") + pd.offsets.MonthEnd(0)


def load_returns_csv(path: Path, prefer_delisting: bool = True) -> pd.DataFrame:
    """Load (permno, yyyymm, ret[, dlret]) -> date, ticker, ret (delisting-adjusted if possible)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    for need in ("permno", "yyyymm", "ret"):
        if need not in cols:
            raise ValueError(f"{path} missing column '{need}'; have {list(df.columns)}")

    out = pd.DataFrame({
        "permno": df[cols["permno"]].astype(int),
        "yyyymm": df[cols["yyyymm"]].astype(int),
        "ret": pd.to_numeric(df[cols["ret"]], errors="coerce"),
    })
    if prefer_delisting and "dlret" in cols:
        dl = pd.to_numeric(df[cols["dlret"]], errors="coerce")
        # CRSP convention: (1+ret)*(1+dlret)-1 when both present
        adj = (1.0 + out["ret"].fillna(0.0)) * (1.0 + dl.fillna(0.0)) - 1.0
        use_dl = dl.notna()
        out.loc[use_dl, "ret"] = adj[use_dl]
        logger.info("applied dlret adjustment on %d rows", int(use_dl.sum()))

    out["date"] = _yyyymm_to_month_end(out["yyyymm"])
    out["ticker"] = out["permno"].astype(str)
    return out[["date", "ticker", "permno", "yyyymm", "ret"]]


def fetch_wrds_monthly_returns(
    username: str,
    password: Optional[str] = None,
    start_yyyymm: int = 198001,
) -> pd.DataFrame:
    """Pull CRSP monthly stock file via WRDS (requires `pip install wrds`)."""
    try:
        import wrds
    except ImportError as e:
        raise ImportError("WRDS requested but wrds package not installed: pip install wrds") from e

    logger.info("connecting to WRDS as %s", username)
    db = wrds.Connection(wrds_username=username, wrds_password=password or None)
    # msf: monthly stock file; msenames for share filters optional
    sql = f"""
        SELECT a.permno, a.date, a.ret, a.retx, b.dlret
        FROM crsp.msf AS a
        LEFT JOIN crsp.msedelist AS b
          ON a.permno = b.permno
         AND date_trunc('month', a.date) = date_trunc('month', b.dlstdt)
        WHERE a.date >= '{str(start_yyyymm)[:4]}-{str(start_yyyymm)[4:]}-01'
          AND a.ret IS NOT NULL
    """
    # Simpler reliable pull: msf only, then merge delist separately if needed
    sql = f"""
        SELECT permno, date, ret
        FROM crsp.msf
        WHERE date >= '{str(start_yyyymm)[:4]}-01-01'
          AND ret IS NOT NULL
    """
    raw = db.raw_sql(sql, date_cols=["date"])
    db.close()

    raw["yyyymm"] = raw["date"].dt.year * 100 + raw["date"].dt.month
    raw["ret"] = pd.to_numeric(raw["ret"], errors="coerce")
    out = pd.DataFrame({
        "permno": raw["permno"].astype(int),
        "yyyymm": raw["yyyymm"].astype(int),
        "ret": raw["ret"],
    })
    logger.info("WRDS returned %d monthly rows", len(out))
    return out


def build_streversal(returns: pd.DataFrame) -> pd.DataFrame:
    """STreversal = prior 1-month return (signed later by OSAP convention; here raw lag).

    OSAP signs predictors so higher => higher expected return; short-term reversal
    is typically signed as *minus* last month's return. We emit -ret_{t-1}.
    """
    r = returns.sort_values(["ticker", "date"]).copy()
    r["STreversal"] = -r.groupby("ticker")["ret"].shift(1)
    return r[["date", "ticker", "permno", "yyyymm", "STreversal"]]


def resolve_returns(cfg: Dict[str, Any], raw: Path) -> Path:
    """Write data/raw/returns.csv from env path, WRDS, or existing file."""
    out = raw / "returns.csv"
    prefer_dl = bool(cfg.get("returns", {}).get("prefer_delisting_adjusted", True))

    configured = cfg.get("returns", {}).get("csv") or os.environ.get("RETURNS_CSV", "").strip()
    if configured:
        src = Path(configured)
        if not src.is_absolute():
            src = (raw.parent.parent / src).resolve()
        df = load_returns_csv(src, prefer_delisting=prefer_dl)
        export = df[["permno", "yyyymm", "ret"]]
        export.to_csv(out, index=False)
        logger.info("copied returns from %s -> %s (%d rows)", src, out, len(export))
        return out

    if out.exists() and out.stat().st_size > 0:
        logger.info("using existing %s", out)
        return out

    user = os.environ.get("WRDS_USERNAME", "").strip()
    password = os.environ.get("WRDS_PASSWORD", "").strip() or None
    if user:
        pulled = fetch_wrds_monthly_returns(user, password=password)
        pulled.to_csv(out, index=False)
        logger.info("wrote WRDS returns -> %s", out)
        return out

    raise FileNotFoundError(
        "No returns file found. Either:\n"
        "  1) set RETURNS_CSV in .env to a CSV with columns permno,yyyymm,ret\n"
        "  2) set WRDS_USERNAME / WRDS_PASSWORD and pip install wrds\n"
        "  3) place data/raw/returns.csv yourself (CRSP monthly)\n"
        "STreversal and the agent panel need this file; OSAP signals alone are not enough."
    )
