"""Kenneth French Data Library download (FF5 + momentum).

Primary path: ZIP files from
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Fallback: pandas_datareader's famafrench reader.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger("real_data.french")

FRENCH_FTP = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
# Monthly CSV zips on the French site
DATASET_ZIPS = {
    "F-F_Research_Data_5_Factors_2x3":
        "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "F-F_Momentum_Factor": "F-F_Momentum_Factor_CSV.zip",
}


def _month_end_index(idx: pd.Index) -> pd.DatetimeIndex:
    """French files use YYYYMM integers (or PeriodIndex) -> month-end."""
    if isinstance(idx, pd.PeriodIndex):
        return (idx.to_timestamp("M") + pd.offsets.MonthEnd(0))
    # integer YYYYMM
    s = pd.Series(idx).astype(str).str.replace(r"\D", "", regex=True)
    return pd.to_datetime(s, format="%Y%m") + pd.offsets.MonthEnd(0)


def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the quirky Ken French monthly CSV.

    Skips the header prose and stops at the annual block.
    """
    lines = text.splitlines()
    # Find first line that looks like a header with factor names
    start = None
    for i, line in enumerate(lines):
        low = line.lower()
        if "mkt-rf" in low or ("mom" in low
                               and "rf" not in low.split(",")[0].lower()):
            # momentum file header is often just ",Mom"
            start = i
            break
        if line.strip().startswith(",") and any(
            k in low for k in ("mkt", "smb", "hml", "rmw", "cma", "mom")
        ):
            start = i
            break
    if start is None:
        # FF5: first data-looking line after blank; try classic layout
        for i, line in enumerate(lines):
            if "Mkt-RF" in line or "Mkt-Rf" in line:
                start = i
                break
    if start is None:
        raise ValueError("could not find French CSV header")

    # Truncate at annual section / copyright footer
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            # blank line often separates monthly from annual
            # peek ahead: if next non-empty looks like a year-only block, cut
            for j in range(i + 1, min(i + 5, len(lines))):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                if (nxt.lower().startswith("annual")
                        or "copyright" in nxt.lower()):
                    end = i
                    break
                # annual rows are YYYY (4 digits) not YYYYMM
                first = nxt.split(",")[0].strip()
                if first.isdigit() and len(first) == 4:
                    end = i
                break
            if end != len(lines):
                break
        if (stripped.lower().startswith("annual")
                or "copyright" in stripped.lower()):
            end = i
            break

    chunk = "\n".join(lines[start:end])
    df = pd.read_csv(io.StringIO(chunk), index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    df.index = _month_end_index(df.index)
    df.index.name = "date"
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]
    return df / 100.0  # percent -> decimal


def download_zip_dataset(name: str, dest_dir: Path) -> pd.DataFrame:
    zip_name = DATASET_ZIPS[name]
    url = f"{FRENCH_FTP}/{zip_name}"
    logger.info("GET %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / zip_name
    zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"no CSV inside {zip_name}")
        text = zf.read(members[0]).decode("latin-1")
        (dest_dir / members[0]).write_text(text, encoding="utf-8")
    return _parse_french_csv(text)


def download_via_datareader(name: str, start: str) -> pd.DataFrame:
    from pandas_datareader import data as pdr

    raw = pdr.DataReader(name, "famafrench", start=start)[0] / 100.0
    raw.index = _month_end_index(raw.index)
    raw.index.name = "date"
    raw.columns = [c.strip().lower().replace("-", "_") for c in raw.columns]
    return raw


def load_french_factors(
    start: str = "1963-07",
    raw_dir: Optional[Path] = None,
    prefer_direct: bool = True,
) -> pd.DataFrame:
    """FF5 + momentum, decimal monthly returns, month-end index."""
    cache = raw_dir or Path("data/raw")
    frames = {}

    for name in ("F-F_Research_Data_5_Factors_2x3", "F-F_Momentum_Factor"):
        df = None
        if prefer_direct:
            try:
                df = download_zip_dataset(name, cache / "french")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "direct French download failed for %s: %s", name, e)
        if df is None:
            logger.info("falling back to pandas_datareader for %s", name)
            df = download_via_datareader(name, start=start)
        frames[name] = df

    ff5 = frames["F-F_Research_Data_5_Factors_2x3"]
    mom = frames["F-F_Momentum_Factor"]
    # momentum column naming varies (mom / mom   )
    mom_col = [c for c in mom.columns if "mom" in c][0]
    out = ff5.join(mom[[mom_col]].rename(
        columns={mom_col: "mom"}), how="inner")
    # drop rf for controls (kept in file if needed)
    start_ts = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    out = out.loc[out.index >= start_ts]
    return out


def run_french_download(cfg: Dict[str, Any], raw: Path) -> Path:
    fcfg = cfg.get("french", {})
    factors = load_french_factors(
        start=str(fcfg.get("start", "1963-07")),
        raw_dir=raw,
        prefer_direct=True,
    )
    out = raw / "french_factors.csv"
    factors.to_csv(out)
    logger.info("wrote French factors %s -> %s", factors.shape, out)
    return out
