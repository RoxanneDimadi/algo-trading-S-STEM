"""Build the point-in-time panel and sync files into the research agent."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .returns import build_streversal, load_returns_csv

logger = logging.getLogger("real_data.panel")


def _yyyymm_to_month_end(yyyymm: pd.Series) -> pd.Series:
    s = yyyymm.astype(int).astype(str)
    return pd.to_datetime(s, format="%Y%m") + pd.offsets.MonthEnd(0)


def load_osap_wide(path: Path, signals: List[str]) -> pd.DataFrame:
    usecols = ["permno", "yyyymm", *signals]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    missing = [s for s in signals if s not in df.columns]
    if missing:
        raise ValueError(f"signals missing from OSAP file {path}: {missing}")
    df["date"] = _yyyymm_to_month_end(df["yyyymm"])
    df["ticker"] = df["permno"].astype(int).astype(str)
    return df


def build_panel(
    signals_csv: Path,
    returns_csv: Path,
    signals: List[str],
    add_streversal: bool = True,
    min_names_per_month: int = 50,
) -> pd.DataFrame:
    """Join OSAP signals to returns; attach fwd_ret = next month's return."""
    osap_signals = [s for s in signals if s != "STreversal"]
    sig = load_osap_wide(signals_csv, osap_signals)
    ret = load_returns_csv(returns_csv, prefer_delisting=True)

    if add_streversal or "STreversal" in signals:
        st = build_streversal(ret)
        sig = sig.merge(
            st[["date", "ticker", "STreversal"]],
            on=["date", "ticker"],
            how="left",
        )

    panel = sig.merge(ret[["date", "ticker", "ret"]],
                      on=["date", "ticker"], how="inner")
    panel = panel.sort_values(["ticker", "date"])
    panel["fwd_ret"] = panel.groupby("ticker")["ret"].shift(-1)

    # Drop thin months
    counts = panel.groupby("date")["ticker"].transform("size")
    panel = panel.loc[counts >= min_names_per_month].reset_index(drop=True)
    logger.info(
        "panel: %d rows, %d months, %d names",
        len(panel), panel["date"].nunique(), panel["ticker"].nunique(),
    )
    return panel


def augment_osap_with_streversal(signals_csv: Path, returns_csv: Path,
                                 out_path: Path) -> Path:
    """Add STreversal to the OSAP wide file so the loader can request it."""
    wide = pd.read_csv(signals_csv)
    ret = load_returns_csv(returns_csv, prefer_delisting=True)
    st = build_streversal(ret)
    st_key = st[["permno", "yyyymm", "STreversal"]
                ].drop_duplicates(["permno", "yyyymm"])
    if "STreversal" in wide.columns:
        wide = wide.drop(columns=["STreversal"])
    merged = wide.merge(st_key, on=["permno", "yyyymm"], how="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logger.info("wrote OSAP+STreversal -> %s", out_path)
    return out_path


def sync_to_agent(
    raw: Path,
    agent_raw: Path,
    files: Optional[List[str]] = None,
) -> List[Path]:
    """Copy ingest artifacts into multisignal-alpha/data/raw/."""
    agent_raw.mkdir(parents=True, exist_ok=True)
    default = [
        "signed_predictors_dl_wide.csv",
        "SignalDoc.csv",
        "returns.csv",
        "french_factors.csv",
    ]
    copied = []
    for name in files or default:
        src = raw / name
        if not src.exists():
            logger.warning("skip missing %s", src)
            continue
        dst = agent_raw / name
        # skip unchanged files (the OSAP wide CSV is ~450MB; don't rewrite it
        # on every run when nothing changed)
        if dst.exists():
            s, d = src.stat(), dst.stat()
            if s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime):
                logger.info("unchanged, skipping %s", name)
                copied.append(dst)
                continue
        shutil.copy2(src, dst)
        logger.info("synced %s -> %s", src, dst)
        copied.append(dst)
    return copied


def write_agent_factor_config(agent_root: Path, n_factors: int) -> Path:
    """Generate a COMPLETE runnable agent config for factor mode.

    Starts from the agent's own configs/config.yaml (so every model /
    walk-forward / cost setting stays the single source of truth there) and
    swaps in the panel_csv data section. n_quantiles adapts to the size of
    the cross-section: quintiles need >= 25 names to be meaningful; below
    that we fall back to terciles (documented in the generated header).
    """
    import yaml

    base_path = agent_root / "configs" / "config.yaml"
    with open(base_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["data"] = {
        "mode": "panel_csv",
        "panel_csv": {
            "panel": "data/raw/factor_panel.csv",
            "meta": "data/raw/factor_meta.csv",
            "french_factors_csv": "data/raw/french_factors.csv",
        },
    }
    n_q = 5 if n_factors >= 25 else 3
    cfg.setdefault("evaluation", {})["n_quantiles"] = n_q
    cfg.setdefault("run", {})["output_dir"] = "results_factor"
    models = cfg.setdefault("models", {})
    if n_factors >= 25:
        # full-universe run: robust thresholds for a growing factor count
        cfg["evaluation"]["min_names_per_date"] = 20
        smoke = False
    else:
        # SMOKE-TEST settings for a tiny cross-section: thresholds relaxed to
        # the structural minimum, PULSE interactions off (P must stay below
        # the per-date observation count) and ridge carries identification
        # (obs_margin: 0). Results are mechanical checks, not evidence.
        cfg["evaluation"]["min_names_per_date"] = 3
        models.setdefault("pulse", {})["interactions"] = False
        models.setdefault("pulse", {})["obs_margin"] = 0
        smoke = True

    header = (
        "# GENERATED by real-data/scripts/build_factor_panel.py -- edit\n"
        "# configs/config.yaml for shared settings and regenerate this file.\n"
        f"# Cross-section at generation time: {n_factors} factors -> "
        f"n_quantiles={n_q}\n"
        + ("# !! SMOKE-TEST CONFIG: fewer than 25 factors. Statistical\n"
           "# !! thresholds are relaxed to the structural minimum; treat the\n"
           "# !! output as a plumbing check only. For the real run, set\n"
           "# !! osap.portfolio_signals: all in real-data and rebuild.\n"
           if smoke else
           "# Full-universe settings (quintile sorts, min 20 names/date).\n")
    )
    out = agent_root / "configs" / "config_factor.yaml"
    out.write_text(header + yaml.safe_dump(cfg, sort_keys=False),
                   encoding="utf-8")
    logger.info("wrote %s (n_quantiles=%d for %d factors)",
                out, n_q, n_factors)
    return out


def write_agent_config_snippet(agent_root: Path, signals: List[str]) -> Path:
    """Drop a small yaml the agent can merge / copy from for osap mode."""
    sigs = list(signals)
    text = f"""# Generated by real-data ingest - merge into
# configs/config.yaml or use as an overlay.
data:
  mode: osap
  osap:
    signals: {sigs}
    signals_csv: data/raw/signed_predictors_dl_wide.csv
    signal_doc_csv: data/raw/SignalDoc.csv
    returns_csv: data/raw/returns.csv
    french_factors_csv: data/raw/french_factors.csv
"""
    out = agent_root / "configs" / "config_osap_overlay.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info("wrote %s", out)
    return out
