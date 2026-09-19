"""Download Chen-Zimmermann OSAP firm signals + SignalDoc.

Uses the openassetpricing package.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("real_data.osap")


def _openap(release: Optional[Any] = None):
    import openassetpricing as oap

    if release in (None, "", "null", "None"):
        return oap.OpenAP()
    return oap.OpenAP(int(release) if str(release).isdigit() else release)


def download_signal_doc(out_path: Path,
                        release: Optional[Any] = None) -> pd.DataFrame:
    openap = _openap(release)
    doc = openap.dl_signal_doc("pandas")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.to_csv(out_path, index=False)
    logger.info("wrote SignalDoc (%d rows) -> %s", len(doc), out_path)
    return doc


def download_signals(
    signals: List[str],
    out_path: Path,
    release: Optional[Any] = None,
) -> pd.DataFrame:
    """Download selected firm predictors as a wide (permno, yyyymm, ...) CSV.

    Pass signals as a list; openassetpricing rejects a bare string.
    """
    if not signals:
        raise ValueError("signals list is empty")
    # OSAP omits CRSP-licensed fields; strip them if they are left in config
    crsp_only = {"Price", "Size", "STreversal"}
    want = [s for s in signals if s not in crsp_only]
    skipped = [s for s in signals if s in crsp_only]
    if skipped:
        logger.warning(
            "skipping CRSP-only fields (build from returns later): %s",
            skipped)

    openap = _openap(release)
    logger.info("downloading OSAP signals %s (release=%s)",
                want, release or "latest")
    df = openap.dl_signal("pandas", want)
    if "permno" not in df.columns or "yyyymm" not in df.columns:
        raise RuntimeError(f"unexpected OSAP columns: {list(df.columns)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("wrote %d rows x %d cols -> %s",
                len(df), df.shape[1], out_path)
    return df


def download_portfolios(
    signals: List[str] | str,
    out_path: Path,
    form: str = "op",
    release: Optional[Any] = None,
    signal_doc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Long-short / sorted portfolio returns from OSAP.

    ``signals`` may be a list of acronyms, or the string ``"all"`` to pull
    the full predictor universe -- portfolio returns are freely distributed
    (unlike firm-level CRSP fields), so "all" is the recommended setting for
    the factor-mode panel: ~200 LS series instead of a handful.
    """
    openap = _openap(release)
    crsp_only = {"Price", "Size", "STreversal"}

    if isinstance(signals, str) and signals.lower() == "all":
        try:
            logger.info(
                "downloading OSAP portfolios form=%s for ALL predictors", form)
            df = openap.dl_port(form, "pandas")
        except TypeError as exc:
            # older openassetpricing requires an explicit list: take every
            # Predictor-class acronym from SignalDoc
            if signal_doc is None:
                raise RuntimeError(
                    "openassetpricing needs an explicit signal list and no "
                    "SignalDoc was provided; download SignalDoc first"
                ) from exc
            cols = {c.lower(): c for c in signal_doc.columns}
            acr = cols["acronym"]
            cat = cols.get("cat.signal")
            doc = signal_doc
            if cat:
                doc = doc[doc[cat].astype(str).str.lower() == "predictor"]
            want = sorted(set(doc[acr].dropna().astype(str)) - crsp_only)
            logger.info(
                "falling back to explicit list of %d predictors", len(want))
            df = openap.dl_port(form, "pandas", want)
    else:
        want = [s for s in signals if s not in crsp_only]
        logger.info(
            "downloading OSAP portfolios form=%s signals=%s", form, want)
        df = openap.dl_port(form, "pandas", want)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    n_sig = df["signalname"].nunique() if "signalname" in df.columns else "?"
    logger.info("wrote portfolios (%d rows, %s signals) -> %s",
                len(df), n_sig, out_path)
    return df


def run_osap_download(cfg: Dict[str, Any], raw: Path) -> Dict[str, Path]:
    ocfg = cfg.get("osap", {})
    release = ocfg.get("release")
    signals = list(ocfg.get("signals", []))

    paths = {
        "signal_doc": raw / "SignalDoc.csv",
        "signals": raw / "signed_predictors_dl_wide.csv",
    }
    doc = download_signal_doc(paths["signal_doc"], release=release)
    download_signals(signals, paths["signals"], release=release)

    if ocfg.get("download_portfolios", True):
        paths["portfolios"] = raw / \
            f"osap_portfolios_{ocfg.get('portfolio_form', 'op')}.csv"
        # portfolio_signals: "all" (recommended for factor mode) or a list;
        # defaults to the firm-signal list for backwards compatibility
        port_signals = ocfg.get("portfolio_signals", signals)
        download_portfolios(
            port_signals,
            paths["portfolios"],
            form=ocfg.get("portfolio_form", "op"),
            release=release,
            signal_doc=doc,
        )
    return paths
