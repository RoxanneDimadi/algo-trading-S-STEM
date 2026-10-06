"""Post-publication decay analysis (the honesty layer).

McLean & Pontiff (2016): anomaly returns fall ~26% after the original sample
ends and ~58% after publication -- but predictability persists rather than
vanishing. This module measures exactly that pattern for each signal by
splitting its long-short return series at the signal's sample_end and
pub_date and reporting per-segment annualized performance.

For synthetic data the dates come from the generator's ground-truth meta;
for OSAP data they come from SignalDoc.csv (see src/data/loaders.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.stats import annualized_stats


def segment_performance(ls: pd.Series, sample_end, pub_date,
                        nw_lags: int = 6) -> pd.DataFrame:
    """Annualized stats for in-sample / post-sample / post-publication."""
    sample_end, pub_date = pd.Timestamp(sample_end), pd.Timestamp(pub_date)
    segs = {
        "in_sample": ls[ls.index <= sample_end],
        "post_sample": ls[(ls.index > sample_end) & (ls.index <= pub_date)],
        "post_publication": ls[ls.index > pub_date],
    }
    rows = []
    for name, s in segs.items():
        st = annualized_stats(s, nw_lags)
        rows.append({"segment": name, "ann_return": st["ann_return"],
                     "sharpe": st["sharpe"], "nw_tstat": st["nw_tstat"],
                     "n_months": st["n_months"]})
    out = pd.DataFrame(rows).set_index("segment")
    is_ret = out.loc["in_sample", "ann_return"]
    if pd.notna(is_ret) and abs(is_ret) > 1e-12:
        out["pct_of_in_sample"] = out["ann_return"] / is_ret
    return out


def decay_table(ls_by_signal: dict[str, pd.Series], meta: pd.DataFrame,
                nw_lags: int = 6) -> pd.DataFrame:
    """One row per signal: in-sample vs post-sample vs post-pub, plus the
    McLean-Pontiff-style retention ratios."""
    rows = []
    for sig, ls in ls_by_signal.items():
        if sig not in meta.index:
            continue
        seg = segment_performance(ls, meta.loc[sig, "sample_end"],
                                  meta.loc[sig, "pub_date"], nw_lags)
        rows.append({
            "signal": sig,
            "in_sample_sharpe": seg.loc["in_sample", "sharpe"],
            "post_sample_sharpe": seg.loc["post_sample", "sharpe"],
            "post_pub_sharpe": seg.loc["post_publication", "sharpe"],
            "post_sample_retention": seg.loc["post_sample", "pct_of_in_sample"]
            if "pct_of_in_sample" in seg else np.nan,
            "post_pub_retention":
            seg.loc["post_publication", "pct_of_in_sample"]
            if "pct_of_in_sample" in seg else np.nan,
        })
    return pd.DataFrame(rows).set_index("signal")
