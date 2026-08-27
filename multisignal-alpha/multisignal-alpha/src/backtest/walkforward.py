"""Purged, embargoed walk-forward splits (Lopez de Prado, AFML ch. 7).

Why purging exists: with a 1-month label horizon, the label at training date
t is realized over (t, t+1]. If the test fold starts at t+1, the last
training label OVERLAPS the first test period -- information leaks across
the boundary. Purging drops `purge` periods between train end and test start
so no training label's realization window touches the test window.

RULE: purge >= label horizon. This repo's horizon is 1 month, so purge=1.
If you extend fwd_ret to h months, raise purge to h.

`embargo` drops additional periods from the train end -- a conservative
buffer against serial dependence beyond the label window.

All model selection (including optuna tuning) must happen INSIDE the train
window of each fold. Test folds are touched exactly once, by the final
predict call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def walkforward_splits(dates, min_train: int = 120, test_size: int = 12,
                       purge: int = 1, embargo: int = 0,
                       expanding: bool = True) -> list[Fold]:
    """Chronological folds over the unique panel dates.

    Layout per fold (index positions):
        train = [start, i - 1 - purge - embargo]
        test  = [i, i + test_size - 1]
    with start = 0 (expanding) or i - purge - embargo - min_train (rolling).
    """
    if purge < 1:
        raise ValueError("purge must be >= 1 (the label horizon).")
    dts = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
    folds: list[Fold] = []
    i = min_train + purge + embargo  # first test index with a full train window
    fid = 0
    while i + test_size <= len(dts):
        train_end = i - 1 - purge - embargo
        train_start = 0 if expanding else max(0, train_end + 1 - min_train)
        train = dts[train_start: train_end + 1]
        test = dts[i: i + test_size]

        # -- self-checks: fail loudly rather than leak quietly ---------------
        assert len(train) > 0 and len(test) > 0
        assert train.max() < test.min(), "train/test overlap"
        gap = np.searchsorted(dts, test.min()) - np.searchsorted(dts, train.max()) - 1
        assert gap >= purge + embargo, f"purge violated: gap={gap}"

        folds.append(Fold(fid, train, test))
        fid += 1
        i += test_size
    if not folds:
        raise ValueError("No folds produced -- panel too short for the "
                         "requested min_train/test_size.")
    return folds
