"""Walk-forward backtest engine.

For each fold: fit the model on the (purged) train window, predict the
cross-section on each test date, and hand every out-of-sample prediction to
the SAME quantile long-short constructor used for single-signal evaluation.
One portfolio machinery for everything means the ML-vs-linear comparison
cannot be an artifact of different construction choices.

Turnover is computed on the concatenated out-of-sample weight path (fold
boundaries included -- the strategy is one continuous portfolio), and costs
are charged on traded notional exactly as in evaluation/portfolio.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.ic import mean_ic
from ..evaluation.portfolio import (apply_costs, portfolio_returns,
                                    score_to_weights, turnover_series)
from ..utils.stats import annualized_stats
from .walkforward import walkforward_splits


def run_model_backtest(panel: pd.DataFrame, feature_cols: list[str],
                       model_factory, wf_cfg: dict, eval_cfg: dict,
                       label_col: str = "fwd_ret",
                       collect_importance: bool = True) -> dict:
    """Train/predict through the walk-forward; evaluate the OOS forecast as a
    portfolio score.

    Returns a dict with the prediction panel, gross/net series, annualized
    stats, per-fold records, and (if available) average feature importances.
    """
    df = panel.dropna(subset=[*feature_cols, label_col]).copy()
    folds = walkforward_splits(
        df["date"].unique(),
        min_train=int(wf_cfg.get("min_train", 120)),
        test_size=int(wf_cfg.get("test_size", 12)),
        purge=int(wf_cfg.get("purge", 1)),
        embargo=int(wf_cfg.get("embargo", 0)),
        expanding=bool(wf_cfg.get("expanding", True)),
    )

    preds, fold_rows, importances = [], [], []
    for fold in folds:
        tr = df[df["date"].isin(fold.train_dates)]
        te = df[df["date"].isin(fold.test_dates)]
        if tr.empty or te.empty:
            continue
        model = model_factory()
        if getattr(model, "requires_dates", False):
            # models with per-date (cross-sectional) objectives need to know
            # which rows share a formation date -- see src/models/icnet.py
            model.fit(tr[feature_cols], tr[label_col].to_numpy(),
                      dates=tr["date"].to_numpy())
        else:
            model.fit(tr[feature_cols], tr[label_col].to_numpy())
        out = te[["date", "ticker", label_col]].copy()
        if getattr(model, "requires_dates", False):
            out["prediction"] = model.predict(te[feature_cols],
                                              dates=te["date"].to_numpy())
        else:
            out["prediction"] = model.predict(te[feature_cols])
        preds.append(out)
        fold_rows.append({
            "fold": fold.fold_id,
            "train_start": fold.train_dates.min(), "train_end": fold.train_dates.max(),
            "test_start": fold.test_dates.min(), "test_end": fold.test_dates.max(),
            "n_train": len(tr), "n_test": len(te),
        })
        if collect_importance and hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            tot = imp.sum()
            importances.append(imp / tot if tot > 0 else imp)
        elif collect_importance and hasattr(model, "coef_"):
            importances.append(np.abs(np.asarray(model.coef_, dtype=float)))

    pred_panel = pd.concat(preds, ignore_index=True)
    pred_panel = pred_panel.merge(
        panel[["date", "ticker", label_col]].drop_duplicates(),
        on=["date", "ticker", label_col], how="left",
    )

    # --- portfolio on the OOS forecast --------------------------------------
    w = score_to_weights(pred_panel, "prediction",
                         n_q=int(eval_cfg.get("n_quantiles", 5)),
                         min_names=int(eval_cfg.get("min_names_per_date", 50)))
    gross = portfolio_returns(w, pred_panel, fwd_col=label_col)
    traded = turnover_series(w)
    net = apply_costs(gross, traded,
                      float(eval_cfg.get("cost_bps_per_side", 10.0)))
    nw_lags = int(eval_cfg.get("nw_lags", 6))

    oos_ic = mean_ic(pred_panel.rename(columns={label_col: "fwd_ret"}),
                     "prediction", "fwd_ret", nw_lags=nw_lags,
                     min_names=int(eval_cfg.get("min_names_per_date", 30)))

    result = {
        "pred_panel": pred_panel,
        "series": {"gross": gross, "net": net, "traded": traded, "weights": w},
        "gross": annualized_stats(gross, nw_lags),
        "net": annualized_stats(net, nw_lags),
        "oos_ic": oos_ic,
        "avg_one_way_turnover": float(traded.mean() / 2.0) if len(traded) else np.nan,
        "folds": pd.DataFrame(fold_rows),
    }
    if importances:
        result["feature_importance"] = pd.Series(
            np.mean(importances, axis=0), index=feature_cols
        ).sort_values(ascending=False)
    return result


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Side-by-side OOS table for the ML-vs-linear headline comparison."""
    rows = []
    for name, r in results.items():
        rows.append({
            "model": name,
            "oos_IC": r["oos_ic"]["ic_mean"],
            "oos_ICIR": r["oos_ic"]["icir"],
            "ann_ret_gross": r["gross"]["ann_return"],
            "sharpe_gross": r["gross"]["sharpe"],
            "ann_ret_net": r["net"]["ann_return"],
            "sharpe_net": r["net"]["sharpe"],
            "nw_t_net": r["net"]["nw_tstat"],
            "one_way_turnover": r["avg_one_way_turnover"],
            "n_oos_months": r["net"]["n_months"],
        })
    return pd.DataFrame(rows).set_index("model")
