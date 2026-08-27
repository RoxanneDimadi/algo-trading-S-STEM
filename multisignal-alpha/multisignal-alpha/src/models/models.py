"""Model factories: the linear benchmark and the ML challenger.

The project's headline comparison (Gu-Kelly-Xiu in miniature) is
LightGBM vs. an elastic net on IDENTICAL inputs. The ML model's job is to
capture nonlinear interactions among signals the linear model cannot; it is
NOT to rescue signals that carry no information (evaluation precedes
modeling for exactly that reason).

Both factories return fresh sklearn-style estimators so each walk-forward
fold trains from scratch -- no state leaks across folds.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import ElasticNet


def make_linear(cfg: dict):
    """Elastic net on rank-normalized signals: the benchmark to beat.

    With rank-normalized features (already comparable scales), no extra
    standardization step is needed. alpha is kept tiny by default: the point
    of the benchmark is 'best linear combination', not aggressive shrinkage.
    """
    return ElasticNet(
        alpha=float(cfg.get("alpha", 1e-4)),
        l1_ratio=float(cfg.get("l1_ratio", 0.5)),
        max_iter=10_000,
        random_state=0,
    )


def make_lgbm(cfg: dict):
    """Gradient-boosted trees: the nonlinear challenger (GKX's best family
    alongside neural nets, and the pragmatic default for tabular panels)."""
    return lgb.LGBMRegressor(
        n_estimators=int(cfg.get("n_estimators", 300)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        num_leaves=int(cfg.get("num_leaves", 31)),
        min_child_samples=int(cfg.get("min_child_samples", 100)),
        subsample=float(cfg.get("subsample", 0.8)),
        subsample_freq=1,
        colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
        objective="regression",
        importance_type="gain",
        n_jobs=-1,
        random_state=0,
        verbose=-1,
    )


def tune_lgbm(X_train, y_train, X_val, y_val, base_cfg: dict,
              n_trials: int = 25):
    """Optional optuna search. CRITICAL DISCIPLINE: this must only ever see
    the train/validation split handed to it by the walk-forward -- tuning on
    anything that overlaps a test fold is leakage, full stop. The pipeline
    enforces that by construction; keep it that way.

    Every completed study is one more 'trial' for the deflated-Sharpe
    accounting. Log it.
    """
    import optuna  # lazy: optional dependency
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(base_cfg)
        params.update(
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 50, 300),
        )
        model = make_lgbm(params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return float(np.mean((pred - y_val) ** 2))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(base_cfg)
    best.update(study.best_params)
    return best
