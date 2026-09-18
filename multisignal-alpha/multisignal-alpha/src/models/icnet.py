"""IC-Net: a from-scratch model whose LOSS FUNCTION is the economics.

Why build a custom model at all
-------------------------------
Off-the-shelf regressors (ElasticNet, LightGBM) minimize pointwise MSE. But
this project never monetizes the *level* of a return forecast -- it sorts the
cross-section and trades the spread. The quantity that maps to P&L is the
per-date cross-sectional RANK correlation between forecast and forward
return (the IC), not the mean squared error. MSE spends model capacity on
exactly the components the long-short portfolio nets out: the market level
and the volatility scale of each month.

IC-Net encodes that prior directly. It is a small multilayer perceptron
implemented in pure NumPy (forward pass, backprop, and Adam written out by
hand -- no torch, no sklearn) trained to MAXIMIZE the average per-date
cross-sectional Pearson correlation between its predictions and forward
returns:

    maximize  (1/T) * sum_t corr_t( f(X_t; W), y_t )  -  l2 * ||W||^2

Three properties fall out of this objective, each with a financial meaning:

  1. **Per-date demeaning is built into corr_t** -- the model is
     cross-sectionally (market-)neutral by construction. It cannot waste
     capacity predicting whether next month is an up month, because the
     objective is invariant to adding a constant to all predictions on a
     date.
  2. **Scale invariance in y** -- corr_t is unchanged if a month's returns
     are all scaled by its volatility. High-vol months therefore do not
     dominate training the way they dominate MSE. (On rank-normalized
     targets Pearson ~= Spearman, so this is a differentiable IC surrogate.)
  3. **Listwise, not pointwise** -- the training signal is "did you order
     the cross-section correctly on this date," which is literally the job.

Discipline notes
----------------
* Early stopping uses a CHRONOLOGICAL tail of the training dates as
  validation -- never the test fold (same rule as optuna tuning).
* Deterministic seed; small capacity by default (one hidden layer of 16
  tanh units on ~6 features). The point is the objective, not the size.
* Exposes `feature_importances_` (first-layer path weights) and
  `requires_dates = True`; the backtest engine passes formation dates so
  the loss can group by date.

Gradient of the per-date objective (for the curious / the interviewer):
with p~ = p - mean(p), y~ = y - mean(y), sp = ||p~||, sy = ||y~||,
corr = p~.y~ / (sp*sy):

    d corr / d p = y~/(sp*sy) - corr * p~ / sp^2

which has zero mean by construction (the demeaning projection is automatic
because y~ and p~ are demeaned).
"""
from __future__ import annotations

import numpy as np


class ICNet:
    """NumPy MLP trained to maximize mean per-date cross-sectional IC."""

    requires_dates = True

    def __init__(self, hidden: int = 16, l2: float = 1e-4, lr: float = 0.01,
                 max_epochs: int = 300, patience: int = 25,
                 val_fraction: float = 0.15, min_names: int = 30,
                 seed: int = 0):
        self.hidden = int(hidden)
        self.l2 = float(l2)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.min_names = int(min_names)
        self.seed = int(seed)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _slices_by_date(dates: np.ndarray):
        """Contiguous [start, stop) row slices per unique date.
        Assumes rows sorted by date (the caller sorts)."""
        _uniq, starts = np.unique(dates, return_index=True)
        order = np.argsort(starts)
        starts = starts[order]
        stops = np.append(starts[1:], len(dates))
        return list(zip(starts, stops))

    def _forward(self, X):
        Z = X @ self.W1 + self.b1
        H = np.tanh(Z)
        p = H @ self.w2 + self.b2
        return H, p

    def _objective_and_grad(self, X, y, slices):
        """Mean per-date corr and its gradient w.r.t. all parameters."""
        n, _ = X.shape
        H, p = self._forward(X)
        g_p = np.zeros(n)
        total, used = 0.0, 0
        for a, b in slices:
            if b - a < self.min_names:
                continue
            ps, ys = p[a:b], y[a:b]
            pt = ps - ps.mean()
            yt = ys - ys.mean()
            sp = np.sqrt(pt @ pt)
            sy = np.sqrt(yt @ yt)
            if sp < 1e-10 or sy < 1e-10:
                continue
            corr = float(pt @ yt) / (sp * sy)
            total += corr
            used += 1
            g_p[a:b] = yt / (sp * sy) - corr * pt / (sp * sp)
        if used == 0:
            raise ValueError("No usable dates (all cross-sections too small).")
        mean_corr = total / used
        g_p /= used  # d(mean corr)/dp; we ASCEND this

        # backprop (ascent direction, so no sign flip here; Adam adds it)
        dw2 = H.T @ g_p - 2 * self.l2 * self.w2
        db2 = g_p.sum()
        dH = np.outer(g_p, self.w2)
        dZ = dH * (1.0 - H * H)
        dW1 = X.T @ dZ - 2 * self.l2 * self.W1
        db1 = dZ.sum(axis=0)
        return mean_corr, (dW1, db1, dw2, db2)

    def _mean_corr(self, X, y, slices) -> float:
        _, p = self._forward(X)
        vals = []
        for a, b in slices:
            if b - a < self.min_names:
                continue
            pt = p[a:b] - p[a:b].mean()
            yt = y[a:b] - y[a:b].mean()
            sp, sy = np.sqrt(pt @ pt), np.sqrt(yt @ yt)
            if sp > 1e-10 and sy > 1e-10:
                vals.append(float(pt @ yt) / (sp * sy))
        return float(np.mean(vals)) if vals else np.nan

    # ------------------------------------------------------------------- API
    def fit(self, X, y, dates=None):
        if dates is None:
            raise ValueError(
                "ICNet requires formation dates: fit(X, y, dates=...)")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        dates = np.asarray(dates)

        # sort rows by date so per-date slices are contiguous
        order = np.argsort(dates, kind="stable")
        X, y, dates = X[order], y[order], dates[order]

        # chronological train/validation split BY DATE (inside-train only;
        # the walk-forward guarantees none of this touches a test fold)
        uniq = np.unique(dates)
        n_val = max(6, int(round(len(uniq) * self.val_fraction)))
        val_dates = set(uniq[-n_val:])
        is_val = np.isin(dates, list(val_dates))
        Xtr, ytr, dtr = X[~is_val], y[~is_val], dates[~is_val]
        Xva, yva, dva = X[is_val], y[is_val], dates[is_val]
        tr_slices = self._slices_by_date(dtr)
        va_slices = self._slices_by_date(dva)

        # init (Xavier) + Adam state
        rng = np.random.default_rng(self.seed)
        k = X.shape[1]
        self.W1 = rng.normal(0, np.sqrt(1.0 / k), size=(k, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0, np.sqrt(1.0 / self.hidden), size=self.hidden)
        self.b2 = 0.0
        params = ["W1", "b1", "w2", "b2"]
        m = {p: np.zeros_like(getattr(self, p), dtype=float) for p in params}
        v = {p: np.zeros_like(getattr(self, p), dtype=float) for p in params}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        best_val, best_state, since_best = -np.inf, None, 0
        self.history_ = []
        for epoch in range(1, self.max_epochs + 1):
            _, grads = self._objective_and_grad(Xtr, ytr, tr_slices)
            for p, g in zip(params, grads):
                m[p] = beta1 * m[p] + (1 - beta1) * g
                v[p] = beta2 * v[p] + (1 - beta2) * (g * g)
                mhat = m[p] / (1 - beta1 ** epoch)
                vhat = v[p] / (1 - beta2 ** epoch)
                # gradient ASCENT on mean corr
                setattr(self, p, getattr(self, p) +
                        self.lr * mhat / (np.sqrt(vhat) + eps))

            val_corr = self._mean_corr(Xva, yva, va_slices)
            self.history_.append(val_corr)
            if np.isfinite(val_corr) and val_corr > best_val + 1e-5:
                best_val, since_best = val_corr, 0
                best_state = {p: getattr(self, p).copy() if p != "b2"
                              else float(self.b2) for p in params}
            else:
                since_best += 1
                if since_best >= self.patience:
                    break

        if best_state is not None:  # restore best-on-validation weights
            for p in params:
                setattr(self, p, best_state[p])
        self.best_val_ic_ = best_val
        self.n_epochs_ = epoch

        # first-layer path importance: sum_h |W1[k,h]| * |w2[h]|
        self.feature_importances_ = (
            np.abs(self.W1) * np.abs(self.w2)).sum(axis=1)
        return self

    def predict(self, X, dates=None):
        # `dates` is unused here; it exists for parity with the
        # other model interfaces, which do need it.
        # pylint: disable=unused-argument
        X = np.asarray(X, dtype=float)
        _, p = self._forward(X)
        return p


def make_icnet(cfg: dict):
    """Factory matching the signature style of make_linear / make_lgbm."""
    return ICNet(
        hidden=int(cfg.get("hidden", 16)),
        l2=float(cfg.get("l2", 1e-4)),
        lr=float(cfg.get("lr", 0.01)),
        max_epochs=int(cfg.get("max_epochs", 300)),
        patience=int(cfg.get("patience", 25)),
        val_fraction=float(cfg.get("val_fraction", 0.15)),
        min_names=int(cfg.get("min_names", 30)),
        seed=int(cfg.get("seed", 0)),
    )
