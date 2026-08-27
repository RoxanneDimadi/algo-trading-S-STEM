"""PULSE: Per-date Update of Latent Signal Efficacy.

The hypothesis
--------------
Standard ML asset pricing treats signal efficacy as STATIONARY: fit one model
on decades of data, as if momentum's beta today equals its 1999 beta. The
decay literature says otherwise -- McLean & Pontiff (2016): returns fall
~26% post-sample, ~58% post-publication; "Anomaly Time" (JF 2024): efficacy
concentrates when information is fresh; factor momentum (Gupta & Kelly 2019;
Ehsani & Linnainmaa, JF 2022): factor returns are themselves autocorrelated,
i.e., CURRENT efficacy predicts NEAR-FUTURE efficacy.

PULSE takes those facts literally. Each signal's true predictive coefficient
is modeled as a latent STATE with mean-reverting dynamics whose resting
point is zero -- "alpha dies by default" as a prior, not a footnote:

    state:        beta[k,t] = a * beta[k,t-1] + w_t,      w ~ N(0, q_k)
    observation:  lam[k,t]  = beta[k,t] + v_t,            v ~ N(0, r[k,t])

where lam[k,t] is the date-t cross-sectional OLS coefficient of forward
returns on the (interaction-expanded) signals -- the Fama-MacBeth first pass
-- and r[k,t] is that coefficient's SAMPLING variance, which the regression
itself provides. Known heteroskedastic observation noise is what makes a
Kalman filter the principled estimator here rather than an ad-hoc moving
average (docs/math/10 proves the filter's steady state IS an exponentially
weighted average -- with the weight chosen by the data instead of by hand,
per-signal).

Prediction at formation date t uses the filtered state given information
through t-1 only:

    p[i,t] = sum_k  E[beta[k,t] | lam up to t-1]  *  z[k,i,t].

Point-in-time note (see docs/math/10 for the lag bookkeeping): lam[k,s]
requires the return over (s, s+1], so it becomes known at s+1. The filter
consumes observations strictly before the formation date; inside a test fold
states evolve by the PRIOR dynamics only (no test-fold information updates
them), preserving the engine's purge guarantees exactly.

Nonlinearity enters through basis expansion: the design includes all
pairwise products z_i * z_j, each with its own filtered efficacy -- so PULSE
can track, e.g., a momentum x value interaction whose strength itself drifts.

Hyperparameters (mean-reversion a, state-noise scale) are chosen by
maximizing the one-step-ahead predictive log-likelihood of the observed
coefficient series ON TRAINING DATA ONLY -- the filter's own out-of-sample
criterion, no peeking.

Everything below is NumPy from scratch: the per-date regressions, the
filter, the likelihood, the grid search. ~150 lines of auditable math.
"""
from __future__ import annotations

import numpy as np


def expand_interactions(X: np.ndarray, names: list[str]):
    """[z_1..z_K]  ->  [z_1..z_K, z_i*z_j for i<j], with names."""
    cols = [X]
    out_names = list(names)
    K = X.shape[1]
    for i in range(K):
        for j in range(i + 1, K):
            cols.append((X[:, i] * X[:, j])[:, None])
            out_names.append(f"{names[i]}*{names[j]}")
    return np.hstack(cols), out_names


class PulseModel:
    """Kalman-filtered time-varying-efficacy forecaster."""

    requires_dates = True

    def __init__(self, interactions: bool = True, ridge: float = 1e-6,
                 a_grid=(0.97, 0.99, 1.0), q_scale_grid=(0.002, 0.01, 0.05),
                 min_names: int = 60):
        self.interactions = bool(interactions)
        self.ridge = float(ridge)
        self.a_grid = tuple(a_grid)
        self.q_scale_grid = tuple(q_scale_grid)
        self.min_names = int(min_names)

    # ---------------------------------------------------------- pass 1: lam_t
    def _per_date_coefs(self, Z, y, dates):
        """Cross-sectional OLS per date -> (dates_used, LAM[T,P], R[T,P]).

        R holds each coefficient's sampling variance s^2 * diag((Z'Z)^-1):
        the KNOWN observation noise of the state-space model.
        """
        uniq, starts = np.unique(dates, return_index=True)
        order = np.argsort(starts)
        starts = starts[order]
        stops = np.append(starts[1:], len(dates))
        P = Z.shape[1]
        lam_rows, r_rows, used_dates = [], [], []
        I = np.eye(P)
        for (a, b), dt in zip(zip(starts, stops), uniq[order]):
            n = b - a
            if n < max(self.min_names, P + 10):
                continue
            Zi, yi = Z[a:b], y[a:b]
            G = Zi.T @ Zi + self.ridge * n * I
            Ginv = np.linalg.inv(G)
            beta = Ginv @ (Zi.T @ yi)
            resid = yi - Zi @ beta
            s2 = float(resid @ resid) / max(n - P, 1)
            lam_rows.append(beta)
            r_rows.append(np.maximum(s2 * np.diag(Ginv), 1e-14))
            used_dates.append(dt)
        return (np.array(used_dates), np.array(lam_rows), np.array(r_rows))

    # ---------------------------------------------------------- the filter
    @staticmethod
    def _filter_1d(lam, r, a, q, m0=0.0, P0=None):
        """Scalar Kalman filter for one coefficient series.

        Returns filtered means m[t] = E[beta_t | lam_1..t], variances,
        and the one-step predictive log-likelihood sum(log N(lam_t; a*m_{t-1},
        a^2 P_{t-1} + q + r_t)) -- the model-selection criterion.
        """
        T = len(lam)
        m = np.empty(T)
        Pv = np.empty(T)
        prev_m = m0
        prev_P = (10.0 * (np.median(r) + q)) if P0 is None else P0  # diffuse
        ll = 0.0
        for t in range(T):
            # predict
            mp = a * prev_m
            Pp = a * a * prev_P + q
            # innovation likelihood
            S = Pp + r[t]
            ll += -0.5 * (np.log(2 * np.pi * S) + (lam[t] - mp) ** 2 / S)
            # update
            K = Pp / S
            prev_m = mp + K * (lam[t] - mp)
            prev_P = (1.0 - K) * Pp
            m[t], Pv[t] = prev_m, prev_P
        return m, Pv, ll

    # ---------------------------------------------------------------- fit
    def fit(self, X, y, dates=None):
        if dates is None:
            raise ValueError("PULSE requires formation dates: fit(X, y, dates=...)")
        Xa = np.asarray(X, dtype=float)
        base_names = (list(X.columns) if hasattr(X, "columns")
                      else [f"x{i}" for i in range(Xa.shape[1])])
        y = np.asarray(y, dtype=float).ravel()
        dates = np.asarray(dates)
        order = np.argsort(dates, kind="stable")
        Xa, y, dates = Xa[order], y[order], dates[order]

        Z, names = (expand_interactions(Xa, base_names) if self.interactions
                    else (Xa, list(base_names)))
        self.feature_names_ = names
        self._base_names = base_names

        dts, LAM, R = self._per_date_coefs(Z, y, dates)
        Pdim = LAM.shape[1]

        # hyperparameters by training predictive likelihood, per coefficient
        # family (shared a and q_scale across coefficients; q_k scales with
        # each coefficient's own observation-noise level).
        best = (None, None, -np.inf)
        med_r = np.median(R, axis=0)
        for a in self.a_grid:
            for qs in self.q_scale_grid:
                ll = 0.0
                for k in range(Pdim):
                    _, _, l = self._filter_1d(LAM[:, k], R[:, k], a, qs * med_r[k])
                    ll += l
                if ll > best[2]:
                    best = (a, qs, ll)
        self.a_, self.q_scale_ = best[0], best[1]

        # final filtering pass; store the whole path for diagnostics
        M = np.empty_like(LAM)
        V = np.empty_like(LAM)
        for k in range(Pdim):
            M[:, k], V[:, k], _ = self._filter_1d(
                LAM[:, k], R[:, k], self.a_, self.q_scale_ * med_r[k])
        self.filter_dates_ = dts
        self.filter_history_ = M          # E[beta_t | data through t]
        self.filter_var_ = V
        # state at the last training observation, for forward propagation
        self.state_mean_ = M[-1].copy()
        self.state_var_ = V[-1].copy()
        self._med_r = med_r
        return self

    # ------------------------------------------------------------- predict
    def predict(self, X, dates=None):
        """Forecast with the PRIOR-propagated state (no test-fold updating).

        The state h months past the training end is a^h * state_mean_ --
        the decay prior gently shrinks efficacy toward zero the longer the
        filter goes without evidence, which is exactly the McLean-Pontiff
        prior in action. If dates are not supplied, one step is assumed.
        """
        Xa = np.asarray(X, dtype=float)
        Z, _ = (expand_interactions(Xa, self._base_names) if self.interactions
                else (Xa, self._base_names))
        if dates is None:
            return Z @ (self.a_ * self.state_mean_)
        dates = np.asarray(dates)
        uniq = np.sort(np.unique(dates))
        step = {dt: h + 1 for h, dt in enumerate(uniq)}   # months past train end
        out = np.empty(len(Z))
        for dt in uniq:
            mask = dates == dt
            out[mask] = Z[mask] @ ((self.a_ ** step[dt]) * self.state_mean_)
        return out


def make_pulse(cfg: dict):
    return PulseModel(
        interactions=bool(cfg.get("interactions", True)),
        ridge=float(cfg.get("ridge", 1e-6)),
        a_grid=tuple(cfg.get("a_grid", (0.97, 0.99, 1.0))),
        q_scale_grid=tuple(cfg.get("q_scale_grid", (0.002, 0.01, 0.05))),
    )
