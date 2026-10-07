"""A differentiable, cost-aware trading policy agent.

What this is, and why it replaces ML4T's RL chapter
---------------------------------------------------
Forecasting says WHICH stocks look good; an agent decides WHAT TO HOLD,
period after period, while paying for every change of mind. ML4T (ch. 22)
attacks this with model-free deep RL (DQN). The modern research consensus
goes another way: when the trading simulation is DIFFERENTIABLE -- returns,
holdings, and costs are all smooth functions of the policy parameters --
policy gradients can be computed EXACTLY by backpropagating through the
simulator, instead of estimated noisily from sampled episodes. Lineage:

  * Moody & Saffell (2001), "Learning to Trade via Direct Reinforcement" --
    the original: optimize the performance measure directly through the
    trading recurrence.
  * Garleanu & Pedersen (2013, JF), "Dynamic Trading with Predictable
    Returns and Transaction Costs" -- with quadratic costs the OPTIMAL
    policy is partial adjustment: trade a fraction of the way from current
    holdings toward an "aim" portfolio each period, and the aim tilts
    toward SLOW (persistent) signals as costs rise.
  * Zhang, Zohren & Roberts (2020) -- deep networks trained end-to-end on
    the Sharpe ratio of NET returns.
  * Kelly & Malamud -- maximum-Sharpe-ratio regression (signals -> weights
    directly), the same philosophy in closed form.

This module implements that consensus in ~200 lines of NumPy. The policy is
deliberately the GP structure -- an economically-derived inductive bias, not
an arbitrary network:

    score_t,i = sum_k theta_k * z_k,i,t            (learned signal blend)
    aim_t     = 2 * demean(score_t) / ||demean(score_t)||_1   (dollar-neutral,
                                                        $1 long / $1 short)
    w_t       = (1 - gamma) * w_{t-1} + gamma * aim_t   (partial adjustment)
    net_t     = <w_t, y_t> - (c/1e4) * sum_i |w_t,i - w_{t-1,i}|

Trainable parameters: theta (one per signal) and gamma = sigmoid(g), the
TRADING SPEED. Objective: the Sharpe ratio of net_t over the training
window, maximized by Adam with EXACT gradients (forward-mode sensitivity
propagation through the recursion -- derived in docs/math/11). |.| is
smoothed as softabs(x) = sqrt(x^2 + eps) so the cost term is differentiable.

Because the recursion w_t = (1-gamma) w_{t-1} + gamma aim_t unrolls to an
exponentially weighted average of past aims, the agent is, provably, a
LEARNED-SPEED EWMA of a LEARNED aim portfolio -- the two dials GP theory
says matter, with the data choosing both.

Falsifiable structure (tested against planted truth in tests/):
  * gamma should DECREASE as the cost parameter increases (GP comparative
    static: trade slower when trading is expensive);
  * the learned blend should tilt toward PERSISTENT signals as costs rise
    (GP: "aim in front of the target");
  * at high costs the agent's net Sharpe should beat the myopic gamma = 1
    (full-rebalance) policy on the same aim.

Honest scope notes: linear price impact only (costs proportional to traded
notional; no square-root impact), previous weights not drift-adjusted
between rebalances (same simplification as the evaluation stack, listed in
the backlog).

Unbalanced panels (real data: factors start and stop at different dates) are
handled by MASKING. A NaN forward return marks a name as untradable that
period: it is excluded from the demeaning and the L1 normalization, its
weight is forced to zero (an existing position is closed and the trade is
charged), and it contributes nothing to returns or gradients. A complete
panel has an all-true mask, which reduces exactly to the unmasked policy.
"""
from __future__ import annotations

import numpy as np


def _softabs(x: np.ndarray, eps: float) -> np.ndarray:
    return np.sqrt(x * x + eps)


def _mask_row(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(tradable mask as 0/1 floats, returns with untradable set to 0)."""
    ok = np.isfinite(y)
    return ok.astype(float), np.where(ok, y, 0.0)


class DiffPolicyAgent:
    """Garleanu-Pedersen-structured policy trained by exact gradients."""

    def __init__(self, cost_bps_per_side: float = 10.0, epochs: int = 300,
                 lr: float = 0.05, softabs_eps: float = 1e-8, seed: int = 0,
                 gamma_init: float = 0.30, sharpe_eps: float = 1e-9,
                 impact_bps: float = 0.0):
        # impact_bps: square-root market impact -- cost per name of
        # (impact_bps/1e4) * |dw|^{3/2}. Real trading fees are NOT linear in
        # size; this concavity-in-price / convexity-in-cost term makes large
        # trades disproportionately expensive INSIDE the differentiated loss.
        self.impact = float(impact_bps)
        self.cost = float(cost_bps_per_side)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.eps = float(softabs_eps)
        self.seed = int(seed)
        self.gamma_init = float(gamma_init)
        self.sharpe_eps = float(sharpe_eps)

    # ------------------------------------------------------------ simulate
    def _roll(self, Z, Y, theta, gamma, w0=None, with_grads=False):
        """Run the policy through the panel.

        Z: (K, T, N) signals; Y: (T, N) forward returns, NaN = untradable.
        Returns net/gross/traded series (T,), final weights, and -- if
        with_grads -- the exact gradient of each net_t w.r.t. params
        (theta_1..K, g), computed by forward-mode accumulation:
            dw_t/dp = (1-gamma) dw_{t-1}/dp + gamma dA_t/dp  (+ speed term).
        """
        K, T, N = Z.shape
        P = K + 1
        c = self.cost / 1e4
        w_prev = np.zeros(N) if w0 is None else w0.copy()
        dw_prev = np.zeros((P, N)) if with_grads else None

        net = np.empty(T)
        gross = np.empty(T)
        traded = np.empty(T)
        dnet = np.empty((P, T)) if with_grads else None

        for t in range(T):
            m, Yt = _mask_row(Y[t])
            n = m.sum()
            Zt = Z[:, t, :]                       # (K, N)
            s = theta @ Zt                        # scores
            s_t = (s - (s @ m) / max(n, 1.0)) * m
            sa_raw = _softabs(s_t, self.eps)
            D = (sa_raw * m).sum()
            if D > 0:
                A = 2.0 * s_t / D          # aim: dollar-neutral, gross 2
            else:
                A = np.zeros(N)

            w = ((1.0 - gamma) * w_prev + gamma * A) * m
            delta = w - w_prev
            da = _softabs(delta, self.eps)
            act = (m > 0) | (w_prev != 0)       # tradable or being closed
            gross[t] = w @ Yt
            traded[t] = da[act].sum()
            impact_t = float(((delta * delta + self.eps) ** 0.75)[act].sum())
            net[t] = gross[t] - c * traded[t] - (self.impact / 1e4) * impact_t

            if with_grads:
                # dA/dtheta_k via quotient rule; Zt demeaned over tradable
                Ztil = (Zt - (Zt @ m)[:, None] / max(n, 1.0)) * m  # (K, N)
                if D > 0:
                    sprime = (s_t / sa_raw) * m                  # softabs'
                    dD = Ztil @ sprime                           # (K,)
                    dA_dth = 2.0 * (Ztil * D - np.outer(dD, s_t)) / \
                        (D * D)  # (K, N)
                else:
                    dA_dth = np.zeros((K, N))
                dw = np.empty((P, N))
                dw[:K] = ((1.0 - gamma) * dw_prev[:K] + gamma * dA_dth) * m
                # speed parameter g: gamma = sigmoid(g)
                dgam = gamma * (1.0 - gamma)
                dw[K] = ((1.0 - gamma) * dw_prev[K] + (A - w_prev) * dgam) * m
                ddelta = dw - dw_prev
                marginal = c * (delta / da) + (self.impact / 1e4) * (
                    1.5 * delta * (delta * delta + self.eps) ** -0.25)
                dnet[:, t] = dw @ Yt - ddelta @ marginal
                dw_prev = dw
            w_prev = w

        out = {"net": net, "gross": gross, "traded": traded, "w_last": w_prev}
        if with_grads:
            out["dnet"] = dnet
        return out

    # ------------------------------------------------------------ objective
    def _sharpe_and_grad(self, net, dnet):
        """J = mean/std (population std) and dJ/dp via
        dJ/dnet_t = 1/(T s) - m (net_t - m) / (T s^3)."""
        T = len(net)
        m = net.mean()
        s = net.std() + self.sharpe_eps
        J = m / s
        dJ_dnet = 1.0 / (T * s) - m * (net - m) / (T * s ** 3)
        return J, dnet @ dJ_dnet

    # ------------------------------------------------------------------ fit
    def fit(self, Z, Y):
        """Learn (theta, gamma) by Adam ascent on training net Sharpe."""
        K = Z.shape[0]
        rng = np.random.default_rng(self.seed)
        theta = 0.2 * np.ones(K) + 0.01 * rng.standard_normal(K)
        g = float(np.log(self.gamma_init / (1 - self.gamma_init)))
        p = np.concatenate([theta, [g]])
        m = np.zeros_like(p)
        v = np.zeros_like(p)
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.history_ = []
        for e in range(1, self.epochs + 1):
            gamma = 1.0 / (1.0 + np.exp(-p[-1]))
            out = self._roll(Z, Y, p[:K], gamma, with_grads=True)
            J, grad = self._sharpe_and_grad(out["net"], out["dnet"])
            self.history_.append(J)
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            p = p + self.lr * (m / (1 - b1 ** e)) / \
                (np.sqrt(v / (1 - b2 ** e)) + eps)
            # Project theta to the unit L1 sphere: the aim normalization makes
            # theta's SCALE a null direction of the objective, and Adam's
            # scale-free steps random-walk null directions (for K=1 this can
            # silently flip the book's sign). Projection removes the null
            # direction; real (perpendicular) gradients are unaffected.
            n1 = np.abs(p[:K]).sum()
            if n1 > 1e-12:
                p[:K] = p[:K] / n1
        self.theta_ = p[:K]
        self.gamma_ = float(1.0 / (1.0 + np.exp(-p[-1])))
        return self

    # ----------------------------------------------------------------- eval
    def roll(self, Z, Y, w0=None, gamma=None, theta=None):
        """Simulate with learned (or overridden) parameters; no gradients.
        gamma=1.0 override gives the MYOPIC full-rebalance benchmark on the
        same aim portfolio -- the control the agent must beat under costs."""
        th = self.theta_ if theta is None else np.asarray(theta, float)
        ga = self.gamma_ if gamma is None else float(gamma)
        return self._roll(Z, Y, th, ga, w0=w0, with_grads=False)


def aim_weights(Z_t: np.ndarray, theta: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """The aim portfolio for ONE cross-section, given a learned signal blend.

    Same score -> demean -> normalize map as `DiffPolicyAgent._roll`'s aim
    step, exposed standalone for live inference: at the live decision date
    there is no forward return yet (it hasn't happened), so the full roll
    (which needs Y) cannot run. This only needs today's signals and the
    already-fit theta. Deliberately does NOT apply gamma partial adjustment
    -- that belongs to whatever holds the live inventory (the execution
    evaluator, using the broker's actual current position), not to a second,
    redundant smoothing pass here. Returns an (N,) dollar-neutral aim
    ($1 long / $1 short gross) aligned with Z_t's columns.
    """
    Z_t = np.asarray(Z_t, float)
    theta = np.asarray(theta, float)
    s = theta @ Z_t
    s_t = s - s.mean()
    sa = _softabs(s_t, eps)
    total = sa.sum()
    if total <= eps:
        return np.zeros_like(s_t)
    return 2.0 * s_t / total


def panel_to_matrices(panel, signal_cols, fwd_col: str = "fwd_ret"):
    """Long panel -> (Z (K,T,N), Y (T,N), dates, tickers).

    Y keeps NaN where a name has no forward return that month; the policy
    treats those names as untradable (see module notes on masking), so an
    unbalanced real-data panel keeps every month. Residual signal NaNs are
    filled with the cross-sectional neutral 0."""
    df = panel.dropna(subset=[fwd_col])
    Ypv = df.pivot_table(index="date", columns="ticker", values=fwd_col)
    dates, tickers = Ypv.index, Ypv.columns
    Z = np.stack([
        df.pivot_table(index="date", columns="ticker", values=c)
          .reindex(index=dates, columns=tickers).fillna(0.0).to_numpy()
        for c in signal_cols
    ])
    return Z, Ypv.to_numpy(), dates, tickers


class MultiSpeedPolicyAgent(DiffPolicyAgent):
    """Garleanu-Pedersen's FULL structure: a trading speed PER SIGNAL.

    The single-speed agent smooths the combined aim with one gamma, which
    (documented null, docs/06 §4) makes the slow-signal aim tilt cancel.
    GP's actual optimum gives each return-predicting factor its own
    adjustment rate. Policy:

        u_k,t = (1 - gamma_k) u_{k,t-1} + gamma_k * demean(z_k,t)
        w_t   = 2 * (sum_k theta_k u_k,t) / || sum_k theta_k u_k,t ||_1

    i.e., per-signal EMAs with learned speeds gamma_k = sigmoid(g_k),
    blended by learned theta and normalized to a $1/$1 dollar-neutral book.
    2K parameters; exact forward-mode gradients as before (each u_k depends
    only on its own g_k, so the sensitivity recursion stays cheap).
    """

    def _roll(self, Z, Y, theta, gamma, w0=None, with_grads=False):
        # here `gamma` is a VECTOR (K,); scalar input broadcasts (myopic=1.0)
        K, T, N = Z.shape
        gamma = np.broadcast_to(np.asarray(gamma, float), (K,)).copy()
        P = 2 * K
        c = self.cost / 1e4
        u = np.zeros((K, N))
        cu = np.zeros((K, N))                    # du_k / dg_k
        w_prev = np.zeros(N) if w0 is None else w0.copy()
        dw_prev = np.zeros((P, N)) if with_grads else None
        net = np.empty(T)
        gross = np.empty(T)
        traded = np.empty(T)
        dnet = np.empty((P, T)) if with_grads else None

        for t in range(T):
            m, Yt = _mask_row(Y[t])
            n = m.sum()
            Zt = Z[:, t, :]
            A = (Zt - (Zt @ m)[:, None] / max(n, 1.0)) * m            # (K,N)
            if with_grads:
                cu = ((1 - gamma)[:, None] * cu
                      + (A - u) * (gamma * (1 - gamma))[:, None]) * m
            # an untradable name's EMA state resets rather than going stale
            u = ((1 - gamma)[:, None] * u + gamma[:, None] * A) * m
            wt = theta @ u                                            # (N,)
            wta = _softabs(wt, self.eps)
            D = (wta * m).sum()
            w = 2.0 * wt / D if D > 0 else np.zeros(N)
            delta = w - w_prev
            da = _softabs(delta, self.eps)
            act = (m > 0) | (w_prev != 0)       # tradable or being closed
            gross[t] = w @ Yt
            traded[t] = da[act].sum()
            impact_t = float(((delta * delta + self.eps) ** 0.75)[act].sum())
            net[t] = gross[t] - c * traded[t] - (self.impact / 1e4) * impact_t

            if with_grads:
                ss = (wt / wta) * m                            # softsign
                # dwt/dp (P,N)
                V = np.vstack([u, theta[:, None] * cu])
                if D > 0:
                    dD = V @ ss                                       # (P,)
                    dw = 2.0 * (V * D - np.outer(dD, wt)) / (D * D)
                else:
                    dw = np.zeros((P, N))
                ddelta = dw - dw_prev
                marginal = c * (delta / da) + (self.impact / 1e4) * (
                    1.5 * delta * (delta * delta + self.eps) ** -0.25)
                dnet[:, t] = dw @ Yt - ddelta @ marginal
                dw_prev = dw
            w_prev = w

        out = {"net": net, "gross": gross, "traded": traded, "w_last": w_prev}
        if with_grads:
            out["dnet"] = dnet
        return out

    def fit(self, Z, Y):
        K = Z.shape[0]
        rng = np.random.default_rng(self.seed)
        theta = 0.2 * np.ones(K) + 0.01 * rng.standard_normal(K)
        g = np.full(K, float(np.log(self.gamma_init / (1 - self.gamma_init))))
        p = np.concatenate([theta, g])
        m = np.zeros_like(p)
        v = np.zeros_like(p)
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.history_ = []
        for e in range(1, self.epochs + 1):
            gamma = 1.0 / (1.0 + np.exp(-p[K:]))
            out = self._roll(Z, Y, p[:K], gamma, with_grads=True)
            J, grad = self._sharpe_and_grad(out["net"], out["dnet"])
            self.history_.append(J)
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            p = p + self.lr * (m / (1 - b1 ** e)) / \
                (np.sqrt(v / (1 - b2 ** e)) + eps)
            n1 = np.abs(p[:K]).sum()   # same null-direction projection
            if n1 > 1e-12:
                p[:K] = p[:K] / n1
        self.theta_ = p[:K]
        self.gamma_ = 1.0 / (1.0 + np.exp(-p[K:]))    # vector now
        return self
