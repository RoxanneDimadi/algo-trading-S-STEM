"""Walk-forward and engine correctness: the anti-leak structural tests."""
# Requesting a pytest fixture shadows the fixture function's name by
# design -- that is how pytest injects it.
# pylint: disable=redefined-outer-name
import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_model_backtest
from src.backtest.walkforward import walkforward_splits
from src.models.models import make_lgbm, make_linear
from src.utils.stats import month_end_freq


def test_walkforward_no_overlap_and_purge():
    dates = pd.date_range("2000-01-31", periods=200, freq=month_end_freq())
    folds = walkforward_splits(dates, min_train=100, test_size=12, purge=2,
                               embargo=1)
    all_test = []
    for f in folds:
        assert f.train_dates.max() < f.test_dates.min()
        gap = (list(dates).index(f.test_dates.min())
               - list(dates).index(f.train_dates.max()) - 1)
        assert gap >= 3  # purge + embargo
        all_test.extend(f.test_dates)
    assert len(all_test) == len(set(all_test)), "test folds must not overlap"


def test_walkforward_rejects_zero_purge():
    dates = pd.date_range("2000-01-31", periods=200, freq=month_end_freq())
    with pytest.raises(ValueError):
        walkforward_splits(dates, purge=0)


def test_models_backtest_runs_and_lgbm_beats_linear_on_interaction(world_cfg):
    panel, wcfg, ecfg = world_cfg
    feats = [c for c in panel.columns if c.startswith("sig_")]
    lin = run_model_backtest(panel, feats,
                             lambda: make_linear({"alpha": 1e-4}), wcfg, ecfg)
    lgbm = run_model_backtest(panel, feats, lambda: make_lgbm({}), wcfg, ecfg)
    # both should find the planted structure OOS
    assert lin["oos_ic"]["ic_mean"] > 0
    assert lgbm["oos_ic"]["ic_mean"] > 0
    # the generator plants a nonlinear interaction: trees should exploit it
    assert lgbm["oos_ic"]["ic_mean"] > lin["oos_ic"]["ic_mean"], (
        "LightGBM should beat the linear benchmark when a nonlinear "
        "interaction is planted (the GKX comparison in miniature)"
    )


@pytest.fixture(scope="session")
def world_cfg(world, cfg):
    panel, _, _ = world
    wcfg = dict(cfg["walkforward"], min_train=96, test_size=12)
    ecfg = cfg["evaluation"]
    return panel, wcfg, ecfg


def test_icnet_from_scratch_finds_planted_structure(world_cfg):
    """The custom NumPy model (per-date IC objective) must detect the planted
    cross-sectional structure out-of-sample. We assert detection, not victory:
    which model wins is an empirical question the pipeline reports honestly."""
    from src.models.icnet import make_icnet
    panel, wcfg, ecfg = world_cfg
    feats = [c for c in panel.columns if c.startswith("sig_")]
    res = run_model_backtest(
        panel, feats,
        lambda: make_icnet({"hidden": 16, "max_epochs": 200, "patience": 20}),
        wcfg, ecfg)
    assert res["oos_ic"]["ic_mean"] > 0.01
    assert res["oos_ic"]["ic_tstat"] > 2.0
    # importances exist and are finite (first-layer path weights)
    assert "feature_importance" in res
    assert res["feature_importance"].notna().all()


def test_icnet_fit_does_not_crash_on_short_training_window():
    """val_fraction's floor of 6 validation dates is sized for realistic
    walk-forward windows; on a short one (small min_train, or a short
    initial fold) that floor used to consume the ENTIRE training set,
    leaving none to train on and raising inside fit(). A short window must
    now degrade gracefully (fewer epochs via early stopping) rather than
    crash, for any number of unique dates down to 1."""
    from src.models.icnet import make_icnet

    rng = np.random.default_rng(0)
    n_names, K = 80, 3
    for n_dates in (1, 2, 6, 10):
        dates = np.repeat(np.arange(n_dates), n_names)
        X = rng.standard_normal((n_dates * n_names, K))
        y = rng.standard_normal(n_dates * n_names)
        m = make_icnet({"hidden": 8, "max_epochs": 5})
        m.fit(X, y, dates=dates)  # must not raise
        assert np.isfinite(m.predict(X)).all()


def test_pulse_tracks_planted_decay_and_predicts_oos(world_cfg, cfg, world):
    """The headline claim, made falsifiable: PULSE's filtered efficacy states
    must TRACK the planted time-varying betas (correlate with the true path,
    register the post-publication step-down, and keep the placebo near zero),
    and its forecasts must carry OOS power through the identical harness."""
    from src.models.pulse import make_pulse

    panel, wcfg, ecfg = world_cfg
    _, _, meta = world
    feats = list(meta.index)

    d = panel.dropna(subset=[*feats, "fwd_ret"])
    m = make_pulse(cfg["models"]["pulse"]).fit(
        d[feats], d["fwd_ret"].to_numpy(), dates=d["date"].to_numpy())
    eff = pd.DataFrame(m.filter_history_,
                       index=pd.DatetimeIndex(m.filter_dates_),
                       columns=m.feature_names_)
    scfg = cfg["data"]["synthetic"]
    for sig in ["sig_momentum", "sig_liquidity"]:
        beta = float(scfg["signals"][sig]["beta"])
        true = pd.Series(beta, index=eff.index)
        true[eff.index > pd.Timestamp(scfg["signals"][sig]["sample_end"])] = \
            beta * float(scfg["decay_post_sample"])
        true[eff.index > pd.Timestamp(scfg["signals"][sig]["pub_date"])] = \
            beta * float(scfg["decay_post_pub"])
        # The filter estimates coefficients on RANK-normalized signals -- a
        # positive rescaling of beta (docs/math/04, check 3) -- so we test
        # SHAPE (correlation and step-down), which rescaling preserves.
        corr = np.corrcoef(eff[sig], true)[0, 1]
        assert corr > 0.4, (
            f"{sig}: filtered path fails to track planted decay "
            f"(corr={corr:.2f})")
        pre = eff[sig][eff.index <= pd.Timestamp(
            scfg["signals"][sig]["sample_end"])].mean()
        post = eff[sig][eff.index > pd.Timestamp(
            scfg["signals"][sig]["pub_date"])].mean()
        assert post < pre, f"{sig}: no post-publication efficacy step-down"

    assert eff["sig_dead"].abs().mean() < 0.5 * \
        eff["sig_momentum"].abs().mean()

    res = run_model_backtest(panel, feats,
                             lambda: make_pulse(cfg["models"]["pulse"]),
                             wcfg, ecfg)
    assert res["oos_ic"]["ic_mean"] > 0.02
    assert res["oos_ic"]["ic_tstat"] > 2.0


def test_pulse_predict_decay_uses_calendar_gap_not_date_position():
    """predict()'s h ('months past training end') must be the actual
    calendar gap from the last fitted date, not the position of a test date
    within whatever `dates` array happens to be passed in. A purged
    walk-forward fold's first test date is purge+embargo months after the
    last training date (src/backtest/walkforward.py) -- enumerating test
    dates from 0 would silently drop that gap and under-decay the state."""
    import pandas as pd
    from src.models.pulse import PulseModel

    rng = np.random.default_rng(0)
    n_dates, n_names = 150, 80
    dates = pd.date_range("2000-01-31", periods=n_dates, freq=month_end_freq())
    rows_date = np.repeat(dates, n_names)
    X = rng.standard_normal((n_dates * n_names, 1))
    y = 0.5 * X[:, 0] + 0.1 * rng.standard_normal(n_dates * n_names)

    m = PulseModel(interactions=False, a_grid=(0.9,), q_scale_grid=(0.01,))
    m.fit(X, y, dates=rows_date)
    last_train = pd.Timestamp(m.filter_dates_[-1])

    # a purge=1 fold's first test date is 2 months past the last train date
    test_date = last_train + pd.DateOffset(months=2)
    Xt = rng.standard_normal((5, 1))
    pred = m.predict(Xt, dates=np.array([test_date] * 5))
    assert np.allclose(pred, Xt @ (m.a_ ** 2 * m.state_mean_))
    assert not np.allclose(pred, Xt @ (m.a_ ** 1 * m.state_mean_))


def test_agent_learns_garleanu_pedersen_comparative_statics(world):
    """Planted-truth validation of the agent's ECONOMICS, not just its P&L:
    (1) learned trading speed falls as costs rise; (2) the learned signal
    blend tilts toward the persistent signal (sig_value, rho=.98) relative
    to the fast one (sig_momentum, rho=.90) as costs rise; (3) under high
    costs the agent beats the myopic full-rebalance policy on its own aim."""
    from src.agent.policy import DiffPolicyAgent, panel_to_matrices

    panel, _, meta = world
    feats = list(meta.index)
    Z, Y, _dates, _ = panel_to_matrices(panel, feats)
    Ztr, Ytr = Z[:, :180, :], Y[:180]

    free = DiffPolicyAgent(cost_bps_per_side=0.0,
                           epochs=250, seed=3).fit(Ztr, Ytr)
    mid = DiffPolicyAgent(cost_bps_per_side=40.0,
                          epochs=250, seed=3).fit(Ztr, Ytr)
    dear = DiffPolicyAgent(cost_bps_per_side=100.0,
                           epochs=250, seed=3).fit(Ztr, Ytr)

    # (1) GP comparative static: trading speed falls MONOTONICALLY in cost
    assert free.gamma_ > mid.gamma_ > dear.gamma_, (
        free.gamma_, mid.gamma_, dear.gamma_)

    # (2) DOCUMENTED NULL RESULT (docs/06 §5): in this single-shared-speed
    # policy class the learned blend does NOT tilt toward the persistent
    # signal as costs rise -- GP's aim-tilt prediction requires per-signal
    # trading speeds. We assert only that the blend stays sane (all-positive
    # loadings on the planted-positive signals' side of the simplex).
    th = np.abs(mid.theta_) / np.abs(mid.theta_).sum()
    assert th[feats.index("sig_momentum")] > th[feats.index("sig_dead")]

    # (3) smoothing beats myopic full rebalancing at meaningful cost,
    # out of sample, on the identical learned aim
    test_out = mid.roll(Z[:, 180:, :], Y[180:])
    myop_out = mid.roll(Z[:, 180:, :], Y[180:], gamma=1.0)

    def sharpe(x):
        return x.mean() / (x.std() + 1e-12)
    assert sharpe(test_out["net"]) > sharpe(
        myop_out["net"]), (sharpe(test_out["net"]),
                           sharpe(myop_out["net"]))


def test_multispeed_agent_cost_protection_and_speed_response(world):
    """Per-signal-speed agent (GP's full structure): (1) the dominant fast
    signal's learned speed falls with cost; (2) at high cost the agent
    beats the myopic full-rebalance control decisively out of sample.
    (The raw-theta slow-signal tilt remains null even here -- documented in
    docs/06 with the effective-exposure metric as the real-data lens.)"""
    from src.agent.policy import MultiSpeedPolicyAgent, panel_to_matrices

    panel, _, meta = world
    feats = list(meta.index)
    Z, Y, _, _ = panel_to_matrices(panel, feats)
    Ztr, Ytr = Z[:, :180, :], Y[:180]

    free = MultiSpeedPolicyAgent(
        cost_bps_per_side=0.0, epochs=200, seed=3).fit(Ztr, Ytr)
    dear = MultiSpeedPolicyAgent(
        cost_bps_per_side=60.0, epochs=200, seed=3).fit(Ztr, Ytr)

    im = feats.index("sig_momentum")
    assert dear.gamma_[im] < free.gamma_[
        im], (free.gamma_[im], dear.gamma_[im])

    ag = dear.roll(Z[:, 180:, :], Y[180:])
    my = dear.roll(Z[:, 180:, :], Y[180:], gamma=1.0)

    def s(x):
        return x.mean() / (x.std() + 1e-12)
    assert s(ag["net"]) > s(my["net"]) + 0.05, (s(ag["net"]), s(my["net"]))


def test_msrr_closed_form_recovered_by_zero_cost_agent(world):
    """At zero cost the numerically-trained agent must recover the CLOSED
    FORM: max-Sharpe over factor returns (MSRR). Validates both the
    optimizer and the theory in one shot; also checks MSRR zeroes the
    placebo."""
    from src.agent.policy import DiffPolicyAgent, panel_to_matrices
    from src.agent.msrr import msrr_theta

    panel, _, meta = world
    feats = list(meta.index)
    Z, Y, _, _ = panel_to_matrices(panel, feats)
    Ztr, Ytr = Z[:, :180, :], Y[:180]

    th_msrr, _ = msrr_theta(Ztr, Ytr)
    agent = DiffPolicyAgent(cost_bps_per_side=0.0,
                            epochs=250, seed=3).fit(Ztr, Ytr)
    th_a = agent.theta_ / np.abs(agent.theta_).sum()
    cos = th_a @ th_msrr / (np.linalg.norm(th_a) * np.linalg.norm(th_msrr))
    assert cos > 0.98, cos
    assert abs(th_msrr[feats.index("sig_dead")]) < 0.05

    def s(x):
        return x.mean() / (x.std() + 1e-12)
    S_a = s(agent.roll(Ztr, Ytr, gamma=1.0)["gross"])
    S_m = s(agent.roll(Ztr, Ytr, gamma=1.0, theta=th_msrr)["gross"])
    assert abs(S_a - S_m) / max(S_m, 1e-9) < 0.05, (S_a, S_m)


def test_msrr_handles_single_signal():
    """np.cov(F, rowvar=False) collapses to a 0-d scalar when F has a single
    column (K=1, e.g. a composed single-forecast aim); np.trace/np.eye on
    that scalar used to raise. K=1 must return a usable, unit-L1-norm theta."""
    from src.agent.msrr import msrr_theta

    rng = np.random.default_rng(0)
    Z = rng.standard_normal((1, 60, 40))
    Y = rng.standard_normal((60, 40))
    theta, info = msrr_theta(Z, Y)
    assert theta.shape == (1,)
    assert np.isclose(np.abs(theta).sum(), 1.0)
    assert np.isfinite(info["monthly_sharpe_factor_space"])


def test_sqrt_impact_slows_trading(world):
    """Square-root impact (convex cost in trade size) inside the loss must
    push the learned policy toward slower, smaller trading."""
    from src.agent.policy import DiffPolicyAgent, panel_to_matrices

    panel, _, meta = world
    Z, Y, _, _ = panel_to_matrices(panel, list(meta.index))
    Ztr, Ytr = Z[:, :180, :], Y[:180]
    lin = DiffPolicyAgent(cost_bps_per_side=10.0,
                          epochs=200, seed=3).fit(Ztr, Ytr)
    imp = DiffPolicyAgent(cost_bps_per_side=10.0, impact_bps=60.0,
                          epochs=200, seed=3).fit(Ztr, Ytr)
    assert imp.gamma_ < lin.gamma_, (lin.gamma_, imp.gamma_)
    t_lin = lin.roll(Ztr, Ytr)["traded"].mean()
    t_imp = imp.roll(Ztr, Ytr)["traded"].mean()
    assert t_imp < t_lin, (t_lin, t_imp)


def test_backtest_agent_carries_inventory_across_fold_boundaries(world_cfg, cfg):
    """backtest_agent's docstring claims the book is ONE continuous portfolio
    across fold boundaries, via w_carry -> w0 of the next fold's roll(). This
    was previously untested by anything in this suite.

    If a fold boundary silently reset to flat (w0=None instead of w_carry),
    that fold's first-date traded notional would equal a full cold-start
    rebalance from cash: traded = gamma * 2 (roll()'s w_prev=0, so
    delta = w = gamma*A, and ||A||_1 = 2 by construction, docs/06 Sec. 3).
    Each fold's own learned gamma (recovered from params_by_fold) gives the
    exact cold-start value to compare against -- no guessing a threshold."""
    from src.agent.backtest import backtest_agent
    from src.backtest.walkforward import walkforward_splits

    panel, wcfg, ecfg = world_cfg
    feats = [c for c in panel.columns if c.startswith("sig_")]
    res = backtest_agent(panel, feats, wcfg, ecfg, cfg["agent"])
    traded = res["series"]["traded"]
    gamma_by_fold = res["params_by_fold"]["gamma"]

    dates = panel["date"].sort_values().unique()
    folds = walkforward_splits(dates, min_train=wcfg["min_train"],
                               test_size=wcfg["test_size"], purge=wcfg["purge"],
                               embargo=wcfg["embargo"], expanding=wcfg["expanding"])
    assert len(folds) >= 2, "need >= 2 folds to exercise a boundary at all"

    # ||A||_1 (the aim's gross) is only APPROXIMATELY 2: softabs(x, eps) =
    # sqrt(x^2+eps) smooths the normalization denominator too, so the
    # cold-start value is close to, not bit-for-bit, 2*gamma.
    cold_start_equiv = 2.0 * gamma_by_fold.loc[folds[0].fold_id]
    cold_start = traded.loc[folds[0].test_dates.min()]
    assert np.isclose(cold_start, cold_start_equiv, rtol=1e-3), (
        "fold 1 starts from literal cash (w0=None): traded must be "
        "approximately the cold-start value")

    for f in folds[1:]:
        boundary_traded = traded.loc[f.test_dates.min()]
        this_fold_cold_start = 2.0 * gamma_by_fold.loc[f.fold_id]
        assert boundary_traded < this_fold_cold_start * (1 - 1e-3), (
            f"fold {f.fold_id} opened with a cold-start-sized rebalance "
            f"(traded={boundary_traded:.4f} >= cold-start {this_fold_cold_start:.4f}) "
            "-- inventory was not carried forward from the previous fold's "
            "ending weights"
        )


def _unbalanced_world(K=3, T=48, N=7, seed=11):
    """Random signals/returns where names start late and stop early, the
    shape of the real OSAP factor panel."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((K, T, N))
    Y = 0.02 * rng.standard_normal((T, N))
    Y[:10, 0] = np.nan          # starts late
    Y[30:, 1] = np.nan          # stops early
    Y[15:25, 2] = np.nan        # gap
    return Z, Y


@pytest.mark.parametrize("multi", [False, True])
def test_agent_mask_untradable_name_has_no_effect(multi):
    """A name that is never tradable must leave every output and gradient
    of the policy unchanged -- masking reduces exactly to the smaller panel."""
    from src.agent.policy import DiffPolicyAgent, MultiSpeedPolicyAgent

    Z, Y = _unbalanced_world()
    Zp = np.concatenate([Z, np.random.default_rng(0).standard_normal(
        (Z.shape[0], Z.shape[1], 1))], axis=2)
    Yp = np.concatenate([Y, np.full((Y.shape[0], 1), np.nan)], axis=1)
    cls = MultiSpeedPolicyAgent if multi else DiffPolicyAgent
    agent = cls(cost_bps_per_side=20.0, impact_bps=30.0)
    theta = np.array([0.5, -0.3, 0.2])
    gamma = np.array([0.3, 0.6, 0.9]) if multi else 0.4
    a = agent._roll(Z, Y, theta, gamma, with_grads=True)
    b = agent._roll(Zp, Yp, theta, gamma, with_grads=True)
    for k in ("net", "gross", "traded", "dnet"):
        np.testing.assert_allclose(a[k], b[k], rtol=1e-10, atol=1e-14)
    assert b["w_last"][-1] == 0.0


def test_agent_mask_closes_positions_and_keeps_all_months():
    """panel_to_matrices must keep months where some names are missing, and
    the policy must hold nothing in a name while it is untradable."""
    from src.agent.policy import DiffPolicyAgent, panel_to_matrices

    Z, Y = _unbalanced_world()
    K, T, N = Z.shape
    dates = pd.date_range("2000-01-31", periods=T, freq=month_end_freq())
    rows = [{"date": dates[t], "ticker": f"f{i}", "fwd_ret": Y[t, i],
             **{f"s{k}": Z[k, t, i] for k in range(K)}}
            for t in range(T) for i in range(N) if np.isfinite(Y[t, i])]
    Z2, Y2, d2, _ = panel_to_matrices(pd.DataFrame(rows),
                                      [f"s{k}" for k in range(K)])
    assert len(d2) == T, "months with a missing name were dropped"
    assert np.isnan(Y2).sum() == np.isnan(Y).sum()

    agent = DiffPolicyAgent(cost_bps_per_side=10.0)
    out = agent._roll(Z2, Y2, np.array([0.5, -0.3, 0.2]), 0.4)
    assert np.isfinite(out["net"]).all()
    # name f1 stops at t=30: by the end it must be flat
    w_end = agent._roll(Z2[:, :31], Y2[:31], np.array([0.5, -0.3, 0.2]),
                        0.4)["w_last"]
    assert w_end[1] == 0.0


@pytest.mark.parametrize("multi", [False, True])
def test_agent_masked_gradients_match_finite_differences(multi):
    """The exact forward-mode gradients must survive masking: compare
    d(sum net)/d(param) against central finite differences."""
    from src.agent.policy import DiffPolicyAgent, MultiSpeedPolicyAgent

    Z, Y = _unbalanced_world()
    cls = MultiSpeedPolicyAgent if multi else DiffPolicyAgent
    agent = cls(cost_bps_per_side=20.0, impact_bps=30.0)
    K = Z.shape[0]
    theta = np.array([0.5, -0.3, 0.2])
    g = np.array([-0.5, 0.2, 1.0]) if multi else np.array([-0.3])
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))

    def total(th, gg):
        gam = sig(gg) if multi else float(sig(gg[0]))
        return agent._roll(Z, Y, th, gam)["net"].sum()

    gam0 = sig(g) if multi else float(sig(g[0]))
    analytic = agent._roll(Z, Y, theta, gam0, with_grads=True)["dnet"].sum(1)
    h = 1e-6
    numeric = []
    for k in range(K):
        e = np.zeros(K)
        e[k] = h
        numeric.append((total(theta + e, g) - total(theta - e, g)) / (2 * h))
    for j in range(len(g)):
        e = np.zeros(len(g))
        e[j] = h
        numeric.append((total(theta, g + e) - total(theta, g - e)) / (2 * h))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-7)
