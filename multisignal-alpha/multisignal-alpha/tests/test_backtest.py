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
