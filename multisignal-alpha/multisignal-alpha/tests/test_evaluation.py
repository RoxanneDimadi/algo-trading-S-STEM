"""Statistical-correctness tests for the evaluation harness.

The philosophy (roadmap §3, weeks 3-4): validate the machinery on data where
the truth is KNOWN before pointing it at real data. A harness that cannot
(a) find a planted signal, (b) clear a placebo, and (c) scream at a leak is
not a harness.
"""
import numpy as np

from src.data.panel import demonstrate_lookahead
from src.evaluation.decay import segment_performance
from src.evaluation.deflated_sharpe import deflated_sharpe, expected_max_sharpe
from src.evaluation.fama_macbeth import fama_macbeth
from src.evaluation.ic import mean_ic
from src.evaluation.portfolio import evaluate_signal_portfolio


def test_planted_signal_detected(world):
    panel, _, _ = world
    r = mean_ic(panel, "sig_momentum")
    assert r["ic_mean"] > 0.01
    assert r["ic_tstat"] > 2.0, "harness failed to detect a real planted signal"


def test_placebo_signal_not_detected(world):
    panel, _, _ = world
    r = mean_ic(panel, "sig_dead")
    assert abs(r["ic_tstat"]) < 2.0, (
        "PLACEBO FAILURE: the harness reports a zero-beta signal as "
        "significant -- something in the alignment or the stats is broken"
    )


def test_lookahead_demonstration_is_absurd(world):
    panel, _, _ = world
    demo = demonstrate_lookahead(panel, seed=1)
    assert demo.loc["leaky_feature", "IC"] > 0.25, "leak should be blatant"
    assert abs(demo.loc["honest_noise", "IC"]) < 0.02


def test_longshort_monotone_with_planted_beta(world):
    panel, _, meta = world
    res_big = evaluate_signal_portfolio(panel, "sig_momentum")
    res_zero = evaluate_signal_portfolio(panel, "sig_dead")
    assert res_big["gross"]["sharpe"] > res_zero["gross"]["sharpe"]
    assert res_big["gross"]["nw_tstat"] > 2.0
    assert abs(res_zero["gross"]["nw_tstat"]) < 2.0


def test_costs_reduce_returns(world):
    panel, _, _ = world
    res = evaluate_signal_portfolio(panel, "sig_momentum", cost_bps_per_side=25)
    assert res["net"]["ann_return"] < res["gross"]["ann_return"]
    assert res["avg_one_way_turnover"] > 0


def test_fama_macbeth_recovers_planted_betas(world):
    panel, _, meta = world
    fm = fama_macbeth(panel, list(meta.index))
    # strongest planted signal significant; placebo not
    assert fm.loc["sig_momentum", "nw_tstat"] > 2.0
    assert abs(fm.loc["sig_dead", "nw_tstat"]) < 2.0
    # coefficient ordering should follow planted beta ordering (roughly)
    assert fm.loc["sig_momentum", "mean_coef"] > fm.loc["sig_value", "mean_coef"]


def test_decay_pattern_recovered(world):
    panel, _, meta = world
    res = evaluate_signal_portfolio(panel, "sig_momentum")
    seg = segment_performance(res["series"]["gross"],
                              meta.loc["sig_momentum", "sample_end"],
                              meta.loc["sig_momentum", "pub_date"])
    assert seg.loc["in_sample", "sharpe"] > seg.loc["post_publication", "sharpe"], \
        "generator plants decay; the decay module must recover it"


def test_deflated_sharpe_sanity():
    # more trials => bigger hurdle => lower DSR for the same observed SR
    few = deflated_sharpe(0.15, 240, 0.0, 3.0, [0.15, 0.05])
    many = deflated_sharpe(0.15, 240, 0.0, 3.0,
                           [0.15, 0.05, 0.02, -0.03, 0.08, 0.11, -0.05, 0.01])
    assert many["dsr"] <= few["dsr"]
    assert expected_max_sharpe(0.01, 10) > expected_max_sharpe(0.01, 2)
