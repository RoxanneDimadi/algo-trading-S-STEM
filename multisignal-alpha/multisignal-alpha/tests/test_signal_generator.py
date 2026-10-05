"""Tests for src.execution.signal_generator: agent/models -> execution bridge."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.agent.policy import aim_weights
from src.execution.agent_evaluator import AgentPolicyConfig, CostAwareAgentEvaluator
from src.execution.alpaca_bridge import AlpacaConfig, AlpacaExecutionBridge
from src.execution.ledger import ExecutionLedger
from src.execution.signal_generator import (
    SignalGeneratorConfig,
    build_signals,
    fit_live_agent,
    latest_cross_section,
    load_permno_ticker_map,
    run_live_signals,
)


# ---------------------------------------------------------------- aim_weights

def test_aim_weights_is_dollar_neutral_and_signed():
    Z = np.array([[1.0, 2.0, -1.0, -2.0]])   # one signal, four names
    theta = np.array([1.0])
    aim = aim_weights(Z, theta)
    assert aim.shape == (4,)
    assert np.isclose(np.abs(aim).sum(), 2.0)
    # ranking by score should be preserved: name 1 (score 2) > name 0 > name 2 > name 3
    assert aim[1] > aim[0] > aim[2] > aim[3]


def test_aim_weights_zero_when_all_scores_identical():
    Z = np.array([[5.0, 5.0, 5.0]])
    aim = aim_weights(Z, np.array([1.0]))
    assert np.allclose(aim, 0.0)


# ------------------------------------------------------------ permno mapping

def test_load_permno_ticker_map(tmp_path):
    p = tmp_path / "map.csv"
    p.write_text("permno,symbol\n10001,AAPL\n10002,MSFT\n")
    m = load_permno_ticker_map(str(p))
    assert m == {"10001": "AAPL", "10002": "MSFT"}


def test_load_permno_ticker_map_rejects_bad_schema(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("foo,bar\n1,2\n")
    with pytest.raises(ValueError):
        load_permno_ticker_map(str(p))


# ------------------------------------------------------- latest_cross_section

def _toy_panel():
    rows = []
    for d, vals in [("2020-01-31", [1.0, -1.0]), ("2020-02-29", [2.0, -2.0, np.nan])]:
        for i, v in enumerate(vals):
            rows.append({"date": pd.Timestamp(d), "ticker": f"T{i}", "sig": v})
    return pd.DataFrame(rows)


def test_latest_cross_section_picks_max_date_and_drops_nan():
    panel = _toy_panel()
    as_of, Z, tickers = latest_cross_section(panel, ["sig"])
    assert as_of == pd.Timestamp("2020-02-29")
    assert Z.shape == (1, 2)   # the NaN row (T2) is dropped
    assert list(tickers) == ["T0", "T1"]


# ------------------------------------------------------------- build_signals

def test_build_signals_thresholds_maps_and_caps():
    panel = _toy_panel()
    theta = np.array([1.0])
    cfg = SignalGeneratorConfig(min_abs_aim_weight=0.5, max_names=1)
    perm_map = {"T0": "AAPL", "T1": "MSFT"}
    as_of, signals, skipped = build_signals(panel, ["sig"], theta, cfg, perm_map,
                                            max_position_pct=0.5)
    assert as_of == pd.Timestamp("2020-02-29")
    # dollar-neutral aim over [2.0, -2.0] -> [+1.0, -1.0]; both pass threshold,
    # but max_names=1 keeps only the larger |aim| one
    assert len(signals) == 1
    assert signals[0]["ticker"] in ("AAPL", "MSFT")
    assert signals[0]["action"] in ("buy", "sell")
    assert abs(signals[0]["signal_strength"]) <= 1.0
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "cut by max_names"


def test_build_signals_skips_unmapped_permno():
    panel = _toy_panel()
    theta = np.array([1.0])
    cfg = SignalGeneratorConfig(min_abs_aim_weight=0.0, max_names=None)
    perm_map = {"T0": "AAPL"}   # T1 unmapped
    _as_of, signals, skipped = build_signals(panel, ["sig"], theta, cfg, perm_map, 0.5)
    assert len(signals) == 1 and signals[0]["ticker"] == "AAPL"
    assert any(s["reason"] == "no permno->ticker mapping" for s in skipped)


def test_build_signals_dedupes_symbol_collisions():
    # T0 (aim +1.0) and T1 (aim -1.0) both map to AAPL: a bad/stale mapping,
    # or a CRSP permno change. Only the larger |aim weight| name should
    # reach the evaluator -- never both, which could double-trade AAPL.
    panel = _toy_panel()
    theta = np.array([1.0])
    cfg = SignalGeneratorConfig(min_abs_aim_weight=0.0, max_names=None)
    perm_map = {"T0": "AAPL", "T1": "AAPL"}
    _as_of, signals, skipped = build_signals(panel, ["sig"], theta, cfg, perm_map, 0.5)
    assert len(signals) == 1
    assert signals[0]["ticker"] == "AAPL"
    assert signals[0]["permno"] == "T0"   # |aim|=1.0 for both; T0 kept (inserted first)
    assert any("duplicate symbol" in s["reason"] for s in skipped)


def test_build_signals_without_map_uses_raw_ticker():
    panel = _toy_panel()
    theta = np.array([1.0])
    cfg = SignalGeneratorConfig(min_abs_aim_weight=0.0, max_names=None)
    _as_of, signals, _skipped = build_signals(panel, ["sig"], theta, cfg, None, 0.5)
    tickers = {s["ticker"] for s in signals}
    assert tickers == {"T0", "T1"}


# ------------------------------------------------------------- fit_live_agent

def test_fit_live_agent_requires_enough_history(world):
    panel, _factors, meta = world
    signal_cols = list(meta.index)
    short_panel = panel[panel["date"] <= panel["date"].unique()[10]]
    cfg = SignalGeneratorConfig(min_train_months=24, agent_epochs=5)
    with pytest.raises(ValueError):
        fit_live_agent(short_panel, signal_cols, cfg)


def test_fit_live_agent_fits_theta_on_full_history(world):
    panel, _factors, meta = world
    signal_cols = list(meta.index)
    cfg = SignalGeneratorConfig(min_train_months=24, agent_epochs=10, agent_seed=1)
    theta, fit_meta = fit_live_agent(panel, signal_cols, cfg)
    assert theta.shape == (len(signal_cols),)
    assert np.isclose(np.abs(theta).sum(), 1.0)  # L1-projected, per DiffPolicyAgent.fit
    assert set(fit_meta["theta"].keys()) == set(signal_cols)


# ------------------------------------------------------------ run_live_signals

@pytest.fixture
def mem_ledger():
    return ExecutionLedger(db_path=":memory:", jsonl_path=None)


@pytest.fixture
def mock_bridge(mem_ledger):
    return AlpacaExecutionBridge(
        config=AlpacaConfig(api_key="K", secret_key="S", paper=True), ledger=mem_ledger,
    )


@pytest.fixture
def mock_evaluator(mock_bridge, mem_ledger):
    cfg = AgentPolicyConfig(
        cost_bps=5.0, gamma_speed=1.0, impact_bps=0.0, alpha_hurdle_bps=0.0,
        max_position_pct=0.20, max_order_notional=50000.0, min_trade_notional=50.0,
        allow_short=True, default_base_alpha_bps=50.0,
    )
    return CostAwareAgentEvaluator(bridge=mock_bridge, config=cfg, ledger=mem_ledger)


def _order_resp(symbol, status=201):
    from unittest.mock import MagicMock
    body = {"id": f"ord_{symbol}", "status": "accepted", "filled_qty": "0"}
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"X-Request-ID": f"req_{symbol}"}
    resp.text = json.dumps(body)
    resp.json.return_value = body
    return resp


def test_run_live_signals_dry_run_never_submits(mock_evaluator, mock_bridge, mem_ledger):
    from unittest.mock import patch
    panel = _toy_panel()
    theta = np.array([1.0])
    cfg = SignalGeneratorConfig(min_abs_aim_weight=0.0, max_names=None)
    perm_map = {"T0": "AAPL", "T1": "MSFT"}

    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000", "buying_power": "100000"}), \
         patch.object(mock_bridge, "get_position", return_value=None), \
         patch.object(mock_bridge, "get_latest_price", return_value=200.0):
        report = run_live_signals(
            panel, ["sig"], mock_evaluator, mock_bridge,
            config=cfg, theta=theta, permno_ticker_map=perm_map, dry_run=True,
        )

    assert report["n_evaluated"] == 2
    assert all(r["order_submitted"] is False for r in report["results"])
    # no orders were actually placed
    assert mem_ledger.get_recent_orders() == []


def test_run_live_signals_submits_approved_orders(mock_evaluator, mock_bridge, mem_ledger):
    from unittest.mock import patch
    panel = _toy_panel()
    theta = np.array([1.0])
    cfg = SignalGeneratorConfig(min_abs_aim_weight=0.0, max_names=None,
                                strategy_name="diffpolicy_agent")
    perm_map = {"T0": "AAPL", "T1": "MSFT"}

    def fake_request(method, url, **kwargs):
        symbol = "AAPL" if "AAPL" in json.dumps(kwargs.get("json", {})) else "MSFT"
        return _order_resp(symbol)

    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000", "buying_power": "100000"}), \
         patch.object(mock_bridge, "get_position", return_value=None), \
         patch.object(mock_bridge, "get_latest_price", return_value=200.0), \
         patch.object(mock_bridge.session, "request", side_effect=fake_request):
        report = run_live_signals(
            panel, ["sig"], mock_evaluator, mock_bridge,
            config=cfg, theta=theta, permno_ticker_map=perm_map, dry_run=False,
        )

    approved = [r for r in report["results"] if r["approved"]]
    assert len(approved) == 2
    assert all(r["order_submitted"] and r["order_success"] for r in approved)

    orders = mem_ledger.get_recent_orders(limit=10)
    assert {o["symbol"] for o in orders} == {"AAPL", "MSFT"}

    decisions = mem_ledger.get_recent_decisions(limit=10)
    assert all(d["signal_source"] == "diffpolicy_agent" for d in decisions)
