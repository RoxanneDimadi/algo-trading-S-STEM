"""Tests for src.execution."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.execution.agent_evaluator import AgentPolicyConfig, CostAwareAgentEvaluator
from src.execution.alpaca_bridge import AlpacaConfig, AlpacaExecutionBridge
from src.execution.ledger import ExecutionLedger, TradeDirective
from src.execution.webhook_listener import create_webhook_app


@pytest.fixture
def mem_ledger():
    return ExecutionLedger(db_path=":memory:", jsonl_path=None)


@pytest.fixture
def mock_bridge(mem_ledger):
    return AlpacaExecutionBridge(
        config=AlpacaConfig(api_key="TEST_KEY_ID", secret_key="TEST_SECRET_KEY", paper=True),
        ledger=mem_ledger,
    )


@pytest.fixture
def mock_evaluator(mock_bridge, mem_ledger):
    cfg = AgentPolicyConfig(
        cost_bps=10.0,
        gamma_speed=0.50,
        impact_bps=2.0,
        alpha_hurdle_bps=5.0,
        max_position_pct=0.20,
        max_order_notional=25000.0,
        min_trade_notional=50.0,
        allow_short=False,
        default_base_alpha_bps=40.0,
    )
    return CostAwareAgentEvaluator(bridge=mock_bridge, config=cfg, ledger=mem_ledger)


def test_bridge_paper_and_live_url_switching():
    assert AlpacaConfig(paper=True).get_effective_base_url() == "https://paper-api.alpaca.markets"
    assert AlpacaConfig(paper=False).get_effective_base_url() == "https://api.alpaca.markets"
    assert (
        AlpacaConfig(base_url="https://mock-alpaca.internal:8080/").get_effective_base_url()
        == "https://mock-alpaca.internal:8080"
    )


def test_bridge_extracts_x_request_id_on_success(mock_bridge, mem_ledger):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"X-Request-ID": "alpaca_req_9988776655"}
    mock_resp.text = json.dumps({"id": "ord_123", "status": "accepted", "symbol": "AAPL", "filled_qty": "0"})
    mock_resp.json.return_value = {"id": "ord_123", "status": "accepted", "symbol": "AAPL", "filled_qty": "0"}

    with patch.object(mock_bridge.session, "request", return_value=mock_resp):
        result = mock_bridge.submit_order(TradeDirective(symbol="AAPL", side="buy", qty=15.0))
        assert result.success is True
        assert result.x_request_id == "alpaca_req_9988776655"
        assert result.alpaca_order_id == "ord_123"

        audits = mem_ledger.get_recent_api_audits(limit=10)
        assert audits[0]["x_request_id"] == "alpaca_req_9988776655"
        assert audits[0]["endpoint"] == "/v2/orders"
        assert mem_ledger.get_recent_orders()[0]["symbol"] == "AAPL"


def test_bridge_extracts_x_request_id_on_error(mock_bridge, mem_ledger):
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.headers = {"X-Request-ID": "err_req_403_insufficient_bp"}
    mock_resp.text = json.dumps({"message": "insufficient buying power", "code": 40310000})
    mock_resp.json.return_value = {"message": "insufficient buying power", "code": 40310000}

    with patch.object(mock_bridge.session, "request", return_value=mock_resp):
        result = mock_bridge.submit_order(TradeDirective(symbol="MSFT", side="buy", qty=500.0))
        assert result.success is False
        assert result.x_request_id == "err_req_403_insufficient_bp"
        assert "insufficient buying power" in (result.error_message or "")

        row = mem_ledger.get_by_x_request_id("err_req_403_insufficient_bp")
        assert row is not None and row["success"] == 0


def test_bridge_handles_network_timeout(mock_bridge, mem_ledger):
    with patch.object(
        mock_bridge.session, "request",
        side_effect=requests.exceptions.Timeout("Connection timed out"),
    ):
        result = mock_bridge.submit_order(TradeDirective(symbol="NVDA", side="buy", qty=10.0))
        assert result.success is False
        assert "Connection timed out" in str(result.error_message)
        assert mem_ledger.get_recent_api_audits()[0]["success"] == 0


def test_evaluator_gp_partial_adjustment_sizing(mock_evaluator, mock_bridge):
    # $100k book, flat; aim 20% of AAPL @ $200 = 100 shares; gamma=0.5 => 50
    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000", "buying_power": "200000", "cash": "100000"}), \
         patch.object(mock_bridge, "get_position", return_value=None):
        decision = mock_evaluator.evaluate_signal({
            "ticker": "AAPL", "action": "buy", "price": 200.0, "signal_strength": 1.0,
        })
        assert decision.approved is True
        assert decision.final_directive.qty == 50.0
        assert decision.final_directive.side == "buy"


def test_evaluator_cost_hurdle_filters_weak_noise(mock_evaluator, mock_bridge):
    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000", "buying_power": "200000"}), \
         patch.object(mock_bridge, "get_position", return_value=None):
        decision = mock_evaluator.evaluate_signal({
            "ticker": "GOOGL", "action": "buy", "price": 150.0, "signal_strength": 0.05,
        })
        assert decision.approved is False
        assert "SKIPPED" in decision.reason


def test_evaluator_short_selling_protection(mock_evaluator, mock_bridge):
    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000"}), \
         patch.object(mock_bridge, "get_position", return_value=None):
        decision = mock_evaluator.evaluate_signal({
            "ticker": "TSLA", "action": "sell", "price": 250.0, "signal_strength": -1.0,
        })
        assert decision.approved is False
        assert "Short selling is disabled" in decision.reason


@pytest.fixture
def test_app(mock_bridge, mock_evaluator, mem_ledger):
    app = create_webhook_app(
        bridge=mock_bridge,
        evaluator=mock_evaluator,
        webhook_passphrase="test_secret_passphrase_123",
        ledger=mem_ledger,
    )
    app.config["TESTING"] = True
    return app.test_client()


def test_webhook_auth_failure(test_app):
    resp = test_app.post("/webhook", json={
        "ticker": "AAPL", "action": "buy", "price": 200.0, "passphrase": "wrong_password",
    })
    assert resp.status_code == 401
    assert resp.get_json()["status"] == "unauthorized"


def test_webhook_e2e_successful_trade(test_app, mock_bridge):
    mock_order_resp = MagicMock()
    mock_order_resp.status_code = 201
    mock_order_resp.headers = {"X-Request-ID": "alpaca_e2e_success_7788"}
    body = {"id": "alp_ord_999", "status": "accepted", "filled_qty": "0"}
    mock_order_resp.text = json.dumps(body)
    mock_order_resp.json.return_value = body

    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000", "buying_power": "200000"}), \
         patch.object(mock_bridge, "get_position", return_value=None), \
         patch.object(mock_bridge.session, "request", return_value=mock_order_resp):
        resp = test_app.post("/webhook", json={
            "passphrase": "test_secret_passphrase_123",
            "ticker": "SPY",
            "action": "buy",
            "price": 500.0,
            "signal_strength": 1.0,
        })
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["status"] == "success"
        assert data["x_request_id"] == "alpaca_e2e_success_7788"
        assert data["alpaca_order_id"] == "alp_ord_999"


def test_health_and_audit_endpoints(test_app, mock_bridge):
    mock_clock = MagicMock()
    mock_clock.status_code = 200
    mock_clock.headers = {"X-Request-ID": "clock_req_001"}
    mock_clock.text = json.dumps({"is_open": True})
    mock_clock.json.return_value = {"is_open": True}

    with patch.object(mock_bridge.session, "request", return_value=mock_clock):
        data = test_app.get("/health").get_json()
        assert data["status"] == "healthy"
        assert data["alpaca_connectivity"] == "connected"
        assert data["x_request_id"] == "clock_req_001"

    audit = test_app.get("/audit").get_json()
    assert audit["status"] == "success"
    assert audit["count"] >= 1


def test_evaluator_rebalance_with_existing_position(mock_evaluator, mock_bridge):
    # hold 30 @ $200 (6%); aim 20%; gamma 0.5 -> target 13% = 65 shares; buy 35
    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000", "buying_power": "200000"}), \
         patch.object(mock_bridge, "get_position", return_value={"qty": "30", "current_price": "200.0", "market_value": "6000.0"}):
        decision = mock_evaluator.evaluate_signal({
            "ticker": "AAPL", "action": "buy", "price": 200.0, "signal_strength": 1.0,
        })
        assert decision.approved is True
        assert decision.final_directive.qty == 35.0


def test_order_types_validation(mock_bridge):
    res = mock_bridge.submit_order(TradeDirective(symbol="AAPL", side="buy", qty=10.0, order_type="limit"))
    assert res.success is False and "limit_price" in str(res.error_message)

    res2 = mock_bridge.submit_order(TradeDirective(symbol="AAPL", side="buy", qty=10.0, order_type="stop"))
    assert res2.success is False and "stop_price" in str(res2.error_message)


def test_webhook_positions_and_orders_endpoints(test_app, mock_bridge):
    with patch.object(mock_bridge, "get_positions", return_value=[{"symbol": "SPY", "qty": "50"}]):
        assert test_app.get("/positions").get_json()["positions"][0]["symbol"] == "SPY"

    assert test_app.get("/orders").status_code == 200
    assert test_app.get("/decisions").status_code == 200

    with patch.object(mock_bridge, "cancel_all_orders", return_value=(True, "cancel_req_999")):
        resp = test_app.post("/cancel_all", json={"passphrase": "test_secret_passphrase_123"})
        assert resp.status_code == 200
        assert resp.get_json()["x_request_id"] == "cancel_req_999"
