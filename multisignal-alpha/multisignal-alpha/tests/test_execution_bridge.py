"""Unit and integration tests for Alpaca Execution Bridge and Webhook Listener."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.execution.agent_evaluator import (
    AgentDecision,
    AgentPolicyConfig,
    CostAwareAgentEvaluator,
)
from src.execution.alpaca_bridge import (
    AlpacaConfig,
    AlpacaExecutionBridge,
    ExecutionResult,
)
from src.execution.ledger import (
    ApiAuditEntry,
    ExecutionLedger,
    OrderLedgerEntry,
    TradeDirective,
)
from src.execution.webhook_listener import create_webhook_app


@pytest.fixture
def mem_ledger():
    """In-memory SQLite ledger fixture."""
    return ExecutionLedger(db_path=":memory:", jsonl_path=None)


@pytest.fixture
def mock_bridge(mem_ledger):
    """Bridge configured for paper trading with in-memory ledger."""
    cfg = AlpacaConfig(
        api_key="TEST_KEY_ID",
        secret_key="TEST_SECRET_KEY",
        paper=True,
    )
    return AlpacaExecutionBridge(config=cfg, ledger=mem_ledger)


@pytest.fixture
def mock_evaluator(mock_bridge, mem_ledger):
    """Agent evaluator with predictable test parameters."""
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


# =============================================================================
# 1. Alpaca Bridge Configuration & Base URL Tests
# =============================================================================

def test_bridge_paper_and_live_url_switching():
    """Verify clean URL swapping between paper and live trading environments."""
    paper_cfg = AlpacaConfig(paper=True)
    assert paper_cfg.get_effective_base_url() == "https://paper-api.alpaca.markets"

    live_cfg = AlpacaConfig(paper=False)
    assert live_cfg.get_effective_base_url() == "https://api.alpaca.markets"

    custom_cfg = AlpacaConfig(base_url="https://mock-alpaca.internal:8080/")
    assert custom_cfg.get_effective_base_url() == "https://mock-alpaca.internal:8080"


# =============================================================================
# 2. X-Request-ID Extraction & Audit Logging
# =============================================================================

def test_bridge_extracts_x_request_id_on_success(mock_bridge, mem_ledger):
    """Verify X-Request-ID is extracted from response headers and logged."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"X-Request-ID": "alpaca_req_9988776655"}
    mock_resp.text = json.dumps({"id": "ord_123", "status": "accepted", "symbol": "AAPL", "filled_qty": "0"})
    mock_resp.json.return_value = {"id": "ord_123", "status": "accepted", "symbol": "AAPL", "filled_qty": "0"}

    with patch.object(mock_bridge.session, "request", return_value=mock_resp):
        directive = TradeDirective(symbol="AAPL", side="buy", qty=15.0, order_type="market")
        result = mock_bridge.submit_order(directive)

        assert result.success is True
        assert result.x_request_id == "alpaca_req_9988776655"
        assert result.alpaca_order_id == "ord_123"
        assert result.status == "accepted"

        # Check ledger audit trail
        audits = mem_ledger.get_recent_api_audits(limit=10)
        assert len(audits) >= 1
        assert audits[0]["x_request_id"] == "alpaca_req_9988776655"
        assert audits[0]["status_code"] == 200
        assert audits[0]["endpoint"] == "/v2/orders"

        # Check order ledger
        orders = mem_ledger.get_recent_orders(limit=10)
        assert len(orders) >= 1
        assert orders[0]["x_request_id"] == "alpaca_req_9988776655"
        assert orders[0]["symbol"] == "AAPL"


def test_bridge_extracts_x_request_id_on_error(mock_bridge, mem_ledger):
    """Verify X-Request-ID is captured and logged even when Alpaca returns 403 or 422."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.headers = {"X-Request-ID": "err_req_403_insufficient_bp"}
    mock_resp.text = json.dumps({"message": "insufficient buying power", "code": 40310000})
    mock_resp.json.return_value = {"message": "insufficient buying power", "code": 40310000}

    with patch.object(mock_bridge.session, "request", return_value=mock_resp):
        directive = TradeDirective(symbol="MSFT", side="buy", qty=500.0, order_type="market")
        result = mock_bridge.submit_order(directive)

        assert result.success is False
        assert result.status_code == 403
        assert result.x_request_id == "err_req_403_insufficient_bp"
        assert "insufficient buying power" in (result.error_message or "")

        # Verify audit ledger has exact failure record with X-Request-ID
        audit_by_id = mem_ledger.get_by_x_request_id("err_req_403_insufficient_bp")
        assert audit_by_id is not None
        assert audit_by_id["status_code"] == 403
        assert audit_by_id["success"] == 0


def test_bridge_handles_network_timeout(mock_bridge, mem_ledger):
    """Verify graceful handling when broker connection times out."""
    with patch.object(mock_bridge.session, "request", side_effect=requests.exceptions.Timeout("Connection timed out")):
        directive = TradeDirective(symbol="NVDA", side="buy", qty=10.0)
        result = mock_bridge.submit_order(directive)

        assert result.success is False
        assert "Connection timed out" in str(result.error_message)

        audits = mem_ledger.get_recent_api_audits(limit=10)
        assert len(audits) >= 1
        assert audits[0]["success"] == 0
        assert "Connection timed out" in audits[0]["error_message"]


# =============================================================================
# 3. Cost-Aware Agent Decision Evaluator Tests
# =============================================================================

def test_evaluator_gp_partial_adjustment_sizing(mock_evaluator, mock_bridge):
    """Verify Gârleanu-Pedersen partial adjustment (gamma speed) calculates correct delta."""
    # Mock account with $100,000 equity and zero current position
    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000", "buying_power": "200000", "cash": "100000"}), \
         patch.object(mock_bridge, "get_position", return_value=None):

        # Signal for AAPL at $200. Max position 20% = $20,000 (100 shares aim).
        # Gamma = 0.50 -> partial adjustment target = 50 shares ($10,000).
        raw_signal = {
            "ticker": "AAPL",
            "action": "buy",
            "price": 200.0,
            "signal_strength": 1.0,
        }
        decision = mock_evaluator.evaluate_signal(raw_signal)

        assert decision.approved is True
        assert decision.final_directive is not None
        assert decision.final_directive.symbol == "AAPL"
        assert decision.final_directive.side == "buy"
        assert decision.final_directive.qty == 50.0  # 100 aim * 0.50 gamma = 50 shares


def test_evaluator_cost_hurdle_filters_weak_noise(mock_evaluator, mock_bridge):
    """Verify that a noisy/marginal signal is rejected by the cost-aware hurdle."""
    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000", "buying_power": "200000"}), \
         patch.object(mock_bridge, "get_position", return_value=None):

        # Extremely weak signal (0.05 strength).
        # Gross alpha = 0.05 * 40 bps = 2.0 bps.
        # Cost = 10 bps linear + impact > 10 bps. Net benefit < 0 -> below 5 bps hurdle.
        raw_signal = {
            "ticker": "GOOGL",
            "action": "buy",
            "price": 150.0,
            "signal_strength": 0.05,
        }
        decision = mock_evaluator.evaluate_signal(raw_signal)

        assert decision.approved is False
        assert "SKIPPED" in decision.reason
        assert decision.final_directive is None


def test_evaluator_short_selling_protection(mock_evaluator, mock_bridge):
    """Verify short selling is blocked when allow_short is False and no shares are held."""
    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000"}), \
         patch.object(mock_bridge, "get_position", return_value=None):

        raw_signal = {
            "ticker": "TSLA",
            "action": "sell",
            "price": 250.0,
            "signal_strength": -1.0,
        }
        decision = mock_evaluator.evaluate_signal(raw_signal)

        assert decision.approved is False
        assert "Short selling is disabled" in decision.reason


# =============================================================================
# 4. Webhook Listener Flask API & End-to-End Tests
# =============================================================================

@pytest.fixture
def test_app(mock_bridge, mock_evaluator, mem_ledger):
    """Flask test client fixture."""
    app = create_webhook_app(
        bridge=mock_bridge,
        evaluator=mock_evaluator,
        webhook_passphrase="test_secret_passphrase_123",
        ledger=mem_ledger,
    )
    app.config["TESTING"] = True
    return app.test_client()


def test_webhook_auth_failure(test_app):
    """Verify unauthorized requests are rejected with HTTP 401."""
    payload = {"ticker": "AAPL", "action": "buy", "price": 200.0, "passphrase": "wrong_password"}
    resp = test_app.post("/webhook", json=payload)
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["status"] == "unauthorized"


def test_webhook_e2e_successful_trade(test_app, mock_bridge):
    """Verify complete end-to-end webhook flow: Alert -> Evaluator -> Bridge -> Order & X-Request-ID."""
    mock_order_resp = MagicMock()
    mock_order_resp.status_code = 201
    mock_order_resp.headers = {"X-Request-ID": "alpaca_e2e_success_7788"}
    mock_order_resp.text = json.dumps({
        "id": "alp_ord_999",
        "client_order_id": "test_order_1",
        "symbol": "SPY",
        "status": "accepted",
        "filled_qty": "0",
    })
    mock_order_resp.json.return_value = {
        "id": "alp_ord_999",
        "client_order_id": "test_order_1",
        "symbol": "SPY",
        "status": "accepted",
        "filled_qty": "0",
    }

    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000", "buying_power": "200000"}), \
         patch.object(mock_bridge, "get_position", return_value=None), \
         patch.object(mock_bridge.session, "request", return_value=mock_order_resp):

        payload = {
            "passphrase": "test_secret_passphrase_123",
            "ticker": "SPY",
            "action": "buy",
            "price": 500.0,
            "signal_strength": 1.0,
            "strategy": "TradingView_Momentum_Breakout",
        }
        resp = test_app.post("/webhook", json=payload)

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["decision"] == "approved"
        assert data["action_taken"] == "order_submitted"
        assert data["symbol"] == "SPY"
        assert data["x_request_id"] == "alpaca_e2e_success_7788"
        assert data["alpaca_order_id"] == "alp_ord_999"


def test_health_and_audit_endpoints(test_app, mock_bridge):
    """Verify /health and /audit diagnostic endpoints work."""
    mock_clock_resp = MagicMock()
    mock_clock_resp.status_code = 200
    mock_clock_resp.headers = {"X-Request-ID": "clock_req_001"}
    mock_clock_resp.text = json.dumps({"is_open": True, "next_open": "2026-08-31T09:30:00-04:00"})
    mock_clock_resp.json.return_value = {"is_open": True, "next_open": "2026-08-31T09:30:00-04:00"}

    with patch.object(mock_bridge.session, "request", return_value=mock_clock_resp):
        resp = test_app.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["paper_mode"] is True
        assert data["alpaca_connectivity"] == "connected"
        assert data["x_request_id"] == "clock_req_001"

    # Verify audit endpoint returns ledger entries
    audit_resp = test_app.get("/audit")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.get_json()
    assert audit_data["status"] == "success"
    assert audit_data["count"] >= 1


def test_evaluator_rebalance_with_existing_position(mock_evaluator, mock_bridge):
    """Verify evaluator computes incremental delta when already holding shares."""
    # Already holding 30 shares of AAPL at $200 ($6,000 / 6% weight)
    # Target aim = 20% ($20,000 -> 100 shares).
    # Partial adjustment target: 0.50 * 0.06 + 0.50 * 0.20 = 0.13 (13% = 65 shares).
    # Delta to buy = 65 - 30 = 35 shares.
    with patch.object(mock_bridge, "get_account", return_value={"portfolio_value": "100000", "buying_power": "200000"}), \
         patch.object(mock_bridge, "get_position", return_value={"qty": "30", "current_price": "200.0", "market_value": "6000.0"}):

        raw_signal = {
            "ticker": "AAPL",
            "action": "buy",
            "price": 200.0,
            "signal_strength": 1.0,
        }
        decision = mock_evaluator.evaluate_signal(raw_signal)

        assert decision.approved is True
        assert decision.final_directive.qty == 35.0
        assert decision.final_directive.side == "buy"


def test_order_types_validation(mock_bridge):
    """Verify limit and stop orders enforce required price arguments."""
    limit_directive_no_price = TradeDirective(symbol="AAPL", side="buy", qty=10.0, order_type="limit")
    res = mock_bridge.submit_order(limit_directive_no_price)
    assert res.success is False
    assert "requires a positive limit_price" in str(res.error_message)

    stop_directive_no_price = TradeDirective(symbol="AAPL", side="buy", qty=10.0, order_type="stop")
    res2 = mock_bridge.submit_order(stop_directive_no_price)
    assert res2.success is False
    assert "requires a positive stop_price" in str(res2.error_message)


def test_webhook_positions_and_orders_endpoints(test_app, mock_bridge):
    """Verify /positions, /orders, /decisions, and /cancel_all endpoints."""
    # 1. /positions
    with patch.object(mock_bridge, "get_positions", return_value=[{"symbol": "SPY", "qty": "50", "market_value": "25000"}]):
        pos_resp = test_app.get("/positions")
        assert pos_resp.status_code == 200
        pos_data = pos_resp.get_json()
        assert pos_data["count"] == 1
        assert pos_data["positions"][0]["symbol"] == "SPY"

    # 2. /orders
    orders_resp = test_app.get("/orders")
    assert orders_resp.status_code == 200

    # 3. /decisions
    decisions_resp = test_app.get("/decisions")
    assert decisions_resp.status_code == 200

    # 4. /cancel_all
    with patch.object(mock_bridge, "cancel_all_orders", return_value=(True, "cancel_req_999")):
        cancel_resp = test_app.post(
            "/cancel_all",
            json={"passphrase": "test_secret_passphrase_123"},
            headers={"Content-Type": "application/json"}
        )
        assert cancel_resp.status_code == 200
        assert cancel_resp.get_json()["status"] == "success"
        assert cancel_resp.get_json()["x_request_id"] == "cancel_req_999"

