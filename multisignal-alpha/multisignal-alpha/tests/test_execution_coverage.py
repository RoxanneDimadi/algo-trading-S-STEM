"""Extra cases for account helpers, sizing edge paths, and webhook quirks."""
# Requesting a pytest fixture shadows the fixture function's name by
# design -- that is how pytest injects it.
# pylint: disable=redefined-outer-name
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.execution.agent_evaluator import (AgentPolicyConfig,
                                           CostAwareAgentEvaluator)
from src.execution.alpaca_bridge import AlpacaConfig, AlpacaExecutionBridge
from src.execution.ledger import (
    AgentDecisionEntry,
    ApiAuditEntry,
    ExecutionLedger,
    OrderLedgerEntry,
    TradeDirective,
)
from src.execution.webhook_listener import create_webhook_app


@pytest.fixture
def mem_ledger():
    return ExecutionLedger(db_path=":memory:", jsonl_path=None)


@pytest.fixture
def mock_bridge(mem_ledger):
    cfg = AlpacaConfig(api_key="K", secret_key="S", paper=True)
    return AlpacaExecutionBridge(config=cfg, ledger=mem_ledger)


@pytest.fixture
def evaluator(mock_bridge, mem_ledger):
    cfg = AgentPolicyConfig(
        cost_bps=5.0,
        gamma_speed=1.0,  # full rebalance for simpler sizing asserts
        impact_bps=0.0,
        alpha_hurdle_bps=0.0,
        max_position_pct=0.20,
        max_order_notional=50000.0,
        min_trade_notional=50.0,
        allow_short=False,
        default_base_alpha_bps=50.0,
    )
    return CostAwareAgentEvaluator(bridge=mock_bridge, config=cfg,
                                   ledger=mem_ledger)


def _ok_resp(payload, x_request_id="req_ok", status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"X-Request-ID": x_request_id}
    resp.text = json.dumps(payload)
    resp.json.return_value = payload
    return resp


def test_get_account_success_and_failure(mock_bridge):
    with patch.object(mock_bridge.session, "request",
                      return_value=_ok_resp({"equity": "100000"})):
        account = mock_bridge.get_account()
        assert account["equity"] == "100000"

    bad = _ok_resp({"message": "unauthorized"},
                   x_request_id="req_fail", status=401)
    with patch.object(mock_bridge.session, "request", return_value=bad):
        with pytest.raises(RuntimeError, match="account fetch failed"):
            mock_bridge.get_account()


def test_get_positions_list_and_dict_and_error(mock_bridge):
    with patch.object(mock_bridge.session, "request",
                      return_value=_ok_resp([{"symbol": "AAPL"}])):
        assert mock_bridge.get_positions()[0]["symbol"] == "AAPL"

    with patch.object(mock_bridge.session, "request",
                      return_value=_ok_resp({"symbol": "SPY"})):
        assert mock_bridge.get_positions()[0]["symbol"] == "SPY"

    bad = _ok_resp({"message": "boom"}, status=500)
    with patch.object(mock_bridge.session, "request", return_value=bad):
        with pytest.raises(RuntimeError, match="positions fetch failed"):
            mock_bridge.get_positions()


def test_get_position_found_missing_and_error(mock_bridge):
    with patch.object(mock_bridge.session, "request",
                      return_value=_ok_resp({"qty": "10"})):
        assert mock_bridge.get_position("aapl")["qty"] == "10"

    missing = _ok_resp({"message": "not found"}, status=404)
    with patch.object(mock_bridge.session, "request", return_value=missing):
        assert mock_bridge.get_position("ZZZ") is None

    bad = _ok_resp({"message": "server"}, status=500)
    with patch.object(mock_bridge.session, "request", return_value=bad):
        with pytest.raises(RuntimeError, match="position AAPL failed"):
            mock_bridge.get_position("AAPL")


def test_get_clock_success_and_failure(mock_bridge):
    with patch.object(mock_bridge.session, "request",
                      return_value=_ok_resp({"is_open": True})):
        assert mock_bridge.get_clock()["is_open"] is True

    bad = _ok_resp({"message": "down"}, status=503)
    with patch.object(mock_bridge.session, "request", return_value=bad):
        with pytest.raises(RuntimeError, match="clock fetch failed"):
            mock_bridge.get_clock()


def test_submit_notional_order_and_invalid_qty(mock_bridge):
    ok = _ok_resp({"id": "n1", "status": "accepted",
                  "filled_qty": "0"}, status=201)
    with patch.object(mock_bridge.session, "request", return_value=ok):
        directive = TradeDirective(
            symbol="AAPL", side="buy", qty=0.0, notional=1500.0)
        result = mock_bridge.submit_order(directive)
        assert result.success is True
        assert result.alpaca_order_id == "n1"

    bad = TradeDirective(symbol="AAPL", side="buy", qty=0.0, notional=None)
    result2 = mock_bridge.submit_order(bad)
    assert result2.success is False
    assert "qty or notional" in result2.error_message


def test_submit_limit_and_stop_limit_orders(mock_bridge):
    ok = _ok_resp({"id": "L1", "status": "new", "filled_qty": "0"}, status=200)
    with patch.object(mock_bridge.session, "request",
                      return_value=ok) as mock_req:
        d = TradeDirective(symbol="MSFT", side="buy", qty=5,
                           order_type="limit", limit_price=400.0)
        assert mock_bridge.submit_order(d).success is True
        payload = mock_req.call_args.kwargs["json"]
        assert payload["limit_price"] == "400.0"

    with patch.object(mock_bridge.session, "request",
                      return_value=ok) as mock_req:
        d2 = TradeDirective(
            symbol="MSFT", side="sell", qty=5, order_type="stop_limit",
            limit_price=390.0, stop_price=395.0,
        )
        assert mock_bridge.submit_order(d2).success is True
        payload = mock_req.call_args.kwargs["json"]
        assert payload["stop_price"] == "395.0"
        assert payload["limit_price"] == "390.0"

    bad_sl = TradeDirective(symbol="MSFT", side="buy",
                            qty=1, order_type="stop_limit", limit_price=100.0)
    assert "stop_price" in (
        mock_bridge.submit_order(bad_sl).error_message or "")


def test_cancel_and_get_orders(mock_bridge):
    ok = MagicMock()
    ok.status_code = 204
    ok.headers = {"X-Request-ID": "cancel_1"}
    ok.text = ""
    ok.json.return_value = {}
    with patch.object(mock_bridge.session, "request", return_value=ok):
        success, x_req = mock_bridge.cancel_order("ord_1")
        assert success is True
        assert x_req == "cancel_1"

    ok2 = _ok_resp([{"id": "1"}, {"id": "2"}])
    with patch.object(mock_bridge.session, "request", return_value=ok2):
        orders = mock_bridge.get_orders(status="open", limit=10)
        assert len(orders) == 2

    with patch.object(mock_bridge.session, "request",
                      return_value=_ok_resp({"bad": True}, status=500)):
        assert mock_bridge.get_orders() == []

    with patch.object(mock_bridge.session, "request",
                      return_value=_ok_resp([], status=207)):
        success, _ = mock_bridge.cancel_all_orders()
        assert success is True


def test_non_json_response_body_is_audited(mock_bridge):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"x-request-id": "lower_case_id"}
    resp.text = "not-json"
    resp.json.side_effect = ValueError("no json")
    with patch.object(mock_bridge.session, "request", return_value=resp):
        # the test drives the private request path on purpose
        # pylint: disable=protected-access
        status, data, x_req, _err = mock_bridge._request(
            "GET", "/v2/clock")
        assert status == 200
        assert x_req == "lower_case_id"
        assert "raw_text" in data


def test_evaluator_rejects_missing_symbol_and_bad_action(evaluator):
    d1 = evaluator.evaluate_signal({"action": "buy", "price": 10})
    assert d1.approved is False
    assert "Missing ticker" in d1.reason

    d2 = evaluator.evaluate_signal(
        {"ticker": "AAPL", "action": "explode", "price": 10})
    assert d2.approved is False
    assert "Unknown action" in d2.reason


def test_evaluator_account_fallback_and_nonpositive_equity(evaluator,
                                                           mock_bridge):
    with patch.object(mock_bridge, "get_account",
                      side_effect=RuntimeError("down")), \
            patch.object(mock_bridge, "get_position", return_value=None):
        d = evaluator.evaluate_signal(
            {"ticker": "AAPL", "action": "buy", "price": 100.0,
             "signal_strength": 1.0})
        # Fallback portfolio 100k still allows a sized trade
        assert d.approved is True

    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "0"}), \
            patch.object(mock_bridge, "get_position", return_value=None):
        d2 = evaluator.evaluate_signal(
            {"ticker": "AAPL", "action": "buy", "price": 100.0})
        assert "non-positive" in d2.reason


def test_evaluator_missing_price_and_position_query_failure(evaluator,
                                                            mock_bridge):
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "100000"}), \
            patch.object(mock_bridge, "get_position",
                         side_effect=RuntimeError("pos fail")):
        d = evaluator.evaluate_signal({"ticker": "AAPL", "action": "buy"})
        assert "Unable to determine current price" in d.reason


def test_evaluator_close_position_and_notional_aim(evaluator, mock_bridge):
    # Close existing long: aim=0, gamma=1 => sell all
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "100000"}), \
            patch.object(mock_bridge, "get_position",
                         return_value={"qty": "40", "current_price": "50",
                                       "market_value": "2000"}):
        d = evaluator.evaluate_signal(
            {"ticker": "AAPL", "action": "close", "price": 50.0})
        assert d.approved is True
        assert d.final_directive.side == "sell"
        assert d.final_directive.qty == 40.0

    # Explicit notional aim
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "200000"}), \
            patch.object(mock_bridge, "get_position", return_value=None):
        d2 = evaluator.evaluate_signal({
            "ticker": "AAPL", "action": "buy", "price": 100.0,
            "notional": 10000.0, "signal_strength": 1.0,
        })
        assert d2.approved is True
        assert d2.final_directive.qty == 100.0


def test_evaluator_requested_qty_and_buying_power_downsize(evaluator,
                                                           mock_bridge):
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "500"}), \
            patch.object(mock_bridge, "get_position", return_value=None):
        d = evaluator.evaluate_signal({
            "ticker": "AAPL", "action": "buy", "price": 100.0,
            "quantity": 50, "signal_strength": 1.0,
        })
        assert d.approved is True
        # 95% of $500 BP / $100 = 4 shares
        assert d.final_directive.qty == 4.0


def test_evaluator_insufficient_buying_power_rejects(evaluator, mock_bridge):
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "10"}), \
            patch.object(mock_bridge, "get_position", return_value=None):
        d = evaluator.evaluate_signal({
            "ticker": "AAPL", "action": "buy", "price": 100.0,
            "quantity": 20, "signal_strength": 1.0,
        })
        assert d.approved is False
        assert "Insufficient buying power" in d.reason


def test_evaluator_max_notional_downsize(evaluator, mock_bridge):
    # max_order_notional=50000; 1000 shares * $100 would be $100k -> downsize
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "1000000",
                                    "buying_power": "1000000"}), \
            patch.object(mock_bridge, "get_position", return_value=None):
        d = evaluator.evaluate_signal({
            "ticker": "AAPL", "action": "buy", "price": 100.0,
            "quantity": 1000, "signal_strength": 1.0,
        })
        assert d.approved is True
        assert d.final_directive.qty == 500.0  # 50000 / 100


def test_evaluator_sell_caps_to_available_shares(mock_bridge):
    cfg = AgentPolicyConfig(
        cost_bps=0.0, gamma_speed=1.0, impact_bps=0.0, alpha_hurdle_bps=0.0,
        max_position_pct=0.20, allow_short=False, default_base_alpha_bps=50.0,
        min_trade_notional=1.0,
    )
    ev = CostAwareAgentEvaluator(
        bridge=mock_bridge, config=cfg, ledger=mock_bridge.ledger)
    # Hold 10 shares; close aims to 0 => sell 10 (already covered).
    # Here sell toward flat with a qty request.
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "100000"}), \
            patch.object(mock_bridge, "get_position",
                         return_value={"qty": "10", "current_price": "50",
                                       "market_value": "500"}):
        d = ev.evaluate_signal(
            {"ticker": "AAPL", "action": "flat", "price": 50.0})
        assert d.approved is True
        assert d.final_directive.side == "sell"
        assert d.final_directive.qty == 10.0


def test_agent_decision_to_dict_includes_directive(evaluator, mock_bridge):
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "200000"}), \
            patch.object(mock_bridge, "get_position", return_value=None):
        d = evaluator.evaluate_signal(
            {"ticker": "AAPL", "action": "buy", "price": 100.0,
             "signal_strength": 1.0})
        payload = d.to_dict()
        assert payload["approved"] is True
        assert payload["final_directive"]["symbol"] == "AAPL"


def test_ledger_jsonl_and_order_status_update(tmp_path):
    db = tmp_path / "ledger.db"
    jsonl = tmp_path / "audit.jsonl"
    ledger = ExecutionLedger(db_path=str(db), jsonl_path=str(jsonl))

    ledger.record_api_call(ApiAuditEntry(
        method="GET", endpoint="/v2/clock", status_code=200, success=True))
    ledger.record_agent_decision(AgentDecisionEntry(
        symbol="AAPL", approved=False, reason="SKIPPED: dust"))
    ledger.record_order(OrderLedgerEntry(
        client_order_id="c1", symbol="AAPL", side="buy", qty=1.0,
        status="new"))
    ledger.update_order_status(
        "c1", status="filled", filled_qty=1.0, filled_avg_price=10.5)

    assert jsonl.exists()
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3
    orders = ledger.get_recent_orders()
    assert orders[0]["status"] == "filled"
    assert orders[0]["filled_avg_price"] == 10.5
    assert ledger.get_recent_decisions()
    assert ledger.get_by_x_request_id("missing") is None


@pytest.fixture
def test_app(mock_bridge, evaluator, mem_ledger):
    app = create_webhook_app(
        bridge=mock_bridge,
        evaluator=evaluator,
        webhook_passphrase="secret",
        ledger=mem_ledger,
    )
    app.config["TESTING"] = True
    return app.test_client()


def test_webhook_header_auth_and_invalid_json(test_app, mock_bridge):
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "200000"}), \
            patch.object(mock_bridge, "get_position", return_value=None), \
            patch.object(mock_bridge.session, "request",
                         return_value=_ok_resp(
                             {"id": "o1", "status": "accepted",
                              "filled_qty": "0"}, status=201)):
        resp = test_app.post(
            "/api/v1/webhook",
            data=json.dumps({"ticker": "AAPL", "action": "buy",
                            "price": 100.0, "signal_strength": 1.0}),
            headers={"Content-Type": "application/json",
                     "X-Webhook-Secret": "secret"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "success"

    bad = test_app.post("/webhook", data="{{{not json", headers={
                        "Content-Type": "application/json",
                        "X-Webhook-Secret": "secret"})
    assert bad.status_code == 400


def test_webhook_rejected_vs_skipped_status_codes(test_app, mock_bridge):
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "200000"}), \
            patch.object(mock_bridge, "get_position", return_value=None):
        # Rejected (unknown action)
        r1 = test_app.post(
            "/webhook", json={"passphrase": "secret", "ticker": "AAPL",
                              "action": "nope", "price": 10})
        assert r1.status_code == 422
        assert r1.get_json()["decision"] == "rejected"

        # Skipped (weak signal under hurdle) -- raise the hurdle via a
        # temporary evaluator config already at 0;
        # use dust: tiny strength with a high price / tiny quantity
        # will not skip under gamma=1 with a 0 hurdle.
        # Force a skip by dust notional via the existing tiny delta:
        # hold at target already.
        with patch.object(mock_bridge, "get_position",
                          return_value={"qty": "200", "current_price": "100",
                                        "market_value": "20000"}):
            r2 = test_app.post("/webhook", json={
                "passphrase": "secret", "ticker": "AAPL", "action": "buy",
                "price": 100.0, "signal_strength": 1.0,
            })
            # aim 20% = 200 shares, already there => dust/skip
            assert r2.status_code == 200
            assert r2.get_json()["decision"] == "skipped"


def test_webhook_account_endpoint_and_cancel_unauthorized(
        test_app, mock_bridge):
    with patch.object(mock_bridge, "get_account",
                      return_value={"equity": "1"}):
        r = test_app.get("/account")
        assert r.status_code == 200
        assert r.get_json()["status"] == "success"

    with patch.object(mock_bridge, "get_account",
                      side_effect=RuntimeError("nope")):
        r2 = test_app.get("/account")
        assert r2.status_code == 500

    with patch.object(mock_bridge, "get_positions",
                      side_effect=RuntimeError("nope")):
        r3 = test_app.get("/positions")
        assert r3.status_code == 500

    unauthorized = test_app.post("/cancel_all", json={"passphrase": "wrong"})
    assert unauthorized.status_code == 401


def test_public_mode_webhook_without_passphrase(mock_bridge, evaluator,
                                                mem_ledger):
    app = create_webhook_app(
        bridge=mock_bridge,
        evaluator=evaluator,
        webhook_passphrase=None,
        ledger=mem_ledger,
    )
    client = app.test_client()
    with patch.object(mock_bridge, "get_account",
                      return_value={"portfolio_value": "100000",
                                    "buying_power": "200000"}), \
            patch.object(mock_bridge, "get_position", return_value=None), \
            patch.object(mock_bridge.session, "request",
                         return_value=_ok_resp(
                             {"id": "p1", "status": "accepted",
                              "filled_qty": "0"}, status=201)):
        resp = client.post(
            "/webhook", json={"ticker": "AAPL", "action": "buy",
                              "price": 100.0, "signal_strength": 1.0})
        assert resp.status_code == 200


def test_health_reports_alpaca_error_status(
        mock_bridge, evaluator, mem_ledger):
    app = create_webhook_app(bridge=mock_bridge, evaluator=evaluator,
                             webhook_passphrase="secret", ledger=mem_ledger)
    client = app.test_client()
    bad = _ok_resp({"message": "auth failed"},
                   status=401, x_request_id="h_err")
    with patch.object(mock_bridge.session, "request", return_value=bad):
        resp = client.get("/health")
        data = resp.get_json()
        assert data["alpaca_connectivity"] == "error"
        assert data["x_request_id"] == "h_err"
