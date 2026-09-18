"""Flask app: TradingView webhooks in, Alpaca orders out (after agent gate)."""
from __future__ import annotations

import hmac
import json
import logging
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, request

from src.execution.agent_evaluator import (AgentDecision,
                                           CostAwareAgentEvaluator)
from src.execution.alpaca_bridge import AlpacaExecutionBridge
from src.execution.ledger import ExecutionLedger

logger = logging.getLogger("execution.webhook_listener")


def create_webhook_app(
    bridge: Optional[AlpacaExecutionBridge] = None,
    evaluator: Optional[CostAwareAgentEvaluator] = None,
    webhook_passphrase: Optional[str] = None,
    ledger: Optional[ExecutionLedger] = None,
) -> Flask:
    app = Flask(__name__)

    app_ledger = ledger or (bridge.ledger if bridge else ExecutionLedger())
    app_bridge = bridge or AlpacaExecutionBridge(ledger=app_ledger)
    app_evaluator = evaluator or CostAwareAgentEvaluator(
        bridge=app_bridge, ledger=app_ledger)
    passphrase = webhook_passphrase

    def _verify_auth(req, data: Dict[str, Any]) -> Tuple[bool, str]:
        if not passphrase:
            return True, "no passphrase configured"

        token = data.get("passphrase") or data.get(
            "secret") or data.get("token")
        if token and hmac.compare_digest(str(token), passphrase):
            return True, "ok"

        header = (
            req.headers.get("X-Webhook-Secret")
            or req.headers.get("X-Passphrase")
            or req.headers.get("Authorization", "").replace(
                "Bearer ", "").strip()
        )
        if header and hmac.compare_digest(header, passphrase):
            return True, "ok"

        return False, "bad or missing passphrase"

    @app.route("/webhook", methods=["POST"])
    @app.route("/api/v1/webhook", methods=["POST"])
    def handle_webhook():
        try:
            raw_data = request.get_json(force=True, silent=True)
            if not raw_data:
                raw_data = json.loads(request.data.decode("utf-8"))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("bad webhook body: %s", e)
            return jsonify({"status": "error",
                            "error": f"Invalid JSON payload: {e}"}), 400

        if not isinstance(raw_data, dict):
            return jsonify({"status": "error",
                            "error": "Payload must be a JSON object"}), 400

        ok, auth_msg = _verify_auth(request, raw_data)
        if not ok:
            logger.warning("unauthorized webhook: %s", auth_msg)
            return jsonify({"status": "unauthorized", "error": auth_msg}), 401

        logger.info("webhook for %s", raw_data.get(
            "ticker") or raw_data.get("symbol"))

        try:
            decision: AgentDecision = app_evaluator.evaluate_signal(raw_data)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.exception("evaluate_signal failed: %s", e)
            return jsonify({"status": "error",
                            "error": f"Agent evaluation failed: {e}"}), 500

        if not decision.approved:
            code = 200 if "SKIPPED" in decision.reason else 422
            kind = "skipped" if "SKIPPED" in decision.reason else "rejected"
            return jsonify({
                "status": kind,
                "decision": kind,
                "reason": decision.reason,
                "symbol": decision.symbol,
                "metrics": decision.metrics,
                "action_taken": "none",
            }), code

        directive = decision.final_directive
        if not directive:
            return jsonify({
                "status": "error",
                "error": "approved but no directive",
            }), 500

        result = app_bridge.submit_order(directive)
        return jsonify({
            "status": "success" if result.success else "failed",
            "decision": "approved",
            "action_taken": "order_submitted",
            "symbol": result.symbol,
            "side": result.side,
            "qty": result.qty,
            "order_type": result.order_type,
            "order_status": result.status,
            "client_order_id": result.client_order_id,
            "alpaca_order_id": result.alpaca_order_id,
            "x_request_id": result.x_request_id,
            "filled_qty": result.filled_qty,
            "filled_avg_price": result.filled_avg_price,
            "reason": decision.reason,
            "error_message": result.error_message,
            "agent_metrics": decision.metrics,
        }), (200 if result.success else 502)

    @app.route("/health", methods=["GET"])
    @app.route("/api/v1/health", methods=["GET"])
    def health_check():
        alpaca_status = "connected"
        alpaca_error = None
        market_clock = None
        x_request_id = None
        try:
            # the listener owns this bridge; the audited request path
            # is deliberately reused here
            # pylint: disable=protected-access
            status_code, clock_data, x_req, err = app_bridge._request(
                "GET", "/v2/clock")
            x_request_id = x_req
            if status_code == 200:
                market_clock = clock_data
            else:
                alpaca_status = "error"
                alpaca_error = err
        except Exception as e:  # pylint: disable=broad-exception-caught
            alpaca_status = "unreachable"
            alpaca_error = str(e)

        return jsonify({
            "service": "execution_bridge_webhook_listener",
            "status": "healthy",
            "paper_mode": app_bridge.config.paper,
            "base_url": app_bridge.config.get_effective_base_url(),
            "alpaca_connectivity": alpaca_status,
            "alpaca_error": alpaca_error,
            "x_request_id": x_request_id,
            "market_clock": market_clock,
            "auth_configured": bool(passphrase),
        }), 200

    @app.route("/account", methods=["GET"])
    @app.route("/api/v1/account", methods=["GET"])
    def get_account_status():
        try:
            account = app_bridge.get_account()
            return jsonify({"status": "success",
                            "paper": app_bridge.config.paper,
                            "account": account}), 200
        except Exception as e:  # pylint: disable=broad-exception-caught
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/positions", methods=["GET"])
    @app.route("/api/v1/positions", methods=["GET"])
    def get_positions():
        try:
            positions = app_bridge.get_positions()
            return jsonify({"status": "success", "count": len(positions),
                            "positions": positions}), 200
        except Exception as e:  # pylint: disable=broad-exception-caught
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/orders", methods=["GET"])
    @app.route("/api/v1/orders", methods=["GET"])
    def get_orders():
        limit = int(request.args.get("limit", 50))
        orders = app_ledger.get_recent_orders(limit=limit)
        return jsonify({"status": "success", "count": len(orders),
                        "orders": orders}), 200

    @app.route("/audit", methods=["GET"])
    @app.route("/api/v1/audit", methods=["GET"])
    def get_audit_trail():
        limit = int(request.args.get("limit", 50))
        audits = app_ledger.get_recent_api_audits(limit=limit)
        return jsonify({"status": "success", "count": len(audits),
                        "audit_trail": audits}), 200

    @app.route("/decisions", methods=["GET"])
    @app.route("/api/v1/decisions", methods=["GET"])
    def get_agent_decisions():
        limit = int(request.args.get("limit", 50))
        decisions = app_ledger.get_recent_decisions(limit=limit)
        return jsonify({"status": "success", "count": len(decisions),
                        "decisions": decisions}), 200

    @app.route("/cancel_all", methods=["POST"])
    @app.route("/api/v1/cancel_all", methods=["POST"])
    def cancel_all_orders():
        ok, auth_msg = _verify_auth(
            request, request.get_json(force=True, silent=True) or {})
        if not ok:
            return jsonify({"status": "unauthorized", "error": auth_msg}), 401
        success, x_req = app_bridge.cancel_all_orders()
        return jsonify({
            "status": "success" if success else "failed",
            "action": "cancel_all_orders",
            "x_request_id": x_req,
        }), (200 if success else 500)

    return app
