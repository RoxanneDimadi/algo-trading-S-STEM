"""TradingView Webhook Listener and API Server.

Lightweight HTTP service that ingests trading signals from TradingView
webhooks, authenticates payloads, coordinates with the cost-aware agent policy,
and executes approved directives through the Alpaca execution bridge.
"""
from __future__ import annotations

import hmac
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, request

from src.execution.agent_evaluator import (
    AgentDecision,
    AgentPolicyConfig,
    CostAwareAgentEvaluator,
)
from src.execution.alpaca_bridge import AlpacaConfig, AlpacaExecutionBridge
from src.execution.ledger import ExecutionLedger

logger = logging.getLogger("execution.webhook_listener")


def create_webhook_app(
    bridge: Optional[AlpacaExecutionBridge] = None,
    evaluator: Optional[CostAwareAgentEvaluator] = None,
    webhook_passphrase: Optional[str] = None,
    ledger: Optional[ExecutionLedger] = None,
) -> Flask:
    """Factory creating the Flask webhook receiver and agent management server."""
    app = Flask(__name__)

    # Default components if not explicitly provided
    app_ledger = ledger or (bridge.ledger if bridge else ExecutionLedger())
    app_bridge = bridge or AlpacaExecutionBridge(ledger=app_ledger)
    app_evaluator = evaluator or CostAwareAgentEvaluator(bridge=app_bridge, ledger=app_ledger)
    passphrase = webhook_passphrase

    def _verify_auth(req: request, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify authentication via header token or payload passphrase."""
        if not passphrase:
            return True, "No passphrase configured (public mode)"

        # Check payload passphrase field
        payload_token = data.get("passphrase") or data.get("secret") or data.get("token")
        if payload_token and hmac.compare_digest(str(payload_token), passphrase):
            return True, "Authenticated via payload passphrase"

        # Check HTTP headers
        header_secret = (
            req.headers.get("X-Webhook-Secret")
            or req.headers.get("X-Passphrase")
            or req.headers.get("Authorization", "").replace("Bearer ", "").strip()
        )
        if header_secret and hmac.compare_digest(header_secret, passphrase):
            return True, "Authenticated via HTTP header"

        return False, "Invalid or missing webhook passphrase/secret"

    # -------------------------------------------------------------------------
    # Webhook Signal Ingestion Endpoint
    # -------------------------------------------------------------------------

    @app.route("/webhook", methods=["POST"])
    @app.route("/api/v1/webhook", methods=["POST"])
    def handle_webhook():
        """Receive TradingView alert webhook, evaluate with agent, and route to Alpaca."""
        # 1. Parse JSON payload
        try:
            raw_data = request.get_json(force=True, silent=True)
            if not raw_data:
                # Handle raw text body containing json
                raw_text = request.data.decode("utf-8")
                raw_data = json.loads(raw_text)
        except Exception as e:
            logger.error("Failed to parse incoming webhook payload: %s", e)
            return jsonify({
                "status": "error",
                "error": f"Invalid JSON payload: {str(e)}"
            }), 400

        if not isinstance(raw_data, dict):
            return jsonify({
                "status": "error",
                "error": "Payload must be a JSON object"
            }), 400

        # 2. Authentication check
        is_auth, auth_msg = _verify_auth(request, raw_data)
        if not is_auth:
            logger.warning("Unauthorized webhook request: %s", auth_msg)
            return jsonify({
                "status": "unauthorized",
                "error": auth_msg
            }), 401

        logger.info("Received valid webhook signal for symbol: %s",
                    raw_data.get("ticker") or raw_data.get("symbol"))

        # 3. Evaluate signal with cost-aware agent policy
        try:
            decision: AgentDecision = app_evaluator.evaluate_signal(raw_data)
        except Exception as e:
            logger.exception("Unexpected error in agent signal evaluation: %s", e)
            return jsonify({
                "status": "error",
                "error": f"Agent evaluation failed: {str(e)}"
            }), 500

        # 4. Handle Agent Decision
        if not decision.approved:
            status_code = 200 if "SKIPPED" in decision.reason else 422
            return jsonify({
                "status": "skipped" if "SKIPPED" in decision.reason else "rejected",
                "decision": "skipped" if "SKIPPED" in decision.reason else "rejected",
                "reason": decision.reason,
                "symbol": decision.symbol,
                "metrics": decision.metrics,
                "action_taken": "none",
            }), status_code

        # 5. Route approved trade directive to Alpaca Execution Bridge
        directive = decision.final_directive
        if not directive:
            return jsonify({
                "status": "error",
                "error": "Agent approved trade but produced no directive"
            }), 500

        exec_result = app_bridge.submit_order(directive)

        response_payload = {
            "status": "success" if exec_result.success else "failed",
            "decision": "approved",
            "action_taken": "order_submitted",
            "symbol": exec_result.symbol,
            "side": exec_result.side,
            "qty": exec_result.qty,
            "order_type": exec_result.order_type,
            "order_status": exec_result.status,
            "client_order_id": exec_result.client_order_id,
            "alpaca_order_id": exec_result.alpaca_order_id,
            "x_request_id": exec_result.x_request_id,
            "filled_qty": exec_result.filled_qty,
            "filled_avg_price": exec_result.filled_avg_price,
            "reason": decision.reason,
            "error_message": exec_result.error_message,
            "agent_metrics": decision.metrics,
        }

        http_code = 200 if exec_result.success else 502
        return jsonify(response_payload), http_code

    # -------------------------------------------------------------------------
    # Health & Diagnostics Endpoints
    # -------------------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    @app.route("/api/v1/health", methods=["GET"])
    def health_check():
        """Health check verifying listener state and Alpaca connectivity."""
        alpaca_status = "connected"
        alpaca_error = None
        market_clock = None
        x_request_id = None

        try:
            status_code, clock_data, x_req, err = app_bridge._request("GET", "/v2/clock")
            x_request_id = x_req
            if status_code == 200:
                market_clock = clock_data
            else:
                alpaca_status = "error"
                alpaca_error = err
        except Exception as e:
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
        """Get live Alpaca account balance, buying power, and paper trading state."""
        try:
            account = app_bridge.get_account()
            return jsonify({
                "status": "success",
                "paper": app_bridge.config.paper,
                "account": account,
            }), 200
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/positions", methods=["GET"])
    @app.route("/api/v1/positions", methods=["GET"])
    def get_positions():
        """Get all currently held Alpaca positions."""
        try:
            positions = app_bridge.get_positions()
            return jsonify({
                "status": "success",
                "count": len(positions),
                "positions": positions,
            }), 200
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/orders", methods=["GET"])
    @app.route("/api/v1/orders", methods=["GET"])
    def get_orders():
        """Get recent orders from the execution ledger."""
        limit = int(request.args.get("limit", 50))
        orders = app_ledger.get_recent_orders(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(orders),
            "orders": orders,
        }), 200

    @app.route("/audit", methods=["GET"])
    @app.route("/api/v1/audit", methods=["GET"])
    def get_audit_trail():
        """Get recent API calls and their Alpaca X-Request-IDs."""
        limit = int(request.args.get("limit", 50))
        audits = app_ledger.get_recent_api_audits(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(audits),
            "audit_trail": audits,
        }), 200

    @app.route("/decisions", methods=["GET"])
    @app.route("/api/v1/decisions", methods=["GET"])
    def get_agent_decisions():
        """Get recent cost-aware agent decisions."""
        limit = int(request.args.get("limit", 50))
        decisions = app_ledger.get_recent_decisions(limit=limit)
        return jsonify({
            "status": "success",
            "count": len(decisions),
            "decisions": decisions,
        }), 200

    @app.route("/cancel_all", methods=["POST"])
    @app.route("/api/v1/cancel_all", methods=["POST"])
    def cancel_all_orders():
        """Emergency endpoint to cancel all open orders."""
        is_auth, auth_msg = _verify_auth(request, request.get_json(force=True, silent=True) or {})
        if not is_auth:
            return jsonify({"status": "unauthorized", "error": auth_msg}), 401

        success, x_req = app_bridge.cancel_all_orders()
        return jsonify({
            "status": "success" if success else "failed",
            "action": "cancel_all_orders",
            "x_request_id": x_req,
        }), 200 if success else 500

    return app
