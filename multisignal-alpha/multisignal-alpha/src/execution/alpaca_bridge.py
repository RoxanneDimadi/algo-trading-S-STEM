"""Alpaca Execution Bridge.

Dedicated broker communication layer that translates validated trade
directives into Alpaca REST API orders, manages paper/live environments,
and extracts and logs X-Request-ID headers for all requests.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.execution.ledger import (
    ApiAuditEntry,
    ExecutionLedger,
    OrderLedgerEntry,
    TradeDirective,
)

logger = logging.getLogger("execution.alpaca_bridge")


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca API connectivity."""
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True
    base_url: Optional[str] = None
    timeout_seconds: float = 10.0
    max_retries: int = 2
    retry_delay_seconds: float = 0.5

    def get_effective_base_url(self) -> str:
        """Return the base URL based on paper toggle or explicit override."""
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"


@dataclass
class ExecutionResult:
    """Outcome of an order execution request sent to Alpaca."""
    success: bool
    status: str                        # "submitted", "filled", "rejected", "failed", "simulated"
    symbol: str
    side: str
    qty: float
    order_type: str = "market"
    client_order_id: Optional[str] = None
    alpaca_order_id: Optional[str] = None
    x_request_id: Optional[str] = None
    status_code: int = 0
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlpacaExecutionBridge:
    """Bridge for routing orders to Alpaca REST API and auditing all traffic."""

    def __init__(self, config: Optional[AlpacaConfig] = None, ledger: Optional[ExecutionLedger] = None):
        self.config = config or AlpacaConfig()
        self.ledger = ledger or ExecutionLedger()
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        """Format required Alpaca authentication and content headers."""
        return {
            "APCA-API-KEY-ID": self.config.api_key,
            "APCA-API-SECRET-KEY": self.config.secret_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        client_order_id: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any], Optional[str], Optional[str]]:
        """Perform HTTP request to Alpaca, log X-Request-ID, and record in ledger.

        Returns:
            (status_code, parsed_json_or_error_dict, x_request_id, error_message)
        """
        base_url = self.config.get_effective_base_url()
        url = f"{base_url}{endpoint}"
        headers = self._get_headers()
        req_body_str = json.dumps(json_data) if json_data else None

        start_time = time.perf_counter()
        status_code = 0
        x_request_id: Optional[str] = None
        error_message: Optional[str] = None
        response_body_str: Optional[str] = None
        parsed_data: Dict[str, Any] = {}
        success = False

        try:
            resp = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_data,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
            status_code = resp.status_code
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract Alpaca's audit identifier from headers (case-insensitive)
            x_request_id = resp.headers.get("X-Request-ID") or resp.headers.get("x-request-id")
            response_body_str = resp.text

            try:
                parsed_data = resp.json() if resp.text else {}
            except Exception:
                parsed_data = {"raw_text": resp.text}

            if 200 <= status_code < 300:
                success = True
                logger.info(
                    "Alpaca API %s %s -> HTTP %d (X-Request-ID: %s, Latency: %.1fms)",
                    method.upper(), endpoint, status_code, x_request_id or "none", latency_ms
                )
            else:
                error_message = (
                    parsed_data.get("message")
                    or parsed_data.get("error")
                    or f"HTTP {status_code}: {resp.text}"
                )
                logger.error(
                    "Alpaca API %s %s -> HTTP %d (X-Request-ID: %s, Latency: %.1fms) Error: %s",
                    method.upper(), endpoint, status_code, x_request_id or "none", latency_ms, error_message
                )

        except requests.exceptions.RequestException as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            error_message = f"Network/Connection Error: {str(e)}"
            status_code = 0
            logger.error(
                "Alpaca API %s %s -> Network Failure (Latency: %.1fms): %s",
                method.upper(), endpoint, latency_ms, error_message
            )
            parsed_data = {"error": error_message}

        # Immutable audit logging of every single call
        audit_entry = ApiAuditEntry(
            method=method.upper(),
            endpoint=endpoint,
            status_code=status_code,
            x_request_id=x_request_id,
            latency_ms=latency_ms,
            request_body=req_body_str,
            response_body=response_body_str,
            client_order_id=client_order_id,
            error_message=error_message,
            success=success,
        )
        self.ledger.record_api_call(audit_entry)

        return status_code, parsed_data, x_request_id, error_message

    # -------------------------------------------------------------------------
    # Account & Portfolio Ingestion
    # -------------------------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        """Fetch current account status, buying power, equity, and cash."""
        status_code, data, x_req, err = self._request("GET", "/v2/account")
        if status_code == 200:
            return data
        raise RuntimeError(f"Failed to fetch Alpaca account (HTTP {status_code}, X-Request-ID: {x_req}): {err}")

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch all currently open positions."""
        status_code, data, x_req, err = self._request("GET", "/v2/positions")
        if status_code == 200 and isinstance(data, list):
            return data
        if status_code == 200 and isinstance(data, dict):
            return [data]
        raise RuntimeError(f"Failed to fetch Alpaca positions (HTTP {status_code}, X-Request-ID: {x_req}): {err}")

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch position for a specific symbol, or None if not held."""
        status_code, data, x_req, err = self._request("GET", f"/v2/positions/{symbol.upper()}")
        if status_code == 200:
            return data
        if status_code == 404:
            return None
        raise RuntimeError(f"Failed to fetch position for {symbol} (HTTP {status_code}, X-Request-ID: {x_req}): {err}")

    def get_clock(self) -> Dict[str, Any]:
        """Fetch current market clock (is_open, next_open, next_close)."""
        status_code, data, x_req, err = self._request("GET", "/v2/clock")
        if status_code == 200:
            return data
        raise RuntimeError(f"Failed to fetch Alpaca market clock (HTTP {status_code}, X-Request-ID: {x_req}): {err}")

    # -------------------------------------------------------------------------
    # Order Routing & Execution
    # -------------------------------------------------------------------------

    def submit_order(self, directive: TradeDirective) -> ExecutionResult:
        """Route a validated trade directive to Alpaca REST API.

        Formats order payload, attaches unique client_order_id, captures
        Alpaca X-Request-ID, and synchronizes the ledger.
        """
        symbol = directive.symbol.upper().strip()
        side = directive.side.lower().strip()
        order_type = directive.order_type.lower().strip()
        time_in_force = directive.time_in_force.lower().strip()

        # Generate deterministic or random unique client_order_id for idempotency
        client_order_id = directive.client_order_id or f"agent_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        directive.client_order_id = client_order_id

        # Format REST payload according to Alpaca V2 order specifications
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": client_order_id,
        }

        # Quantity vs Notional specification
        if directive.qty > 0:
            # Round fractional or integer quantities
            payload["qty"] = str(round(directive.qty, 4) if directive.qty != int(directive.qty) else int(directive.qty))
        elif directive.notional and directive.notional > 0:
            payload["notional"] = str(round(directive.notional, 2))
        else:
            err = "Invalid order: either qty or notional must be strictly positive"
            logger.error(err)
            return ExecutionResult(
                success=False,
                status="rejected",
                symbol=symbol,
                side=side,
                qty=directive.qty,
                order_type=order_type,
                client_order_id=client_order_id,
                error_message=err,
            )

        if order_type in ("limit", "stop_limit"):
            if directive.limit_price is None or directive.limit_price <= 0:
                err = f"Order type {order_type} requires a positive limit_price"
                return ExecutionResult(
                    success=False,
                    status="rejected",
                    symbol=symbol,
                    side=side,
                    qty=directive.qty,
                    order_type=order_type,
                    client_order_id=client_order_id,
                    error_message=err,
                )
            payload["limit_price"] = str(directive.limit_price)

        if order_type in ("stop", "stop_limit"):
            if directive.stop_price is None or directive.stop_price <= 0:
                err = f"Order type {order_type} requires a positive stop_price"
                return ExecutionResult(
                    success=False,
                    status="rejected",
                    symbol=symbol,
                    side=side,
                    qty=directive.qty,
                    order_type=order_type,
                    client_order_id=client_order_id,
                    error_message=err,
                )
            payload["stop_price"] = str(directive.stop_price)

        # Dispatch POST /v2/orders
        status_code, resp_data, x_request_id, error_message = self._request(
            method="POST",
            endpoint="/v2/orders",
            json_data=payload,
            client_order_id=client_order_id,
        )

        success = (status_code == 200 or status_code == 201)
        alpaca_order_id = resp_data.get("id") if isinstance(resp_data, dict) else None
        order_status = resp_data.get("status", "rejected" if not success else "accepted")

        filled_qty = float(resp_data.get("filled_qty", 0.0)) if isinstance(resp_data, dict) else 0.0
        filled_avg_price = (
            float(resp_data["filled_avg_price"])
            if isinstance(resp_data, dict) and resp_data.get("filled_avg_price") is not None
            else None
        )

        # Record in trade ledger
        ledger_entry = OrderLedgerEntry(
            client_order_id=client_order_id,
            alpaca_order_id=alpaca_order_id,
            x_request_id=x_request_id,
            symbol=symbol,
            side=side,
            qty=directive.qty,
            order_type=order_type,
            status=order_status,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            limit_price=directive.limit_price,
            stop_price=directive.stop_price,
            error_message=error_message,
            raw_response=json.dumps(resp_data) if resp_data else None,
        )
        self.ledger.record_order(ledger_entry)

        return ExecutionResult(
            success=success,
            status=order_status,
            symbol=symbol,
            side=side,
            qty=directive.qty,
            order_type=order_type,
            client_order_id=client_order_id,
            alpaca_order_id=alpaca_order_id,
            x_request_id=x_request_id,
            status_code=status_code,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
            limit_price=directive.limit_price,
            stop_price=directive.stop_price,
            error_message=error_message,
            raw_response=resp_data,
        )

    def cancel_order(self, order_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel an open order by ID or client_order_id."""
        status_code, data, x_req, err = self._request("DELETE", f"/v2/orders/{order_id}")
        success = (status_code == 204 or status_code == 200)
        return success, x_req

    def cancel_all_orders(self) -> Tuple[bool, Optional[str]]:
        """Cancel all open orders."""
        status_code, data, x_req, err = self._request("DELETE", "/v2/orders")
        success = (status_code == 200 or status_code == 207)
        return success, x_req

    def get_orders(self, status: str = "open", limit: int = 50) -> List[Dict[str, Any]]:
        """Get orders from Alpaca."""
        status_code, data, x_req, err = self._request("GET", "/v2/orders", params={"status": status, "limit": limit})
        if status_code == 200 and isinstance(data, list):
            return data
        return []
