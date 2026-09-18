"""SQLite + JSONL audit trail for orders, decisions, and broker API calls."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("execution.ledger")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ApiAuditEntry:
    id: Optional[int] = None
    timestamp: str = field(default_factory=_utc_now_iso)
    method: str = ""
    endpoint: str = ""
    status_code: int = 0
    x_request_id: Optional[str] = None
    latency_ms: float = 0.0
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    client_order_id: Optional[str] = None
    error_message: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TradeDirective:
    symbol: str
    side: str
    qty: float
    order_type: str = "market"
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    notional: Optional[float] = None
    strategy: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDecisionEntry:
    id: Optional[int] = None
    timestamp: str = field(default_factory=_utc_now_iso)
    signal_source: str = "tradingview_webhook"
    symbol: str = ""
    raw_signal: Dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    reason: str = ""
    current_position_qty: float = 0.0
    current_price: float = 0.0
    target_position_qty: float = 0.0
    order_qty: float = 0.0
    expected_alpha_bps: float = 0.0
    expected_cost_bps: float = 0.0
    net_benefit_bps: float = 0.0
    portfolio_value: float = 0.0
    directive: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderLedgerEntry:
    id: Optional[int] = None
    timestamp: str = field(default_factory=_utc_now_iso)
    client_order_id: str = ""
    alpaca_order_id: Optional[str] = None
    x_request_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    qty: float = 0.0
    order_type: str = "market"
    status: str = "pending"
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    error_message: Optional[str] = None
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionLedger:
    """Thread-safe store; one connection so :memory: works across calls."""

    def __init__(
        self,
        db_path: str = "data/ledger.db",
        jsonl_path: Optional[str] = "data/audit_log.jsonl",
    ):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_storage()

    def _init_storage(self) -> None:
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(
                os.path.abspath(self.db_path)), exist_ok=True)
        if self.jsonl_path:
            os.makedirs(os.path.dirname(
                os.path.abspath(self.jsonl_path)), exist_ok=True)

        with self._lock:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = self._conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    method TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    x_request_id TEXT,
                    latency_ms REAL NOT NULL,
                    request_body TEXT,
                    response_body TEXT,
                    client_order_id TEXT,
                    error_message TEXT,
                    success INTEGER NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    signal_source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    raw_signal TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    current_position_qty REAL NOT NULL,
                    current_price REAL NOT NULL,
                    target_position_qty REAL NOT NULL,
                    order_qty REAL NOT NULL,
                    expected_alpha_bps REAL NOT NULL,
                    expected_cost_bps REAL NOT NULL,
                    net_benefit_bps REAL NOT NULL,
                    portfolio_value REAL NOT NULL,
                    directive TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS order_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    client_order_id TEXT UNIQUE NOT NULL,
                    alpaca_order_id TEXT,
                    x_request_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filled_qty REAL NOT NULL,
                    filled_avg_price REAL,
                    limit_price REAL,
                    stop_price REAL,
                    error_message TEXT,
                    raw_response TEXT
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_x_req_id "
                "ON api_audit(x_request_id)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_client_order_id "
                "ON api_audit(client_order_id)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_client_id "
                "ON order_ledger(client_order_id)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_order_alpaca_id "
                "ON order_ledger(alpaca_order_id)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_symbol "
                "ON agent_decisions(symbol)")
            self._conn.commit()

    def record_api_call(self, entry: ApiAuditEntry) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO api_audit (
                    timestamp, method, endpoint, status_code, x_request_id,
                    latency_ms, request_body, response_body, client_order_id,
                    error_message, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.timestamp, entry.method, entry.endpoint,
                    entry.status_code, entry.x_request_id, entry.latency_ms,
                    entry.request_body, entry.response_body,
                    entry.client_order_id, entry.error_message,
                    1 if entry.success else 0,
                ),
            )
            row_id = cur.lastrowid
            self._conn.commit()
        if self.jsonl_path:
            self._append_jsonl("api_audit", entry.to_dict())
        return row_id

    def record_agent_decision(self, decision: AgentDecisionEntry) -> int:
        raw = json.dumps(decision.raw_signal) if decision.raw_signal else "{}"
        directive = json.dumps(
            decision.directive) if decision.directive else None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO agent_decisions (
                    timestamp, signal_source, symbol, raw_signal, approved,
                    reason, current_position_qty, current_price,
                    target_position_qty, order_qty, expected_alpha_bps,
                    expected_cost_bps, net_benefit_bps, portfolio_value,
                    directive
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    decision.timestamp, decision.signal_source,
                    decision.symbol, raw, 1 if decision.approved else 0,
                    decision.reason,
                    decision.current_position_qty, decision.current_price,
                    decision.target_position_qty, decision.order_qty,
                    decision.expected_alpha_bps, decision.expected_cost_bps,
                    decision.net_benefit_bps, decision.portfolio_value,
                    directive,
                ),
            )
            row_id = cur.lastrowid
            self._conn.commit()
        if self.jsonl_path:
            self._append_jsonl("agent_decision", decision.to_dict())
        return row_id

    def record_order(self, order: OrderLedgerEntry) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO order_ledger (
                    timestamp, client_order_id, alpaca_order_id, x_request_id,
                    symbol, side, qty, order_type, status, filled_qty,
                    filled_avg_price, limit_price, stop_price, error_message,
                    raw_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    alpaca_order_id = excluded.alpaca_order_id,
                    x_request_id = excluded.x_request_id,
                    status = excluded.status,
                    filled_qty = excluded.filled_qty,
                    filled_avg_price = excluded.filled_avg_price,
                    error_message = excluded.error_message,
                    raw_response = excluded.raw_response""",
                (
                    order.timestamp, order.client_order_id,
                    order.alpaca_order_id,
                    order.x_request_id, order.symbol, order.side, order.qty,
                    order.order_type, order.status, order.filled_qty,
                    order.filled_avg_price, order.limit_price,
                    order.stop_price,
                    order.error_message, order.raw_response,
                ),
            )
            row_id = cur.lastrowid
            self._conn.commit()
        if self.jsonl_path:
            self._append_jsonl("order_ledger", order.to_dict())
        return row_id

    def update_order_status(
        self,
        client_order_id: str,
        status: str,
        filled_qty: float = 0.0,
        filled_avg_price: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """UPDATE order_ledger
                   SET status = ?, filled_qty = ?, filled_avg_price = ?,
                       error_message = ?
                   WHERE client_order_id = ?""",
                (status, filled_qty, filled_avg_price,
                 error_message, client_order_id),
            )
            self._conn.commit()

    def get_recent_api_audits(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM api_audit ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_recent_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM agent_decisions ORDER BY id DESC LIMIT ?",
                (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_recent_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM order_ledger ORDER BY id DESC LIMIT ?",
                (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_by_x_request_id(
            self, x_request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM api_audit WHERE x_request_id = ?",
                (x_request_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def _append_jsonl(self, event_type: str, data: Dict[str, Any]) -> None:
        if not self.jsonl_path:
            return
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event_type": event_type, **data}) + "\n")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("jsonl write failed: %s", e)
