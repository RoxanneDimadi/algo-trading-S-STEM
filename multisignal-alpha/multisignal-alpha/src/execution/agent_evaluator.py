"""Live-signal gate in front of the Alpaca bridge.

Incoming alerts are sized with the same GP partial-adjustment idea as the
research agent (gamma speed toward an aim weight), then dropped when expected
net alpha does not clear the cost hurdle.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from src.execution.alpaca_bridge import AlpacaExecutionBridge
from src.execution.ledger import (AgentDecisionEntry, ExecutionLedger,
                                  TradeDirective)

logger = logging.getLogger("execution.agent_evaluator")


@dataclass
class AgentPolicyConfig:
    cost_bps: float = 10.0
    gamma_speed: float = 0.35          # partial adjustment speed in (0, 1]
    impact_bps: float = 2.0
    alpha_hurdle_bps: float = 5.0      # skip if net benefit below this
    max_position_pct: float = 0.20
    max_order_notional: float = 50000.0
    min_trade_notional: float = 50.0
    allow_short: bool = False
    target_annual_vol: float = 0.15
    default_base_alpha_bps: float = 30.0


@dataclass
class AgentDecision:
    approved: bool
    reason: str
    symbol: str
    raw_signal: Dict[str, Any]
    final_directive: Optional[TradeDirective] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if self.final_directive:
            out["final_directive"] = self.final_directive.to_dict()
        return out


class CostAwareAgentEvaluator:
    def __init__(
        self,
        bridge: AlpacaExecutionBridge,
        config: Optional[AgentPolicyConfig] = None,
        ledger: Optional[ExecutionLedger] = None,
    ):
        self.bridge = bridge
        self.config = config or AgentPolicyConfig()
        self.ledger = ledger or bridge.ledger

    def evaluate_signal(self, raw_signal: Dict[str, Any]) -> AgentDecision:
        # A gauntlet of independent rejection gates; each one returns
        # early with its own reason, which is clearer than nesting.
        # pylint: disable=too-many-locals, too-many-return-statements
        symbol = str(raw_signal.get("ticker") or raw_signal.get(
            "symbol", "")).upper().strip()
        if not symbol:
            return self._reject(
                raw_signal, symbol, "Missing ticker/symbol in signal payload")

        action = str(raw_signal.get("action")
                     or raw_signal.get("side", "")).lower().strip()
        if action not in ("buy", "sell", "long", "short", "flat", "close",
                          "hold"):
            return self._reject(
                raw_signal, symbol, f"Unknown action/side: {action}")

        signal_strength = float(raw_signal.get(
            "signal_strength", raw_signal.get("strength", 1.0)))
        signal_strength = max(-1.0, min(1.0, signal_strength))

        requested_qty = float(raw_signal.get("quantity")
                              or raw_signal.get("qty") or 0.0)
        requested_notional = float(raw_signal.get("notional") or 0.0)
        signal_price = float(raw_signal.get(
            "price") or raw_signal.get("close") or 0.0)
        order_type = str(raw_signal.get(
            "order_type", "market")).lower().strip()
        limit_price = float(raw_signal["limit_price"]) if raw_signal.get(
            "limit_price") else None
        stop_price = float(raw_signal["stop_price"]) if raw_signal.get(
            "stop_price") else None
        time_in_force = str(raw_signal.get(
            "time_in_force", "day")).lower().strip()

        try:
            account = self.bridge.get_account()
            portfolio_value = float(account.get(
                "portfolio_value", account.get("equity", 100000.0)))
            cash = float(account.get("cash", 100000.0))
            buying_power = float(account.get("buying_power", cash))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("account lookup failed (%s); using defaults", e)
            portfolio_value = 100000.0
            buying_power = 100000.0

        if portfolio_value <= 0:
            return self._reject(
                raw_signal, symbol,
                f"Portfolio value is non-positive: {portfolio_value}")

        current_shares = 0.0
        current_price = signal_price
        current_market_value = 0.0
        try:
            position = self.bridge.get_position(symbol)
            if position:
                current_shares = float(position.get("qty", 0.0))
                current_price = float(position.get(
                    "current_price", signal_price or 1.0))
                current_market_value = float(position.get(
                    "market_value", current_shares * current_price))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("position lookup failed for %s: %s", symbol, e)

        if current_price <= 0:
            if signal_price > 0:
                current_price = signal_price
            else:
                return self._reject(
                    raw_signal, symbol,
                    "Unable to determine current price for asset")

        current_weight = current_market_value / portfolio_value

        if (action in ("sell", "short") and not self.config.allow_short
                and current_shares <= 0):
            return self._reject(
                raw_signal, symbol,
                "Short selling is disabled by policy configuration")

        aim_weight = self._compute_aim_weight(
            action, signal_strength, requested_qty, requested_notional,
            current_price, portfolio_value,
        )

        # w_t = (1 - gamma) w_{t-1} + gamma * aim
        gamma = self.config.gamma_speed
        target_weight = (1.0 - gamma) * current_weight + gamma * aim_weight
        max_w = self.config.max_position_pct
        min_w = -max_w if self.config.allow_short else 0.0
        target_weight = max(min_w, min(max_w, target_weight))

        target_shares = (target_weight * portfolio_value) / current_price
        delta_shares = target_shares - current_shares
        order_shares = round(delta_shares)
        order_notional = abs(order_shares * current_price)

        if (order_shares == 0
                or order_notional < self.config.min_trade_notional):
            reason = (
                f"trade too small ({abs(delta_shares):.2f} shares / "
                f"${order_notional:.2f} < "
                f"${self.config.min_trade_notional:.2f})"
            )
            return self._skip(
                raw_signal, symbol, reason, current_shares, current_price,
                target_shares, portfolio_value, current_weight, target_weight,
            )

        order_side = "buy" if order_shares > 0 else "sell"
        abs_order_shares = abs(order_shares)

        dw = abs(target_weight - current_weight)
        total_cost_bps = self.config.cost_bps + \
            self.config.impact_bps * (dw ** 0.5)
        gross_alpha_bps = abs(signal_strength) * \
            self.config.default_base_alpha_bps
        net_benefit_bps = gross_alpha_bps - total_cost_bps

        if (net_benefit_bps < self.config.alpha_hurdle_bps
                and action not in ("flat", "close")):
            reason = (
                f"net benefit {net_benefit_bps:.1f} bps below hurdle "
                f"{self.config.alpha_hurdle_bps:.1f} bps "
                f"(alpha {gross_alpha_bps:.1f}, cost {total_cost_bps:.1f})"
            )
            return self._skip(
                raw_signal, symbol, reason, current_shares, current_price,
                target_shares, portfolio_value, current_weight, target_weight,
                cost_bps=total_cost_bps, alpha_bps=gross_alpha_bps,
                net_benefit=net_benefit_bps,
            )

        if order_notional > self.config.max_order_notional:
            downsized = int(self.config.max_order_notional / current_price)
            logger.info("%s order capped by max notional: %d -> %d shares",
                        symbol, abs_order_shares, max(1, downsized))
            abs_order_shares = max(1, downsized)
            order_notional = abs_order_shares * current_price

        if order_side == "buy" and order_notional > buying_power:
            affordable = int((buying_power * 0.95) / current_price)
            if affordable <= 0:
                return self._reject(
                    raw_signal, symbol,
                    f"Insufficient buying power: "
                    f"need ${order_notional:.2f}, "
                    f"have ${buying_power:.2f}",
                )
            logger.info("%s buy sized down for buying power: %d -> %d",
                        symbol, abs_order_shares, affordable)
            abs_order_shares = affordable

        if (order_side == "sell" and not self.config.allow_short
                and current_shares <= 0):
            return self._reject(
                raw_signal, symbol,
                "Short selling is disabled by policy configuration")

        if (order_side == "sell" and not self.config.allow_short
                and abs_order_shares > current_shares):
            abs_order_shares = int(current_shares)
            if abs_order_shares <= 0:
                return self._skip(
                    raw_signal, symbol, "already flat", current_shares,
                    current_price, target_shares, portfolio_value,
                    current_weight, target_weight,
                )

        directive = TradeDirective(
            symbol=symbol,
            side=order_side,
            qty=float(abs_order_shares),
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            strategy=raw_signal.get("strategy", "cost_aware_gp_agent"),
            notes=f"gamma={gamma:.2f}, net={net_benefit_bps:.1f}bps",
        )

        reason = (
            f"{order_side.upper()} {abs_order_shares} {symbol} "
            f"(w*={target_weight * 100:.2f}%, net {net_benefit_bps:.1f} bps)"
        )
        metrics = {
            "symbol": symbol,
            "portfolio_value": portfolio_value,
            "current_price": current_price,
            "current_shares": current_shares,
            "current_weight": current_weight,
            "aim_weight": aim_weight,
            "target_weight": target_weight,
            "target_shares": target_shares,
            "order_shares": abs_order_shares,
            "order_side": order_side,
            "order_notional": order_notional,
            "gamma_speed": gamma,
            "expected_cost_bps": total_cost_bps,
            "expected_alpha_bps": gross_alpha_bps,
            "net_benefit_bps": net_benefit_bps,
        }

        self.ledger.record_agent_decision(AgentDecisionEntry(
            symbol=symbol,
            raw_signal=raw_signal,
            approved=True,
            reason=reason,
            current_position_qty=current_shares,
            current_price=current_price,
            target_position_qty=target_shares,
            order_qty=float(abs_order_shares),
            expected_alpha_bps=gross_alpha_bps,
            expected_cost_bps=total_cost_bps,
            net_benefit_bps=net_benefit_bps,
            portfolio_value=portfolio_value,
            directive=directive.to_dict(),
        ))

        return AgentDecision(
            approved=True,
            reason=reason,
            symbol=symbol,
            raw_signal=raw_signal,
            final_directive=directive,
            metrics=metrics,
        )

    def _compute_aim_weight(
        self,
        action: str,
        signal_strength: float,
        requested_qty: float,
        requested_notional: float,
        current_price: float,
        portfolio_value: float,
    ) -> float:
        if action in ("flat", "close", "hold"):
            return 0.0

        if requested_qty > 0:
            weight = (requested_qty * current_price) / portfolio_value
            return -weight if action in ("sell", "short") else weight

        if requested_notional > 0:
            weight = requested_notional / portfolio_value
            return -weight if action in ("sell", "short") else weight

        direction = -1.0 if action in ("sell", "short") else 1.0
        return direction * abs(signal_strength) * self.config.max_position_pct

    def _reject(self, raw_signal: Dict[str, Any], symbol: str,
                reason: str) -> AgentDecision:
        logger.warning("reject %s: %s", symbol or "UNKNOWN", reason)
        tagged = f"REJECTED: {reason}"
        self.ledger.record_agent_decision(AgentDecisionEntry(
            symbol=symbol or "UNKNOWN",
            raw_signal=raw_signal,
            approved=False,
            reason=tagged,
        ))
        return AgentDecision(approved=False, reason=tagged, symbol=symbol,
                             raw_signal=raw_signal)

    def _skip(
        self,
        raw_signal: Dict[str, Any],
        symbol: str,
        reason: str,
        current_shares: float,
        current_price: float,
        target_shares: float,
        portfolio_value: float,
        current_weight: float,
        target_weight: float,
        cost_bps: float = 0.0,
        alpha_bps: float = 0.0,
        net_benefit: float = 0.0,
    ) -> AgentDecision:
        logger.info("skip %s: %s", symbol, reason)
        tagged = f"SKIPPED: {reason}"
        self.ledger.record_agent_decision(AgentDecisionEntry(
            symbol=symbol,
            raw_signal=raw_signal,
            approved=False,
            reason=tagged,
            current_position_qty=current_shares,
            current_price=current_price,
            target_position_qty=target_shares,
            expected_alpha_bps=alpha_bps,
            expected_cost_bps=cost_bps,
            net_benefit_bps=net_benefit,
            portfolio_value=portfolio_value,
        ))
        return AgentDecision(
            approved=False,
            reason=tagged,
            symbol=symbol,
            raw_signal=raw_signal,
            metrics={
                "current_weight": current_weight,
                "target_weight": target_weight,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "cost_bps": cost_bps,
                "alpha_bps": alpha_bps,
                "net_benefit_bps": net_benefit,
            },
        )
