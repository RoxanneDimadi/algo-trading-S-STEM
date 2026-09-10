"""Cost-Aware Agent Decision and Policy Evaluator.

Evaluates raw incoming signals against live portfolio inventory, expected
trading costs, market impact, and risk bounds using Gârleanu-Pedersen partial
adjustment and hurdle filtering before passing finalized directives to the bridge.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

from src.execution.alpaca_bridge import AlpacaExecutionBridge
from src.execution.ledger import AgentDecisionEntry, ExecutionLedger, TradeDirective

logger = logging.getLogger("execution.agent_evaluator")


@dataclass
class AgentPolicyConfig:
    """Policy and risk parameters for the trading agent evaluator."""
    cost_bps: float = 10.0                 # One-way transaction cost assumption (bps)
    gamma_speed: float = 0.35              # Gârleanu-Pedersen partial adjustment speed (0 < gamma <= 1)
    impact_bps: float = 2.0                # Market impact scaling parameter (bps)
    alpha_hurdle_bps: float = 5.0          # Minimum net alpha advantage required to trigger a trade
    max_position_pct: float = 0.20         # Maximum position weight in single name (20% default)
    max_order_notional: float = 50000.0    # Maximum single order dollar value ($)
    min_trade_notional: float = 50.0       # Minimum trade dollar threshold to avoid dust orders ($)
    allow_short: bool = False              # Whether short positions are permitted
    target_annual_vol: float = 0.15        # Target portfolio volatility for sizing
    default_base_alpha_bps: float = 30.0   # Baseline expected gross alpha for full-strength signal


@dataclass
class AgentDecision:
    """The structured decision produced by the cost-aware agent."""
    approved: bool
    reason: str
    symbol: str
    raw_signal: Dict[str, Any]
    final_directive: Optional[TradeDirective] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        if self.final_directive:
            res["final_directive"] = self.final_directive.to_dict()
        return res


class CostAwareAgentEvaluator:
    """Evaluates raw trading signals and determines optimal order sizing."""

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
        """Evaluate an incoming TradingView signal against portfolio state and cost policy.

        Steps:
        1. Parse and validate the incoming signal payload.
        2. Query live account status and current position from Alpaca.
        3. Compute current portfolio weight and desired aim weight.
        4. Apply GP partial adjustment: w_target = (1 - gamma) * w_curr + gamma * w_aim.
        5. Calculate expected transaction costs (spread/commission + impact).
        6. Perform alpha vs cost hurdle check: only trade if net advantage > hurdle.
        7. Enforce risk constraints (max position, buying power, min trade size).
        8. Format finalized TradeDirective and record decision in ledger.
        """
        symbol = str(raw_signal.get("ticker") or raw_signal.get("symbol", "")).upper().strip()
        if not symbol:
            return self._reject(raw_signal, symbol, "Missing ticker/symbol in signal payload")

        # Extract signal direction / action
        action = str(raw_signal.get("action") or raw_signal.get("side", "")).lower().strip()
        if action not in ("buy", "sell", "long", "short", "flat", "close", "hold"):
            return self._reject(raw_signal, symbol, f"Unknown action/side: {action}")

        # Signal strength: normalized [-1.0, 1.0] or default 1.0 for buy, -1.0 for sell
        signal_strength = float(raw_signal.get("signal_strength", raw_signal.get("strength", 1.0)))
        signal_strength = max(-1.0, min(1.0, signal_strength))

        # Explicit desired shares or notional if provided by TradingView alert
        requested_qty = float(raw_signal.get("quantity") or raw_signal.get("qty") or 0.0)
        requested_notional = float(raw_signal.get("notional") or 0.0)
        signal_price = float(raw_signal.get("price") or raw_signal.get("close") or 0.0)
        order_type = str(raw_signal.get("order_type", "market")).lower().strip()
        limit_price = float(raw_signal["limit_price"]) if raw_signal.get("limit_price") else None
        stop_price = float(raw_signal["stop_price"]) if raw_signal.get("stop_price") else None
        time_in_force = str(raw_signal.get("time_in_force", "day")).lower().strip()

        # Fetch live portfolio and account state from Alpaca bridge
        try:
            account = self.bridge.get_account()
            portfolio_value = float(account.get("portfolio_value", account.get("equity", 100000.0)))
            cash = float(account.get("cash", 100000.0))
            buying_power = float(account.get("buying_power", cash))
        except Exception as e:
            logger.warning("Failed to fetch account info from Alpaca (%s), using fallback defaults", e)
            portfolio_value = 100000.0
            cash = 100000.0
            buying_power = 100000.0

        if portfolio_value <= 0:
            return self._reject(raw_signal, symbol, f"Portfolio value is non-positive: {portfolio_value}")

        # Fetch current position for the target symbol
        current_shares = 0.0
        current_price = signal_price
        current_market_value = 0.0
        try:
            position = self.bridge.get_position(symbol)
            if position:
                current_shares = float(position.get("qty", 0.0))
                current_price = float(position.get("current_price", signal_price or 1.0))
                current_market_value = float(position.get("market_value", current_shares * current_price))
        except Exception as e:
            logger.warning("Failed to query position for %s: %s", symbol, e)

        # Fallback price if none is available
        if current_price <= 0:
            if signal_price > 0:
                current_price = signal_price
            else:
                return self._reject(raw_signal, symbol, "Unable to determine current price for asset")

        current_weight = current_market_value / portfolio_value

        # Short selling check upfront when shorting is disabled
        if action in ("sell", "short") and not self.config.allow_short and current_shares <= 0:
            return self._reject(raw_signal, symbol, "Short selling is disabled by policy configuration")

        # Determine target aim weight from signal
        aim_weight = self._compute_aim_weight(
            action=action,
            signal_strength=signal_strength,
            requested_qty=requested_qty,
            requested_notional=requested_notional,
            current_price=current_price,
            portfolio_value=portfolio_value,
        )

        # Apply Gârleanu-Pedersen Partial Adjustment:
        # w_target = (1 - gamma) * w_current + gamma * w_aim
        gamma = self.config.gamma_speed
        target_weight = (1.0 - gamma) * current_weight + gamma * aim_weight

        # Enforce max position weight constraint
        max_w = self.config.max_position_pct
        min_w = -max_w if self.config.allow_short else 0.0
        target_weight = max(min_w, min(max_w, target_weight))

        # Target dollar allocation and target share count
        target_notional = target_weight * portfolio_value
        target_shares = target_notional / current_price

        # Desired change in shares (delta)
        delta_shares = target_shares - current_shares

        # Round to integer shares if dealing with non-fractional asset
        order_shares = round(delta_shares)
        order_notional = abs(order_shares * current_price)

        # ---------------------------------------------------------------------
        # Sizing and Dust Filters
        # ---------------------------------------------------------------------
        if order_shares == 0 or order_notional < self.config.min_trade_notional:
            reason = (
                f"Partial adjustment resulted in negligible change ({abs(delta_shares):.2f} shares / "
                f"${order_notional:.2f} < ${self.config.min_trade_notional:.2f} min threshold). No trade needed."
            )
            return self._skip(
                raw_signal=raw_signal,
                symbol=symbol,
                reason=reason,
                current_shares=current_shares,
                current_price=current_price,
                target_shares=target_shares,
                portfolio_value=portfolio_value,
                current_weight=current_weight,
                target_weight=target_weight,
            )

        # Order side
        order_side = "buy" if order_shares > 0 else "sell"
        abs_order_shares = abs(order_shares)

        # ---------------------------------------------------------------------
        # Cost-Benefit / Hurdle Evaluation
        # ---------------------------------------------------------------------
        # 1. Turnover weight delta: dw = |target_w - current_w|
        dw = abs(target_weight - current_weight)

        # 2. Linear transaction costs + square-root market impact cost
        linear_cost_bps = self.config.cost_bps
        impact_cost_bps = self.config.impact_bps * (abs(dw) ** 0.5)
        total_cost_bps = linear_cost_bps + impact_cost_bps

        # 3. Expected Gross Alpha benefit over typical holding horizon
        gross_alpha_bps = abs(signal_strength) * self.config.default_base_alpha_bps

        # 4. Net Benefit calculation
        net_benefit_bps = gross_alpha_bps - total_cost_bps

        if net_benefit_bps < self.config.alpha_hurdle_bps and action not in ("flat", "close"):
            reason = (
                f"Expected net benefit ({net_benefit_bps:.1f} bps) below hurdle "
                f"({self.config.alpha_hurdle_bps:.1f} bps). Gross Alpha: {gross_alpha_bps:.1f} bps, "
                f"Estimated Total Cost: {total_cost_bps:.1f} bps. Trade filtered to save fees."
            )
            return self._skip(
                raw_signal=raw_signal,
                symbol=symbol,
                reason=reason,
                current_shares=current_shares,
                current_price=current_price,
                target_shares=target_shares,
                portfolio_value=portfolio_value,
                current_weight=current_weight,
                target_weight=target_weight,
                cost_bps=total_cost_bps,
                alpha_bps=gross_alpha_bps,
                net_benefit=net_benefit_bps,
            )

        # ---------------------------------------------------------------------
        # Risk & Buying Power Safety Controls
        # ---------------------------------------------------------------------
        if order_notional > self.config.max_order_notional:
            # Downsize order to max allowable notional
            downsized_shares = int(self.config.max_order_notional / current_price)
            logger.info("Order for %s exceeds max notional ($%.2f), downsized from %d to %d shares",
                        symbol, order_notional, abs_order_shares, downsized_shares)
            abs_order_shares = max(1, downsized_shares)
            order_notional = abs_order_shares * current_price

        # Check buying power for buys
        if order_side == "buy" and order_notional > buying_power:
            # Downsize to 95% of available buying power to provide safety buffer
            affordable_shares = int((buying_power * 0.95) / current_price)
            if affordable_shares <= 0:
                return self._reject(
                    raw_signal,
                    symbol,
                    f"Insufficient buying power: required ${order_notional:.2f}, available ${buying_power:.2f}"
                )
            logger.info("Downsized buy order for %s from %d to %d shares due to buying power",
                        symbol, abs_order_shares, affordable_shares)
            abs_order_shares = affordable_shares

        # Prevent selling short if disallowed
        if order_side == "sell" and not self.config.allow_short and current_shares <= 0:
            return self._reject(raw_signal, symbol, "Short selling is disabled by policy configuration")

        if order_side == "sell" and not self.config.allow_short and abs_order_shares > current_shares:
            # Cap sell to available long shares
            abs_order_shares = int(current_shares)
            if abs_order_shares <= 0:
                return self._skip(
                    raw_signal=raw_signal,
                    symbol=symbol,
                    reason="Position is already closed/flat. Nothing to sell.",
                    current_shares=current_shares,
                    current_price=current_price,
                    target_shares=target_shares,
                    portfolio_value=portfolio_value,
                    current_weight=current_weight,
                    target_weight=target_weight,
                )

        # Format finalized directive
        directive = TradeDirective(
            symbol=symbol,
            side=order_side,
            qty=float(abs_order_shares),
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            strategy=raw_signal.get("strategy", "cost_aware_gp_agent"),
            notes=f"GP speed: {gamma:.2f}, Net benefit: {net_benefit_bps:.1f}bps",
        )

        approval_reason = (
            f"Approved: {order_side.upper()} {abs_order_shares} shares of {symbol} "
            f"(target weight {target_weight*100:.2f}%, net alpha {net_benefit_bps:.1f} bps after {total_cost_bps:.1f} bps costs)"
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

        # Record decision in ledger
        decision_entry = AgentDecisionEntry(
            symbol=symbol,
            raw_signal=raw_signal,
            approved=True,
            reason=approval_reason,
            current_position_qty=current_shares,
            current_price=current_price,
            target_position_qty=target_shares,
            order_qty=float(abs_order_shares),
            expected_alpha_bps=gross_alpha_bps,
            expected_cost_bps=total_cost_bps,
            net_benefit_bps=net_benefit_bps,
            portfolio_value=portfolio_value,
            directive=directive.to_dict(),
        )
        self.ledger.record_agent_decision(decision_entry)

        return AgentDecision(
            approved=True,
            reason=approval_reason,
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
        """Compute the unconstrained aim target portfolio weight."""
        if action in ("flat", "close", "hold"):
            return 0.0

        # If alert specified explicit desired quantity or notional
        if requested_qty > 0:
            desired_notional = requested_qty * current_price
            weight = desired_notional / portfolio_value
            return -weight if action in ("sell", "short") else weight

        if requested_notional > 0:
            weight = requested_notional / portfolio_value
            return -weight if action in ("sell", "short") else weight

        # Standard proportional weight based on signal strength & max position
        direction = -1.0 if action in ("sell", "short") else 1.0
        aim = direction * abs(signal_strength) * self.config.max_position_pct
        return aim

    def _reject(self, raw_signal: Dict[str, Any], symbol: str, reason: str) -> AgentDecision:
        """Helper to create a rejected decision and log to ledger."""
        logger.warning("Signal rejected for %s: %s", symbol or "UNKNOWN", reason)
        decision_entry = AgentDecisionEntry(
            symbol=symbol or "UNKNOWN",
            raw_signal=raw_signal,
            approved=False,
            reason=f"REJECTED: {reason}",
            current_position_qty=0.0,
            current_price=0.0,
            target_position_qty=0.0,
            order_qty=0.0,
            expected_alpha_bps=0.0,
            expected_cost_bps=0.0,
            net_benefit_bps=0.0,
            portfolio_value=0.0,
            directive=None,
        )
        self.ledger.record_agent_decision(decision_entry)
        return AgentDecision(approved=False, reason=f"REJECTED: {reason}", symbol=symbol, raw_signal=raw_signal)

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
        """Helper to record a trade skipped/filtered by cost policy."""
        logger.info("Signal skipped for %s: %s", symbol, reason)
        decision_entry = AgentDecisionEntry(
            symbol=symbol,
            raw_signal=raw_signal,
            approved=False,
            reason=f"SKIPPED: {reason}",
            current_position_qty=current_shares,
            current_price=current_price,
            target_position_qty=target_shares,
            order_qty=0.0,
            expected_alpha_bps=alpha_bps,
            expected_cost_bps=cost_bps,
            net_benefit_bps=net_benefit,
            portfolio_value=portfolio_value,
            directive=None,
        )
        self.ledger.record_agent_decision(decision_entry)
        return AgentDecision(
            approved=False,
            reason=f"SKIPPED: {reason}",
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
