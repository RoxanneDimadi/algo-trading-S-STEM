"""CLI runner to launch the Alpaca Execution Bridge & Webhook Listener service."""
from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.agent_evaluator import (
    AgentPolicyConfig,
    CostAwareAgentEvaluator,
)
from src.execution.alpaca_bridge import AlpacaConfig, AlpacaExecutionBridge
from src.execution.ledger import ExecutionLedger
from src.execution.webhook_listener import create_webhook_app


def load_config(config_path: str = "configs/execution_config.yaml") -> dict:
    """Load config from YAML file with environment variable fallbacks."""
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Environment variable overrides
    broker_cfg = cfg.get("broker", {})
    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID") or broker_cfg.get("api_key", "")
    secret_key = (
        os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY") or broker_cfg.get("secret_key", "")
    )
    paper_env = os.environ.get("ALPACA_PAPER")
    paper = (paper_env.lower() in ("true", "1", "yes")) if paper_env is not None else broker_cfg.get("paper", True)
    base_url = os.environ.get("ALPACA_BASE_URL") or broker_cfg.get("base_url")

    broker_cfg["api_key"] = api_key
    broker_cfg["secret_key"] = secret_key
    broker_cfg["paper"] = paper
    broker_cfg["base_url"] = base_url
    cfg["broker"] = broker_cfg

    webhook_cfg = cfg.get("webhook", {})
    passphrase = os.environ.get("WEBHOOK_PASSPHRASE") or webhook_cfg.get("passphrase", "")
    webhook_cfg["passphrase"] = passphrase
    cfg["webhook"] = webhook_cfg

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run Alpaca Execution Bridge & Webhook Listener")
    parser.add_argument("--config", default="configs/execution_config.yaml", help="Path to execution config YAML")
    parser.add_argument("--host", default=None, help="Server host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: 8000)")
    parser.add_argument("--paper", action="store_true", default=None, help="Force paper trading mode")
    parser.add_argument("--live", action="store_true", default=False, help="Force LIVE trading mode (caution!)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("execution.server")

    cfg = load_config(args.config)

    # CLI Overrides
    if args.live:
        cfg["broker"]["paper"] = False
        logger.warning("!!! CAUTION: RUNNING IN LIVE BROKER TRADING MODE !!!")
    elif args.paper is not None:
        cfg["broker"]["paper"] = args.paper

    host = args.host or cfg.get("webhook", {}).get("host", "0.0.0.0")
    port = args.port or cfg.get("webhook", {}).get("port", 8000)
    passphrase = cfg.get("webhook", {}).get("passphrase", "")

    # Initialize components
    broker_cfg = cfg.get("broker", {})
    alpaca_config = AlpacaConfig(
        api_key=broker_cfg.get("api_key", ""),
        secret_key=broker_cfg.get("secret_key", ""),
        paper=broker_cfg.get("paper", True),
        base_url=broker_cfg.get("base_url"),
        timeout_seconds=broker_cfg.get("timeout_seconds", 10.0),
    )

    ledger_cfg = cfg.get("ledger", {})
    ledger = ExecutionLedger(
        db_path=ledger_cfg.get("db_path", "data/ledger.db"),
        jsonl_path=ledger_cfg.get("jsonl_path", "data/audit_log.jsonl"),
    )

    bridge = AlpacaExecutionBridge(config=alpaca_config, ledger=ledger)

    agent_cfg = cfg.get("agent_policy", {})
    policy_config = AgentPolicyConfig(
        gamma_speed=agent_cfg.get("gamma_speed", 0.35),
        cost_bps=agent_cfg.get("cost_bps", 10.0),
        impact_bps=agent_cfg.get("impact_bps", 2.0),
        alpha_hurdle_bps=agent_cfg.get("alpha_hurdle_bps", 5.0),
        max_position_pct=agent_cfg.get("max_position_pct", 0.20),
        max_order_notional=agent_cfg.get("max_order_notional", 50000.0),
        min_trade_notional=agent_cfg.get("min_trade_notional", 50.0),
        allow_short=agent_cfg.get("allow_short", False),
        default_base_alpha_bps=agent_cfg.get("default_base_alpha_bps", 30.0),
    )

    evaluator = CostAwareAgentEvaluator(bridge=bridge, config=policy_config, ledger=ledger)

    app = create_webhook_app(
        bridge=bridge,
        evaluator=evaluator,
        webhook_passphrase=passphrase,
        ledger=ledger,
    )

    mode_str = "PAPER TRADING" if alpaca_config.paper else "LIVE TRADING"
    logger.info("=" * 65)
    logger.info(" Starting Alpaca Execution Bridge & Webhook Listener")
    logger.info(" Environment: %s", mode_str)
    logger.info(" Base URL:    %s", alpaca_config.get_effective_base_url())
    logger.info(" Listening on: http://%s:%d/webhook", host, port)
    logger.info(" Auth:        %s", "Enabled (passphrase required)" if passphrase else "Disabled (public)")
    logger.info(" Agent Gamma: %.2f (GP partial adjustment speed)", policy_config.gamma_speed)
    logger.info(" Cost Model:  %.1f bps linear + %.1f bps impact", policy_config.cost_bps, policy_config.impact_bps)
    logger.info("=" * 65)

    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
