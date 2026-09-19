"""Start the webhook listener / Alpaca bridge."""
from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PACKAGE_ROOT)

# Read .env next to this package, if there is one, so the credentials in it
# reach os.environ before load_config() looks for them. override=False, so a
# variable already exported in the shell still wins over the file.
load_dotenv(os.path.join(PACKAGE_ROOT, ".env"), override=False)

# The package lives one level up; the bootstrap above has to run
# before these imports resolve.
# pylint: disable=wrong-import-position
from src.execution.webhook_listener import create_webhook_app
from src.execution.ledger import ExecutionLedger
from src.execution.alpaca_bridge import AlpacaConfig, AlpacaExecutionBridge
from src.execution.agent_evaluator import (AgentPolicyConfig,
                                           CostAwareAgentEvaluator)


def load_config(config_path: str = "configs/execution_config.yaml") -> dict:
    """Merge configs/execution_config.yaml with the environment.

    Environment variables win over the YAML file, and .env has already been
    folded into the environment at import time, so the precedence is:
    exported shell variable, then .env, then the YAML file.
    """
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    broker = cfg.setdefault("broker", {})
    broker["api_key"] = (
        os.environ.get("ALPACA_API_KEY")
        or os.environ.get("APCA_API_KEY_ID")
        or broker.get("api_key", "")
    )
    broker["secret_key"] = (
        os.environ.get("ALPACA_SECRET_KEY")
        or os.environ.get("APCA_API_SECRET_KEY")
        or broker.get("secret_key", "")
    )
    paper_env = os.environ.get("ALPACA_PAPER")
    if paper_env is not None:
        broker["paper"] = paper_env.lower() in ("true", "1", "yes")
    else:
        broker.setdefault("paper", True)
    broker["base_url"] = os.environ.get(
        "ALPACA_BASE_URL") or broker.get("base_url")

    webhook = cfg.setdefault("webhook", {})
    webhook["passphrase"] = os.environ.get(
        "WEBHOOK_PASSPHRASE") or webhook.get("passphrase", "")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Alpaca paper/live webhook server")
    parser.add_argument("--config", default="configs/execution_config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--paper", action="store_true", default=None)
    parser.add_argument("--live", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("execution.server")

    cfg = load_config(args.config)
    if args.live:
        cfg["broker"]["paper"] = False
        log.warning("live trading mode enabled")
    elif args.paper is not None:
        cfg["broker"]["paper"] = True

    host = args.host or cfg.get("webhook", {}).get("host", "0.0.0.0")
    port = args.port or cfg.get("webhook", {}).get("port", 8000)
    passphrase = cfg.get("webhook", {}).get("passphrase", "")

    broker = cfg.get("broker", {})
    alpaca = AlpacaConfig(
        api_key=broker.get("api_key", ""),
        secret_key=broker.get("secret_key", ""),
        paper=broker.get("paper", True),
        base_url=broker.get("base_url"),
        timeout_seconds=broker.get("timeout_seconds", 10.0),
    )
    ledger_cfg = cfg.get("ledger", {})
    ledger = ExecutionLedger(
        db_path=ledger_cfg.get("db_path", "data/ledger.db"),
        jsonl_path=ledger_cfg.get("jsonl_path", "data/audit_log.jsonl"),
    )
    bridge = AlpacaExecutionBridge(config=alpaca, ledger=ledger)

    ap = cfg.get("agent_policy", {})
    policy = AgentPolicyConfig(
        gamma_speed=ap.get("gamma_speed", 0.35),
        cost_bps=ap.get("cost_bps", 10.0),
        impact_bps=ap.get("impact_bps", 2.0),
        alpha_hurdle_bps=ap.get("alpha_hurdle_bps", 5.0),
        max_position_pct=ap.get("max_position_pct", 0.20),
        max_order_notional=ap.get("max_order_notional", 50000.0),
        min_trade_notional=ap.get("min_trade_notional", 50.0),
        allow_short=ap.get("allow_short", False),
        default_base_alpha_bps=ap.get("default_base_alpha_bps", 30.0),
    )
    evaluator = CostAwareAgentEvaluator(
        bridge=bridge, config=policy, ledger=ledger)
    app = create_webhook_app(
        bridge=bridge, evaluator=evaluator,
        webhook_passphrase=passphrase, ledger=ledger,
    )

    mode = "paper" if alpaca.paper else "live"
    log.info(
        "listening on http://%s:%d/webhook (%s, %s, gamma=%.2f)",
        host, port, mode, alpaca.get_effective_base_url(), policy.gamma_speed,
    )
    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
