"""Shared execution_config.yaml loading, with env vars overriding secrets.

Used by both the inbound webhook server (scripts/run_execution_server.py)
and the model-driven signal generator (scripts/run_agent_signals.py) so the
two entry points agree on broker credentials, paper/live mode, and the
agent's risk limits.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load_execution_config(config_path: str = "configs/execution_config.yaml") -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
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
    broker["base_url"] = os.environ.get("ALPACA_BASE_URL") or broker.get("base_url")

    webhook = cfg.setdefault("webhook", {})
    webhook["passphrase"] = os.environ.get("WEBHOOK_PASSPHRASE") or webhook.get("passphrase", "")
    return cfg
