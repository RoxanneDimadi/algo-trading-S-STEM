"""POST a fake TradingView alert at a running webhook server."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import requests


def send_alert(url: str, payload: Dict[str, Any],
               passphrase: str = "") -> None:
    headers = {"Content-Type": "application/json"}
    if passphrase:
        payload = {**payload, "passphrase": passphrase}
        headers["X-Webhook-Secret"] = passphrase

    print(f"POST {url}")
    print(json.dumps(payload, indent=2))
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10.0)
        print(f"HTTP {resp.status_code}")
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:  # pylint: disable=broad-exception-caught
            print(resp.text)
    except requests.exceptions.RequestException as e:
        print(f"request failed: {e}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Send a test webhook")
    p.add_argument("--url", default="http://127.0.0.1:8000/webhook")
    p.add_argument("--passphrase", default="change_this_secret_passphrase")
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--action", default="buy",
                   choices=["buy", "sell", "flat", "close"])
    p.add_argument("--qty", type=float, default=10.0)
    p.add_argument("--price", type=float, default=225.50)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--order_type", default="market",
                   choices=["market", "limit", "stop"])
    p.add_argument("--limit_price", type=float, default=None)
    p.add_argument(
        "--scenario",
        choices=["buy", "sell", "close", "low_alpha_noise", "custom"],
        default="buy",
    )
    args = p.parse_args()

    if args.scenario == "buy":
        payload = {
            "ticker": args.symbol,
            "action": "buy",
            "quantity": args.qty,
            "price": args.price,
            "signal_strength": args.strength,
            "order_type": args.order_type,
            "limit_price": args.limit_price,
            "strategy": "tv_test",
        }
    elif args.scenario == "sell":
        payload = {
            "ticker": args.symbol,
            "action": "sell",
            "quantity": args.qty,
            "price": args.price,
            "signal_strength": -0.8,
            "order_type": "market",
            "strategy": "tv_test",
        }
    elif args.scenario == "close":
        payload = {
            "ticker": args.symbol,
            "action": "close",
            "price": args.price,
            "strategy": "tv_test",
        }
    elif args.scenario == "low_alpha_noise":
        payload = {
            "ticker": args.symbol,
            "action": "buy",
            "quantity": 1.0,
            "price": args.price,
            "signal_strength": 0.05,
            "strategy": "noise",
        }
    else:
        payload = {
            "ticker": args.symbol,
            "action": args.action,
            "quantity": args.qty,
            "price": args.price,
            "signal_strength": args.strength,
            "order_type": args.order_type,
        }

    send_alert(args.url, payload, args.passphrase)


if __name__ == "__main__":
    main()
