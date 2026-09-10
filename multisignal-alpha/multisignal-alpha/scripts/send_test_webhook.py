"""Helper script to simulate and dispatch TradingView webhook alerts."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import requests


def send_alert(url: str, payload: Dict[str, Any], passphrase: str = "") -> None:
    """Send JSON alert payload to the webhook endpoint and print the response."""
    headers = {"Content-Type": "application/json"}
    if passphrase:
        payload["passphrase"] = passphrase
        headers["X-Webhook-Secret"] = passphrase

    print(f"\n[>>>] Sending Webhook to: {url}")
    print(f"Payload:\n{json.dumps(payload, indent=2)}")

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10.0)
        x_req = resp.headers.get("X-Request-ID")
        print(f"\n[<<<] HTTP Status: {resp.status_code}")
        if x_req:
            print(f"[<<<] Alpaca X-Request-ID: {x_req}")
        try:
            resp_json = resp.json()
            print("Response JSON:\n" + json.dumps(resp_json, indent=2))
        except Exception:
            print(f"Raw Response: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"[!] Connection Error: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Send test TradingView alert webhook")
    parser.add_argument("--url", default="http://127.0.0.1:8000/webhook", help="Webhook receiver URL")
    parser.add_argument("--passphrase", default="change_this_secret_passphrase", help="Secret passphrase")
    parser.add_argument("--symbol", default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--action", default="buy", choices=["buy", "sell", "flat", "close"], help="Signal action")
    parser.add_argument("--qty", type=float, default=10.0, help="Explicit share quantity")
    parser.add_argument("--price", type=float, default=225.50, help="Simulated current price")
    parser.add_argument("--strength", type=float, default=1.0, help="Signal strength [-1.0 to 1.0]")
    parser.add_argument("--order_type", default="market", choices=["market", "limit", "stop"], help="Order type")
    parser.add_argument("--limit_price", type=float, default=None, help="Limit price (if limit order)")
    parser.add_argument("--scenario", choices=["buy", "sell", "close", "low_alpha_noise", "custom"], default="buy")

    args = parser.parse_args()

    if args.scenario == "buy":
        payload = {
            "ticker": args.symbol,
            "action": "buy",
            "quantity": args.qty,
            "price": args.price,
            "signal_strength": args.strength,
            "order_type": args.order_type,
            "limit_price": args.limit_price,
            "strategy": "TV_Momentum_Breakout_V2",
            "comment": "RSI breakout above 70 with volume surge",
        }
    elif args.scenario == "sell":
        payload = {
            "ticker": args.symbol,
            "action": "sell",
            "quantity": args.qty,
            "price": args.price,
            "signal_strength": -0.8,
            "order_type": "market",
            "strategy": "TV_Trend_Exit",
            "comment": "Trailing stop hit",
        }
    elif args.scenario == "close":
        payload = {
            "ticker": args.symbol,
            "action": "close",
            "price": args.price,
            "strategy": "TV_Rebalance_Flat",
            "comment": "Flattening position",
        }
    elif args.scenario == "low_alpha_noise":
        payload = {
            "ticker": args.symbol,
            "action": "buy",
            "quantity": 1.0,
            "price": args.price,
            "signal_strength": 0.05,  # Very weak signal (5% strength) -> should be filtered by cost hurdle
            "strategy": "Noise_Generator",
            "comment": "Marginal signal that should be filtered out by cost-aware agent",
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
