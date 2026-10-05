"""Glue: model forecasts -> agent aim weights -> the execution bridge.

Closes the loop the rest of the repo leaves open. Everywhere else, "the
agent" (`src/agent/policy.py`) and "the models" (`src/models/`) only ever
run inside a purged walk-forward *backtest* -- they are evaluated, never
deployed. And `src/execution/` only ever *receives* signals, from an
external source (a TradingView alert, `scripts/send_test_webhook.py`'s
fake one) that already names a real symbol and a price.

This module is what sits in between for OSAP/CRSP real data:

    latest cross-section of signals
        -> DiffPolicyAgent's learned theta (score -> dollar-neutral aim)
        -> permno -> tradable ticker symbol
        -> CostAwareAgentEvaluator.evaluate_signal (the existing risk gate,
           sized against the LIVE Alpaca position, not a backtest's)
        -> AlpacaExecutionBridge.submit_order

Only `theta` (the learned signal blend) is reused from the research agent.
`gamma` is deliberately NOT: the backtest's gamma is whatever was optimal
in-sample over the fit window, while live execution should use the
evaluator's own `gamma_speed` (an operator-set risk knob in
execution_config.yaml, reviewed independently of the research fit). See
`aim_weights` in `src/agent/policy.py` for the "aim = model, speed =
execution" split this mirrors (docs/06 Sec. 8b).

Honest limits, carried over from the research side (docs/06 Sec. 6): OSAP
signals are monthly and published with a real lag, so "live" here means
"the most recent complete month available," not intraday. There is no
permno -> ticker mapping in the OSAP/CRSP files themselves (CRSP license);
callers MUST supply one (`load_permno_ticker_map`) or every name is
skipped. MultiSpeedPolicyAgent is not wired here: its per-signal EMA state
only exists after rolling through history, which this module does not
carry between runs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..agent.policy import DiffPolicyAgent, aim_weights, panel_to_matrices
from .agent_evaluator import CostAwareAgentEvaluator
from .alpaca_bridge import AlpacaExecutionBridge

logger = logging.getLogger("execution.signal_generator")


@dataclass
class SignalGeneratorConfig:
    min_abs_aim_weight: float = 0.0     # drop names whose |aim weight| is below this
    max_names: Optional[int] = 50       # cap how many names get sent to the evaluator
    agent_epochs: int = 150
    agent_lr: float = 0.05
    agent_seed: int = 0
    cost_bps_per_side: float = 10.0     # fit-time cost; live hurdle is the evaluator's own
    min_train_months: int = 24
    strategy_name: str = "diffpolicy_agent"


def load_permno_ticker_map(path: str) -> Dict[str, str]:
    """permno (or whatever id the panel's `ticker` column holds) -> a real,
    tradable symbol. Required for OSAP data: the OSAP/CRSP files ship permno,
    not a ticker (CRSP license), so there is nothing to derive this from.

    Accepts a CSV with a permno-like column (permno/id/ticker_id) and a
    symbol-like column (symbol/ticker); case-insensitive.
    """
    df = pd.read_csv(path, dtype=str)
    cols = {c.lower().strip(): c for c in df.columns}
    key_col = cols.get("permno") or cols.get("ticker_id") or cols.get("id")
    sym_col = cols.get("symbol") or cols.get("ticker")
    if key_col is None or sym_col is None or key_col == sym_col:
        raise ValueError(
            "permno->ticker map needs a permno/id column and a distinct "
            f"symbol/ticker column; found columns {list(df.columns)}"
        )
    out: Dict[str, str] = {}
    for k, v in zip(df[key_col], df[sym_col]):
        if pd.isna(k) or pd.isna(v) or not str(v).strip():
            continue
        out[str(k).strip()] = str(v).strip().upper()
    return out


def latest_cross_section(
    panel: pd.DataFrame, signal_cols: List[str],
    date_col: str = "date", ticker_col: str = "ticker",
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """The most recent date's signals, as (K, N) -- independent of fwd_ret,
    which does not exist yet for that date (it hasn't happened)."""
    latest_date = panel[date_col].max()
    sub = panel.loc[panel[date_col] == latest_date].dropna(subset=signal_cols)
    if sub.empty:
        raise ValueError(f"No names with complete signals at the latest date {latest_date}")
    Z_t = sub[signal_cols].to_numpy(float).T
    tickers = sub[ticker_col].astype(str).to_numpy()
    return latest_date, Z_t, tickers


def fit_live_agent(
    panel: pd.DataFrame, signal_cols: List[str], config: SignalGeneratorConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit the agent's signal blend on every month with a realized forward
    return (i.e. everything except the live, undetermined, latest month)."""
    Z, Y, dates, _tickers = panel_to_matrices(panel, signal_cols)
    if Z.shape[1] < config.min_train_months:
        raise ValueError(
            f"Only {Z.shape[1]} months have a realized forward return; need "
            f">= {config.min_train_months} before the agent's blend is fit, "
            "not fit fresh on every live run."
        )
    agent = DiffPolicyAgent(
        cost_bps_per_side=config.cost_bps_per_side,
        epochs=config.agent_epochs, lr=config.agent_lr, seed=config.agent_seed,
    )
    agent.fit(Z, Y)
    meta = {
        "theta": dict(zip(signal_cols, agent.theta_.tolist())),
        "gamma_fit": agent.gamma_,
        "n_train_months": int(Z.shape[1]),
        "last_train_date": str(pd.Timestamp(dates[-1]).date()),
    }
    return agent.theta_, meta


def build_signals(
    panel: pd.DataFrame,
    signal_cols: List[str],
    theta: np.ndarray,
    config: SignalGeneratorConfig,
    permno_ticker_map: Optional[Dict[str, str]] = None,
    max_position_pct: float = 0.20,
) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Latest-cross-section aim weights -> webhook-shaped signal payloads.

    Returns (as_of, signals, skipped). `signals` is ready to hand one-by-one
    to `CostAwareAgentEvaluator.evaluate_signal`. `skipped` is an honest
    record of every name the model scored but that never reached the
    evaluator (unmapped permno, below threshold, or cut by max_names) --
    without it, a quiet universe-shrink would be invisible in the audit log.
    """
    as_of, Z_t, permnos = latest_cross_section(panel, signal_cols)
    aims = aim_weights(Z_t, theta)

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for permno, aim in zip(permnos, aims):
        if permno_ticker_map is not None:
            symbol = permno_ticker_map.get(str(permno))
            if symbol is None:
                skipped.append({"permno": permno, "symbol": None,
                                "aim_weight": float(aim),
                                "reason": "no permno->ticker mapping"})
                continue
        else:
            symbol = str(permno)
        rows.append({"permno": permno, "symbol": symbol, "aim_weight": float(aim)})

    rows.sort(key=lambda r: abs(r["aim_weight"]), reverse=True)

    kept: List[Dict[str, Any]] = []
    for row in rows:
        if abs(row["aim_weight"]) < config.min_abs_aim_weight:
            skipped.append({**row, "reason": "below min_abs_aim_weight"})
        else:
            kept.append(row)

    if config.max_names is not None and len(kept) > config.max_names:
        for row in kept[config.max_names:]:
            skipped.append({**row, "reason": "cut by max_names"})
        kept = kept[: config.max_names]

    signals = []
    for row in kept:
        aim = row["aim_weight"]
        strength = max(-1.0, min(1.0, aim / max_position_pct)) if max_position_pct > 0 else 0.0
        action = "flat" if abs(aim) < 1e-9 else ("buy" if aim > 0 else "sell")
        signals.append({
            "ticker": row["symbol"],
            "permno": row["permno"],
            "action": action,
            "signal_strength": strength,
            "aim_weight": aim,
            "strategy": config.strategy_name,
            "as_of": str(pd.Timestamp(as_of).date()),
        })
    return as_of, signals, skipped


def run_live_signals(
    panel: pd.DataFrame,
    signal_cols: List[str],
    evaluator: CostAwareAgentEvaluator,
    bridge: AlpacaExecutionBridge,
    config: Optional[SignalGeneratorConfig] = None,
    theta: Optional[np.ndarray] = None,
    permno_ticker_map: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Fit (or reuse) theta, turn today's cross-section into signals, run
    each through the existing risk-gated evaluator, and -- unless
    `dry_run` -- submit approved orders through the existing Alpaca bridge.

    Deliberately does not reimplement sizing, cost hurdles, position caps,
    or order submission: those stay exactly as tested in
    `agent_evaluator.py` / `alpaca_bridge.py`. This function only supplies
    the per-name `signal_strength` those already-trusted gates consume.
    """
    config = config or SignalGeneratorConfig()
    fit_meta: Dict[str, Any] = {}
    if theta is None:
        theta, fit_meta = fit_live_agent(panel, signal_cols, config)

    max_position_pct = getattr(evaluator.config, "max_position_pct", 0.20)
    as_of, signals, skipped = build_signals(
        panel, signal_cols, theta, config, permno_ticker_map, max_position_pct,
    )
    logger.info(
        "live signals as of %s: %d candidates, %d evaluated, %d skipped",
        as_of, len(signals) + len(skipped), len(signals), len(skipped),
    )

    results = []
    for raw in signals:
        decision = evaluator.evaluate_signal(raw, signal_source=config.strategy_name)
        row: Dict[str, Any] = {
            "symbol": decision.symbol,
            "permno": raw.get("permno"),
            "aim_weight": raw["aim_weight"],
            "action": raw["action"],
            "approved": decision.approved,
            "reason": decision.reason,
            "order_submitted": False,
        }
        if decision.approved and decision.final_directive and not dry_run:
            order = bridge.submit_order(decision.final_directive)
            row.update({
                "order_submitted": True,
                "order_status": order.status,
                "order_success": order.success,
                "alpaca_order_id": order.alpaca_order_id,
                "error_message": order.error_message,
            })
        results.append(row)

    return {
        "as_of": as_of,
        "fit_meta": fit_meta,
        "n_candidates": len(signals) + len(skipped),
        "n_evaluated": len(signals),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "results": results,
        "dry_run": dry_run,
    }
