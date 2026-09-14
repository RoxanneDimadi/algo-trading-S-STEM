"""Path helpers and config loading."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    load_env()
    cfg_path = Path(path) if path else ROOT / "configs" / "data_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    release = os.environ.get("OSAP_RELEASE", "").strip()
    if release:
        cfg.setdefault("osap", {})["release"] = int(release) if release.isdigit() else release

    signals = os.environ.get("OSAP_SIGNALS", "").strip()
    if signals:
        cfg.setdefault("osap", {})["signals"] = [s.strip() for s in signals.split(",") if s.strip()]

    french_start = os.environ.get("FRENCH_START", "").strip()
    if french_start:
        cfg.setdefault("french", {})["start"] = french_start

    returns_csv = os.environ.get("RETURNS_CSV", "").strip()
    if returns_csv:
        cfg.setdefault("returns", {})["csv"] = returns_csv

    agent_raw = os.environ.get("AGENT_RAW_DIR", "").strip()
    if agent_raw:
        cfg.setdefault("paths", {})["agent_raw"] = agent_raw
    return cfg


def resolve(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p)


def raw_dir(cfg: Dict[str, Any]) -> Path:
    d = resolve(cfg.get("paths", {}).get("raw", "data/raw"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_dir(cfg: Dict[str, Any]) -> Path:
    d = resolve(cfg.get("paths", {}).get("processed", "data/processed"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_raw_dir(cfg: Dict[str, Any]) -> Path:
    d = resolve(cfg.get("paths", {}).get("agent_raw", "../multisignal-alpha/multisignal-alpha/data/raw"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def agent_root(cfg: Dict[str, Any]) -> Path:
    # agent_raw is .../data/raw -> parent.parent is agent root
    return agent_raw_dir(cfg).parent.parent


def signal_list(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg.get("osap", {}).get("signals", []))
