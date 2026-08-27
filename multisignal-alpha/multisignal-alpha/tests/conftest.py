"""Shared fixture: a small synthetic panel with known ground truth.

Kept small (240 months x 200 names) so the suite runs in seconds while still
having the statistical power to separate planted signals from the placebo.
"""
import pytest
import yaml

from src.data.synthetic import make_synthetic_panel
from src.utils.stats import rank_normalize_cross_section


@pytest.fixture(scope="session")
def cfg():
    with open("configs/config.yaml") as f:
        c = yaml.safe_load(f)
    c["data"]["synthetic"]["n_tickers"] = 200
    c["data"]["synthetic"]["start"] = "2003-01-31"
    # Small panel = less statistical power; strengthen the planted
    # interaction so the trees-vs-linear mechanism test is well-powered.
    c["data"]["synthetic"]["interaction"]["beta"] = 0.012
    c["data"]["synthetic"]["idio_vol_monthly"] = 0.07
    return c


@pytest.fixture(scope="session")
def world(cfg):
    panel, factors, meta = make_synthetic_panel(cfg, seed=7)
    panel = rank_normalize_cross_section(panel, list(meta.index))
    return panel, factors, meta
