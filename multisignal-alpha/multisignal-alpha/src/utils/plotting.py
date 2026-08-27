"""Figures for the research note. Matplotlib only, Agg backend (headless)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cumulative_ls(series_by_signal: dict[str, pd.Series], path: str,
                       title: str = "Single-signal long-short (gross, cumulative)"):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, s in series_by_signal.items():
        ax.plot(s.index, s.cumsum(), label=name, lw=1.4)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title)
    ax.set_ylabel("cumulative monthly return (sum)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_rolling_ic(rolling_by_signal: dict[str, pd.Series], path: str,
                    window: int):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, s in rolling_by_signal.items():
        ax.plot(s.index, s, label=name, lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Rolling {window}m mean IC (stability behind ICIR)")
    ax.set_ylabel("mean IC")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_decay(ls: pd.Series, sample_end, pub_date, path: str, signal: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ls.index, ls.cumsum(), lw=1.4, color="tab:blue")
    ax.axvline(pd.Timestamp(sample_end), color="tab:orange", ls="--",
               label="sample end")
    ax.axvline(pd.Timestamp(pub_date), color="tab:red", ls="--",
               label="publication")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{signal}: cumulative long-short with decay markers "
                 "(McLean-Pontiff pattern)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_model_comparison(net_by_model: dict[str, pd.Series], path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, s in net_by_model.items():
        ax.plot(s.index, s.cumsum(), label=f"{name} (net)", lw=1.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Out-of-sample combined strategies, net of costs "
                 "(purged walk-forward)")
    ax.set_ylabel("cumulative monthly return (sum)")
    ax.legend(fontsize=9)
    _save(fig, path)
