"""Cost sweep: when does the learned trading speed earn its keep?

Trains the agent at several cost levels (single chronological split on the
full panel for speed; the walk-forward result at the config cost is the
headline -- this sweep maps the comparative static around it) and records:
the learned gamma, and the OOS net-Sharpe advantage over the myopic
gamma=1 control on the identical learned aim.

    python3 scripts/agent_cost_sweep.py
"""
import sys
import pathlib

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The project package lives one level up; the bootstrap above
# has to run before these imports resolve.
# pylint: disable=wrong-import-position
from src.agent.policy import DiffPolicyAgent, panel_to_matrices
from src.utils.stats import rank_normalize_cross_section
from src.data.synthetic import make_synthetic_panel


matplotlib.use("Agg")


COSTS = [0, 10, 25, 50, 100]
TRAIN_MONTHS = 180


def sharpe_ann(x):
    return float(np.sqrt(12) * x.mean() / (x.std() + 1e-12))


def main():
    with open(ROOT / "configs/config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    panel, _, meta = make_synthetic_panel(cfg, seed=cfg["run"]["seed"])
    sig = list(meta.index)
    panel = rank_normalize_cross_section(panel, sig)
    Z, Y, _dates, _ = panel_to_matrices(panel, sig)
    tr, te = slice(0, TRAIN_MONTHS), slice(TRAIN_MONTHS, None)

    rows = []
    for c in COSTS:
        a = DiffPolicyAgent(cost_bps_per_side=c, epochs=150,
                            seed=cfg["agent"].get("seed", 0)).fit(
                                Z[:, tr, :], Y[tr])
        ag = a.roll(Z[:, te, :], Y[te])
        my = a.roll(Z[:, te, :], Y[te], gamma=1.0)
        rows.append({"cost_bps": c, "learned_gamma": a.gamma_,
                     "agent_net_sharpe": sharpe_ann(ag["net"]),
                     "myopic_net_sharpe": sharpe_ann(my["net"]),
                     "advantage": (sharpe_ann(ag["net"])
                                   - sharpe_ann(my["net"])),
                     "agent_turnover": float(ag["traded"].mean() / 2),
                     "myopic_turnover": float(my["traded"].mean() / 2)})
    tbl = pd.DataFrame(rows).set_index("cost_bps")
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    tbl.to_csv(ROOT / "results/tables/agent_cost_sweep.csv")
    print(tbl.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(tbl.index, tbl["learned_gamma"], "o-")
    axes[0].set_xlabel("cost (bps per side)")
    axes[0].set_ylabel("learned gamma")
    axes[0].set_title("Trading speed falls with cost (GP)")
    axes[1].plot(tbl.index, tbl["advantage"], "o-", color="tab:green")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("cost (bps per side)")
    axes[1].set_ylabel("net Sharpe: agent - myopic")
    axes[1].set_title("Smoothing's payoff grows with cost")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/agent_cost_sweep.png", dpi=140)


if __name__ == "__main__":
    main()
