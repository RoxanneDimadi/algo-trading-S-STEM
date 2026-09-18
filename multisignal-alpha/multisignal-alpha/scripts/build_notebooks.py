"""Build the four workflow notebooks.

Notebooks are generated from code so they stay in sync with src/ and never
rot as hand-edited JSON. Re-run after changing the pipeline:

    python3 scripts/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

SHIM = """\
# Path shim: make the repo root importable when running from notebooks/
import sys, pathlib
ROOT = pathlib.Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.width", 140)
"""

LOAD = """\
import yaml
from src.data.synthetic import make_synthetic_panel
from src.utils.stats import rank_normalize_cross_section

cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
panel, factors, meta = make_synthetic_panel(cfg, seed=cfg["run"]["seed"])
signal_cols = list(meta.index)
panel = rank_normalize_cross_section(panel, signal_cols)
print(panel["date"].nunique(), "months x", panel["ticker"].nunique(), "names")
meta
"""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [], "source": s}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NB1 = notebook([
    md("# 01 — Data & point-in-time leak checks\n\n"
       "Build the panel (synthetic by default: planted betas, known truth), "
       "then run the leak diagnostics **before any evaluation**. The rule "
       "(roadmap §0): point-in-time correctness in both directions — no "
       "forward leakage, no stale information."),
    code(SHIM),
    code(LOAD),
    md("## Predictive vs contemporaneous\n\nThe only tradeable number is the "
       "IC against **forward** returns. The contemporaneous column is shown "
       "to make the comparison explicit and auditable."),
    code("from src.data.panel import leak_report\n"
         "leak_report(panel, signal_cols)"),
    md("## The deliberate lookahead demonstration\n\nA feature contaminated "
       "with the return it claims to predict produces an *absurd* IC "
       "(~0.4 when honest single signals live near 0.02–0.06). If a real "
       "feature ever looks like this, audit the timestamps."),
    code("from src.data.panel import demonstrate_lookahead\n"
         "demonstrate_lookahead(panel, seed=cfg['run']['seed'])"),
    md("## Staleness profile ('Anomaly Time', JF 2024)\n\nIC as the signal "
       "goes stale. Fresh information should dominate; a steep drop means "
       "formation timing is itself a research variable."),
    code("from src.data.panel import staleness_experiment\n"
         "stale = staleness_experiment(panel, signal_cols, max_lag=6)\n"
         "stale.pivot(index='staleness_months', columns='signal', "
         "values='IC').round(4)"),
])

NB2 = notebook([
    md("# 02 — Per-signal evaluation & decay\n\nStatistical test (IC/ICIR, "
       "Fama–MacBeth) → economic test (quintile long-short, gross and net) → "
       "decay segmentation (McLean–Pontiff). Evaluation **precedes** "
       "modeling; the placebo `sig_dead` must fail."),
    code(SHIM),
    code(LOAD),
    md("## IC, ICIR, and the quintile long-short"),
    code("from src.evaluation.ic import mean_ic\n"
         "from src.evaluation.portfolio import "
         "evaluate_signal_portfolio, summary_row\n"
         "import pandas as pd\n\n"
         "rows, series = [], {}\n"
         "for c in signal_cols:\n"
         "    res = evaluate_signal_portfolio(panel, c,\n"
         "        n_q=cfg['evaluation']['n_quantiles'],\n"
         "        cost_bps_per_side=cfg['evaluation']['cost_bps_per_side'])\n"
         "    icr = mean_ic(panel, c)\n"
         "    row = summary_row(res); row.update(IC=icr['ic_mean'],\n"
         "        IC_t=icr['ic_tstat'], ICIR=icr['icir'])\n"
         "    rows.append(row); series[c] = res['series']['gross']\n"
         "tab = pd.DataFrame(rows).set_index('signal')\n"
         "tab.round(3)"),
    md("Read the table against the planted truth in `meta`: net Sharpe "
       "ordering should follow `true_beta`, and `sig_dead` should be "
       "indistinguishable from zero (|t| < 2)."),
    md("## Fama–MacBeth: marginal predictive power"),
    code("from src.evaluation.fama_macbeth import fama_macbeth\n"
         "fama_macbeth(panel, signal_cols).round(4)"),
    md("## Decay: in-sample vs post-sample vs post-publication"),
    code("from src.evaluation.decay import decay_table\n"
         "decay_table(series, meta).round(3)"),
    code("import matplotlib.pyplot as plt\n"
         "lead = tab['sharpe_gross'].idxmax()\n"
         "ls = series[lead]\n"
         "ax = ls.cumsum().plot(figsize=(9, 4), lw=1.4,\n"
         "    title=f'{lead}: cumulative L/S with decay markers')\n"
         "ax.axvline(meta.loc[lead, 'sample_end'], ls='--', "
         "color='orange', label='sample end')\n"
         "ax.axvline(meta.loc[lead, 'pub_date'], ls='--', "
         "color='red', label='publication')\n"
         "ax.legend(); plt.show()"),
])

NB3 = notebook([
    md("# 03 — ML combination vs linear benchmark (GKX in miniature)\n\n"
       "Purged walk-forward: elastic net vs LightGBM on **identical inputs**. "
       "The synthetic DGP plants a nonlinear interaction "
       "(momentum × value), so trees *should* win — and we can verify they "
       "win **for the right reason** via feature importances."),
    code(SHIM),
    code(LOAD),
    code("from src.backtest.engine import run_model_backtest, "
         "comparison_table\n"
         "from src.models.models import make_linear, make_lgbm\n"
         "from src.models.icnet import make_icnet\n\n"
         "wcfg, ecfg = cfg['walkforward'], cfg['evaluation']\n"
         "results = {\n"
         "    'elasticnet': run_model_backtest(panel, signal_cols,\n"
         "        lambda: make_linear(cfg['models']['linear']), wcfg, ecfg),\n"
         "    'lightgbm':  run_model_backtest(panel, signal_cols,\n"
         "        lambda: make_lgbm(cfg['models']['lgbm']), wcfg, "
         "ecfg),\n"
         "    'icnet':     run_model_backtest(panel, signal_cols,\n"
         "        lambda: make_icnet(cfg['models']['icnet']), wcfg, "
         "ecfg),\n"
         "}\n"
         "comparison_table(results).round(3)"),
    md("Note the turnover column: the tree model trades more, so its **net** "
       "edge is smaller than its gross edge. Costs are part of the answer, "
       "not a footnote."),
    code(
        "results['lightgbm']['feature_importance']"
        ".rename('importance').to_frame().round(3)"),
    code("import matplotlib.pyplot as plt\n"
         "for name, r in results.items():\n"
         "    r['series']['net'].cumsum().plot(label=f'{name} (net)', "
         "lw=1.6, figsize=(9, 4.5))\n"
         "plt.axhline(0, color='k', lw=0.6); plt.legend()\n"
         "plt.title('Out-of-sample, net of costs (purged "
         "walk-forward)'); plt.show()"),
])

NB4 = notebook([
    md("# 04 — Factor controls & deflated Sharpe (the credibility layer)\n\n"
       "Regress the net OOS strategies on the factor panel (Newey–West "
       "errors, formation-aligned timing) and deflate the headline Sharpe "
       "for every configuration examined."),
    code(SHIM),
    code(LOAD),
    code("from src.backtest.engine import run_model_backtest, "
         "comparison_table\n"
         "from src.models.models import make_linear, make_lgbm\n"
         "from src.models.icnet import make_icnet\n"
         "wcfg, ecfg = cfg['walkforward'], cfg['evaluation']\n"
         "results = {\n"
         "    'elasticnet': run_model_backtest(panel, signal_cols,\n"
         "        lambda: make_linear(cfg['models']['linear']), wcfg, ecfg),\n"
         "    'lightgbm':  run_model_backtest(panel, signal_cols,\n"
         "        lambda: make_lgbm(cfg['models']['lgbm']), wcfg, "
         "ecfg),\n"
         "    'icnet':     run_model_backtest(panel, signal_cols,\n"
         "        lambda: make_icnet(cfg['models']['icnet']), wcfg, "
         "ecfg),\n"
         "}\n"
         "comp = comparison_table(results); comp.round(3)"),
    md("## Factor-controlled alpha\n\nLow R² with surviving alpha = genuine "
       "information. Alpha that vanishes under controls = a known factor in "
       "disguise (reporting that is *also* a strong result)."),
    code("from src.evaluation.factor_controls import alpha_regression\n"
         "import pandas as pd\n"
         "pd.DataFrame([\n"
         "    {'model': name, **{k: v for k, v in\n"
         "        alpha_regression(r['series']['net'], factors).items()\n"
         "        if k != 'betas'}}\n"
         "    for name, r in results.items()\n"
         "]).set_index('model').round(3)"),
    md("## Deflated Sharpe (Bailey–López de Prado)\n\nTrials = every "
       "single-signal book examined + both models. Undercounting trials is "
       "how the deflation gets gamed — count honestly."),
    code("import numpy as np\n"
         "from src.evaluation.deflated_sharpe import deflated_sharpe\n"
         "from src.evaluation.portfolio import evaluate_signal_portfolio\n\n"
         "single_srs = [evaluate_signal_portfolio(panel, c)"
         "['net']['monthly_sharpe']\n"
         "              for c in signal_cols]\n"
         "trials = [r['net']['monthly_sharpe'] for r in "
         "results.values()] + single_srs\n"
         "best = comp['sharpe_net'].idxmax(); b = results[best]['net']\n"
         "deflated_sharpe(b['monthly_sharpe'], b['n_months'], b['skew'],\n"
         "                b['kurtosis'], trials)"),
    md("**Caveat that belongs in every writeup:** synthetic Sharpes are "
       "pedagogically inflated (the betas are planted and known). On real "
       "OSAP data expect numbers an order of magnitude humbler — that is "
       "the honest result the project is designed to report."),
])

if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    for name, nb in {
        "01_data_and_leak_checks.ipynb": NB1,
        "02_signal_eval_and_decay.ipynb": NB2,
        "03_ml_vs_linear.ipynb": NB3,
        "04_controls_and_dsr.ipynb": NB4,
    }.items():
        (NB_DIR / name).write_text(json.dumps(nb, indent=1))
        print("wrote", NB_DIR / name)
