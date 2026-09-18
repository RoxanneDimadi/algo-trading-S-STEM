# multisignal-alpha

**A cross-sectional equity return-prediction project built to demonstrate quantitative-research discipline** — replicable signals, correct statistics, honest decay characterization, and an ML-vs-linear comparison in the Gu–Kelly–Xiu tradition.

The claim is deliberately *not* "I found alpha." The claim is: **here is a real, partially-decayed set of documented signals; here is exactly how much has been arbitraged away; here is what a nonlinear model recovers over a linear benchmark on identical inputs; and here is what survives factor controls, transaction costs, and multiple-testing deflation.** That framing is the job.

---

## Quickstart

```bash
pip install -r requirements.txt
make test     # 53 statistical-correctness tests (placebo, leak, purge, ...)
make demo     # full pipeline on synthetic data with PLANTED signals
```

`make demo` writes tables to `results/tables/`, figures to `results/figures/`, and a digest to `results/summary.md`. The four notebooks in `notebooks/` walk the same pipeline interactively (regenerate them anytime with `make notebooks`).

## Why the demo data is synthetic — and why that's a feature

The repo ships with a simulator (`src/data/synthetic.py`) that plants:

- five signals with **known monthly betas** (momentum-, liquidity-, volatility-, value-, quality-like), with McLean–Pontiff-style decay applied after each signal's synthetic sample-end and publication dates;
- one signal with **beta exactly zero** (`sig_dead`) — the placebo;
- one **nonlinear interaction** (momentum × value) that a linear model cannot capture and gradient-boosted trees should.

This is the validation step most projects skip: *prove the harness on data where the truth is known before pointing it at real data.* A harness that can't find a planted signal, can't clear a placebo, or doesn't scream at a deliberate leak is not a harness. The test suite (`tests/`) makes those checks permanent.

**Verified behavior on the shipped config** (your `make demo` will reproduce it):
the placebo evaluates to noise (|t| < 2, net Sharpe ≈ 0); per-signal ICs order
exactly by planted beta; the deliberate-leak feature produces an absurd IC
(~0.40 vs ~0.03 for honest signals); the decay module recovers the planted
post-publication haircut; and LightGBM beats the elastic net out-of-sample
because — per its feature importances — it exploits the planted interaction;
and IC-Net, a from-scratch NumPy model trained on a per-date cross-sectional
correlation objective (`docs/04_custom_model_proposal.md`), leads the
comparison with roughly half LightGBM's turnover; and PULSE, a from-scratch
Kalman-filtered efficacy model (`docs/05_pulse_model.md`) whose filtered
states provably track the planted decay staircase, leads the four-model
table (with the documented caveat that planted time-varying betas are its
home turf -- the real-data run is the actual test). A differentiable
trading agent (Garleanu-Pedersen partial adjustment trained end-to-end on
net Sharpe, `docs/06_trading_agent.md`) closes the loop from forecast to
position: its learned trading speed falls monotonically with costs, and at
100 bps it keeps a strategy alive that myopic rebalancing kills. At zero
cost the agent provably recovers the MSRR closed form (cosine 1.000), and
composing PULSE's point-in-time forecasts as its aim nearly doubles net
Sharpe over the static blend (5.35 vs 2.96; scripts/compose_pulse_agent.py).

> **Caveat that belongs in every conversation about this repo:** synthetic
> Sharpes are pedagogically inflated (the betas are planted and known). On
> real data expect numbers an order of magnitude humbler — reporting those
> honestly is the point of the project.

## Real data without WRDS: factor mode

Firm-level CRSP returns are license-blocked without WRDS, so `data.mode: osap`
stays a documented-but-gated path. The **supported real-data path** trades
OSAP's freely-distributed long-short *portfolio* returns instead: each of the
~200 published anomalies becomes a tradable asset, with factor-momentum and
publication-status features (the McLean-Pontiff angle, point-in-time safe).
This is the factor-timing setting of Ehsani-Linnainmaa / Gupta-Kelly -- and
PULSE's natural habitat on real data.

```bash
cd ../real-data
python scripts/run_all.py --skip-returns   # OSAP + French + factor panel
python scripts/validate_data.py            # readiness report per data mode
cd ../multisignal-alpha/multisignal-alpha
python -m src.pipeline --config configs/config_factor.yaml
```

Outputs land in `results_factor/`. See `../../docs/USER_GUIDE.md` for the complete
walkthrough and how to read every table.

## Repository map

```
configs/config.yaml        # every tunable number, in one auditable place
src/
  data/synthetic.py        # planted-signal DGP (the validation testbed)
  data/loaders.py          # OSAP + CRSP-returns + French-factor loaders
  data/panel.py            # point-in-time alignment + leak diagnostics
  evaluation/ic.py         # per-date rank IC, ICIR, rolling IC (NW inference)
  evaluation/portfolio.py  # quantile L/S, turnover, transaction costs
  evaluation/fama_macbeth.py
  evaluation/decay.py      # in-sample vs post-sample vs post-publication
  evaluation/factor_controls.py  # alpha regression, NW errors, R²
  evaluation/deflated_sharpe.py  # Bailey–López de Prado DSR
  models/models.py         # elastic-net benchmark, LightGBM, optional optuna
  models/icnet.py          # from-scratch NumPy net, per-date IC objective (docs/04)
  models/pulse.py          # Kalman-filtered time-varying signal efficacy (docs/05)
  agent/policy.py          # differentiable cost-aware trading policy (docs/06)
  agent/backtest.py        # walk-forward for the stateful agent (continuous book)
  agent/msrr.py            # closed-form Maximum Sharpe Ratio Regression (the aim)
  backtest/walkforward.py  # purged/embargoed splits with self-checks
  backtest/engine.py       # walk-forward train/predict → portfolio → costs
  pipeline.py              # end-to-end orchestrator (python -m src.pipeline)
notebooks/                 # 01 data+leaks, 02 eval+decay, 03 ML-vs-linear, 04 controls+DSR
tests/                     # the statistical-correctness gate
docs/                      # research findings + roadmap + backlog + model proposal
```

The user guide and the proof-driven lesson plan live at the repository root,
in `../../docs/`: `USER_GUIDE.md` and `math/` (every formula derived, with
predictions checked against `make demo` output).

## Methodology (the defensible core)

- **Predictive, lagged tests only.** Signal at *t* against the return over
  (*t*, *t*+1] — never contemporaneous. `leak_report` makes the comparison
  explicit; `demonstrate_lookahead` shows what a leak looks like so you
  recognize one.
- **Point-in-time in both directions.** No forward leakage, and no stale
  information either — the *Anomaly Time* (JF 2024) lesson; the staleness
  experiment measures IC as information ages.
- **Fama–MacBeth** for marginal cross-sectional power; **quintile long-short**
  (±$1, equal-weighted legs) for the economic test, gross **and net** of a
  per-side cost on traded notional.
- **Newey–West everywhere** a time-series mean is tested (returns, ICs, alphas).
- **Purged, embargoed walk-forward** (purge ≥ label horizon, enforced with a
  `ValueError`); all tuning inside the train window.
- **Factor-controlled alpha** with formation-aligned factor timing.
- **Deflated Sharpe** with honest trial accounting: every single-signal book
  examined plus every model counts as a trial.

## Anti-overfitting checklist

- [x] Point-in-time data, both directions (verified by tests + leak tables)
- [x] Predictive, lagged tests everywhere
- [x] Purged/embargoed walk-forward; tuning inside train only
- [x] Realistic transaction costs; results reported net
- [x] Factor-controlled alpha with Newey–West errors
- [x] Deflated Sharpe over the full set of configurations tried
- [x] Decay quantified per signal (in-sample / post-sample / post-publication)
- [x] ML gain attributed (feature importances), not just asserted
- [x] Placebo signal shipped and required to fail

## Moving to real data

Ingest lives in the sibling **`real-data/`** directory (not in this package):

```bash
cd ../real-data          # from repo root: real-data/
pip install -r requirements.txt
copy .env.example .env   # set RETURNS_CSV or WRDS_* as needed
python scripts/run_all.py
python scripts/sync_to_agent.py
```

That writes OSAP signals, SignalDoc, Ken French FF5+Mom, and (if available)
returns into `data/raw/`, and drops `configs/config_osap_overlay.yaml`.

1. **Signals** — `openassetpricing` subset matching config (Mom12m, Illiquidity,
   IdioVol3F, BM, GP). Or place `signed_predictors_dl_wide.csv` + `SignalDoc.csv`
   under `data/raw/` yourself.
2. **Returns** — `data/raw/returns.csv` with columns `permno, yyyymm, ret`
   (CRSP via WRDS or your own CSV). **Price, Size, and STreversal are not in
   the public OSAP file**; `real-data` builds STreversal as `-ret_{t-1}`.
3. **Factors** — prefer local `data/raw/french_factors.csv` from `real-data`;
   `load_french_factors()` falls back to `pandas-datareader` if missing.
4. Set `data.mode: osap` in `configs/config.yaml` (or merge the overlay) and
   re-run `make demo` / `python -m src.pipeline`.

The loaders (`src/data/loaders.py`) implement signal-at-*t* → return-over-
(*t*, *t*+1] alignment. Returns still need WRDS or a CSV you supply.

Tooling note: the original Quantopian libraries (alphalens/pyfolio/zipline)
are unmaintained; if you want tear sheets, install Stefan Jansen's
`-reloaded` forks in a separate environment (parts still prefer `numpy<2`).
This repo deliberately self-implements its evaluation machinery so every
statistical choice is visible and testable.

## The mathematics, derived

`../../docs/math/` (repo root `docs/math/`; start at `00_index.md`) is a nine-chapter, proof-driven lesson
plan assuming no finance background: correlation and the IC, portfolio
algebra, Newey-West, shrinkage, boosting, the IC-Net objective and its full
gradient derivation, purging, and the deflated Sharpe ratio. Its signature
move: because the demo data has planted truth, most derivations end with a
numerical prediction checked against pipeline output (e.g., momentum's gross
annual return: predicted 10.14%, measured 10.17%).

## References

Gu, Kelly & Xiu (2020) *Empirical Asset Pricing via Machine Learning*, RFS ·
Chen & Zimmermann (2022) *Open Source Cross-Sectional Asset Pricing*, CFR ·
McLean & Pontiff (2016) JF · Bowles, Reed, Ringgenberg & Thornock (2024)
*Anomaly Time*, JF · Bailey & López de Prado (2014) *The Deflated Sharpe
Ratio* · López de Prado (2018) *Advances in Financial Machine Learning* ·
Kelly & Xiu (2023) *Financial Machine Learning* · Jensen, Kelly & Pedersen
(2023) JF. See `docs/01_research_findings.md` for the annotated shelf,
including the 2023–2026 frontier (virtue-of-complexity debate, LLM signals).

## License

MIT
