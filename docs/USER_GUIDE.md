# Multi-Signal Alpha — Complete User Guide

How to set up, run, and — most importantly — *read* every part of this project: the four predictive models, the differentiable trading agent, the synthetic validation testbed, and the real-data factor mode. Written so that you (or a reviewer cloning the repo cold) can go from zero to interpreted results.

**Contents**
1. [What this project is](#1-what-this-project-is)
2. [Setup](#2-setup)
3. [The two repositories](#3-the-two-repositories)
4. [Running the test suite](#4-running-the-test-suite)
5. [Running the synthetic demo](#5-running-the-synthetic-demo)
6. [Getting real data without WRDS](#6-getting-real-data-without-wrds)
7. [Running the models on real data](#7-running-the-models-on-real-data)
8. [Reading every output](#8-reading-every-output)
9. [The four models](#9-the-four-models)
10. [The trading agent](#10-the-trading-agent)
11. [Configuration reference](#11-configuration-reference)
12. [What the results mean — and what they don't](#12-what-the-results-mean--and-what-they-dont)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What this project is

Multi-Signal Alpha is a cross-sectional equity-return research platform. It predicts next-month returns from documented signals, converts predictions into long-short portfolios, charges transaction costs, and closes the loop with a differentiable trading agent trained end-to-end on net-of-cost Sharpe. The deliberate claim is **not** "I found alpha" — it is: *here is a harness proven on data where the truth is planted, pointed at real data with every statistical correction applied, reporting honestly what survives.*

Three run modes, in order of evidential weight:

| Mode | Data | What it proves |
|---|---|---|
| `synthetic` | Simulator with planted signals | The harness works: finds planted betas, clears the placebo, screams at leaks |
| `panel_csv` (factor) | **Real** OSAP long-short portfolio returns | Real-data behavior of every model and the agent — no WRDS needed |
| `osap` | Firm-level OSAP signals + CRSP returns | The full Gu–Kelly–Xiu setting — **gated on WRDS/CRSP access** |

## 2. Setup

```bash
# agent repo
cd multisignal-alpha/multisignal-alpha
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# data ingest repo (separate venv keeps download deps out of the research env)
cd ../../real-data
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # Windows: copy .env.example .env
```

You do **not** need to edit `.env` for the no-WRDS path — the defaults work.

## 3. The two repositories

```
multisignal-alpha/multisignal-alpha/   the research agent (models, stats, agent)
  configs/config.yaml                  every tunable number — the single source of truth
  configs/config_factor.yaml           GENERATED runnable config for real factor data
  src/                                 pipeline, models, evaluation, agent
  tests/                               53 statistical-correctness tests
  results/          (synthetic runs)   tables/, figures/, summary.md
  results_factor/   (real-data runs)   same layout

real-data/                             ingest only — downloads never mix into research code
  configs/data_config.yaml             what to download and how to build panels
  scripts/run_all.py                   one-shot: OSAP + French + factor panel + sync
  scripts/build_factor_panel.py        the no-WRDS panel builder (+ real decay exhibit)
  scripts/validate_data.py             readiness report: which modes can run right now
  data/raw/                            downloads;  data/processed/  built panels

docs/                                  this guide + the proof-driven math lesson plan
  USER_GUIDE.md                        you are here
  math/                                every formula derived, chapter per file
  mathematical_foundations_lesson_plan.md   the same lesson plan as one document
```

The ingest repo *pushes* files into the agent's `data/raw/` and writes the agent's factor config, so the agent never reaches outward for data.

## 4. Running the test suite

```bash
cd multisignal-alpha/multisignal-alpha
make test        # or: python -m pytest -q tests/
```

Expect **53 passed**. These are not unit tests in the trivial sense — they are the statistical-correctness gate: the placebo signal must evaluate to noise, a deliberately leaked feature must produce an absurd IC, purged walk-forward splits must not overlap, the deflated Sharpe must penalize trial count, the agent's zero-cost limit must recover the MSRR closed form. If any of these fail after a change you made, the change broke the *science*, not just the code.

### Linting

```bash
make lint        # or: pylint --rcfile=../../.pylintrc src scripts tests
```

Code is PEP 8 (79-column lines) and pylint-clean at 10.00/10 under the
repository's `.pylintrc`. That file turns off a handful of checks, each with
its reason written next to it — the significant ones being short
mathematical variable names (`Z`, `LAM`, `R` — the symbols the derivations in
`docs/math/` use), scikit-learn's convention of defining fitted attributes in
`fit()`, and deliberate lazy imports of optional dependencies. Docstring
checks for functions and classes are off (that is PEP 257, not PEP 8);
module docstrings are still required.

GitHub Actions runs both gates — `pylint` and `pytest` — on every pull
request, from `.github/workflows/ci.yml` at the repository root.

## 5. Running the synthetic demo

```bash
make demo        # = python -m src.pipeline --config configs/config.yaml
```

Runtime: several minutes. Outputs go to `results/`. What to check, in order:

1. `results/tables/lookahead_demonstration.csv` — the leaky feature's IC should be ~0.40 vs ~0.03 for honest signals. If leakage ever creeps into real features, this is what it looks like.
2. `results/tables/signal_evaluation.csv` — per-signal ICs should order exactly by planted beta; `sig_dead` (the placebo) should show |t| < 2 and net Sharpe ≈ 0.
3. `results/tables/decay_analysis.csv` — recovered post-sample/post-publication haircuts should match the planted 0.70 / 0.45 multipliers.
4. `results/tables/model_comparison.csv` — LightGBM should beat the elastic net (it can exploit the planted momentum×value interaction; the linear model cannot), and PULSE should lead (planted time-varying betas are its home turf — a documented caveat, not a victory lap).
5. `results/summary.md` — everything above in one digest.

**Synthetic Sharpes are pedagogically inflated.** The betas are planted and known. Real-data numbers are an order of magnitude humbler — that is expected and is the point of running both.

## 6. Getting real data without WRDS

### Why firm-level mode is blocked, honestly

The firm-level path (`data.mode: osap`) needs monthly stock returns keyed by CRSP `permno`. CRSP returns are licensed; without WRDS there is **no legitimate free source** of permno-keyed firm returns, and free price feeds (Yahoo, Stooq) are ticker-keyed with no public permno crosswalk, plus survivorship bias. Rather than fake this, the project gates it and provides a real alternative.

### The factor mode: real data that is actually free

Chen & Zimmermann's Open Source Asset Pricing project freely distributes each published anomaly's **long-short portfolio return series** (~200 predictors, monthly, back to the 1920s for some). Factor mode treats *each anomaly as the tradable asset*:

- **ticker** = the anomaly (e.g. `Mom12m`, `BM`)
- **fwd_ret** = that anomaly's LS return next month
- **features** = point-in-time constructs from the factor's own history:

| Feature | Definition | Literature |
|---|---|---|
| `fmom_1m` | last month's LS return | short-horizon factor momentum |
| `fmom_12_2` | mean LS return over months t−11…t−1 | classic 12-2 momentum, applied to factors |
| `fmom_12m` | mean LS return over t−11…t | Ehsani–Linnainmaa factor momentum |
| `fvol_12m` | 12-month LS volatility | factor low-vol timing |
| `post_pub` | 1 strictly after publication | McLean–Pontiff decay |
| `years_since_pub` | years since publication, clamped at 0 | decay magnitude |

Point-in-time discipline: publication features activate only *after* the paper exists (no negative "years since pub" — that would leak the future), and a post-*sample-end* feature is deliberately excluded because the sample end only becomes public knowledge at publication. The reasoning is documented in `real-data/src/factor_panel.py`.

This is not a consolation prize — factor timing on the published-anomaly universe is a real research setting (Ehsani–Linnainmaa 2022; Gupta–Kelly 2019), and PULSE (a Kalman filter over time-varying signal efficacy) is *literally designed* for it.

### Commands

```bash
cd real-data
python scripts/run_all.py --skip-returns    # download OSAP + French, build panel, sync
python scripts/validate_data.py             # confirm: factor RUNNABLE
```

`configs/data_config.yaml` ships with `osap.portfolio_signals: all`, so the download covers the full predictor universe (~200 LS series). The panel builder auto-detects that OSAP ships returns in **percent** and converts to decimals (French factors are decimals — the units mismatch is handled for you and logged).

Two artifacts worth knowing about beyond the panel:

- `data/processed/factor_decay.csv` — the **real McLean–Pontiff exhibit**: each factor's annualized return and Sharpe in-sample vs post-sample vs post-publication, with retention ratios. The `post_sample_retention` / `post_pub_retention` columns are ratios of **annualized return**, not of Sharpe. On the currently downloaded five factors, Illiquidity retains only ~12% of its in-sample annualized return post-publication while BM retains ~77%; on Sharpe the same two are ~21% and ~63%. Measured, not simulated — and quote it with the units attached.
- `configs/config_factor.yaml` (written into the *agent* repo) — a complete runnable config. If fewer than 25 factors are on disk it is marked **SMOKE-TEST** in the header and relaxes statistical thresholds to their structural minimum; with the full universe it uses quintile sorts and a 20-name minimum per date.

### If you ever get returns anyway

Nothing WRDS-shaped was removed. Drop a `permno,yyyymm,ret[,dlret]` CSV anywhere and set `RETURNS_CSV` in `.env` (or set WRDS credentials), then `python scripts/run_all.py` — STreversal is constructed, the firm-level panel builds, and `data.mode: osap` unlocks. `validate_data.py` will flip `osap` to RUNNABLE.

## 7. Running the models on real data

```bash
cd multisignal-alpha/multisignal-alpha
python -m src.pipeline --config configs/config_factor.yaml
```

Runtime: a few minutes on the 5-factor panel; longer with the full universe. Outputs land in `results_factor/` with the same layout as the synthetic run. The pipeline stages, in the order they print:

1. **data** — panel shape sanity line (months × names × features).
2. **leak checks** — the same lookahead demonstration runs on real data: a deliberately leaked feature is planted and must still look absurd (IC ~0.27 vs ~0.00 in the verified run). If your honest features ever approach the leaky feature's IC, stop and audit.

   Read `leak_report.csv` with its `note` column. In factor mode `fmom_1m` **is** `ret_t` — last month's factor return, used as this month's signal — so its contemporaneous correlation with `ret` is exactly 1.0 by construction. That is not lookahead (`ret_t` is known at the end of month *t*), and the table says so in the note while suppressing the meaningless t-statistic. A correlation near 1.0 on **any other** signal is the alarm this table exists to raise: the note column flags it as `CHECK ALIGNMENT`.
3. **per-signal evaluation** — IC/ICIR and tercile/quintile long-short economics per feature.
4. **decay** — *skipped* in factor mode, by design: derived features like `fmom_12m` have no publication dates, so a McLean–Pontiff split on them is meaningless. The real decay exhibit lives in `real-data/data/processed/factor_decay.csv` instead.
5. **models** — purged walk-forward comparison of all four models.
6. **agent** — learned trading policy vs the myopic (γ=1) ablation.
7. **controls + DSR** — factor-controlled alpha (FF5+Mom) of each net strategy, deflated Sharpe of the best one.

## 8. Reading every output

All tables in `results*/tables/`, figures in `results*/figures/`, digest in `results*/summary.md`.

### signal_evaluation.csv
One row per feature. Columns and how to read them:

- **IC** — mean per-date Spearman rank correlation between the feature and next-month returns. On real data, 0.02 is weak, 0.05 respectable, 0.10+ strong for a single feature. (Verified factor run: `fmom_12m` IC ≈ 0.12.)
- **IC_t** — Newey–West t-stat of the mean IC. |t| ≥ 2 is the conventional bar; after multiple-testing awareness, treat 2–3 as suggestive, 3+ as solid.
- **ICIR** — mean IC / std IC: *stability* of predictive power. An erratic large IC is harder to monetize than a steady modest one.
- **ann_ret_gross / net, sharpe_gross / net** — annualized long-short economics before/after the configured cost (10 bps per side by default). The gross→net gap is the cost drag; compare it against turnover.
- **nw_t_gross / net** — Newey–West t-stats on the LS mean return.
- **one_way_turnover** — average fraction of the book traded per month. 1.0 means the whole book turns over monthly (e.g. `fmom_1m` ≈ 1.15 — fast signals pay for themselves or die by costs).
- **n_months** — periods with a valid cross-section. A feature like `post_pub` only has months where factors *differ* in publication status.

### fama_macbeth.csv
Multivariate per-date cross-sectional regressions: each feature's *marginal* power holding the others fixed. A feature with strong univariate IC but a dead FM coefficient is redundant with the others — that is a finding, not a failure.

### decay_analysis.csv (synthetic / dated signals) and real-data factor_decay.csv
In-sample vs post-sample vs post-publication annualized return and Sharpe, plus retention ratios. McLean–Pontiff's headline: ~26% decay post-sample, ~58% post-publication *on average*, with predictability persisting rather than vanishing. Compare your per-factor retentions against those anchors.

### model_comparison.csv
One row per model, all evaluated on identical purged walk-forward out-of-sample predictions:

- **oos_IC / oos_ICIR** — rank correlation of the model's *forecast* with realized returns. This is the cleanest model-quality number, before any portfolio construction choices.
- **sharpe_net, nw_t_net** — the economics of trading the forecast with costs.
- **n_oos_months** — out-of-sample months (the first `min_train` months are burn-in).

How to compare: elastic net is the *benchmark*; anything that can't beat it out-of-sample doesn't earn its complexity. On the verified real 5-factor run, IC-Net led (OOS IC 0.114, net Sharpe 0.28) with elastic net close behind, while LightGBM and PULSE trailed — a small cross-section starves both tree splits and per-date state estimation, which is the expected pattern, and the full-universe run is the real referendum.

### agent_vs_myopic.csv
The differentiable agent (learned trading speed γ) against the same aim traded myopically (γ=1). Read **mean_gamma** first: γ near 1 means costs are low enough that partial adjustment barely matters (the verified factor run: γ ≈ 0.95, agent ≈ myopic — an honest null at 10 bps on slow signals). The agent's value shows up as costs rise — `scripts/agent_cost_sweep.py` traces learned γ falling monotonically with cost, and at 100 bps the agent keeps alive a strategy that myopic rebalancing kills.

### factor_controls.csv
Each net strategy regressed on FF5 + momentum. **alpha_ann** is what survives exposure to known factors; **alpha_t** its NW t-stat; **r2** how much of the strategy is just repackaged factor beta. Insignificant alpha on real data (verified run: t ≈ 0.6–1.1) is the *honest* result — the strategies time known factors, so much of their return is explained by them.

### deflated_sharpe.csv
Bailey–López de Prado's DSR: the probability the best model's Sharpe exceeds what the *best of N tried configurations* would show by luck, given non-normality. Trials counted: every single-feature book + all four models + the agent (the paper trail is the config). DSR ≥ 0.95 is the conventional "probably real" bar; the verified run's 0.79 reads as "promising, not proven" — say exactly that.

### Figures
`single_signal_cumulative.png` (LS wealth curves per feature), `rolling_ic.png` (24-month IC stability — watch for regime breaks), `model_comparison_net.png` and `agent_vs_myopic_net.png` (cumulative net wealth), `pulse_efficacy_paths.png` (PULSE's filtered per-feature efficacy states over time — on synthetic data these visibly track the planted decay staircase; on real data they are the model's belief about which features currently work).

## 9. The four models

All trained per walk-forward fold on identical inputs; docs/04–05 hold the derivations.

1. **Elastic net** — the linear benchmark. Its job is to be beaten; when it isn't, that is the headline.
2. **LightGBM** — gradient-boosted trees, the Gu–Kelly–Xiu-style nonlinear learner. Earns its keep by finding interactions (verified on the planted momentum×value term via feature importances).
3. **IC-Net** — from-scratch NumPy net trained directly on a per-date cross-sectional correlation objective rather than MSE: it optimizes *ranking*, which is what a long-short book monetizes. Typically trades less than LightGBM for similar IC.
4. **PULSE** — Kalman filter over time-varying per-signal efficacies (state = "how well does each signal work *right now*"), motivated by McLean–Pontiff decay and factor momentum. Two-pass: per-date cross-sectional regressions produce noisy efficacy observations; the filter smooths them under a mean-reverting prior chosen on the training window.

## 10. The trading agent

The models end at forecasts; the agent ends at *positions*. It is a Gârleanu–Pedersen partial-adjustment policy — trade a learned fraction γ of the gap between current holdings and an aim portfolio — trained end-to-end on **net-of-cost Sharpe** with exact forward-mode gradients (no surrogate loss), with square-root market impact available. Two validated anchors: at zero cost it recovers the MSRR closed form (cosine similarity 1.000 — proof the machinery is right), and composing PULSE's point-in-time forecasts as its aim roughly doubled net Sharpe over a static blend on synthetic data (5.35 vs 2.96, `scripts/compose_pulse_agent.py`). Interpret γ as the cost dial: γ→1 trade fully, γ→0 sit still.

## 11. Configuration reference

Everything lives in `configs/config.yaml` (agent) and `configs/data_config.yaml` (ingest) — no tunable number is hardcoded, so the DSR's "configurations tried" accounting has a paper trail.

Key agent knobs:

| Key | Meaning | Default |
|---|---|---|
| `data.mode` | `synthetic` / `osap` / `panel_csv` | synthetic |
| `evaluation.n_quantiles` | LS sort buckets | 5 |
| `evaluation.cost_bps_per_side` | transaction cost | 10 |
| `evaluation.min_names_per_date` | min cross-section for stats/books (omit → per-module defaults 30/50/60) | unset |
| `walkforward.min_train / test_size / purge` | fold geometry; purge ≥ label horizon | 120 / 12 / 1 |
| `models.pulse.obs_margin` | per-date identification headroom; ≤0 = ridge-identified smoke mode | 10 |
| `agent.epochs / lr` | policy training | 150 / 0.05 |

Key ingest knobs: `osap.portfolio_signals` (`all` recommended), `factor_panel.lookback_months / min_names_per_month / units` (`auto` handles OSAP's percent units).

### A warning about the execution bridge

`src/execution/` places real orders. Two defaults matter:

- `AlpacaConfig.paper` is `True`, so you hit the paper endpoint unless you
  deliberately set it false.
- The webhook listener runs in **public mode with no authentication at all**
  when `WEBHOOK_PASSPHRASE` is unset — anyone who can reach the port can
  submit trade signals. It logs a warning at startup when that happens. Set a
  passphrase, or bind it to localhost, before exposing it anywhere.

## 12. What the results mean — and what they don't

What you **can** claim from a clean run:

- The harness is validated on ground truth: placebo clears, leaks scream, planted decay is recovered, DSR punishes trials. (Synthetic run.)
- On real factor data, trailing factor momentum carries statistically significant cross-sectional information (IC ~0.1, NW t > 5 in the verified 5-factor run), consistent with the factor-momentum literature — *and* survives 10 bps costs at t ≈ 2.3–2.6.
- Real McLean–Pontiff decay is measured, per factor, with retention ratios.
- The full stack — signals → models → costs → agent → controls → deflation — runs on real data with no step skipped.

What you should **not** claim:

- Live, deployable alpha. Factor-controlled alphas are insignificant; DSR < 0.95; the OP long-short portfolios are not directly tradable at these costs (they rebalance hundreds of underlying stocks).
- That synthetic Sharpes generalize. They are pedagogical by construction.
- That the 5-factor smoke run is evidence about model ranking. The full ~200-factor universe is the real comparison.

The strongest sentence this project supports is the one it was built for: *the methodology is airtight, demonstrated end-to-end on data where the truth is known, and reports real-data results with every honesty correction applied.*

## 13. Troubleshooting

- **`No returns file found`** — you ran the firm-level path without WRDS. Use `--skip-returns` and the factor mode; `validate_data.py` explains what is blocked and why.
- **All-NaN signal table / "cross-sections too small"** — your cross-section is below the statistical thresholds. Set `evaluation.min_names_per_date` (and for PULSE, `min_names` + `obs_margin: 0`) or use the generated `config_factor.yaml`, which sizes these automatically.
- **Units look wrong (Sharpes absurd)** — check `validate_data.py`'s units line; OSAP portfolios are percent, French factors decimals. The factor panel converts automatically; hand-built CSVs must be decimals.
- **`signals missing from OSAP file`** — `Price`, `Size`, `STreversal` are CRSP-licensed and never in the public dump; STreversal is built from returns when you have them.
- **A `make demo` run overwrote results you wanted** — synthetic writes to `results/`, factor mode to `results_factor/`; they never collide. Copy out anything you want to keep before re-running a mode.
- **Slow full-universe run** — LightGBM tuning off by default (`models.lgbm.tune: false`); keep it off for iteration, on for the final table.
