# Project Roadmap: A Multi-Signal Cross-Sectional Return-Prediction Project

*Operationalizes the conventions in `01_research_findings.md` into a concrete, buildable plan — re-anchored on the approaches with a documented track record of real, out-of-sample returns. The goal remains a portfolio piece that demonstrates research discipline for a quantitative research role, with the statistical work correct throughout. The deliverable that matters is an honest research note backed by clean, reproducible code.*

---

## 0. Project thesis and framing

**What we're building.** A cross-sectional equity return-prediction system that combines a curated set of **documented, replicable signal families** — anchored on **momentum, liquidity, and volatility** — using a machine-learning model that captures their nonlinear interactions, validated with correct statistics, and characterized honestly for post-publication decay.

**Why this anchor.** This is the Gu–Kelly–Xiu result, the most-replicated finding in empirical asset pricing: ML (trees/neural nets) combining many real characteristics delivers genuine out-of-sample gains over linear methods, and the dominant signals are exactly momentum, liquidity, and volatility. Unlike news sentiment (weak and picked-over), these signals have a real, persistent — though partially decayed — track record. See `01_research_findings.md` §0–1.

**The honest contribution.** Not a fresh edge. The project demonstrates three things an interviewer actually values:
1. You can **replicate real signals correctly** using point-in-time data.
2. You can show the **ML-combination gain** over a linear benchmark on the same inputs (the GKX result, at small scale).
3. You can **characterize the decay honestly** — quantify how much of each signal's historical premium has been arbitraged away, and what survives after factor controls and realistic costs.

**The non-negotiable rule.** Point-in-time correctness, in *both* directions: never use information before it was public (forward leakage), and don't use stale information either (the "Anomaly Time" lesson, Bowles–Reed–Ringgenberg–Thornock, JF 2024: anomaly returns concentrate in the first month after information release, and the June-formation convention hides real signal by using stale data). Most fake edges are leakage; this gets paranoid attention at every stage.

---

## 1. Repository structure to scaffold

```
multisignal-alpha/
├── README.md                       # the research note (write continuously)
├── requirements.txt
├── configs/
│   └── config.yaml                 # universe, dates, costs, holding period, signal list
├── data/
│   ├── raw/                        # OSAP signals + French factors + price data
│   └── processed/                  # aligned date × ticker × signal panel
├── notebooks/
│   ├── 01_data_prep.ipynb          # ingest OSAP + French; align; define universe
│   ├── 02_signal_eval.ipynb        # per-signal IC/ICIR/turnover, quantile sorts, decay
│   ├── 03_model.ipynb              # ML combination vs. linear benchmark
│   └── 04_backtest.ipynb           # purged walk-forward, costs, factor attribution
├── src/
│   ├── data/                       # loaders, point-in-time alignment, universe
│   ├── signals/                    # signal selection + preprocessing (rank-normalize)
│   ├── evaluation/                 # IC, ICIR, turnover, portfolio sort, decay analysis
│   ├── models/                     # linear baseline + gradient-boosted trees
│   ├── backtest/                   # purged walk-forward engine + cost model
│   └── utils/
├── models/                         # saved artifacts
└── results/                        # tearsheets, figures, tables for the README
```

---

## 2. The signal set (small, durable, well-motivated)

Anchor on the documented dominant families, plus a couple of lowly-correlated diversifiers. Keep it to ~8–15 signals — enough for the ML combination to have something to work with, few enough to keep multiple-testing honest:

| Family | Example signals (all in the OSAP set) | Why included |
|---|---|---|
| **Momentum** | 12-1 month cross-sectional momentum; short-term reversal | The standout durable anomaly |
| **Liquidity** | Amihud illiquidity; share turnover | Persistent compensated risk |
| **Volatility** | Idiosyncratic vol; beta; low-vol | Robust across the ML literature |
| **Value** (diversifier) | Book-to-market; earnings yield | Lowly correlated with the core |
| **Quality** (diversifier) | Gross profitability; ROA | Lowly correlated; helps the combine |

All signals are **cross-sectionally rank-normalized into [-1, 1] period by period** (the standard GKX / Kelly–Pruitt–Su preprocessing) so the model sees comparable, outlier-robust inputs.

**Data caveat to plan for on day one:** the `openassetpricing` package delivers 209 of the 212 characteristics; **Price, Size, and short-term reversal (`STreversal`) come from CRSP**, which requires academic WRDS access. If you don't have WRDS: compute short-term reversal directly from a free price feed (it's just lagged 1-month return), use equal-weighted portfolios (avoiding the need for market cap), or swap in another OSAP momentum-family signal. None of these compromises the design.

---

## 3. Week-by-week plan (10–12 weeks)

**Weeks 1–2 — Foundations, data, point-in-time panel.**
- Read Gu–Kelly–Xiu (the method and the dominant-signal finding) and López de Prado on purged CV and the deflated Sharpe. Skim McLean–Pontiff for the decay numbers.
- Stand up the repo + `config.yaml`. `pip install openassetpricing`; pull the chosen signals and the French factors.
- Build the **point-in-time `date × ticker × signal` panel.** Verify both leakage directions: confirm forward shifts don't inflate IC, and confirm using fresh (not stale-June) signal timing where applicable. Document these checks in the README.

**Weeks 3–4 — Evaluation harness + backtester on a placebo.**
- Implement IC, ICIR, factor turnover, and quantile portfolio-sort utilities (or wire `alphalens-reloaded`).
- Implement the **purged/embargoed walk-forward** backtest engine and a transaction-cost model, and validate both on a random placebo signal (IC ≈ 0, Sharpe ≈ 0 net). A correct backtester is itself the impressive artifact.

**Weeks 5–6 — Per-signal evaluation + decay analysis (the honesty core).**
- For each signal: predictive test (forward return on current signal), IC, **ICIR** (rolling-window stability), turnover, and the **quantile long/short** economic test.
- **Decay analysis:** for each signal, plot the cumulative long/short return with the original sample-end and publication dates marked, and compare in-sample vs. post-sample vs. post-publication Sharpe. This is the sophisticated, honest section that sets the project apart — and the OSAP demo code does almost exactly this.

**Weeks 7–9 — The ML combination (the productive contribution).**
- **Linear benchmark first:** combine the signals with a simple linear model (e.g., a cross-sectional regression / elastic net) and record its out-of-sample long/short performance.
- **Then the ML model:** train **gradient-boosted trees (LightGBM)**, tuned with `optuna` *inside* the purged walk-forward (no peeking), to predict forward returns from the signal panel. Form the long/short portfolio on the model's forecast.
- **The headline comparison:** ML vs. linear on identical inputs. Demonstrating the ML model's out-of-sample gain — and attributing it to nonlinear interactions — is the small-scale replication of the GKX result and the centerpiece of the writeup. Report feature importances for interpretability.

**Weeks 10–11 — Robustness, write-up, polish.**
- Robustness battery: subperiod stability, sensitivity to holding period / quantile cutoffs / cost assumptions; **factor-controlled alpha** (regress the strategy on time-shifted French factors, Newey–West errors, report alpha + t-stat + R²); **deflated Sharpe** for the configs tried.
- Write the README as a research paper: thesis → data + point-in-time methodology → per-signal evaluation + decay → ML-vs-linear combination → factor-controlled, cost-adjusted results → honest conclusion.
- Clean `src/`, pin `requirements.txt`, make notebooks run top-to-bottom, export figures to `results/`.

**Week 12 — Buffer / stretch** (§7).

---

## 4. Methodology specifics (correct statistics)

- **Predictive, lagged test** — forward return on current signal; never contemporaneous.
- **Cross-sectional regression** — Fama–MacBeth for per-signal predictiveness; ML model for the combined forecast.
- **Portfolio construction** — quintile sort, long top / short bottom, equal-weighted within legs; report gross *and* net of costs. (Optionally show value-weighted and NYSE-breakpoint variants, which OSAP provides, to demonstrate construction sensitivity.)
- **Controls (credibility multiplier)** — regress strategy returns on the French five factors (+ momentum), **time-shifted to match information timing**, Newey–West SEs; report the **alpha (intercept), its t-stat, and R²**. Low R² with surviving alpha = genuine information; alpha that vanishes under controls = a known factor in disguise (and reporting *that* is still a strong result).
- **Validation** — purged/embargoed walk-forward; deflated Sharpe.

---

## 5. Anti-overfitting checklist (the part that signals seniority)

A gate the project must pass:

- [ ] **Point-in-time data, both directions** — no forward leakage, no stale-information leakage (verified).
- [ ] **Predictive, lagged** test everywhere — no contemporaneous correlation.
- [ ] **Purged / embargoed walk-forward** CV — model tuning happens inside the walk-forward, never on the test period.
- [ ] **Realistic transaction costs + slippage** — results reported net.
- [ ] **Factor-controlled alpha** with Newey–West errors — beta isn't alpha.
- [ ] **Deflated Sharpe ratio** — discount for the number of models/configs tried.
- [ ] **Decay quantified** — in-sample vs. post-sample vs. post-publication, per signal.
- [ ] **ML gain attributed, not just asserted** — beat a linear benchmark on identical inputs; show feature importances.
- [ ] **Honest negative results documented** — what didn't survive is part of the write-up.

---

## 6. Deliverables and success criteria

**Deliverables:** a clean reproducible repo; a README research note written as a paper (with the IC/decay/ML-vs-linear tables and figures); a "methodology and pitfalls" section foregrounding point-in-time handling and the anti-overfitting checklist.

**Success is NOT a high backtest Sharpe.** Success IS:
- The pipeline is correct and point-in-time clean (demonstrable, both directions).
- The signals are real and replicated, and you've **quantified their decay** honestly.
- You've shown the **ML-combination gain over a linear benchmark** out-of-sample and explained *why* it exists.
- The final numbers are **factor-controlled and cost-adjusted**, and you can speak fluently in an interview about every defensive step and what it caught.

A project concluding "these documented signals have decayed ~50% post-publication but a meaningful premium persists; combining them with gradient-boosted trees recovers X% out-of-sample over a linear benchmark; after factor controls and costs the surviving alpha is small but statistically real" is a **strong** outcome. It proves you can extract and validate genuine signal with correct statistics — which is the job.

---

## 7. Data sources and tools (defaults)

- **Replicated signals:** Open Source Asset Pricing — `pip install openassetpricing` (Chen–Zimmermann; supports pandas and polars; **latest data release October 2025**). The foundation. For a global/international extension, the Jensen–Kelly–Pedersen dataset (jkpfactors.com) is the other major open source.
- **Risk factors (controls):** Kenneth French Data Library.
- **Prices/returns:** CRSP at the academic tier (needed for Price/Size/STreversal — see §2 caveat); yfinance/Sharadar for supplementary retail data.
- **Tooling — use the maintained forks:** the original Quantopian packages are unmaintained and break on modern Python, so install **`alphalens-reloaded`** (factor evaluation), **`pyfolio-reloaded`** (risk tearsheets), and **`empyrical-reloaded`** — Stefan Jansen's maintained forks — plus `pandas`/`numpy` and `LightGBM` + `optuna` (the model + tuner). For a batteries-included backtest engine, **`zipline-reloaded`** (research-grade, steeper setup), `vectorbt` (fast, vectorized), or Microsoft `Qlib`. Pin versions in `requirements.txt`; some of this stack still prefers `numpy<2`, which is exactly why pinning matters.

---

## 8. Stretch goals (only after the core is solid)

- **Neural-net comparison:** add a small feed-forward net to the ML-vs-linear horse race, mirroring GKX's full method ladder.
- **Decay predictors:** test whether *ex-ante* features (e.g., publication year, in-sample Sharpe) predict which signals decayed fastest — a Falck–Rej–Thesmar-style mini-analysis.
- **Formation-timing experiment (current-generation touch):** following "Anomaly Time" (JF 2024), compare portfolios formed immediately after information releases vs. the stale annual convention, and show the difference in measured predictability.
- **An LLM-derived feature (the 2025–2026 frontier):** add one text-based feature scored by an LLM and evaluate it inside the *identical* harness as every other signal. If you do this, address the modern lookahead bias head-on: an LLM trained through year Y "knows" outcomes when scoring year Y−1 text, so restrict scoring to post-training-cutoff periods or use a time-appropriate model. Naming and mitigating this bias is itself an interview differentiator.
- **Capacity / crowding awareness:** discuss how trading costs and capacity erode the paper premium, sizing positions accordingly.
- **Second universe:** re-run on international data (via the JKP global dataset) or crypto to show the pipeline generalizes.

---

## 9. Reference shelf

From `01_research_findings.md` §7. The four to read *before* writing code:
1. Gu, Kelly & Xiu (2020) — the method and the dominant-signal finding (the project's backbone).
2. Chen & Zimmermann — Open Source Asset Pricing (the data + how to construct signals correctly).
3. McLean & Pontiff (2016) — the decay numbers (the honesty layer).
4. López de Prado, *Advances in Financial Machine Learning* — purged CV + deflated Sharpe (the anti-overfitting core).

One to read *before interviews*: Kelly & Xiu (2023), *Financial Machine Learning* — the survey that frames how the field currently talks, including the live "virtue of complexity" debate (Kelly–Malamud–Zhou 2024 vs. Nagel 2025 / Buncic 2025). See `01_research_findings.md` §6 for that frontier context.

---

*Sequencing note: build the point-in-time panel and the purged backtester first (Weeks 1–4), before modeling. Then the project's spine is the honest three-step story — real signals → quantified decay → ML-combination gain over a linear benchmark — every step factor-controlled and cost-adjusted. That spine is what an interviewer will probe, and what makes this more than a tutorial follow-along.*
