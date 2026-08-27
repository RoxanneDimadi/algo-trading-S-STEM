# Research Findings: What Actually Produces Real, Replicable Signals — and How Such Projects Are Built and Tested

*A survey of the empirical asset-pricing literature and open-source factor-research repositories, conducted to identify which signal families and project approaches have a documented track record of real, out-of-sample returns — while keeping the statistical work correct. This revision re-centers the project away from news sentiment (a heavily-mined, weak signal) toward the approaches with the strongest evidence of genuine productivity.*

---

## 0. The central finding (read this first)

If you ask "what has actually produced real signals and returns, holding the statistics to a high standard," the literature converges on an answer that is not glamorous but is robust:

**The most productive, replicable approach is to combine a modest set of well-documented signal families — anchored on momentum, liquidity, and volatility — using machine-learning models (gradient-boosted trees and neural networks) that capture the nonlinear interactions linear models miss.**

This is the headline result of Gu, Kelly & Xiu's *Empirical Asset Pricing via Machine Learning* (RFS 2020), the single most-cited and most-replicated paper in the area. Their finding, stated plainly: machine learning delivers large economic gains to investors, in some cases doubling the performance of leading regression-based strategies from the literature; the best methods are trees and neural networks; the gains come from allowing nonlinear predictor interactions missed by other methods; and crucially, all methods agree on the same set of dominant predictive signals, a set that includes variations on momentum, liquidity, and volatility.

Three things make this the right anchor for a serious project:

1. **It is real, not data-mined.** It holds out-of-sample, across decades, and has been independently replicated (there is a public Tidy Finance replication).
2. **It is buildable by a student** because the underlying data is open-source (see §3).
3. **The productive contribution — the ML combination step — is itself the impressive part,** and it can be demonstrated honestly at small scale by showing the ML model beats a linear benchmark on the same signals.

The rest of this document supports that conclusion: what the durable signals are, the honest truth about how much they've decayed, how such projects are structured and tested, and the data that makes it all reproducible.

---

## 1. Which signals have a real, out-of-sample track record

### 1.1 The dominant families

Across the ML asset-pricing literature, the same signal families surface as the dominant predictors regardless of method — **momentum, liquidity, and volatility.** This consensus is the most useful single fact for choosing what to build on, because it tells you where to spend effort:

- **Momentum** (cross-sectional 12-minus-1-month relative strength, plus time-series momentum) is the standout durable anomaly — it has survived out-of-sample, across markets, and across asset classes more robustly than almost anything else.
- **Liquidity** (illiquidity measures like Amihud, turnover, bid–ask spreads) is a persistent compensated risk.
- **Volatility** (idiosyncratic volatility, the low-volatility anomaly, beta) rounds out the trio.

A diversifying handful of **value** and **quality/profitability** characteristics is worth adding because they are in the replicable set and are lowly correlated with the momentum/liquidity/volatility core — which is exactly what helps a combined model.

### 1.2 The honest truth about decay (this is not optional context — it shapes the whole project)

Every one of these signals has weakened since it was published, and a credible project must say so. The canonical evidence is McLean & Pontiff, *Does Academic Research Destroy Stock Return Predictability?* The headline numbers, which you should be able to quote:

- Anomaly returns decline by roughly **26% out-of-sample** (after the original study's sample ends but before publication) and roughly **58% post-publication.**
- The gap between those two figures is informative: it splits the decay into a statistical/data-mining component (~26%) and an arbitrage/learning component (~32%), consistent with investors trading the signal away once it's public.

But — and this is the nuance that justifies building on these signals at all — **predictability persists out-of-sample; it does not vanish.** A re-analysis of the McLean–Pontiff predictors found that in the first three years after the original samples end, the mean return is still 74 bps per month against an in-sample 100 bps — i.e., about 74% of the signal remains. International evidence is even stronger: out-of-sample, international return predictability stays the same (or even goes up) after the end of the original sample, which the authors read as evidence the signals are real (arbitrage-driven decay) rather than pure data-mining.

Momentum is the concrete poster child for "decayed but real": a 2025 working paper on factor crowding characterizes it as returning approximately 10% annually in the 1990s versus closer to 2% today — a large haircut, yet a persistently positive, tradeable premium, and one where crowding is benign rather than prone to violent reversals. (Treat the specific magnitudes as one paper's estimate; the qualitative pattern is the robust part.)

**Implication for the project:** the goal is not to claim a fresh edge. It is to extract a real-but-modest, partially-decayed signal correctly, combine the families to recover what a single decayed signal loses, and characterize the decay honestly. That framing — "here is a real signal, here is exactly how much has been arbitraged away, here is what survives after costs and factor controls" — is precisely what a quant-research interviewer wants to see, and it is far more credible than a sentiment project claiming alpha.

---

## 2. How these projects are built (convergent structure)

The structural conventions are the same whether the signal is sentiment or momentum, so this survey of open-source factor-research repositories still applies in full. The architecture is remarkably convergent across authors and scales.

A representative skeleton:

```
project/
├── configs/                # experiment + backtest parameters
├── data/{raw,processed}/   # raw downloads → cleaned, point-in-time panel
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_signal_construction.ipynb
│   ├── 03_signal_evaluation.ipynb
│   ├── 04_model.ipynb              # combine signals
│   └── 05_backtest.ipynb
├── src/{signals,evaluation,backtest,utils}/
├── models/
├── requirements.txt
└── README.md               # the research note — the part people read
```

The **notebook ordering is the tell** and it recurs independently across projects: data prep → signal construction → **signal evaluation** → model → backtest. One repo runs `Data_Preparation → Alpha_Factor_Generation → Alpha_Factor_Selection → ML_Model → factor_backtest`; another `dataingestion → model_training → backtest`. A larger factor-mining system even makes hypothesis proposal a first-class pipeline component. The lesson: adopt the convergent layout; it's a solved problem and reviewers recognize it.

---

## 3. The data that makes this reproducible (the key enabler)

The single most important practical resource for building a *statistically correct* version of this project is **Chen & Zimmermann's Open Source Asset Pricing** dataset. It is what turns "real signals with correct stats" from aspiration into something a student can actually do:

- It provides data and code that successfully reproduces nearly all cross-sectional stock return predictors — on the order of 200+ published signals — with the replication quality documented against the original papers (for the 161 characteristics that were clearly significant in the original papers, 98% of the long-short portfolios find t-stats above 1.96).
- It is genuinely open and **actively maintained**: all datasets are open source, downloadable directly or via `pip install openassetpricing` (the package supports both pandas and polars), with the **latest data release in October 2025**. The maintainers take point-in-time correctness seriously — e.g., an October 2024 patch fixed a look-ahead bias in one announcement-return signal.
- One practical caveat: the package delivers **209 of the 212 characteristics**; three (Price, Size, and short-term reversal / `STreversal`) must come from CRSP, which requires academic WRDS access — or can be computed directly from a free price feed. Value-weighted portfolio construction likewise needs market cap. Plan for this on day one.
- It ships **multiple portfolio implementations** (equal- vs. value-weighted, NYSE vs. all-stock breakpoints, decile vs. quintile, price and financial-industry filters), so you can study how construction choices change results — itself a sophisticated thing to demonstrate.

One methodological gem built *on top of* this data — from **"Anomaly Time"** (Bowles, Reed, Ringgenberg & Thornock, *Journal of Finance*, Oct 2024), worth designing into the project: the academic convention of forming portfolios once a year in June underestimates predictability because it uses stale information; anomaly returns concentrate in the first month after information release, and forming portfolios immediately after information releases recovers signal (the typical academic rebalance arrives ~80 trading days after the information was released). Point-in-time discipline isn't just about avoiding leakage *forward* — using *stale* information leaks the other way and hides real signal.

Pair this with **Kenneth French's Data Library** for the factor returns used as controls, and you have a complete, free, reproducible foundation.

---

## 4. How the signal is tested (the statistics, done correctly)

The testing recipe is standardized across the credible literature and is the same regardless of which signal you're testing.

**The predictive test.** Regress **forward** returns on the **current** signal. Never contemporaneous. For cross-sectional predictiveness, the workhorse is the **Fama–MacBeth (1973)** two-pass regression: run a cross-sectional regression each period, then test whether the time-series average of the coefficients is reliably non-zero.

**The economic test (portfolio sort).** Sort the universe into quantiles by the signal, long the top quantile and short the bottom, and measure the spread. Report:

- Annualized long–short return **with a t-statistic**.
- **Sharpe ratio.**
- **Factor-model alpha** — the return surviving after controlling for known factors — with its own t-stat. This is the claim to genuine information.
- **Newey–West standard errors** for autocorrelation/heteroskedasticity.

**The signal-quality metrics.** Information Coefficient (**IC**, the rank correlation of signal with forward returns), **ICIR** (IC mean / IC volatility — *stability*), and **factor turnover** (a proxy for trading cost). Real single-signal ICs are *small* — a deep-learning factor study treats an IC of ~0.065 as a strong result. Anyone reporting 0.5 has a leak.

**The validation.** Purged/embargoed **walk-forward** out-of-sample evaluation, and a **deflated Sharpe ratio** to discount for the number of configurations tried.

---

## 5. Direct answers to the earlier questions (still relevant)

**Do most projects generate many signals, then select the best and model off them?** Yes — it's the dominant pattern, a funnel: generate candidates → score on IC/stability/turnover → keep survivors → combine. One project explicitly reduced features from 100+ candidates to 85 high-quality factors using statistical and ML methods. The ML model's job is *combination*, not rescuing an empty signal — which is why evaluation precedes modeling everywhere. In the productive (Gu–Kelly–Xiu) framing, the "many signals" are the documented characteristics and the "model" is the tree/NN that combines them.

**Do all projects use control / "extraneous" variables, or only the best ones?** The "extraneous variables" are control variables for known risk factors (Fama–French). Credible projects always control: they regress strategy returns on the known factors and report the surviving **alpha (intercept) with a t-stat** — proving the signal isn't a known factor in disguise. A generative-AI stock-selection paper shows alphas persist after controlling for common equity risk factors with R² of only 2.5% to 6.2%, indicating that 93-98% of returns are unexplained by the Fama-French factors — that low R² is the point. Careful work even time-shifts the Fama-French factors to match information timing so the control doesn't leak. Naive projects skip controls and report raw returns — and those "edges" are the ones that evaporate. Controls aren't reserved for winners; controlling is *how you determine* which signals are winners.

**Does a data source recur?** Yes, strongly. For replicated signals: **Open Source Asset Pricing** (Chen–Zimmermann); a second major open source is the **Jensen–Kelly–Pedersen global factor dataset** (from *Is There a Replication Crisis in Finance?*, JF 2023), which covers 150+ factors across 90+ countries. For factor/control data: **Kenneth French's Data Library**. For prices/fundamentals: CRSP/Compustat (via WRDS) at the academic tier; yfinance/Sharadar/Qlib for retail. For tooling, the Quantopian-lineage stack recurs more than any single data source — but note an important 2026 reality: **the original `alphalens`/`pyfolio`/`zipline` packages are unmaintained** (Quantopian shut down in 2020) and break on modern Python. The maintained versions are Stefan Jansen's forks — **`alphalens-reloaded`, `pyfolio-reloaded`, `zipline-reloaded`, `empyrical-reloaded`** — which are the ones to install; `vectorbt` and Microsoft `Qlib` are the common modern alternatives. `LightGBM` + `optuna` remain the default model + optimizer.

**Similarities across projects?** The structure (`configs/data/notebooks/src/models`, numbered notebooks), the funnel (generate → filter → combine), the test sequence (predictive Fama–MacBeth → quantile long/short → factor-controlled alpha), the metrics (IC/ICIR/turnover/Sharpe/alpha), the controls (French factors, time-shifted), the validation (purged walk-forward, deflated Sharpe), and the tooling are all shared. Projects differ mainly in *scale* and *how much anti-overfitting discipline is applied inside the funnel* — and that discipline is the variable that separates good from bad.

---

## 6. What's moved since GKX — the 2023–2026 frontier (interview awareness)

GKX (2020) remains the right *buildable anchor*, but a candidate for top firms should be conversant in what came after. Four developments matter:

**(a) The field now has a canonical survey.** Kelly & Xiu's *Financial Machine Learning* (2023) is the reference that organizes the whole area — worth reading before interviews, since it frames ML asset pricing the way practitioners now discuss it (including conditional autoencoders and portfolio-weight-learning approaches that generalize GKX).

**(b) The "virtue of complexity" debate — live and interview-relevant.** Kelly, Malamud & Zhou (*Journal of Finance* 2024) theoretically prove that simple models severely understate return predictability compared to "complex" models in which the number of parameters exceeds the number of observations — a provocative result arguing for heavily overparameterized models with shrinkage ("benign overfitting"). Critiques by Nagel (2025) and Buncic (2025) caution that apparent gains may reflect mechanical volatility-timing artifacts or restrictive implementation choices, with a Kelly–Malamud (2025) response clarifying nominal vs. effective complexity. This is an *active, unresolved* debate — knowing both sides signals you read the current literature, and it's a natural interview conversation starter. For the project itself, the safe methodological stance is the GKX-style comparison (does nonlinearity beat linear on identical inputs?), while *citing* the complexity debate as context.

**(c) Timing matters more than the old conventions assumed.** "Anomaly Time" (JF 2024) — see §3 — showed anomaly returns concentrate in the first month after information release and decay quickly, so portfolio-formation timing is itself a research variable. Building fresh-information timing into the pipeline is a current-generation touch.

**(d) LLM-based signal research is the newest wave.** Since Lopez-Lira & Tang's *Can ChatGPT Forecast Stock Price Movements?* (2023), a fast-growing literature uses LLMs to extract signals from text, and LLM-agent systems now automate parts of factor mining itself (e.g., systems that propose, code, and backtest candidate factors in a loop). Top firms are actively experimenting here. For a summer project this is best treated as a *stretch extension*, not the core: the statistical discipline is identical, and an LLM feature only means something if evaluated inside the same rigorous harness. A candidate who can say "my pipeline is signal-agnostic — here's where an LLM-derived feature would slot in, and here's the leakage risk specific to LLM training-data cutoffs" is speaking the current language. (Note the subtle leakage issue: an LLM trained on data through year Y "knows" outcomes when scoring year Y−1 text — a modern lookahead bias that interviewers love to probe.)

---

## 7. Reference shelf

**The productivity evidence (build on these):**
- Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS — ML combination beats linear; dominant signals are momentum, liquidity, volatility. (Public replication: Tidy Finance.)
- Chen & Zimmermann (2022), *Open Source Cross-Sectional Asset Pricing*, Critical Finance Review — 200+ replicated signals, open data + code (`openassetpricing.com`, `pip install openassetpricing`).

**The honesty layer (decay — quote these):**
- McLean & Pontiff (2016), *Does Academic Research Destroy Stock Return Predictability?*, JF — ~26% post-sample, ~58% post-publication decay.
- Chen & Zimmermann (2022), *Publication Bias in Asset Pricing Research* — predictability persists OOS (~74% remains in first 3 years).
- Falck, Rej & Thesmar, *When do systematic strategies decay?* — ex-ante predictors of which signals decay fastest.

**The 2023–2026 frontier (read to be conversant, cite as context):**
- Kelly & Xiu (2023), *Financial Machine Learning* — the canonical survey of the field.
- Kelly, Malamud & Zhou (2024), *The Virtue of Complexity in Return Prediction*, JF — plus the Nagel (2025) / Buncic (2025) critiques and the Kelly–Malamud (2025) response.
- Jensen, Kelly & Pedersen (2023), *Is There a Replication Crisis in Finance?*, JF — global factor evidence + open dataset (jkpfactors.com).
- Bowles, Reed, Ringgenberg & Thornock (2024), *Anomaly Time*, JF — formation timing and stale information.
- Lopez-Lira & Tang (2023), *Can ChatGPT Forecast Stock Price Movements?* — the LLM-signal starting point.

**The methodology canon:**
- Fama & French (factor models / controls); Fama & MacBeth (1973) (cross-sectional regression).
- Grinold & Kahn, *Active Portfolio Management* (IC, signal quality).
- López de Prado, *Advances in Financial Machine Learning* (purged CV, deflated Sharpe, multiple-testing).
- Jansen, *Machine Learning for Algorithmic Trading* (+ GitHub) — the alphalens/zipline/pyfolio workflow; use his maintained `-reloaded` package forks.

**Representative repositories (structure):** `OpenSourceAP/CrossSection` (the data + signal-construction code), `nuglifeleoji/Factor-Research` (generate→select funnel), `xalioh/alpha-signal-platform` (clean minimal pipeline). For firm-characteristic construction from raw WRDS data, `PyAnomaly` is a maintained Python option.

---

*Next document: `02_project_roadmap.md` — a concrete, scaffolded plan for a multi-signal cross-sectional return-prediction project anchored on the durable signal families and the ML-combination approach, built on Open Source Asset Pricing data with correct statistics throughout.*
