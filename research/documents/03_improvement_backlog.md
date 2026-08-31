# Improvement Backlog: What's Missing, What's a Known Limitation, What's Next

*A prioritized register of gaps in `multisignal-alpha` as shipped. P0 items are prerequisites for trusting any real-data result; P1 items strengthen the research story; P2 is engineering polish; P3 are the stretch extensions from the roadmap. Known-limitation items at the end are deliberate simplifications that should be disclosed, not silently fixed. Treating this list as part of the project — and saying so in interviews — is itself a signal of research maturity.*

---

## P0 — Required before any real-data result is trustworthy

**1. Validate the real-data loaders live.** `src/data/loaders.py` (OSAP wide CSV, CRSP returns, SignalDoc, French factors) is written with the correct signal-at-*t* → return-over-(*t*, *t*+1] alignment but has never touched real files (built in a network-restricted sandbox). First session with real data: check join row-counts against expectations, date ranges, the fraction of permno-months lost in the merge, and that SignalDoc's `SampleEndYear`/`Year` columns parse for every signal you use. Budget a full day; data plumbing always costs one.

**2. Universe filters.** The pipeline currently evaluates whatever the panel contains. Real CRSP data is dominated by microcaps whose "returns" are untradeable: without filters, every result is inflated. Add config-driven filters — minimum price (e.g., $5), minimum market cap or NYSE-percentile cutoff, share codes 10/11, exchange codes 1/2/3 — applied *before* quantile formation. This is the single largest gap between the synthetic demo and a defensible real-data number.

**3. Missing-data policy.** Synthetic data is complete; the OSAP panel is not (coverage varies enormously by signal and era). The current `dropna` approach silently shrinks the universe to stocks with *all* signals — a selection bias. Implement the GKX convention: impute missing characteristics with the cross-sectional median (i.e., rank-normalized value 0), and report per-signal coverage over time so the reader can see what the model actually saw.

**4. Delisting returns.** CRSP returns without delisting-return adjustment overstate performance (the classic survivorship-adjacent bias, worst for small/distressed names that short legs love). The returns loader should document the expectation that `ret` already incorporates delisting returns, and the WRDS pull script should construct it that way (merge `dlret`).

**5. Replication sanity check against OSAP's published portfolios.** Before trusting your own portfolio construction, reproduce Chen–Zimmermann's *published* long-short returns for 2–3 signals from their portfolio-returns files and compare against your quintile construction on the raw characteristics. Material disagreement means a construction bug, not a discovery. This is the cheapest available ground truth for real data and the natural first real-data notebook.

**6. Deflated Sharpe becomes meaningful only on real data.** On synthetic data the DSR saturates at 1.0 (planted betas, inflated Sharpes) — currently a decorative number. On real data, log *every* configuration examined (signals tried and discarded, both models, any optuna study) in a persistent trial register (a simple CSV appended by the pipeline) so the DSR's `n_trials` is an audit trail rather than a recollection.

---

## P1 — Strengthens the research story

**7. Quantile monotonicity.** Report all five quantile portfolio returns, not just the Q5–Q1 spread. A spread driven by one extreme bucket is fragile; monotonically increasing returns across quantiles is the pattern reviewers look for. Cheap: `score_to_weights` already computes the buckets — surface them.

**8. Robustness battery (roadmap weeks 10–11, not yet implemented).** Subperiod stability (halves, decades), sensitivity to quantile count (terciles/deciles), holding period (1/3/6 months — requires raising `purge` with horizon), and cost assumptions (5/10/25 bps). One summary table, one config sweep. The walk-forward and cost machinery already parameterize all of this; what's missing is the loop and the table.

**9. Formation-timing experiment (Anomaly Time).** `staleness_experiment` measures IC decay with signal age, but the full version compares *portfolios* formed on fresh vs. deliberately staled signals — the JF 2024 result reproduced in your own harness. Mostly a wrapper around existing pieces.

**10. Value-weighted and NYSE-breakpoint variants.** Only equal-weighted quintiles with all-stock breakpoints exist. EW overweights small caps and flatters most anomalies; showing EW vs. VW side by side (and how the premium shrinks) is exactly the construction-sensitivity sophistication the roadmap promised. Requires market cap — one more reason WRDS access is P0.

**11. Turnover drift correction.** `turnover_series` compares target weights at *t* vs. *t*−1, ignoring that *t*−1 weights drift with returns before rebalancing. Current version slightly misstates traded notional. Fix: drift prior weights by (1 + fwd_ret) and renormalize before differencing. Small, defensible, and a nice detail to be able to discuss.

**12. Complete the GKX model ladder.** Add a small feed-forward net (2–3 hidden layers, the GKX NN3 shape) to the ENet-vs-LGBM horse race, trained inside the same walk-forward. Also exercise the optuna path once (`tune: true`) and confirm the study is logged as trials for the DSR.

**13. Report OOS R² alongside IC.** GKX's headline metric is out-of-sample predictive R² (vs. a zero forecast). Adding it to `comparison_table` makes results directly comparable to the paper's ~0.3–0.4% monthly figures — and forces the honest observation that good strategies have tiny R².

**14. Multiple-testing control across signals.** The DSR deflates the *headline strategy*; the per-signal table has no adjustment. Add Benjamini–Hochberg FDR-adjusted q-values to `signal_evaluation.csv` (and note Harvey–Liu's t > 3 hurdle in the writeup).

---

## P2 — Engineering polish

**15. CI.** A GitHub Actions workflow running `pytest` on push (ubuntu, py3.11/3.12). The test suite is the repo's core claim; CI makes it visible. ~20 lines of YAML.

**16. Packaging.** A minimal `pyproject.toml` so `pip install -e .` works and notebooks lose the path shim. Add `ruff` config for lint/format consistency.

**17. Processed-panel caching.** `data/processed/` exists but is unused; the pipeline rebuilds the panel every run. Cache the aligned panel as parquet keyed on a config hash — matters once real data (minutes to load/join) replaces synthetic (seconds).

**18. Logging + run manifest.** Replace `print` with `logging`; write a run manifest (config hash, git SHA, timestamp, package versions) into `results/` so every table is traceable to the exact code and config that produced it. Reproducibility is the brand.

**19. Seed-robustness check.** One test or script that reruns the demo across 3–5 seeds and confirms conclusions (placebo fails, ordering holds, LGBM ≥ linear) are seed-stable, not one lucky draw.

---

## P3 — Stretch extensions (roadmap §8, in order of payoff)

**20. Decay predictors (Falck–Rej–Thesmar mini-study).** With ~10+ real signals, test whether ex-ante features (publication year, in-sample Sharpe, turnover) predict post-publication retention. Uses `decay_table` output as the dependent variable.

**21. LLM-derived feature.** One text-based feature scored by an LLM, evaluated in the identical harness — with the training-cutoff lookahead bias named and mitigated (score only post-cutoff periods, or use a time-appropriate model). The README should discuss this even before it's built; naming the bias is the differentiator.

**22. Second universe via JKP global factors.** Re-run the pipeline on one non-US region from jkpfactors.com to show the machinery generalizes and the premia travel.

**23. A complexity-debate experiment.** A contained Kelly–Malamud–Zhou-style exercise: random-Fourier-feature ridge with parameter count swept past the observation count, OOS performance vs. complexity plotted — plus the Nagel/Buncic critique acknowledged in the writeup. High interview value; keep it clearly separated from the core claim.

---

## Known limitations to disclose (choices, not bugs)

- **Flat transaction-cost model.** A single bps-per-side charge on traded notional; no market impact, no size/liquidity dependence, no bid-ask spread term. Fine for a research note if stated; a size-dependent cost curve is the upgrade path.
- **No shorting frictions.** The short leg pays no borrow fee and assumes full availability — most generous exactly where anomalies concentrate (small, illiquid names). At minimum, report the long leg standalone as a feasibility check.
- **Synthetic factor panel is noise (except market).** The factor-control machinery is correct, but the synthetic demo can't show alpha *dying* under controls because the planted signals are orthogonal to the fake factors by construction. The real French factors are where that test earns its keep.
- **Monthly frequency only.** No intramonth timing; the Anomaly Time experiment partially addresses this at monthly resolution.
- **Notebooks validated as concatenated scripts,** not through a Jupyter kernel — cosmetic differences (display, widths) possible on first interactive run.
- **`embargo` defaults to 0.** Defensible with purge ≥ horizon and monthly data, but say so rather than let a reader wonder.

---

*Suggested sequencing: items 1–6 in the first real-data week (they gate everything), 7–9 alongside the first real results, 10–14 before the writeup freeze, P2 continuously, P3 only after the core note is drafted. The honest framing for interviews: "here is what the project demonstrates, here is the register of what it doesn't yet — and here is why each item is ordered where it is."*
