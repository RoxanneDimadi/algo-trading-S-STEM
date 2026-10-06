# Improvement Log, 2026-09-17

Every change made in this pass, why it was made, and how it was verified.
State at the end: **53/53 agent tests passing, no regressions**, and the full
pipeline runs end to end on real market data for the first time
(`results_factor/`).

## 1. Real-data connection reworked for the no-WRDS case

**Problem.** `data.mode: osap` was structurally dead without WRDS. It
requires `returns.csv` (CRSP firm-level, license-blocked), so the entire
real-data half of the project could never run. There is no legitimate free
source of permno-keyed firm returns, and ticker-keyed free feeds can't merge
with the permno-keyed OSAP file.

**Solution: factor mode.** OSAP *does* freely distribute every published
anomaly's long-short portfolio returns. The new pipeline treats each anomaly
as the tradable asset, with point-in-time features from its own history
(factor momentum and vol) plus publication-status features, which is the
McLean-Pontiff angle. This is the Ehsani-Linnainmaa / Gupta-Kelly
factor-timing setting rather than a workaround, and it is where PULSE
belongs on real data.

New and changed in `real-data/`:

- `src/factor_panel.py` (new): LS extraction, **automatic percent to decimal
  unit conversion** (OSAP ships percent and French ships decimals, a latent
  footgun that is now handled and logged), point-in-time features with the
  leak reasoning written down (publication features clamp to zero before
  publication; post-sample-end is deliberately excluded as a feature, because
  the sample end only becomes public knowledge at publication), and
  `factor_decay_table()`, which is the **real McLean-Pontiff decay exhibit**
  per factor.
- `scripts/build_factor_panel.py` (new): builds the panel, meta and decay
  exhibit, syncs them to the agent, and generates a complete runnable agent
  config.
- `scripts/validate_data.py` (new): schema, coverage, NaN and unit checks on
  every data file, plus a readiness verdict per mode (synthetic, factor,
  osap) that explains the WRDS gating.
- `src/osap_download.py`: `portfolio_signals: all` downloads the full
  ~200-predictor LS universe, with a SignalDoc-driven fallback for older
  `openassetpricing` versions. The config default is now `all`.
- `scripts/run_all.py`: the factor panel is wired in as the default
  real-data step (`--skip-factor-panel` opts out), and `--skip-returns` now
  yields a fully working real-data path instead of a warning.
- `src/panel_build.py`: `write_agent_factor_config()` generates
  `configs/config_factor.yaml` from the agent's own base config, so that
  file stays the single source of truth. It sizes `n_quantiles` and the
  thresholds to the cross-section and stamps a loud SMOKE-TEST header when
  there are fewer than 25 factors. `sync_to_agent()` now **skips unchanged
  files**, where before it rewrote the 450 MB OSAP CSV on every run.
- `configs/data_config.yaml`: a new `factor_panel:` section (lookback, min
  names, units) and `osap.portfolio_signals`.

New and changed in the agent repo:

- `src/data/loaders.py`: `load_prebuilt_panel()`, a generic
  `data.mode: panel_csv` entry point. Today it takes the factor panel; a
  WRDS firm panel would drop in later with no change to the agent.
- `src/pipeline.py`: `panel_csv` mode, and the decay stage now **skips with
  an explanatory note** when features carry no publication dates. Derived
  features like `fmom_12m` have none, and the real per-factor decay lives in
  `real-data/data/processed/factor_decay.csv`.

**Verified end to end on real data** (5 factors on disk at the time; the
full universe follows a re-download): leak demo IC 0.27 against 0.00 for the
honest control; factor momentum IC around 0.11 to 0.12 with NW t around 5.9;
net Sharpe 0.23 to 0.28 at 10 bps; IC-Net leads the OOS model table; agent
gamma around 0.95, which is effectively myopic and an honest null at low
cost; factor-controlled alphas insignificant; DSR 0.79. On the decay
exhibit, note that the retention columns are ratios of annualized *return*,
not Sharpe: Illiquidity keeps about 12% of its in-sample annualized return
after publication and BM about 77%, while on Sharpe the same two are about
21% and 63%.

## 2. Hardcoded statistical thresholds made config-driven

The `min_names` minimum cross-section sizes were hardcoded in five places
(ic 30, portfolio 50, PULSE 60, IC-Net 30, and implicitly in the engine).
That broke the project's own "every tunable in config" rule, and it made any
cross-section under about 50 names silently NaN or crash. They now run
through one optional knob, `evaluation.min_names_per_date`, threaded into
`ic.py`, `portfolio.py`, `data/panel.py`, `backtest/engine.py` and both
model factories. Module defaults are unchanged when the knob is unset, so
all prior behavior and every test are preserved. PULSE also gains
`obs_margin`, the identification headroom `n >= P + margin`, where a value
of 0 or less is documented ridge-identified smoke mode. `ic_series` now
skips constant cross-sections cleanly instead of spraying scipy warnings.

## 3. Dead weight removed and prevented

- `real-data/.venv/`, **534 MB**, was shipped inside the project archive. It
  was already gitignored; the zip simply included it. Deleted from the
  working copy. Environments are conda now, so there is nothing to recreate
  in the tree: `conda env create -f environment.yml`. Any leftover `.venv/`
  on your machine is safe to delete.
- All `__pycache__/` and `.pytest_cache/` directories removed.
- The 443 MB OSAP wide CSV is duplicated by design (one ingest copy, one
  agent copy) so the agent repo stays self-contained. The sync-skip change
  above at least stops it being rewritten every run. If disk space matters,
  delete `real-data/data/raw/signed_predictors_dl_wide.csv` after syncing
  and `download_osap.py` will restore it on demand.
- `coverage.xml` at the repo root is a build artifact and already
  gitignored, so it is safe to delete at any time.

## 4. Documentation corrected and completed

- **`docs/USER_GUIDE.md`** (new, in `docs/` at the repository root): the
  complete use guide. Setup, both repos, the test suite, the synthetic demo,
  the no-WRDS data story, running on real data, how to read *every* output
  table and figure, the four models, the agent, the config reference, what
  the results claim and don't, and troubleshooting.
- Agent `README.md`: the stale "17 tests" is now 53, and there is a new
  "Real data without WRDS" section with the four-command path.
- `real-data/README.md`: the run section is rewritten around the factor path
  and `validate_data.py`.
- This file.

## 5. Applying these changes to your local copy

The archive `msa_improvements.zip` contains only new and changed files, in
their correct relative paths. From the directory that contains
`multisignal-alpha/` and `real-data/`:

```powershell
Expand-Archive -Path msa_improvements.zip -DestinationPath . -Force
# unix: unzip -o msa_improvements.zip
cd real-data; python scripts/validate_data.py
# unix: cd real-data && python scripts/validate_data.py
cd ../multisignal-alpha/multisignal-alpha; make test    # expect 53 passed
# unix: cd ../multisignal-alpha/multisignal-alpha && make test
```

Then re-download with the full universe when convenient
(`python scripts/download_osap.py`, then
`python scripts/build_factor_panel.py`) and run
`python -m src.pipeline --config configs/config_factor.yaml` for the full
~200-factor result, which is the one to present.

## 6. Known limitations left deliberately in place

- `data.mode: osap` is still gated on returns, and rightly so. The gate and
  what unlocks it (any permno-keyed returns CSV, or WRDS credentials) are
  documented rather than faked with survivorship-biased free feeds.
- OP long-short portfolios are paper portfolios built from hundreds of
  underlying names, so factor-mode cost numbers model the cost of trading
  the *factor books*. The guide says as much.
- The 5-factor run committed in `results_factor/` is a verified smoke test.
  `summary.md` opens with a SMOKE TEST banner, emitted by `src/pipeline.py`
  whenever the cross-section is under 25 names, and
  `configs/config_factor.yaml` carries one in its header. The full-universe
  run is yours to produce, and needs the OSAP download of roughly 440 MB.
