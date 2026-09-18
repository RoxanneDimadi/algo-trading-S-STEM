# Improvement Log — 2026-09-17

Every change made in this pass, why, and how it was verified. Verification state at the end: **53/53 agent tests passing (no regressions)** and **the full pipeline runs end-to-end on real market data for the first time** (`results_factor/`).

## 1. Real-data connection refined for the no-WRDS reality (the big one)

**Problem.** `data.mode: osap` was structurally dead without WRDS: it requires `returns.csv` (CRSP firm-level, license-blocked), so the entire real-data half of the project could never run. There is no legitimate free source of permno-keyed firm returns, and ticker-keyed free feeds can't merge with the permno-keyed OSAP file.

**Solution — factor mode.** OSAP *does* freely distribute every published anomaly's long-short portfolio returns. New pipeline treats each anomaly as the tradable asset with point-in-time features from its own history (factor momentum/vol) plus publication-status features (the McLean–Pontiff angle). This is the Ehsani–Linnainmaa / Gupta–Kelly factor-timing setting — PULSE's natural real-data habitat — not a workaround.

New/changed in `real-data/`:
- `src/factor_panel.py` (new) — LS extraction, **automatic percent→decimal unit conversion** (OSAP ships percent, French decimals — a latent footgun now handled and logged), point-in-time features with documented leak reasoning (publication features clamp to zero pre-publication; post-sample-end deliberately excluded as a feature because sample end is only public knowledge at publication), and `factor_decay_table()` — the **real McLean–Pontiff decay exhibit** per factor.
- `scripts/build_factor_panel.py` (new) — builds panel + meta + decay exhibit, syncs to the agent, and generates a complete runnable agent config.
- `scripts/validate_data.py` (new) — schema/coverage/NaN/units checks on every data file plus a per-mode readiness verdict (synthetic / factor / osap) with the WRDS gating explained.
- `src/osap_download.py` — `portfolio_signals: all` downloads the full ~200-predictor LS universe (with a SignalDoc-driven fallback for older `openassetpricing` versions); config default set to `all`.
- `scripts/run_all.py` — factor panel wired in as the default real-data step (`--skip-factor-panel` to opt out); `--skip-returns` now yields a fully working real-data path instead of a warning.
- `src/panel_build.py` — `write_agent_factor_config()` generates `configs/config_factor.yaml` from the agent's own base config (single source of truth preserved), auto-sizing `n_quantiles`/thresholds to the cross-section and stamping a loud SMOKE-TEST header when < 25 factors; `sync_to_agent()` now **skips unchanged files** (was rewriting the 450 MB OSAP CSV on every run).
- `configs/data_config.yaml` — new `factor_panel:` section (lookback, min names, units) and `osap.portfolio_signals`.

New/changed in the agent repo:
- `src/data/loaders.py` — `load_prebuilt_panel()`: generic `data.mode: panel_csv` entry point (today's factor panel; a WRDS firm panel someday, unchanged agent).
- `src/pipeline.py` — `panel_csv` mode; decay stage now **skips gracefully with an explanatory note** when features carry no publication dates (derived features like `fmom_12m` have none; the real per-factor decay lives in `real-data/data/processed/factor_decay.csv`).

**Verified end-to-end on real data** (5 factors currently on disk; full universe after re-download): leak demo IC 0.27 vs 0.00 honest; factor momentum IC ≈ 0.11–0.12 (NW t ≈ 5.9), net Sharpe ≈ 0.23–0.28 at 10 bps; IC-Net leads the OOS model table; agent γ ≈ 0.95 ≈ myopic (honest null at low cost); factor-controlled alphas insignificant; DSR 0.79. Real decay exhibit: the retention columns are annualized-RETURN ratios -- Illiquidity retains ~12% of its in-sample annualized return post-publication, BM ~77% (on Sharpe the same two are ~21% and ~63%).

## 2. Hardcoded statistical thresholds made config-driven (bug-class fix)

`min_names` minimum cross-section sizes were hardcoded in five places (ic 30, portfolio 50, PULSE 60, IC-Net 30, engine implicit) — violating the project's own "every tunable in config" principle and making any cross-section under ~50 names silently NaN or crash. Now threaded through one optional knob `evaluation.min_names_per_date` (module defaults unchanged when unset, so all prior behavior and tests are preserved) into `ic.py`, `portfolio.py`, `data/panel.py`, `backtest/engine.py`, and both model factories. PULSE additionally gains `obs_margin` (identification headroom `n ≥ P + margin`; `≤ 0` = documented ridge-identified smoke mode). `ic_series` now skips constant cross-sections cleanly instead of spraying scipy warnings.

## 3. Dead weight removed / prevented

- `real-data/.venv/` — **534 MB** shipped inside the project archive (it was already gitignored; the zip just included it). Deleted from the working copy; recreate locally with `python -m venv .venv`. On your machine: safe to delete and recreate the same way.
- All `__pycache__/` and `.pytest_cache/` directories removed.
- The 443 MB OSAP wide CSV is duplicated by design (ingest copy + agent copy) so the agent repo stays self-contained; the sync-skip change above at least stops rewriting it every run. If disk matters, delete `real-data/data/raw/signed_predictors_dl_wide.csv` after syncing — `download_osap.py` restores it on demand.
- `coverage.xml` (repo root) is a build artifact already gitignored; safe to delete anytime.

## 4. Documentation corrected and completed

- **`docs/USER_GUIDE.md` (new, repository root `docs/`)** — the complete use guide: setup, both repos, test suite, synthetic demo, the no-WRDS data story, running on real data, how to read *every* output table and figure, the four models, the agent, config reference, honest claims vs non-claims, troubleshooting.
- Agent `README.md` — stale "17 tests" corrected to 55; new "Real data without WRDS" section with the four-command path.
- `real-data/README.md` — rewritten run section around the factor path + `validate_data.py`.
- This file.

## 5. Applying these changes to your local copy

The archive `msa_improvements.zip` contains only new/changed files in their correct relative paths. From the directory that contains `multisignal-alpha/` and `real-data/`:

```bash
unzip -o msa_improvements.zip     # overwrites the ~15 touched files in place
cd real-data && python scripts/validate_data.py
cd ../multisignal-alpha/multisignal-alpha && make test    # expect 53 passed
```

Then re-download with the full universe when convenient (`python scripts/download_osap.py` → `python scripts/build_factor_panel.py`) and run `python -m src.pipeline --config configs/config_factor.yaml` for the full ~200-factor result — the one worth putting on the poster.

## 6. Known limitations left deliberately in place

- `data.mode: osap` remains gated on returns — correctly so; the gate and its unlock (any permno-keyed returns CSV, or WRDS creds) are documented rather than faked with survivorship-biased free feeds.
- OP long-short portfolios are paper portfolios (hundreds of underlying names); factor-mode cost numbers model trading the *factor books*, and the guide says so.
- The 5-factor run committed in `results_factor/` is a verified smoke test. `summary.md` now opens with a SMOKE TEST banner (emitted by `src/pipeline.py` whenever the cross-section is under 25 names) and `configs/config_factor.yaml` carries one in its header; the full-universe run is yours to produce (needs the OSAP download, ~440 MB).
