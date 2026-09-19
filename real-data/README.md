# real-data

Separate ingest repo for wiring **multisignal-alpha** to real market data:

1. **Open Source Asset Pricing (OSAP)**: Chen & Zimmermann firm characteristics, via `pip install openassetpricing`
2. **Kenneth French Data Library**: FF5 + momentum from [Ken French's site](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
3. **Monthly equity returns**: your own `returns.csv`, or WRDS/CRSP if you have credentials

This directory does **not** mix download logic into the research package. It writes files under `data/raw/`, then copies them into `../multisignal-alpha/multisignal-alpha/data/raw/`.

## Setup

```bash
cd real-data
conda env create -f environment.yml
conda activate msa-ingest
cp .env.example .env     # Windows: copy .env.example .env
```

Needs Miniconda or Anaconda. After a change to `requirements.txt`, refresh
the environment with `conda env update -f environment.yml --prune`.

Edit `.env`:

- `OSAP_SIGNALS`: defaults to the agent config's list (`Mom12m,Illiquidity,IdioVol3F,BM,GP`)
- `RETURNS_CSV`: path to a CRSP-style file with `permno,yyyymm,ret` (optional `dlret`)
- **or** `WRDS_USERNAME` / `WRDS_PASSWORD` and `pip install wrds`

`STreversal` is not in the public OSAP dump (CRSP license). Once returns exist, we build it as `-ret_{t-1}` and merge it into the wide signal file.

## Run

```bash
# everything that can run without WRDS: OSAP + French + the FACTOR PANEL
# (the supported real-data path -- OSAP long-short portfolios as assets)
python scripts/run_all.py --skip-returns

# check what's present / which agent modes are runnable
python scripts/validate_data.py

# pieces
python scripts/download_osap.py
python scripts/download_french.py
python scripts/build_factor_panel.py     # no-WRDS panel + real decay exhibit
python scripts/build_returns.py          # only if you have a returns CSV/WRDS
python scripts/sync_to_agent.py --build-panel
```

### The no-WRDS path (recommended)

`build_factor_panel.py` turns `osap_portfolios_op.csv` (freely licensed)
into a factor-timing panel -- each anomaly's LS portfolio is a "ticker",
features are its own trailing momentum/vol plus publication-status flags --
plus `factor_decay.csv`, the REAL McLean-Pontiff decay measured per factor.
It syncs both into the agent and writes a complete runnable config:

```bash
cd ../multisignal-alpha/multisignal-alpha
python -m src.pipeline --config configs/config_factor.yaml
```

Set `osap.portfolio_signals: all` in `configs/data_config.yaml` (default) so
the download covers the full ~200-predictor universe; with fewer than 25
factors the generated config is marked as a smoke test.

After a successful sync + returns:

1. Open `multisignal-alpha/multisignal-alpha/configs/config.yaml`
2. Set `data.mode: osap` (or merge `configs/config_osap_overlay.yaml`)
3. Run `python -m src.pipeline --config configs/config.yaml` from the agent repo

## Outputs

| File | Source |
|------|--------|
| `data/raw/signed_predictors_dl_wide.csv` | OSAP (+ STreversal if returns available) |
| `data/raw/SignalDoc.csv` | OSAP metadata |
| `data/raw/french_factors.csv` | Ken French FF5 + Mom |
| `data/raw/returns.csv` | Your CSV or WRDS |
| `data/raw/osap_portfolios_op.csv` | OSAP long-short portfolios (sanity) |

## Notes

- OSAP predictors must be requested as a **list** (`["BM", "Mom12m"]`), never a bare string.
- Full `dl_all_signals` needs WRDS; we only pull the configured subset.
- French ZIPs are fetched from the Dartmouth FTP; `pandas_datareader` is the fallback.
- Do not commit `.env` (password). `.env.example` is the template.
- Without `returns.csv`, sync still copies OSAP + French; the overlay omits `STreversal`
  until returns are available. The agent panel (`data.mode: osap`) needs returns.
