# real-data

Separate ingest repo for wiring **multisignal-alpha** to real market data:

1. **Open Source Asset Pricing (OSAP)** — Chen & Zimmermann firm characteristics via `pip install openassetpricing`
2. **Kenneth French Data Library** — FF5 + momentum from [Ken French's site](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
3. **Monthly equity returns** — your `returns.csv`, or WRDS/CRSP if you have credentials

This directory does **not** mix download logic into the research package. It writes files under `data/raw/`, then copies them into `../multisignal-alpha/multisignal-alpha/data/raw/`.

## Setup

```bash
cd real-data
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
```

Edit `.env`:

- `OSAP_SIGNALS` — default matches the agent config (`Mom12m,Illiquidity,IdioVol3F,BM,GP`)
- `RETURNS_CSV` — path to a CRSP-style file with `permno,yyyymm,ret` (optional `dlret`)
- **or** `WRDS_USERNAME` / `WRDS_PASSWORD` and `pip install wrds`

`STreversal` is not in the public OSAP dump (CRSP license). Once returns exist, we build it as `-ret_{t-1}` and merge it into the wide signal file.

## Run

```bash
# everything that can run without WRDS (OSAP + French), then try returns/sync
python scripts/run_all.py

# pieces
python scripts/download_osap.py
python scripts/download_french.py
python scripts/build_returns.py
python scripts/sync_to_agent.py --build-panel
```

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
