# Multi-Signal Alpha

A cross-sectional equity return-prediction research platform: documented signals, four predictive models, honest statistics (costs, factor controls, multiple-testing deflation), and a differentiable trading agent that turns forecasts into positions.

The claim is **not** "I found alpha." It is: *here is a harness proven on data where the truth is planted, pointed at real data with every statistical correction applied, reporting honestly what survives.*

For the full walkthrough (how to read every table and figure, config reference, troubleshooting), see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

---

## Project description

Multi-Signal Alpha predicts next-month returns from documented signals, builds long-short portfolios, charges transaction costs, and closes the loop with a cost-aware trading agent trained end-to-end on net-of-cost Sharpe.

Three run modes, in order of evidential weight:

| Mode | Data | What it proves |
|---|---|---|
| `synthetic` | Simulator with planted signals | The harness works: finds planted betas, clears the placebo, screams at leaks |
| `panel_csv` (factor) | Real OSAP long-short portfolio returns | Real-data behavior of every model and the agent, no WRDS needed |
| `osap` | Firm-level OSAP signals + CRSP returns | The full Gu-Kelly-Xiu setting, **gated on WRDS/CRSP access** |

Four predictive models share the same walk-forward inputs: **elastic net** (linear benchmark), **LightGBM** (nonlinear trees), **IC-Net** (ranking / correlation objective), and **PULSE** (Kalman filter over time-varying signal efficacy). The **trading agent** sits after the models: it learns how fast to trade toward an aim portfolio given costs.

---

## Repository description

This monorepo has two conda environments and a docs tree. Download code never mixes with research code.

```
multisignal-alpha/multisignal-alpha/   research agent (models, stats, agent)
  environment.yml                      conda env: msa-agent
  configs/config.yaml                  synthetic / general config
  configs/config_factor.yaml           generated config for real factor data
  src/                                 pipeline, models, evaluation, agent
  tests/                               statistical-correctness suite
  results/                             synthetic run outputs
  results_factor/                      real-data run outputs

real-data/                             ingest only (downloads and panel build)
  environment.yml                      conda env: msa-ingest
  configs/data_config.yaml             what to download and how to build panels
  scripts/run_all.py                   one-shot: OSAP + French + factor panel + sync
  scripts/validate_data.py             readiness report per data mode

docs/                                  user guide + math lesson plan
```

The ingest side *pushes* panels and a factor config into the agent tree; the agent never reaches outward for data.

---

## Use cases

- **Validate a research harness** on synthetic data where betas, decay, a placebo, and a leak are planted (`make demo`).
- **Run the four models and the agent on free real data** (OSAP factor portfolios) without WRDS (`config_factor.yaml`).
- **Measure signal decay** the McLean-Pontiff way (synthetic recovery; real exhibit under `real-data/`).
- **Study cost-aware trading** with a Gârleanu-Pedersen-style agent (learned trading speed γ vs myopic γ=1).
- **Teach / audit methodology** via the math lesson plan in `docs/math/` and the statistical test suite.

---

## User guide

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed first. Commands below are **PowerShell** (Windows default). Lines marked `# unix:` are macOS/Linux alternatives.

### 1. Create the conda environments

There are **two** environment YAML files, on purpose: ingest dependencies stay out of the results environment.

```powershell
# Research agent (tests, demo, models, agent pipeline)
cd multisignal-alpha/multisignal-alpha
conda env create -f environment.yml
conda activate msa-agent

# Data ingest (downloads and panel build only)
cd ../../real-data
conda env create -f environment.yml
conda activate msa-ingest
copy .env.example .env
# unix: cp .env.example .env
```

| YAML path | Env name | Use for |
|---|---|---|
| `multisignal-alpha/multisignal-alpha/environment.yml` | `msa-agent` | `make test`, `make demo`, model/agent pipeline |
| `real-data/environment.yml` | `msa-ingest` | anything under `real-data/` |

Switch with `conda activate <name>`. After dependency changes: `conda env update -f environment.yml --prune`.

You do **not** need to edit `real-data/.env` for the no-WRDS factor path; defaults work. The agent's `.env.example` is for optional live/paper execution only; the research pipeline does not read it.

### 2. Run the models

Models run inside the research agent environment via the end-to-end pipeline (signals → leak checks → walk-forward models → agent → controls / DSR).

**Synthetic demo** (planted signals; several minutes):

```powershell
conda activate msa-agent
cd multisignal-alpha/multisignal-alpha
make demo
# same as: python -m src.pipeline --config configs/config.yaml
```

Outputs: `results/tables/`, `results/figures/`, `results/summary.md`.

**Real factor data** (after ingest below):

```powershell
conda activate msa-agent
cd multisignal-alpha/multisignal-alpha
python -m src.pipeline --config configs/config_factor.yaml
```

Outputs: `results_factor/` (same layout as synthetic).

**Optional checks before a research run:**

```powershell
make test    # statistical-correctness suite
make lint    # same pylint gate CI runs
# unix (no make): python -m pytest -q tests/
# unix (no make): pylint --rcfile=../../.pylintrc src scripts tests
```

**Ingest real factor data** (no WRDS), from the ingest env:

```powershell
conda activate msa-ingest
cd real-data
python scripts/run_all.py --skip-returns
python scripts/validate_data.py    # confirm factor mode is RUNNABLE
```

### 3. Run the agent

The trading agent is part of the same pipeline as the models. When you run `make demo` or `python -m src.pipeline --config ...`, the pipeline trains and evaluates the agent after the model stage and writes `agent_vs_myopic` tables/figures under `results*/`.

Extra agent-focused scripts (still under `msa-agent`, from `multisignal-alpha/multisignal-alpha/`):

```powershell
python scripts/agent_cost_sweep.py      # how learned γ changes with cost
python scripts/compose_pulse_agent.py   # PULSE forecasts as the agent's aim
```

Paper/live order placement (`src/execution/`, `scripts/run_execution_server.py`) is separate, needs credentials in `.env`, and is **not** required for research runs.

---

## How the agent and models relate (plain language)

Think of two jobs in a row:

1. **Models answer:** "Which names look good next month?"  
   They turn signals into forecasts (scores or predicted returns). Elastic net, LightGBM, IC-Net, and PULSE all do this job in different ways, on the same walk-forward folds.

2. **The agent answers:** "Given those forecasts, what do I hold *this* month, knowing what I held last month and that trading costs money?"  
   A forecast is memoryless. Positions are not: buying or selling today changes tomorrow's costs. The agent learns a trading speed (how much of the gap toward an "aim" portfolio to close each period) by optimizing **net** Sharpe (returns after costs) not just forecast accuracy.

So: **models produce the map; the agent decides how to drive on that map when every turn has a toll.** You can evaluate models on IC and portfolio Sharpe alone; the agent is the step that turns a forecast into a cost-aware position path. In the default pipeline they run together so you see both the forecast quality and the traded result.

---

## Where to go next

| Doc | Start here if you want… |
|---|---|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Full setup, every output table, troubleshooting |
| [`docs/math/00_index.md`](docs/math/00_index.md) | Formula-by-formula math proofs |
| [`multisignal-alpha/multisignal-alpha/README.md`](multisignal-alpha/multisignal-alpha/README.md) | Research-agent quickstart and repo map |
| [`real-data/README.md`](real-data/README.md) | Ingest details and factor-panel construction |
