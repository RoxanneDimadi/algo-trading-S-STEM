# Multi-Signal Alpha

A cross-sectional equity return-prediction research platform: documented signals, four predictive models, honest statistics (costs, factor controls, multiple-testing deflation), and a differentiable, cost-aware trading agent that turns signals into a position path (see Results for how far that is actually wired).

The claim isn't I found alpha." 


---

## Project description

Multi-Signal Alpha predicts next-month returns from documented signals, builds long-short portfolios, charges transaction costs, and closes the loop with a cost-aware trading agent trained end-to-end on net-of-cost Sharpe.

Three run modes, in order of evidential weight:

| Mode | Data | What it proves |
|---|---|---|
| `synthetic` | Simulator with planted signals | It works to find planted betas, clears the placebo, screams at leaks |
| `panel_csv` (factor) | Real OSAP long-short portfolio returns | Real-data behavior of every model and the agent, no WRDS needed |
| `osap` | Firm-level OSAP signals + CRSP returns | The full Gu-Kelly-Xiu setting, **gated on WRDS/CRSP access** |

Four predictive models share the same walk-forward inputs: **elastic net** (linear benchmark), **LightGBM** (nonlinear trees), **IC-Net** (ranking / correlation objective), and **PULSE** (Kalman filter over time-varying signal efficacy). The **trading agent** sits after the models: it learns how fast to trade toward an aim portfolio given costs.

---

## Repository description

This monorepo has two conda environments. Download code never mixes with research code.

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
```

The ingest side *pushes* panels and a factor config into the agent tree; the agent never reaches outward for data.

---

## Use cases

- **Validate a research harness** on synthetic data where betas, decay, a placebo, and a leak are planted (`make demo`).
- **Run the four models and the agent on free real data** (OSAP factor portfolios) without WRDS (`config_factor.yaml`).
- **Measure signal decay** the McLean-Pontiff way (synthetic recovery; real exhibit under `real-data/`).
- **Study cost-aware trading** with a Gârleanu-Pedersen-style agent (learned trading speed γ vs myopic γ=1).


---

## User guide

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed first. Commands below are PowerShell (I have a windows and this was easiest for me). I tried adding unix and making them with `# unix:` are macOS/Linux alternatives.

### 1. Create the conda environments

There are two environment YAML files, on purpose: ingest dependencies stay out of the results environment.

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

You don't need to edit `real-data/.env` for the no-WRDS factor path; defaults work. The agent's `.env.example` is for optional live/paper execution only; the research pipeline does not read it.

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


### What the agent actually is

It is one file: [`src/agent/policy.py`](multisignal-alpha/multisignal-alpha/src/agent/policy.py), about 200 lines of NumPy.

It is not a fifth predictive model. It has one weight per signal, plus one number for trading speed. On the real factor run that is seven numbers in total. 
It was meant to take outputs from the best performing ML model and then paper trade with the Alpaca api, but I didn't finish this in time. 

Each month it does four things:

1. Combines the signals into one score per name, using its learned weights.
2. Turns those scores into a target portfolio — equal dollars long and short.
3. Moves only part of the way from last month's portfolio to that target. How far is the trading speed it learned.
4. Subtracts trading costs from the result.

Training means picking those seven numbers so that step 4 (return after costs, divided by its own volatility) is as high as possible over the training window. The partial-move structure comes from Gârleanu-Pedersen (2013), built in deliberately rather than discovered.


The agent is pretty much trained with which signals to weight and how fast to trade. 



The agent never sees model output. It builds its own target from its own weighted mix of the **raw signals**. The four models and the agent are two separate things reading the same signal data, side by side. The pipeline prints results for both (`model_comparison.csv` and `agent_vs_myopic.csv`), but nothing passes from one to the other.

One script is the exception: [`compose_pulse_agent.py`](multisignal-alpha/multisignal-alpha/scripts/compose_pulse_agent.py) feeds PULSE's forecast in as the agent's target. That is a single backtest exhibit, it only runs on synthetic data, and it hard-codes PULSE instead of picking whichever model did best.

This repository has ever placed an order. 

The code for that path does exist, and the pieces are in the right order:

```
newest month of signals
  -> agent's learned signal weights     (src/execution/signal_generator.py)
  -> look up a real ticker for each name
  -> risk gate: skip trades too small to beat costs  (src/execution/agent_evaluator.py)
  -> submit the order                   (src/execution/alpaca_bridge.py)
```

[`scripts/run_agent_signals.py`](multisignal-alpha/multisignal-alpha/scripts/run_agent_signals.py) drives it, and it has unit tests. It still cannot be run end to end, for these reasons:

| What is missing | Why it blocks |
|---|---|
| **Paid data** | The script only accepts `data.mode=osap`, and that mode needs WRDS/CRSP access this project does not have. |
| **Ticker lookup** | OSAP and CRSP files identify companies by `permno`, not by ticker symbol. Only an example file ships (`permno_ticker_map.csv.example`). Without a real one, every name gets skipped. |
| **The data that works is not tradable** | The one real-data mode that *does* run uses 212 OSAP factor portfolios. Those are not stocks — you cannot buy them. So the run with real results and the run that could trade are not the same run. |
| **The models aren't in the path** | The live code reuses only the agent's signal weights. Elastic net, LightGBM, IC-Net and PULSE are not involved at all. |
| **Trading speed is not carried over** | The learned speed is deliberately dropped. Live code uses a hand-set `gamma_speed: 0.35` from `execution_config.yaml` instead. |
| **The webhook path is something else** | `webhook_listener.py` checks alerts that arrive from outside (e.g. TradingView). Model forecasts never reach it. |
| **Monthly, not live** | OSAP signals are monthly and published late. "Live" here would mean "the most recent finished month," not intraday. |

So: the research side produces measured forecasts. The execution side can place an order if something hands it a signal. Nothing that has actually run connects the two.

### What I want it to become

Not built yet, in the order it needs building:

1. The walk-forward run **picks the best model** on out-of-sample results after costs, not on training fit.
2. That model's forecast becomes the agent's target portfolio. It's the same idea as `compose_pulse_agent.py`, but with the chosen model instead of always PULSE, and on real data instead of synthetic.
3. The agent applies its learned trading speed to turn that target into how much to hold this month.
4. That gets compared to what the Alpaca account currently holds, passed through the risk gate, and sent as paper orders.

Getting to step 4 needs four things that do not exist yet: a universe of real, buyable tickers (either CRSP access plus a ticker lookup, or swapping in liquid ETFs and large caps that the free ingest path can download); a model-selection step that saves the winning model to disk; saved positions carried between runs, so the agent knows what it held last month; and then paper trading for long enough to check whether real fill costs match what the backtest charged.

Until all four exist, every number below is a backtest.

---


## Results

Every number here comes from the committed run outputs: `results_factor/` (real data) and `results/` (synthetic). Re-running the pipeline overwrites them.

### What was run

**Real data run.** 212 OSAP long-short factor portfolios, monthly, December 1926 to November 2024 (1,176 months). Training expands over time: at least 120 months of history, then predict the next 12, then move forward and repeat, dropping the 1 month that straddles the boundary. That leaves 1,044 out-of-sample months, months the model had never seen when it made the call. Costs charged at 10 basis points per side.

**Synthetic run.** Same code, on simulated data where I planted the answer: known signal strengths, a known decay pattern, one signal that is pure noise, and one deliberate data leak. 168 out-of-sample months. Its only job is to show the harness finds what was planted and flags the leak. Synthetic results are not evidence about markets.

### How to read the columns

| Column | What it means | What counts as good |
|---|---|---|
| **IC** | Correlation between the forecast and what actually happened, month by month. | 0.02–0.05 is normal and usable for monthly stock forecasts. Above 0.10 usually means the universe is easier than individual stocks — which is the case here. |
| **ICIR** | Average IC divided by how much the IC bounces around. Consistency, not size. | Above 0.3 is respectable, above 1.0 is strong. |
| **Net Sharpe** | Return after trading costs, divided by volatility, annualized. | Below 0.3 is not worth running. 0.5–1.0 is a real but modest strategy. Above 2.0 on real data almost always means a bug or a leak. |
| **t-stat** | Whether the return could plausibly be luck. Newey-West corrected for the fact that monthly returns are not independent. | Above 2.0 is the conventional bar. Above 3.0 is the honest bar once you account for having tried several models. |
| **Turnover** | Fraction of the portfolio replaced each month. | Higher is not wrong, but it has to earn back its costs. |
| **Alpha** | Annual return left over after subtracting what market, size, value and momentum exposure already explain. | This is the number that says "not just repackaged market risk." |
| **DSR** | Deflated Sharpe: probability the result is real given how many models were tried. | Above 0.95. |

### Real data: the four models

Out-of-sample, after costs, 1,044 months:

| Model | IC | ICIR | Net Sharpe | t-stat | Turnover | Alpha/yr | Alpha t |
|---|---|---|---|---|---|---|---|
| **Elastic net** | 0.129 | 0.39 | **0.59** | 4.89 | 0.74 | 8.4% | 3.86 |
| **IC-Net** | 0.124 | 0.38 | **0.53** | 4.56 | 0.93 | 8.9% | 3.89 |
| LightGBM | 0.082 | 0.36 | 0.41 | 3.81 | 1.11 | 4.2% | 2.44 |
| PULSE | 0.081 | 0.31 | 0.30 | 2.26 | 0.87 | 4.0% | 3.04 |

Deflated Sharpe for the best model (elastic net), accounting for 10 attempts: **0.97**. Above the 0.95 bar.

Reading this honestly:

- **The simplest model won.** Elastic net is linear regression with a penalty term. It beat gradient-boosted trees and both custom models. That is a common outcome in this field, and it is a result, not a failure to tune.
- **LightGBM lost to its own costs.** Its gross Sharpe is 0.64, its net Sharpe is 0.41: it turns over 111% of the portfolio a month and gives back a third of its return. The nonlinearity it found was not worth what it cost to trade.
- **PULSE underperformed here.** It is designed for signals whose usefulness changes over time. Across 98 years of factor data, that flexibility cost more than it gained. It did well on synthetic data, where the decay was planted for it to find.
- **The alphas hold up.** 8–9% a year for elastic net and IC-Net, with t-stats near 3.9, after removing market, size, value and momentum exposure. That is the strongest finding in the project.

### Which models would beat the S&P 500

The S&P 500 has returned roughly 10% a year with about 15–16% volatility over long periods, giving a Sharpe ratio of about 0.4 to 0.5. That is the bar.

Comparing on Sharpe, not on raw return, because these are dollar-neutral long-short portfolios: equal money long and short, so there is no market exposure and the raw return is not comparable to an index.

| Model | Net Sharpe | vs. S&P (0.4–0.5) |
|---|---|---|
| Elastic net | 0.59 | **Above** |
| IC-Net | 0.53 | **Above** |
| LightGBM | 0.41 | Roughly equal |
| PULSE | 0.30 | Below |

So: elastic net and IC-Net clear the bar. LightGBM ties it. PULSE does not.

The more useful point is that elastic net and IC-Net produce 8–9% a year that is *uncorrelated* with the index, with t-stats near 4. Something with a 0.55 Sharpe and no market exposure is more valuable than the comparison suggests, because you can hold it alongside an index fund rather than instead of one.

What that comparison does not include:
- **No borrowing costs on the short side**, and no check that the short legs were actually shortable. On real stocks this is often where a long-short strategy's edge goes.
- **Costs are a flat 10 bps per side, linear in size.** Real costs rise with order size. The agent code has a square-root impact term; the model evaluation does not use it.
- **These are factor portfolios, not stocks.** You cannot buy an OSAP factor portfolio. Reproducing these returns means trading the underlying stocks, which costs more than 10 bps.
- **The sample starts in 1926.** Early decades have wider spreads, less liquidity, and different market structure. The full-sample Sharpe is better than a modern-only Sharpe would be.
- **Four models, one dataset.** The deflated Sharpe corrects for this, and elastic net survives it. But survival is not proof.

Taking all of that together: these are **genuine, statistically significant, cost-aware out-of-sample results on a universe that is easier than real stock trading.** They support the claim that the method works. They do not support the claim that this strategy would make money in a brokerage account.

### The agent's results

On real data the agent lost to the simple alternative:

| Policy | Net Sharpe | Turnover | Trading speed |
|---|---|---|---|
| Agent (learned speed) | 0.31 | 0.04 | 0.03 |
| Rebalance fully every month | 0.55*| 0.62 | 1.00 |

The agent learned a speed of 0.03, meaning it closes 3% of the gap to its target each month and barely trades at all. At 10 basis points, trading is cheap enough that moving slowly costs more in missed return than it saves in fees. Fold-by-fold (`agent_params_by_fold.csv`), the learned speed starts near 0.96 in the early folds and collapses to below 0.001 in later ones — it is fitting the training window's noise, not finding a stable speed.

The cost sweep shows the agent's logic is sound but the cost level here is wrong for it:

| Cost (bps/side) | Learned speed | Agent Sharpe | Full-rebalance Sharpe | Agent advantage |
|---|---|---|---|---|
| 0 | 0.92 | 2.61 | 2.63 | −0.02 |
| 10 | 0.83 | 2.29 | 2.29 | 0.00 |
| 25 | 0.60 | 1.93 | 1.79 | +0.13 |
| 50 | 0.39 | 1.50 | 0.97 | +0.53 |
| 100 | 0.21 | 0.95 | −0.69 | +1.64 |

(Synthetic data, where the agent's Sharpe is inflated. The pattern is what matters.)

This is the behavior theory predicts: as costs rise the agent trades slower, and its advantage grows. At 100 bps it still makes money while full rebalancing loses money. **The agent is worth having when trading is expensive. At 10 bps on this data, it is not**

### Summary

| Question | Answer |
|---|---|
| Does the harness work? | Yes. It recovers planted signals, catches the deliberate leak, and rejects the noise signal. |
| Do the models find real predictive power on real data? | Yes. Elastic net: IC 0.13, net Sharpe 0.59, t = 4.9, 8.4% annual alpha at t = 3.9, deflated Sharpe 0.97. |
| Which model is best? | Elastic net, with IC-Net close behind. The simplest model won. |
| Which models beat the S&P on risk-adjusted return? | Elastic net (0.59) and IC-Net (0.53), against the index's 0.4–0.5. LightGBM ties. PULSE does not. |
| Would this make money in a real account? | Unknown, and not claimed. Short borrowing costs, realistic size-dependent costs, and tradable instruments are all missing. |
| Does the agent help? | Not at 10 bps on this data. It loses to monthly full rebalancing. It wins clearly above 25 bps. |
| Has any of this traded? | No. Zero orders placed. See the bridge gaps above. |
