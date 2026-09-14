# Coverage

```bash
pip install -r requirements-dev.txt
make coverage          # terminal + coverage.xml
make coverage-html     # htmlcov/index.html
```

Config: `.coveragerc`. Fast path for just the broker stack:

```bash
python -m pytest tests/test_execution_bridge.py tests/test_execution_coverage.py \
  --cov=src/execution --cov-report=term-missing -q
```

## Coveralls

1. Put the repo on GitHub and add it at [coveralls.io](https://coveralls.io) (GitHub login).
2. Keep `.github/workflows/coverage.yml` at the **git root**. If the root is the parent `trading agent/` folder, move the workflow there and set `working-directory: multisignal-alpha/multisignal-alpha` (and point `file:` at that folder's `coverage.xml`).
3. Push to `main`/`master` — the action runs pytest with coverage and uploads via `GITHUB_TOKEN`.

Local upload (optional):

```powershell
$env:COVERALLS_REPO_TOKEN = "token_from_coveralls_repo_settings"
make coverage
coveralls --service=github
```

Badge:

```markdown
[![Coverage Status](https://coveralls.io/repos/github/YOUR_USER/YOUR_REPO/badge.svg?branch=main)](https://coveralls.io/github/YOUR_USER/YOUR_REPO?branch=main)
```

Rough numbers from a recent local run: ~96% on `src/execution/`, ~77% on all of `src/`. Untested leftovers are mostly `pipeline.py`, plotting, and the OSAP loader. On some Windows Anaconda setups the LightGBM backtest test can segfault — that's the binary, not coverage.
