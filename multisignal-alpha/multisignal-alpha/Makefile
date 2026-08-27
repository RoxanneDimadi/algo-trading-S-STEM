.PHONY: demo test notebooks clean

# Run the full synthetic-data pipeline end-to-end (placebo-validated demo)
demo:
	python3 -m src.pipeline --config configs/config.yaml

# Run the statistical-correctness test suite
test:
	python3 -m pytest -q tests/

# Rebuild the four workflow notebooks
notebooks:
	python3 scripts/build_notebooks.py

clean:
	rm -rf results/tables/* results/figures/* __pycache__ .pytest_cache
