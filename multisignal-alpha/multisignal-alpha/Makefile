.PHONY: demo test coverage coverage-html coveralls notebooks clean

# Run the full synthetic-data pipeline end-to-end (placebo-validated demo)
demo:
	python -m src.pipeline --config configs/config.yaml

# Run the statistical-correctness test suite
test:
	python -m pytest -q tests/

# Run tests with terminal coverage report (line + branch)
coverage:
	python -m pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=xml \
		-q

# Same as coverage, plus an HTML report at htmlcov/index.html
coverage-html:
	python -m pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-report=html \
		-q

# Upload the most recent coverage.xml to Coveralls (requires COVERALLS_REPO_TOKEN)
coveralls:
	coveralls --service=github

# Rebuild the four workflow notebooks
notebooks:
	python scripts/build_notebooks.py

clean:
	rm -rf results/tables/* results/figures/* __pycache__ .pytest_cache htmlcov .coverage coverage.xml
