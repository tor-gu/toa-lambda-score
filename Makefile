.PHONY: format lint test check

format:
	isort .
	black .

lint:
	black --check .
	isort --check .
	flake8 .

test:
	pytest tests/

check: lint test
