# Developer entry points. Every target is a thin wrapper over a plain command,
# so the same steps work without `make` (see README "Without make").

PY ?= python
COMPOSE ?= docker compose
SIM_MODE ?= fast

.PHONY: help env install up down logs train simulate demo test lint format clean

help: ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

env: ## create .env from the example if missing
	@test -f .env || (cp .env.example .env && echo "created .env - set AWS_SECRET_ACCESS_KEY / ANTHROPIC_API_KEY")

install: ## local dev install (venv recommended)
	$(PY) -m pip install -e ".[dev]"

up: env ## build and start minio + api + watcher
	$(COMPOSE) up -d --build
	@echo "API http://localhost:8000/docs  |  MinIO console http://localhost:9001"

down: ## stop the stack (volumes kept)
	$(COMPOSE) down

logs: ## follow api + watcher logs
	$(COMPOSE) logs -f api watcher

train: ## generate the synthetic dataset and train the detector (local)
	$(PY) -m ml.generate_dataset
	$(PY) -m ml.train

simulate: ## seed (if needed) and run the SAFE ransomware simulator inside the stack
	$(COMPOSE) exec api sh -c "test -f sandbox/watched/.sim_manifest.json || python -m simulator.fake_ransomware seed sandbox/watched --files 120"
	$(COMPOSE) exec api python -m simulator.fake_ransomware attack sandbox/watched --mode $(SIM_MODE)

demo: ## end-to-end: seed -> snapshot -> attack -> detect -> ask agent -> restore -> verify
	$(COMPOSE) exec api python scripts/demo.py

test: ## run the test suite
	$(PY) -m pytest

lint: ## ruff lint + format check
	$(PY) -m ruff check app tests ml simulator scripts
	$(PY) -m ruff format --check app tests ml simulator scripts

format: ## apply ruff formatting and safe fixes
	$(PY) -m ruff format app tests ml simulator scripts
	$(PY) -m ruff check --fix app tests ml simulator scripts

clean: ## stop the stack AND delete its volumes (all backups/metadata!)
	$(COMPOSE) down -v
