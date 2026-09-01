# Thin wrappers around rag/*.py CLIs (ROADMAP §Phase 0/4). Every target here is directly
# runnable without make too - e.g. `uv run python -m rag.eval --tier 1 --config dense`. This
# file exists so the commands ROADMAP documents (`make eval TIER=<n> CONFIG=<name>`) are real.
#
# `make` is not installed on the primary dev machine this project was built on (Windows, no GNU
# Make on PATH) - this file is written for whichever environment runs it next (CI, WSL, a
# teammate's Linux box), and every target has been exercised via its underlying `python -m`
# command directly.

TIER ?= 1
CONFIG ?= dense+bm25+rerank
N ?=
SEED ?= 0

.PHONY: eval goldset review-goldset curate-goldset ingest serve agent router-eval api ui

goldset:
	uv run python -m rag.goldset -n $(or $(N),200) --seed $(SEED)

review-goldset:
	uv run python -m rag.goldset_review $(if $(REDO),--redo)

curate-goldset:
	uv run python -m rag.goldset_curate

eval:
	uv run python -m rag.eval --tier $(TIER) --config $(CONFIG) $(if $(N),-n $(N)) --seed $(SEED)

ingest:
	@echo "Run notebooks/ingest_phase1.ipynb then notebooks/embed_index_phase1.ipynb - not yet a script."

serve:
	uv run python -m rag.cli "$(Q)"

agent:
	uv run python -m rag.agent_cli "$(Q)"

router-eval:
	uv run python -m rag.router_eval $(if $(N),-n $(N)) --seed $(SEED)

api:
	uv run uvicorn rag.api:app --reload

ui:
	uv run streamlit run streamlit_app.py
