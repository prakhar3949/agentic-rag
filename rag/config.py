"""One place for every path, model ID and knob.

Anything a reviewer might ask "where did that number come from?" about lives here, not inline
in a call site (ROADMAP §6: recompute $/query from config, never re-run a sweep to fix a price).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# The .env in this repo uses mixed casing (`GOOGLE_API_key`); every Google SDK reads
# `GOOGLE_API_KEY`. The ingest notebook papers over this in its own first cell, so without the
# same bridge here the CLI would fail with an auth error that looks nothing like its cause.
#
# LANGSMITH_API_key2 is checked BEFORE LANGSMITH_API_key (order here is precedence, first match
# wins): the original key 403'd on every call, including plain reads - traced to the LangSmith
# account being provisioned in the APAC region, whose keys don't authenticate against the default
# (US) API host. `key2` is a fresh key confirmed working once LANGSMITH_ENDPOINT (below) points at
# the APAC host instead. Kept both entries rather than deleting the first, so the fix and the
# failure it replaces are both visible in one place instead of a silently vanished env var.
for _alias, _canonical in (
    ("GOOGLE_API_key", "GOOGLE_API_KEY"),
    ("fireworks_API_key", "FIREWORKS_API_KEY"),
    ("LANGSMITH_API_key2", "LANGSMITH_API_KEY"),
    ("LANGSMITH_API_key", "LANGSMITH_API_KEY"),
    ("TAVILY_API_key", "TAVILY_API_KEY"),
):
    if os.getenv(_alias) and not os.getenv(_canonical):
        os.environ[_canonical] = os.environ[_alias]

# --- data ---------------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
DOCS_PATH = DATA_DIR / "documents.jsonl"
IMAGES_DIR = DATA_DIR / "images"

# --- vector store -------------------------------------------------------------------------
# "local" is embedded Qdrant: no server, storage is a directory. Exact search, no HNSW, payload
# indexes accepted-and-ignored. Set QDRANT_MODE=server in .env to point at a real instance;
# nothing else changes (see notebooks/embed_index_phase1.ipynb for the full caveat).
QDRANT_MODE = os.getenv("QDRANT_MODE", "local")
# Overridable so a second index can be built without disturbing the live one - which matters
# more than it sounds in embedded mode, where the store is single-writer and a running notebook
# kernel locks out every other process. Also the mechanism for holding one index per ablation
# arm side by side.
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", PROJECT_ROOT / "data" / "qdrant"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "arxiv_phase1")

# --- models -------------------------------------------------------------------------------
EMBED_MODEL_ID = "BAAI/bge-m3"

# Same model the ingest notebook captions with. Flagged UNVERIFIED in ROADMAP Phase 0 against
# Google's published catalogue, but empirically real: it captioned 72 figures on this API key.
# That establishes the ID exists, NOT its price - config/pricing.yaml is still all zeros, so no
# $/query figure derived from it means anything yet.
GEN_MODEL_ID = os.getenv("GEN_MODEL_ID", "gemini-3.1-flash-lite")

# Pinned at 0. Phase 4 compares ablation arms by their metric deltas, and a sampling generator
# makes those deltas partly noise - you would be measuring the temperature, not the retriever.
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0"))

# --- retrieval ----------------------------------------------------------------------------
DEFAULT_TOP_K = 5
# Ablation row 1 is dense-only with no reranker, so top_k is the whole context budget. Five
# chunks of ~1000 chars is a small prompt on purpose: row 8 (long-context control, 50 chunks
# stuffed) only means something if the baseline is not already stuffing.

# --- Phase 3: BM25 fusion + reranking -----------------------------------------------------
SPARSE_MODEL_ID = "Qdrant/bm25"
SPARSE_VECTOR_NAME = "bm25"

# Cormack et al.'s original RRF paper found k=60 robust across collections; exposed here rather
# than hardcoded in the fusion math so it is a named, tunable ablation knob, not a magic number.
RRF_K = int(os.getenv("RRF_K", "60"))

# Shared by three things: how many candidates each of dense/BM25 fetch before fusion, how many
# fused candidates the reranker sees, and row 8's "stuffed, no rerank" pool size.
RETRIEVAL_CANDIDATES = int(os.getenv("RETRIEVAL_CANDIDATES", "50"))

RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"

# --- score logging --------------------------------------------------------------------------
# Raw material for Phase 4's eval harness and the Phase 12 ablation table: every retrieval's
# per-chunk scores, so recall@k / config comparisons can be recomputed later without re-running
# retrieval to reconstruct them.
RESULTS_DIR = PROJECT_ROOT / "results"
SCORE_LOG_PATH = RESULTS_DIR / "retrieval_scores.csv"

# --- Phase 4: eval judge -------------------------------------------------------------------
# DeepSeek-V4-Flash-0731 via Fireworks, OpenAI-compatible endpoint. Chosen 2026-08-05 after the
# originally-planned qwen2p5-72b-instruct was retired from Fireworks' serverless catalog - see
# config/pricing.yaml's `eval-judge` entry for the full reasoning (open weight, different model
# family from the Gemini generator, clears the 32B-minimum floor). Dated version string, not the
# bare "deepseek-v4-flash" preview name, so the judge does not drift under a re-run.
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID", "accounts/fireworks/models/deepseek-v4-flash-0731")
JUDGE_BASE_URL = "https://api.fireworks.ai/inference/v1"
JUDGE_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Pinned at 0, same reasoning as GEN_TEMPERATURE: ablation deltas must come from the retriever
# being compared, not sampling noise in the judge scoring it.
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0"))

# Every judge call retries on unparseable output; an unparseable result after retry is `null`,
# never 0.0 (ROADMAP Phase 4) - a parse failure scored as zero is indistinguishable from a
# genuinely bad answer and would silently drag a config's mean down.
JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "2"))

# --- Phase 4: golden set + eval results ------------------------------------------------------
GOLDSET_PATH = RESULTS_DIR / "goldset.jsonl"
# Human verdicts on the goldset, keyed by row id - kept separate from GOLDSET_PATH so the judge's
# raw generated output and a human's accept/edit/reject call never overwrite each other.
GOLDSET_REVIEW_PATH = RESULTS_DIR / "goldset_review.jsonl"
# rag.goldset_curate's output: GOLDSET_PATH filtered/patched by GOLDSET_REVIEW_PATH verdicts.
# Point `rag.eval --goldset` at this once it exists - eval never picks it up implicitly.
GOLDSET_CURATED_PATH = RESULTS_DIR / "goldset_curated.jsonl"
EVAL_RESULTS_DIR = RESULTS_DIR / "eval"

# --- Phase 4: judge validation (Cohen's kappa, PHASE4_NOTES.md) ---------------------------
# rag.judge_validate's three stages: generate pools every atomic claim/chunk verdict the judge
# made over the curated goldset; label is where a human independently calls the same items,
# blind to the judge's verdict; report computes agreement between the two.
JUDGE_POOL_PATH = RESULTS_DIR / "judge_pool.jsonl"
JUDGE_LABELS_PATH = RESULTS_DIR / "judge_validation.jsonl"
JUDGE_REPORT_PATH = RESULTS_DIR / "judge_validation_report.json"

# rag.judge_probe's output: a small hand-built adversarial set with KNOWN ground truth (not
# blind human labels) - the real goldset sample never gave the judge a genuinely unsupported
# claim to score, so kappa against it couldn't show whether the judge can say "no" at all.
JUDGE_PROBE_REPORT_PATH = RESULTS_DIR / "judge_probe_report.json"

# --- Phase 5: agentic layer (planner / router / fanout / synthesis) -----------------------
# The categories actually indexed in Qdrant (PHASE3_NOTES.md) - astronomy/hep-th are on disk
# under arxiv_papers/ but have not been through ingestion yet. This is the router's entire valid
# output space; update this one line, not the router's prompt, once more categories are indexed.
INDEXED_CATEGORIES = ["ai", "cs.CL", "finance"]

# Short, stable descriptions for the router prompt - arXiv category scope doesn't change, so this
# is hand-written metadata, not a runtime knob like the paths above.
CATEGORY_DESCRIPTIONS = {
    "ai": "artificial intelligence and machine learning",
    "cs.CL": "computational linguistics and natural language processing",
    "finance": "quantitative finance and economics",
}

# Planner and router are both cheap-tier, high-volume LLM calls (one each per question) - reuse
# GEN_MODEL_ID rather than adding new model knobs. Same for the synthesizer: ROADMAP's stack
# table calls for a separate "Pro tier for synthesis" model, but no such model is verified
# anywhere in this repo yet (config/pricing.yaml's generator entry is still UNVERIFIED) - adding
# a second unverified model string would be a magic name nothing points at. Synthesis reuses
# GEN_MODEL_ID until a real Pro-tier ID is verified and added, at which point every LLM call site
# in rag/graph.py that should move to it is a one-line change, not a rewrite.

# SqliteSaver, not MemorySaver - matches the project's existing local-file-persistence pattern
# (embedded Qdrant is the same idea) and actually satisfies "checkpointing for replay" across
# process restarts, which an in-memory checkpointer cannot.
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "langgraph_checkpoints.sqlite"

# rag.router_eval's output: router-only accuracy against goldset_curated.jsonl's `category` field
# (free ground truth, ROADMAP §Phase 5's router bullet) - no judge call, no generation.
ROUTER_EVAL_PATH = RESULTS_DIR / "router_eval.json"

# --- Phase 5b: conversational query contextualization -------------------------------------
# ROADMAP §7 cut list: "Multi-turn memory beyond 3 turns - out of scope." rag/contextualize.py
# truncates to the most recent MAX_HISTORY_TURNS before building its rewrite prompt - a stated
# scope decision, not an arbitrary number to raise later without re-checking that line.
MAX_HISTORY_TURNS = 3

# --- Phase 6: web search tool + escalation -------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# One tool, one call, no multi-query (ROADMAP's explicit cut list) - a handful of results is
# enough for "the corpus doesn't cover this, here's what the web says instead," not a second
# retrieval pipeline to rerank or score.
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))

# ROADMAP: "Threshold tuned on the golden set, not guessed." rag/web_eval.py measures the
# reranked top-1 score `Retriever.search()` returns on goldset_curated.jsonl's questions - every
# one of which IS answerable from the indexed corpus by construction (rag/goldset.py generates
# each question from a real, already-indexed chunk). That means the golden set can't teach this
# threshold to recognize an unanswerable question directly - there are no negative examples yet
# (Phase 10 §④'s adversarial set is a later phase). What it CAN do: show how low a genuine
# match's top score gets, so the threshold sits below that with a known, small false-escalation
# rate on real corpus questions, rather than a number picked by eye.
#
# Tuned 2026-08-25 via `python -m rag.web_eval` (n=123, the full curated goldset, 5th
# percentile): threshold=0.4862. Score range across the set was 0.078-0.9999 (mean=0.922,
# median=0.990) - most genuine matches score well above 0.9, so a threshold at the 5th
# percentile only misfires on the 6/123 hardest-but-still-answerable questions
# (false_escalation_rate=0.049). See PHASE6_NOTES.md §2 and results/web_escalation_report.json
# for the full distribution.
WEB_ESCALATION_THRESHOLD = float(os.getenv("WEB_ESCALATION_THRESHOLD", "0.48619261384010315"))

WEB_ESCALATION_REPORT_PATH = RESULTS_DIR / "web_escalation_report.json"

# --- Phase 7: guardrails -------------------------------------------------------------------
# No new numeric threshold here, unlike every prior phase (RRF_K, WEB_ESCALATION_THRESHOLD,
# MAX_HISTORY_TURNS) - all three new LLM-check rails (jailbreak, scope, clarify) are categorical
# judgments (a bool or a label), not a score compared against a tunable cutoff.
#
# rag.guardrail_probe's output: a small hand-built sanity probe set (not Phase 10 §④'s eventual
# 100-probe confusion matrix) confirming each rail actually discriminates.
GUARDRAIL_PROBE_REPORT_PATH = RESULTS_DIR / "guardrail_probe_report.json"

# --- Phase 8: LangSmith observability ------------------------------------------------------
# Resolves ROADMAP Phase 0's open item ("confirm the current LangSmith env vars for the tracing
# toggle and the sampling rate") with an answer read out of the installed `langsmith` SDK's own
# source (langsmith/utils.py's get_env_var(), which checks LANGSMITH_<NAME> then falls back to
# LANGCHAIN_<NAME>), not assumed from memory of an older convention:
#   toggle:         LANGSMITH_TRACING=true          (legacy fallback: LANGCHAIN_TRACING_V2)
#   project:        LANGSMITH_PROJECT=<name>        (legacy fallback: LANGCHAIN_PROJECT)
#   sampling rate:  LANGSMITH_TRACING_SAMPLING_RATE  (legacy fallback: LANGCHAIN_TRACING_SAMPLING_RATE)
# All three are read directly out of os.environ by the langsmith/langchain_core SDKs the first
# time anything traced runs - nothing here needs to be threaded through a client constructor, but
# they DO get cached (functools.lru_cache) on first read, so they must be set before the first
# LLM call in a process, not after. Setting them at rag/config.py's import time (below) guarantees
# that, since every entry point imports this module first.
#
# Tracing defaults ON: this phase's entire point is "trace the full graph." `.env`'s
# `LANGSMITH_TRACING=false` overrides this back off for anyone who wants it off.
os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "true"))

# Three separate projects (ROADMAP §Phase 8: "so retention is configured per project instead of
# globally," and extended retention on `tier3` must be set BEFORE that project's first trace -
# retroactive upgrade is not available). Retention itself is a LangSmith workspace/UI setting,
# not something this repo's code can set - PHASE8_NOTES.md documents the one-time manual step and
# its sequencing risk.
LANGSMITH_PROJECT_DEV = os.getenv("LANGSMITH_PROJECT_DEV", "agentic-rag-dev")
LANGSMITH_PROJECT_TIER2 = os.getenv("LANGSMITH_PROJECT_TIER2", "agentic-rag-tier2")
LANGSMITH_PROJECT_TIER3 = os.getenv("LANGSMITH_PROJECT_TIER3", "agentic-rag-tier3")

# `dev` is the default for anything that doesn't explicitly pick a project (agent_cli, chat,
# every notebook) - set here, once, so importing this module is sufficient. rag/eval.py's main()
# overrides this to tier2/tier3 before running a real sweep (see its own comment for why).
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT_DEV)

# ROADMAP's own control: "sample interactive dev queries at ~10% once Phase 5 is live." Applies
# to the `dev` project default above; rag/eval.py's main() removes this for tier2/tier3 runs -
# every row of an eval sweep must be traced, not a random 10% of them.
LANGSMITH_DEV_SAMPLING_RATE = float(os.getenv("LANGSMITH_DEV_SAMPLING_RATE", "0.1"))
os.environ.setdefault("LANGSMITH_TRACING_SAMPLING_RATE", str(LANGSMITH_DEV_SAMPLING_RATE))

# rag.eval's LangSmith experiment tracking (--langsmith flag) reuses one dataset per goldset
# question set, rather than creating a new one per run - LangSmith's `evaluate()` compares
# experiments run against the SAME dataset over time, which is the whole point of "quality over
# time, not measured once."
LANGSMITH_DATASET_NAME = os.getenv("LANGSMITH_DATASET_NAME", "agentic-rag-goldset")
