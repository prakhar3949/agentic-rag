# Agentic Multimodal RAG — Capstone Roadmap

**Owner:** Prakhar
**Started:** 2026-07-30
**Target:** ~172h tracked work → 9–11 weeks at 20 hrs/week

---

## 0. Project spec (the thing being defended)

A production-grade agentic RAG over ~10,000 multimodal documents that:

1. Handles text + images (figures/charts from source PDFs)
2. Spans 4–5 distinct topic domains with real topic labels
3. Decomposes a query into tasks and fans out to topic-specialist subagents in parallel
4. Escalates to web search when corpus coverage is insufficient
5. Is measured with four purpose-written LLM-judge metrics (context precision, context recall, faithfulness, response relevancy) over an ensemble retriever + cross-encoder reranking
6. Has input / jailbreak / dialogue guardrails with a measured false-positive rate
7. Is fully traced and evaluated in LangSmith with a dashboard

**The deliverable is not the system. The deliverable is the ablation table + failure analysis.**

---

## 1. Stack decisions

| Layer | Choice | Defense one-liner |
|---|---|---|
| Corpus | arXiv, ~2,000 papers × 5 categories (cs.AI, q-fin, q-bio, physics, econ) | Real PDFs with real figures; arXiv categories give free ground-truth topic labels for router evaluation |
| PDF parsing | PyMuPDF (text + embedded images) | Deliberately not layout-aware — parsing depth was traded for eval depth (see §8) |
| Multimodal | VLM-generated figure descriptions for retrieval; original image passed to generator at answer time | Single vector space for retrieval + real visual grounding at generation. ColPali evaluated and rejected (index size, no lexical hybrid) |
| Embeddings | BGE-M3 locally | Free, no rate limits, fast iteration; multilingual + multi-granularity |
| Vector store | Qdrant (Docker) | Native hybrid search, named vectors for multimodal, tunable HNSW (`m`, `ef_construct`) |
| Sparse retrieval | BM25 | Lexical recall for exact terms, IDs, notation embeddings miss |
| Fusion | Reciprocal Rank Fusion | BM25 and cosine scores are on incomparable scales; RRF uses ranks only, no unjustifiable normalization hyperparameter |
| Reranking | `bge-reranker-v2-m3` (cross-encoder), top-50 → top-5 | Cross-attention over (query, doc) beats bi-encoder similarity for precision@k |
| Orchestration | LangGraph | Explicit graph state; `Send` API for dynamic parallel fanout; checkpointing for replay/debug |
| Generation | Gemini (Flash tier for volume, Pro tier for synthesis) | Cost + native multimodal input. **Verify current model IDs and per-MTok prices before budgeting — treat price as a config value** |
| Eval judge | **DeepSeek-V4-Flash-0731 (284B/13B-active MoE, open weight), via Fireworks.** `temperature=0`, dated version string pinned in config | Different model family from the generator → no self-evaluation bias. Open weights → the exact judge version is pinned and reproducible, unlike a proprietary endpoint that updates silently. Reliability quantified via Cohen's κ against hand labels. **Changed 2026-08-05 from the originally-planned Qwen2.5-72B-Instruct**, which was retired from Fireworks' serverless catalog; the only remaining Qwen options on the two configured providers were a likely-proprietary Fireworks proxy (`qwen3p7-plus`, no HF id) and a sub-floor open model (Groq's `qwen/qwen3.6-27b`, 27B) — see `config/pricing.yaml` for the full reasoning |
| Eval metrics | **No eval framework.** Four metrics written directly against the judge; retrieval metrics are plain ranking arithmetic; LangSmith Datasets + custom evaluators do the tracking | Ragas 0.4.3 went dormant — last release 2026-01-13, last commit 2026-02-24 — and its **unpinned** `langchain-community` dependency broke it outright once that package dropped `chat_models.vertexai`. `import ragas` raised `ModuleNotFoundError`. DeepEval swaps one dependency risk for another and ships posthog + sentry telemetry. These four metrics are judge prompts, not algorithms: owning them pins the prompt, the judge version and the parsing, and makes *"what exactly does your faithfulness score measure?"* answerable line by line |
| Guardrails | **Hand-written LLM-prompt nodes, no framework.** Four rails — jailbreak (input), off-topic+sensitive classifier (input), clarification (dialogue), grounding check (output) — each a Pydantic schema + prompt against the same structured-output call the planner/router already use | NeMo Guardrails requires `langchain-core<0.4.0` in every version — a hard, unresolvable conflict with this project's `langchain-core>=1.5.3` (confirmed via `uv`'s resolver, not assumed). `guardrails-ai` and `llm-guard` both installed cleanly with **no** resolver conflict but were dropped anyway: four rails is a small, fully-specified surface, and a hand-written node is exactly as auditable as the planner/router/contextualize nodes already in this codebase, at the cost of zero new dependencies instead of two. The jailbreak rail is the one place in this codebase that fails **closed**, not open, on an internal error — a safety check that silently disables itself on an LLM/parsing failure is a vulnerability, not a degraded-quality answer. See `PHASE7_NOTES.md` |
| Observability | LangSmith | Datasets + Evaluators make each ablation config a first-class tracked experiment with side-by-side comparison |

### Open items to resolve in Phase 0
- [ ] **Verify `gemini-3.1-flash-lite` is a real, currently-priced model ID** against Google's live pricing page — not in this skill's cached catalog, unconfirmed. Fill in `config/pricing.yaml` once checked
- [x] Verify Fireworks pricing for the judge model against `docs.fireworks.ai/serverless/pricing`; filled into `config/pricing.yaml` — `deepseek-v4-flash-0731`, not the retired `qwen2p5-72b-instruct` (see judge row above)
- [x] **LangSmith counts one trace = one root run** (confirmed 2026-07-30). Nested LLM calls within a graph invocation are child runs and do not each bill as a trace. Volume math in Phase 8 assumes this
- [x] Confirm the current LangSmith env vars for the tracing toggle *and* the sampling rate (they're separate settings) — read out of the installed `langsmith` SDK's own source, not assumed: toggle is `LANGSMITH_TRACING` (legacy `LANGCHAIN_TRACING_V2`), sampling rate is `LANGSMITH_TRACING_SAMPLING_RATE` (legacy `LANGCHAIN_TRACING_SAMPLING_RATE`) — see `PHASE8_NOTES.md` §2
- [x] Pick the judge model and provider — `accounts/fireworks/models/deepseek-v4-flash-0731` on Fireworks (2026-08-05)
- [~] Add both judge and generator prices to `config/pricing.yaml` with source URL + verification date — **judge done**; generator (`gemini-3.1-flash-lite`) still TODO
- [x] **LangSmith pricing resolved:** 1 LSU = $1.00. Base = 0.0005 LSU/trace ($0.50/1k, 14-day retention); extended = 0.005 LSU/trace ($5.00/1k, 400-day). Recorded in Phase 8
- [x] **Verify whether extended retention can be applied retroactively.** Checked against LangSmith's current support docs, not left as a guess: it CAN, via an explicit per-workspace checkbox that applies a new setting to existing traces (off by default — new traces only). Lower sequencing risk than assumed, but re-verify before Phase 12's real sweep since this is exactly the kind of setting that can change. A separate, unavoidable behavior found along the way: LangSmith auto-upgrades any feedback-bearing trace to extended retention with no opt-out — relevant to `rag.eval --langsmith`, quantified in `PHASE8_NOTES.md` §6
- [x] Create three LangSmith projects: `dev`, `tier2`, `tier3`, so retention is set per project — named `agentic-rag-dev`/`agentic-rag-tier2`/`agentic-rag-tier3` (`rag/config.py`), auto-created on first trace; `dev`/`tier2` confirmed live via a real verification run, `tier3` auto-creates identically on first real Tier 3 use

---

## 2. Phase plan

### Phase 0 — Scope & foundations · 4h
- [ ] Create `decisions.md` (ADR log). One entry per real decision: context, choice, **alternative rejected and why**
- [ ] Define "document" precisely and write it down (10K docs, shallow per-doc ingestion — see Phase 2)
- [ ] Resolve the three open items above
- [ ] Repo skeleton, `pyproject.toml`, `.env.example`, Makefile targets for `ingest` / `eval` / `serve`

### Phase 1 — Vertical slice, 100 docs · 12h
**Non-negotiable. You will find three architectural mistakes here that would cost days at 10K.**
**Status 2026-08-05: fully implemented and Phase 3 already builds on top of it.** See `PHASE1_NOTES.md`.
- [x] Download 100 arXiv PDFs across all 5 categories
- [x] Parse text + extract embedded images — `notebooks/ingest_phase1.ipynb`
- [x] Caption figures with the VLM
- [x] Chunk, embed, index in Qdrant — `notebooks/embed_index_phase1.ipynb`
- [x] Dense-only retrieval → single generation call → answer with citations — `rag/retrieval.py`, `rag/generation.py`
- [x] End-to-end CLI: question in, cited answer out — `rag/cli.py` (`python -m rag.cli`)
- [x] Write down every assumption this slice broke — `PHASE1_NOTES.md` §3, §8

**Validated on `ai/`, `cs.CL/`, `finance/` (2026-08-19, 60 docs → 54 unique).** `astronomy` and
`hep-th` remain unvalidated: heading conventions, column layouts and title typesetting vary more
by field than by paper, and are expected to shift the `section_source` distribution further.
Re-run the batch cell per category before treating any parse rate as real.

The promised "three architectural mistakes" arrived, and then some — seven, in
`PHASE1_NOTES.md` §3. Three were silent (pipeline reported success while dropping an entire
modality): VLM captions discarded, 100% of tables lost to a PyMuPDF bug, and non-deterministic
output from MuPDF thread-unsafety. **Expanding to a second and third category (2026-08-19) found
three more**, all silent in the same way — pipeline "succeeded" while producing wrong results.
See `PHASE1_NOTES.md` §3b: cross-listed duplicate PDFs indexed twice under different categories,
ordinal-based chunk IDs drifting on ordinary VLM non-determinism (not just a deliberate
re-chunking), and `Qdrant.delete_collection()` not actually purging on-disk state in embedded
mode.

### Phase 2 — Ingestion at scale · 12h
- [ ] arXiv bulk fetch with category metadata, ~2,000 per category
- [ ] **Shallow per-doc extraction:** abstract + intro + conclusion + figure captions. Not every page. Cuts chunk count ~4×, keeps 10K docs intact. Document this tradeoff in the README — it's an engineering decision, not a shortcut
- [ ] Figure captioning via batch API (cheap tier model)
- [ ] Chunking: 2 strategies only (fixed-window w/ overlap, and section-aware). Compare in eval, pick one, stop
- [ ] Metadata schema: `arxiv_id`, `category`, `section`, `modality` (text|figure), `page`, `figure_ref`
- [ ] Idempotent, resumable ingestion (you will re-run this)
- [ ] Log ingestion failure counts by cause — this feeds Phase 10 taxonomy

### Phase 3 — Ensemble retrieval + reranking · 15h
- [x] Dense retrieval (BGE-M3 → Qdrant)
- [x] BM25 index — Qdrant native sparse vector (`fastembed`'s `Qdrant/bm25`, `Modifier.IDF`)
- [x] RRF fusion, `k` parameter exposed as config — `rag/fusion.py`, `config.RRF_K`
- [x] Cross-encoder reranker, top-50 → top-5 — `rag/rerank.py` (`BAAI/bge-reranker-v2-m3`)
- [x] Metadata filtering by category (needed for subagent corpus partitions)
- [x] Every stage returns scores + provenance so Phase 10 can attribute failures — provenance
      fields on `RetrievedChunk` plus `rag/scorelog.py`'s persistent `results/retrieval_scores.csv`
- [x] `ai/`, `cs.CL/`, `finance/` now indexed (542/289/388 chunks — Qdrant verified via
      `rag.retrieval.Retriever`, category filter spot-checked on both new categories). Still
      20 docs/category, not Phase 2's ~2,000/category "at scale" target; `astronomy`/`hep-th`
      remain unindexed. Unblocks Phase 5's router/fanout, which need more than one real category
      to be anything but degenerate (see Phase 5 discussion, 2026-08-19)

### Phase 4 — Eval harness · 23h
**Built before the agent layer, on purpose.**

**Status 2026-08-20:** golden set generation, hand-review, and judge validation (κ + adversarial
probe) all done — see `PHASE4_NOTES.md` for full reasoning and numbers. Golden set now spans all
three indexed categories (150/200 generated, 123 curated — see below). Still open: growing the
last ~50 toward 200, the Phase 10 §④ 40-question adversarial set, and running real Tier 1/2/3
sweeps against `results/goldset_curated.jsonl` — only the Aug-5 smoke tests exist so far (Tier 1
n=20, Tier 2 n=3, one config, and predating both the corpus expansion and the goldset growth, so
not comparable to a fresh run).

**No eval framework — the metrics are ours.** See the stack table: Ragas went dormant and broke
on an unpinned dependency, and swapping it for another framework only moves the risk. What that
actually costs is one file of prompts and one of ranking arithmetic; what it buys is that every
number in the ablation table has a prompt behind it that can be read out loud.

**Judge setup first — do this before generating anything:**
- [x] **Hour one: verify the judge endpoint returns parseable structured output.** Verified 2026-08-05 against `deepseek-v4-flash-0731` (see judge-model-change note below) — 3/3 trials parsed cleanly at `temperature=0`, ~2s after a cold-start warmup call
- [x] `temperature=0`, exact version string pinned in config — `rag/config.py` `JUDGE_MODEL_ID`/`JUDGE_TEMPERATURE`
- [x] Judge client is **separate and untraced** (see Phase 8 tracing hygiene) — `rag/judge.py` is a raw `openai.OpenAI` client against Fireworks, never routed through LangChain/LangSmith instrumentation
- [x] **Every judge call retries on unparseable output, and an unparseable result after retry is recorded as `null`, never as 0.0.** `rag/judge.py`'s `Judge.call_json` retries `JUDGE_MAX_RETRIES` times, returns `JudgeResponse(parsed=None, ...)` on exhaustion; every metric in `rag/metrics_llm.py` propagates that as `score=None`

  **⚠️ Judge model changed 2026-08-05.** The originally-planned `qwen2p5-72b-instruct` was retired from Fireworks' serverless catalog. Of the two providers already configured (Fireworks, Groq), the remaining Qwen options were `qwen3p7-plus` (Fireworks — `kind: CUSTOM_MODEL`, no Hugging Face ID, almost certainly a proprietary Alibaba DashScope proxy, not open weights) and `qwen/qwen3.6-27b` (Groq — confirmed open weight via `hugging_face_id`, but 27B, under the 32B floor). Switched to **`deepseek-v4-flash-0731`** on Fireworks: confirmed open weight (`kind: HF_BASE_MODEL`, checkpoint is `deepseek-ai/DeepSeek-V4-Flash` on Hugging Face), 284B total / ~13B active MoE parameters (clears the floor), different model family from the Gemini generator. Full reasoning in `config/pricing.yaml`'s `eval-judge` entry. Re-verify this pick is still current before the Tier 3 sweep — hosted serverless catalogs change.

**Golden set:**
- [~] Generate ~200 QA pairs with `rag/goldset.py` — sample chunks stratified by category and modality, one judge call per chunk asking for a question answerable *only* from that passage. **150/200 generated (2026-08-20)** (`results/goldset.jsonl`), now spanning `ai` (84), `cs.CL` (32), `finance` (34) — stratified round-robin correctly balanced the smaller categories against `ai`'s larger leftover pool, see `PHASE4_NOTES.md` §10. `goldset.py` excludes chunks already turned into a question on re-run, so growing the set is idempotent, not just additive-with-duplicates. Run `python -m rag.goldset -n <k>` for the remaining ~50
- [x] Reject any generated question containing the source paper's `topic` string — implemented with a one-shot corrective retry before discarding
- [x] **Hand-review ≥50 of them.** Done — all 150 generated rows reviewed via `rag/goldset_review.py` (interactive, blind pass over question + reference answer + source passage): 115 accept / 8 edit / 27 reject. `rag/goldset_curate.py` applies the verdicts → `results/goldset_curated.jsonl` (123 rows: `ai`=66, `cs.CL`=25, `finance`=32) — the set eval actually runs against, never `goldset.jsonl` directly. Disclosed in `README.md` (numbers there predate this growth — needs a refresh, see below)
- [x] Ensure coverage: per-category, and text-only vs figure-requiring questions tagged — `stratified_sample()` round-robins across (category, modality) so figure/table strata aren't drowned out by the corpus's text majority

**Metrics — `rag/metrics_llm.py`, one judge prompt each:**
- [x] **Context precision** — rank-weighted relevance of each retrieved chunk against the reference answer
- [x] **Context recall** — fraction of the reference answer's claims supported by the retrieved context
- [x] **Faithfulness** — decompose the answer into atomic claims, entail each against the context
- [x] **Response relevancy** — does the answer address the question actually asked
- [x] **Every metric returns its per-claim intermediate output alongside the score.** All four return their chunk/claim-level verdicts on `LLMMetricResult.claims`
- [x] Non-LLM retrieval metrics: hit@k, MRR, NDCG — `rag/metrics_retrieval.py`, unit-verified against hand-computed examples
- [x] **Judge validation:** hand-label 50 examples, compute Cohen's κ between judge and labels. Done, two ways — `rag/judge_validate.py` (blind hand-labeling vs. κ: overall 0.247/fair on 46 items; `context_recall`/`faithfulness` both 0.0 but diagnosed as a zero-variance artifact, not unreliability; `context_precision` 0.125→0.308 after a targeted rubric fix) and `rag/judge_probe.py` (10-case adversarial set with ground truth known by construction, confirming negative-case recall the blind sample couldn't test: 10/10 correct). Full reasoning in `PHASE4_NOTES.md`. **Load-bearing** — reported in `README.md`

**Tiered eval loop** — you will spend ~80% of your iterations on Tier 1, so make it fast and free:

| Tier | When | Questions | Metrics | Traced | Cost |
|---|---|---|---|---|---|
| 1 — Dev loop | Every retriever/chunking change | 30 | Retrieval only (hit@k, MRR, NDCG) | No | $0, seconds |
| 2 — Config comparison | Each new ablation arm | 80 subset | All four judge metrics | Yes | Low |
| 3 — Published sweep | **Once**, at the end | 240 (200 + 40 adversarial) | All four judge metrics | Yes | Budget the overage |

- [x] Implement all three tiers as separate make targets — `rag/eval.py` (`--tier {1,2,3} --config NAME`), wrapped by `Makefile`'s `eval` target. Validated end-to-end: Tier 1 on 20 examples (recall@50=1.0, recall@5 MRR/NDCG *higher* than @50 — the reranker correctly promoting the gold chunk within top-5), Tier 2 on 3 examples (all four judge metrics returned, zero nulls)
- [ ] LangSmith Dataset + Evaluators for Tiers 2–3 so each config is a tracked experiment — **key now resolves** (`LANGSMITH_API_KEY` added to `.env`; casing-alias bridge fixed in `rag/config.py`), but **no instrumentation exists yet** — this is still all of Phase 8, unstarted. The CSV output below is the ground-truth artifact either way (Phase 8's own conclusion: "the committed CSV is the artifact, not the trace links"); don't block Tier 2/3 usage on it
- [x] `make eval TIER=<n> CONFIG=<name>` produces one row of the ablation table — writes `results/eval/tier{n}_{config}_{timestamp}.csv`, one row per question, plus a summary dict
- [x] **Store token counts per run** so `$/query` can be recomputed from `config/pricing.yaml` without re-running eval — every Tier 2/3 row carries `gen_input_tokens`/`gen_output_tokens`/`judge_input_tokens`/`judge_output_tokens`

### Phase 5 — LangGraph agentic layer · 25h
**Status 2026-08-24: implemented and verified end-to-end against the live `ai`/`cs.CL`/`finance`
index.** See `PHASE5_NOTES.md` for full reasoning, including two bugs found and fixed while
building this (a thread-safety race in `Retriever`/`Generator`/`Reranker`'s lazy singletons,
first exposed by the parallel fanout; a checkpoint serializer gap for this codebase's own
dataclasses). `rag/eval.py` Tier 2/3 integration (so a sweep can actually produce ablation rows
4/5/10) is explicitly deferred, not attempted half-done — `PHASE5_NOTES.md` §9.
- [x] Graph state schema (query, subtasks, per-branch results, citations, route, scores) —
      `rag/agent_state.py` (`AgentState` TypedDict, `BranchResult` dataclass)
- [x] **Planner/decomposer node** — decomposes query into subtasks. *This is your query rewriting.* Make it bypassable so it becomes an ablation arm — `rag/planner.py`, `use_planner=False` skips the LLM call entirely
- [x] Router: query → relevant topic categories (evaluate against arXiv labels — free ground truth) — `rag/router.py`; `rag/router_eval.py` scores it against `goldset_curated.jsonl`'s `category` field (n=15 sanity check: hit_rate=0.667 vs 0.333 chance, mean_route_len=1.73 — not trivially returning all 3)
- [x] Parallel fanout via `Send` to **3 topic subagents** (same model, topic-specific system prompt, category-filtered retriever) — `rag/graph.py`'s `topic_subagent_node`, one `Send` per router-selected category (only `ai`/`cs.CL`/`finance` indexed so far, `config.INDEXED_CATEGORIES`)
- [x] Synthesizer node: merge branch drafts, dedupe, preserve citations — `rag/generation.py`'s `synthesize()`; regenerates against a freshly-renumbered merged source list rather than splicing branch citation numbers (PHASE5_NOTES §4 — the citation-hallucination risk that would create)
- [x] **Single-agent baseline path** — build this deliberately so the fanout has something to beat — `rag/graph.py`'s `single_agent_node` (ablation row 10). Deliberately keeps the planner's subtasks and router's category scope, isolating fanout as the only variable vs row 5 — a design decision made explicitly with the user, not the literal old Phase 1/3 pipeline (PHASE5_NOTES §3)
- [x] Checkpointing enabled for replay — `SqliteSaver` over `data/langgraph_checkpoints.sqlite` (not `MemorySaver` — replay must survive a process restart); verified by reading a checkpoint back from a separate process

### Phase 5b — Conversational query contextualization · 1.5h
**Status 2026-08-25: implemented and verified end-to-end** (`rag/contextualize.py`,
`rag/agent_cli.py --chat`). See `PHASE5B_NOTES.md` — including a real bug found and fixed before
shipping: reusing one `thread_id` across chat turns silently leaked the previous turn's
`branch_results` into the next one via its `operator.add` reducer.
- [x] Rewrite follow-ups ("what about the second one?") into standalone queries from chat history — `rag/contextualize.py`'s `contextualize_node`, wired as a new `START -> contextualize -> planner` step ahead of everything in Phase 5's graph; fails open to the raw follow-up on an LLM error, skips the call entirely on turn 1 (no history) or `use_context=False`
- [x] Needed for multi-turn, which dialogue rails assume. Distinct problem from decomposition — `rag/agent_cli.py --chat` REPL exercises it end-to-end; history capped at `config.MAX_HISTORY_TURNS`=3 per this section's own cut-list line below, and is caller-managed (a plain Python list), not accumulated via the checkpointer — see `PHASE5B_NOTES.md` §2-3 for why

### Phase 6 — Web search tool + escalation · 6h
**Status 2026-08-25: implemented and verified end-to-end against the live index.** See
`PHASE6_NOTES.md`. No new graph node — escalation is an inline check inside `synthesizer_node`/
`single_agent_node` (both already build each path's final answer), so nothing about the
corpus-only path changed when `use_web=False`.
- [x] Single web search tool — `rag/websearch.py`'s `web_search()`, one Tavily call
      (`langchain-tavily`) over `ddgs` (also a dependency, unused — PHASE6_NOTES §4); fails open
      to `[]` on a missing key or API error
- [x] One escalation rule: reranker top-score below threshold → escalate. Threshold tuned on the
      golden set, not guessed — `rag/websearch.py`'s `should_escalate()`; `rag/web_eval.py`
      measured the reranked top-1 score on all 123 curated goldset questions and set
      `WEB_ESCALATION_THRESHOLD=0.486` at the 5th percentile (median score 0.990,
      false_escalation_rate=4.9% — PHASE6_NOTES §2)
- [x] Web results clearly attributed separately from corpus results in the answer —
      `rag/generation.py`'s `Answer` carries `web_results`/`web_cited`/`web_uncited`/
      `web_invalid` as a fully parallel field set to `chunks`/`cited`/`uncited`/`invalid`, cited
      as `[W1]` via a disjoint regex from the corpus's `[1]` (PHASE6_NOTES §3). Verified against
      PHASE5B_NOTES §5's own documented abstention: "What is the Black-Scholes model used for in
      options pricing?" now gets a real, web-cited answer with `use_web=True`, and reproduces the
      exact original `INSUFFICIENT_CONTEXT` abstention with `--no-web` (PHASE6_NOTES §5)
- [x] No multi-query, no result reranking, no credibility scoring — out of scope. Tavily's own
      ranking is taken as-is, never re-scored

### Phase 7 — Guardrails · 10h
**Status 2026-08-25: implemented and verified end-to-end against the live index.** See
`PHASE7_NOTES.md`, including a real finding surfaced running the notebook (not designed in from
the start): disabling one rail doesn't guarantee that class of risk gets through, since another
rail can independently catch the same content on different grounds (§6).
- [x] Input rail: jailbreak detection — `rag/guardrails.py`'s `jailbreak_node`, the one node in
      this codebase that fails **closed** (not open) on an internal error — every other rail, and
      every node in Phases 5/5b/6, fails open (PHASE7_NOTES §3)
- [x] Input rail: off-topic classifier (**sensitive-topic folded in as an extra label**, not a
      separate rail) — `scope_node`'s `ScopeVerdict` schema (`on_topic`/`off_topic`/`sensitive`);
      `sensitive` blocks exactly like `off_topic`, with a distinct `blocked_reason`
      (PHASE7_NOTES §4)
- [x] One dialogue rail: clarification flow for underspecified queries — `clarify_node`, fails
      open (proceeds with the original question) on error
- [x] Output rail: grounding check (drop it if the latency cost isn't justified — and report that
      decision either way) — `grounding_node` reuses Phase 4's `faithfulness()`/`Judge` verbatim.
      Measured real cost: 7-11s per query where it ran (n=9 timed A/B pass,
      `notebooks/agent_phase7.ipynb` §7). **Kept, but flagged for `--no-grounding` on interactive
      use** — a stated tradeoff, not a silent default (PHASE7_NOTES §5)
- [x] **Measure and report the latency each rail adds** — `jailbreak_latency_s`/`scope_latency_s`/
      `clarify_latency_s`/`grounding_latency_s` on every `AgentState`; grounding's real A/B
      numbers in PHASE7_NOTES §5

### Phase 8 — LangSmith observability · 10h
**Status 2026-08-28: implemented and verified end-to-end.** See `PHASE8_NOTES.md`, including a
real bug found before any of this could be trusted: the account's LangSmith API key 403'd on
every call (even a bare read) — traced to the account being provisioned in LangSmith's APAC
region, whose keys don't authenticate against the default US API host. Fixed via
`LANGSMITH_ENDPOINT`, not a new key (§1).
- [x] Trace the full graph; auto-instrument the LLM client and HTTPX — verified by reading a real
      trace back via the SDK: 29 runs in one tree, every node/conditional-edge named individually,
      plus nested LLM (`ChatGoogleGenerativeAI`) and tool (`tavily_search`, Phase 6) calls, with
      **zero** `@traceable` decorators added anywhere — LangGraph's plain-function nodes trace
      themselves once tracing is on (PHASE8_NOTES §4)
- [x] Custom span attributes: retrieval scores per stage, chosen route, tokens, cost, per-stage
      latency, guardrail verdicts — already present for free, a consequence of this codebase's own
      state design since Phase 5 (nodes return only what changed; the tracer records exactly that
      as each run's outputs). Cost is the one exception — token counts are visible, a $ figure
      needs LangSmith's own model-pricing UI setting (PHASE8_NOTES §4)
- [x] Push metric results as tracked experiment metrics (quality over time, not measured once) —
      `rag/eval.py --langsmith` on a Tier 2/3 run, additive to the CSV (never a replacement — "the
      committed CSV is the artifact, not the trace links," this section's own words). Verified
      with a real n=3 run: all four judge metrics landed as per-example feedback, browsable across
      sweeps in LangSmith's compare view (PHASE8_NOTES §6). Real, disclosed cost consequence found
      along the way: LangSmith auto-upgrades any feedback-bearing trace to extended retention with
      no opt-out, so a `--langsmith` Tier 2 run can't actually stay on base/disposable retention as
      originally planned — quantified at $0.36/80-row sweep, accepted as trivial, `--langsmith`
      kept opt-in (PHASE8_NOTES §6)
- [x] Dashboard. If LangSmith's trace-oriented views aren't enough, export runs via SDK into the
      Streamlit page — LangSmith's own trace list + the compare view above already cover this
      phase's needs; a Streamlit export is deferred to Phase 9 (its own line item) rather than
      built twice (PHASE8_NOTES §7)

**Trace hygiene — 1 trace = 1 root run, so 5,000/month is workable with discipline.**

Volume math:

| Activity | Root runs | Notes |
|---|---|---|
| Tier 1 dev loop | 0 | Offline, untraced, unlimited |
| Tier 2 comparison, per arm | 80 | ~800 total across 10 arms |
| Tier 3 published sweep | 2,400 | 240 questions × 10 configs. ~48% of monthly quota |
| Interactive dev queries | ~200–500 | Sample at 10% |
| **Total for the month you sweep** | **~3,500–3,700** | Inside the free tier |

The nested LLM calls per pipeline run (planner, router, 3 subagents, synthesizer, guardrails) are **child runs** and cost nothing extra — that's what makes this fit.

Controls, in order of leverage:
- [ ] **Judge client untraced — still the #1 control.** The judge runs *outside* the pipeline graph, so a traced judge produces its own root run per metric call: 240 × 4 metrics ≈ 1,000–2,000 extra traces, which would nearly double a sweep. Separate client, tracing disabled. Writing the metrics ourselves makes this easier to guarantee, not harder — the judge client is constructed in one place instead of handed to a framework that decides its own instrumentation
- [ ] **Tracing off entirely during Phases 2–3.** Ingestion and retrieval aren't LLM work
- [ ] **Sample interactive dev queries at ~10%** once Phase 5 is live
- [ ] **Run the Tier 3 sweep early in a calendar month.** One sweep fits comfortably; a *second* one in the same month tips you over

⚠️ Two things blow the budget: tracing the judge, and re-running the full sweep. Both are avoidable — finish upstream work before Tier 3.

**Overage pricing (resolved 2026-07-30). 1 LSU = $1.00.**

| Retention | LSU / trace | Per 1,000 traces | Window |
|---|---|---|---|
| Base | 0.0005 | $0.50 | **14 days** |
| Extended | 0.005 (+0.0045 premium) | $5.00 | 400 days |

| Scenario | Traces | Cost |
|---|---|---|
| Tier 3 sweep, if fully over quota (base) | 2,400 | $1.20 |
| All Tier 2 comparisons (base) | 800 | $0.40 |
| Judge accidentally traced (base) | ~1,500 | $0.75 |
| **Extended-retention premium, Tier 3 only** | 2,400 | **$10.80** |

**Trace overage is negligible; retention is the real line item.** Don't optimize trace counts for cost — optimize them so the free 5,000 covers a clean sweep, which it does.

**⚠️ 14-day base retention is shorter than your gap to defense.** Two consequences, both sequencing-sensitive:
- [ ] **Set up three separate LangSmith projects** — `dev`, `tier2`, `tier3` — so retention is configured per project instead of globally
- [ ] **Decide retention on `tier3` BEFORE running the sweep.** Extended retention is generally applied at ingestion, not retroactively — you likely cannot upgrade traces after the fact. Verify this, because getting it wrong is unrecoverable
- [ ] **$10.80 for 400-day retention on the Tier 3 project is worth paying** if you want to click into a live trace during your defense. Leave `dev` and `tier2` on base — they're disposable
- [ ] **Commit the CSV regardless.** With a 14-day base window this is not optional — it's the primary artifact. Trace links are the demo; the CSV is the evidence

**Retention decision — this matters more than the cost.** Base retention is a short window; extended is the long one. Your defense may be weeks after the Tier 3 sweep runs.
- [ ] **Upgrade retention on the Tier 3 sweep only** (+6 LSU). Leave dev and Tier 2 traces on base retention — they're disposable
- [ ] **Don't depend on LangSmith retention for defense evidence at all.** Export the Tier 3 results — per-question metric scores, token counts, retrieved chunk IDs, latency per stage — to `results/tier3_<date>.csv` and **commit it**. Trace links are a nice-to-have; the committed CSV is the artifact. This also means the ablation table can be regenerated offline, and `$/query` recomputed against updated `pricing.yaml`, with no vendor dependency

### Phase 9 — API + thin UI · 3h
**Status 2026-08-28: implemented and verified end-to-end, including a live screenshot.** See
`PHASE9_NOTES.md`. **Extended the same day, on request**, with per-user in-memory conversation
history (`user_id`-keyed, no login) — a deliberate, disclosed exception to this section's own
"no persisted history" and to §7's cut-list line, not a silent scope change; still no auth (a
self-declared ID, not a password) and no persistence across a server restart. Building it surfaced
a real Phase 7 gap: a correctly-rewritten, obviously on-topic follow-up was getting blocked by
`scope_node`, which had no visibility into the conversation that motivated the rewrite. Fixed by
giving `scope_node`/`clarify_node` the same conversation-transcript context
`contextualize_node` already had; Phase 7's full single-turn probe suite re-verified unchanged
(17/17) after the fix. See `PHASE9_NOTES.md` §4.
- [x] FastAPI: `/query`, `/health` — `rag/api.py`; `Retriever`/`Generator`/`Judge`/the compiled
      graph built once at startup (`lifespan`), not per request. `/query` now also takes a
      `user_id` and `/reset` clears one user's history on demand (the history extension above)
- [x] Streamlit, **one page**: query box, answer, citations, retrieved chunks w/ scores, figure
      thumbnails — `streamlit_app.py`, a pure HTTP client of the API (no direct model/graph
      dependency). Verified live: driven headlessly with Playwright against the running API,
      screenshotted with the answer, citations, and metadata all rendered correctly, zero console
      errors (PHASE9_NOTES §3). Later grew a name/ID field, a running transcript, and a "New
      conversation" button for the history extension
- [x] No auth, no streaming, no persisted history. Screenshot it and move on — still true of
      streaming and auth; persisted history is the one documented, deliberate exception above.
      Every `/query` call still gets a fresh per-turn checkpoint `thread_id`, never a shared one

### Phase 10 — Failure analysis · 29h
**This is the phase that separates the project from a demo.** Detail in §3.

### Phase 11 — Generation-prompt iteration · 3h
- [ ] Driven by the oracle-context arm (§3②). If faithfulness is low with perfect context, the fault is here
- [ ] Iterate: citation grounding, "answer only from provided context," explicit abstention instruction, output structure
- [ ] Log each prompt version as a tracked LangSmith experiment — show the improvement curve

### Phase 12 — Write-up & defense prep · 18h
- [ ] Run the full ablation sweep
- [ ] Architecture diagram (generate from the LangGraph structure)
- [ ] README: spec, stack, ablation table, failure taxonomy chart, known limitations
- [ ] `decisions.md` finalized
- [ ] Pre-write answers to §5 questions
- [ ] Failure case gallery: 8–10 documented cases

---

## 3. Failure analysis detail (Phase 10)

### ① Failure mode taxonomy · 6h
Hand-label ~50 bottom-quartile cases into:

- [ ] Retrieval miss (gold chunk not in top-50 at all)
- [ ] Retrieved but reranked away (in top-50, not top-5)
- [ ] Retrieved and ignored by generator
- [ ] Chunk boundary split the answer
- [ ] Wrong topic route → wrong corpus slice
- [ ] Figure/table content lost in parsing
- [ ] Citation hallucination (right answer, wrong source)
- [ ] Genuinely unanswerable from corpus

**Deliverable:** stacked bar chart of failure mode × config. Turns "faithfulness = 0.87" into "here are my four failure modes and which fix addressed which."

### ② Retrieval ceiling decomposition · 4h
Highest ROI item in the whole project. Report three numbers per config:

| Metric | Isolates |
|---|---|
| recall@50 (pre-rerank) | Can the retriever find it at all? Your ceiling |
| recall@5 (post-rerank) | Is the reranker keeping the right things? |
| faithfulness | Does the generator use what it was given? |

- [ ] Implement all three, per config
- [ ] Every failure now attributable to a stage

### ③ Oracle-context arm · 2h
- [ ] Feed ground-truth contexts directly to the generator, retrieval bypassed
- [ ] That's your generation ceiling. Forecloses "did you try improving retrieval?" in one chart
- [ ] Feeds Phase 11

### ④ Adversarial / negative test set · 5h
~40 hand-written questions:
- [ ] Unanswerable-from-corpus (~10)
- [ ] Multi-hop across topics (~10)
- [ ] False-premise questions (~10)
- [ ] Near-duplicate distractors (~10)
- [ ] **Measure abstention rate.** A system at 0.92 faithfulness on answerable questions that confidently fabricates on unanswerable ones is not production grade — and saying so about your own system is a strong move

### ⑤ Guardrail confusion matrix · 4h
- [ ] 100 probes: 25 jailbreak, 25 off-topic, 25 sensitive, 25 **benign-but-superficially-risky**
- [ ] Real 2×2 with FP and FN rates
- [ ] The false-positive rate on legitimate queries is the number nobody measures

### ⑥ Sliced metrics · 3h
- [ ] Every metric broken down by arXiv category
- [ ] Every metric broken down by text-only vs figure-requiring
- [ ] One slice will be visibly worse. Diagnose, fix, show before/after

### ⑦ Latency & cost waterfall · 2h
- [ ] p50/p95 per stage from LangSmith traces
- [ ] Identify the real bottleneck (usually reranker or slowest fanout branch)

### ⑧ Regression suite in CI · 3h
- [ ] 50-question subset, cheap judge, runs on every push
- [ ] Fails the build if faithfulness drops >3 points

---

## 4. Ablation table (the centerpiece)

| # | Config | recall@50 | recall@5 | Ctx Precision | Ctx Recall | Faithfulness | Answer Rel. | Abstention (neg. set) | p95 latency | $/query |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Dense only | | | | | | | | | |
| 2 | + BM25 (RRF) | | | | | | | | | |
| 3 | + cross-encoder rerank | | | | | | | | | |
| 4 | + planner decomposition | | | | | | | | | |
| 5 | + parallel topic fanout | | | | | | | | | |
| 6 | + web fallback | | | | | | | | | |
| 7 | + guardrails | | | | | | | | | |
| 8 | **Long-context control** (top-50 stuffed, no rerank) | | | | | | | | | |
| 9 | **Oracle context** (generation ceiling) | — | — | — | — | | | | | |
| 10 | **Single-agent baseline** | | | | | | | | | |

Row 8 matters: with a huge context window it's tempting to stuff 50 chunks and skip reranking. Test it. If reranking wins, that's a killer defense point. If long-context wins on quality but costs 8× per query, that's a better one.

Row 4 is your query-rewriting evidence — planner bypassed vs. enabled.

---

## 5. Defense questions to pre-answer

- [ ] Why RRF over weighted score fusion?
- [ ] What's your chunking strategy and what did you measure it against?
- [ ] Faithfulness is 0.87 — what are the failing 13%? *(Requires having read failure cases)*
- [ ] How do you know your LLM judge is trustworthy? *(κ number)*
- [ ] Isn't using the same model to generate and judge biased? *(Different families — say so before they ask)*
- [ ] How do you know the parallel agents help? *(Row 5 vs row 10)*
- [ ] Did you do query rewriting? *(Three answers: planner decomposition, conversational contextualization, and HyDE/multi-query as explicitly-scoped-out with reasoning)*
- [ ] Why not just stuff everything into the long context window? *(Row 8)*
- [ ] What's your p95 latency and where does it go?
- [ ] What does your guardrail miss, and how often does it block legitimate queries?
- [ ] How would you scale to 10M docs? *(Sharding, HNSW tuning, cascade reranking, semantic cache)*
- [ ] What would you do differently / what's next?
- [ ] Why upsert instead of insert, and what makes it safe to re-run? *(Point ID = deterministic UUIDv5 of stable fields, so re-ingestion overwrites a chunk with itself instead of duplicating it)*
- [ ] Why UUIDv5 for chunk IDs instead of a random UUID4 or an incrementing int? *(Derived, not stored — any process holding `arxiv_id`/`ordinal`/`modality`/`figure_ref` recomputes the same ID with no lookup. Trade-off: re-chunking changes ordinals and orphans the old points, so a chunking-strategy change needs a collection drop, not just a re-run)*
- [ ] BGE-M3's max_seq_length is 8,192 tokens — why does that matter, and how did you verify no chunk exceeds it? *(Truncation past the limit is silent, not an error — audited token lengths, not char lengths, before embedding; p95=346, max=790, zero over limit)*
- [ ] Why dense-only in Phase 1 when BGE-M3 also emits sparse and ColBERT-style vectors? *(Isolates dense as a true floor for the ablation table; BM25/sparse arrives Phase 3 as an additional named vector, not a schema migration, because `dense` was named up front)*
- [ ] What does a retrieval call actually return, and what's in a result? *(`query_points()` → `QueryResponse.points`, a ranked `list[ScoredPoint]`; each has `id` (round-trips to the ingested chunk), `score` (cosine, since the collection uses `Distance.COSINE` on unit-normalized vectors), and `payload` (text + metadata — Qdrant is the only store, no second lookup by ID))*

---

## 6. Budget

| Item | Estimate |
|---|---|
| Figure captioning (cheap Gemini tier + batch) | $10–25 |
| Embeddings (BGE-M3 local) | $0 |
| Qwen judge — Tier 2 comparison runs | $20–40 |
| Qwen judge — Tier 3 published sweep (once) | $20–40 |
| Generation across ablation configs | $20–40 |
| Dev/testing queries | $30 |
| LangSmith trace overage | **$0** if the judge stays untraced and Tier 3 runs once (~3,600 of 5,000 free). Worst case, a full re-sweep is $1.20 |
| **LangSmith 400-day retention, `tier3` project only** | **$10.80** — the dominant LangSmith cost; buys live trace links at your defense |
| **Total** | **~$110–190** |

Controls:
- Tier 1 dev loop is $0 (non-LLM retrieval metrics, no tracing) — keep iterations there
- Tier 3 runs **once**. If you find yourself re-running it, something upstream wasn't finished
- Recompute `$/query` from stored token counts × `config/pricing.yaml` — never re-run a sweep to fix a pricing figure

---

## 7. Cut list (documented as decisions, not omissions)

Keeping this in the README is itself a defense asset — it shows scoping judgment.

| Cut | Rationale |
|---|---|
| Layout-aware parsing (Docling/unstructured) | Highest-risk time sink with zero defense payoff |
| Eval framework (Ragas, then DeepEval) | Dropped both. Ragas 0.4.3 is dormant and was **un-importable** in this venv — it pins no upper bound on `langchain-community`, which removed the module it imports. DeepEval trades that for a different unpinned surface plus posthog/sentry telemetry. The four metrics are judge prompts, not algorithms; writing them removed 9 packages and made the judge, the prompt and the parsing all things I pin and can explain |
| Full-text per-doc extraction | Traded parsing depth for eval depth; 10K-doc requirement preserved |
| Standalone HyDE / blind multi-query expansion | Redundant with the planner's decomposition, which is measured directly |
| 5 subagents → 3 | Identical fanout logic; more configs, more eval cost, no additional insight |
| Rich UI (React, auth, streaming, history) | Zero hiring signal. **Partial exception, 2026-08-28:** per-user in-memory conversation history was added to the Phase 9 demo on explicit request — a conscious, disclosed reversal of this line, not a silent scope creep. Still no auth (a self-declared `user_id`, not a login), no streaming, no persistence across a server restart. See `PHASE9_NOTES.md` §4 |
| 6 guardrails → 3 | Sensitive-topic folded into off-topic as a label |
| Web search sophistication | One tool, one escalation rule |
| Reranker / embedding fine-tuning | Rabbit hole, no defense payoff |
| Multi-turn memory beyond 3 turns | Out of scope |
| >2 chunking strategies | Diminishing returns |

---

## 8. Ordering rule

**Eval before agents.** Phase 4 ships before Phase 5. Every subsequent change is then a measurable delta, and the ablation table writes itself. If eval comes last, you get bad numbers with no idea which component owns them.
