# Phase 5 — LangGraph agentic layer: decisions, bugs, and defense notes

Working record for `rag/agent_state.py`, `rag/planner.py`, `rag/router.py`, `rag/graph.py`,
`rag/agent_cli.py`, `rag/router_eval.py`, and `notebooks/agent_phase5.ipynb`. Same purpose as
PHASE1/3/4_NOTES.md: what was decided, what broke, what the evidence was, what's still open.

---

## 1. Architecture

```
                         START
                           |
                       [planner]   -- bypassable (use_planner=False -> subtasks=[question])
                           |
                        [router]   -- LLM call -> categories subset of config.INDEXED_CATEGORIES
                                      fail-open to all indexed categories on empty/invalid output
                           |
                  route_after_router (conditional edge)
                    /                              \
          use_fanout=True                    use_fanout=False
                /                                    \
   Send(topic_subagent, cat) x N              [single_agent]
    (parallel, one per routed category)      (pooled retrieval across
                \                              routed categories,
        [topic_subagent] x N                   one generation call)
                /                                    |
        [synthesizer]                                |
                \                                    /
                          END (AgentState.final_answer)
```

Reuses Phase 1/3 unchanged: `Retriever.search()` (category filter, BM25/RRF, rerank),
`Reranker.rerank()`, `Generator.generate()`. No new retrieval or generation primitives — Phase 5
is orchestration on top of what already existed, plus two small additive changes to
`rag/generation.py` (an optional `system_prompt` kwarg on `generate()`, and a new `synthesize()`
function) and one new `Generator.structured()` helper reused by both the planner and router.

---

## 2. Ablation row mapping (ROADMAP §4)

| Row | Config |
|---|---|
| 4 — + planner decomposition | `use_planner=True, use_fanout=False` |
| 5 — + parallel topic fanout | `use_planner=True, use_fanout=True` (full graph) |
| 10 — single-agent baseline | `use_planner=True, use_fanout=False` — **same config as row 4** |

Row 10 being identical to row 4's config is deliberate, not an oversight — see §3.

---

## 3. The row-10 design decision

ROADMAP requires a "single-agent baseline path... built deliberately so the fanout has something
to beat," and separately asks "how do you know the parallel agents help? (row 5 vs row 10)."
Neither line pins down *what stays the same* between row 5 and row 10 besides fanout itself, and
that choice materially changes what the comparison proves. Two readings were considered:

- **Literal Phase 1/3 baseline** — no planner, no router, whole-corpus search, one call. Already
  fully implemented (`rag.cli`/`rag.eval`'s `dense+bm25+rerank` config), zero new code. But then
  row 5 vs row 10 conflates three differences at once (planner decomposition + router
  category-scoping + fanout structure) — a win doesn't tell you *which one* did the work.
- **Isolate fanout only** (chosen) — row 10 keeps the planner's subtasks and the router's
  category scope, and only removes the parallel-branch structure: every (subtask, category) pair
  is retrieved, pooled into one candidate set, reranked once against the original question
  (`Reranker.rerank()`, same as `Retriever.search()`'s own post-fusion rerank step — not a new
  primitive), and answered with a single generation call. No synthesizer.

This makes fanout the *only* variable between `single_agent_node` and
`topic_subagent_node`/`synthesizer_node` — the row 5 vs row 10 delta in the ablation table
answers "does splitting per-category retrieval into parallel specialist calls beat pooling them"
specifically, not "is the whole new pipeline better than the old one." Decided with the user
2026-08-24 after walking through both readings concretely (what fanout actually does branch by
branch, what each baseline reading would and wouldn't hold constant).

**Trade-off:** this costs slightly more code than the literal-baseline reading (a real
`single_agent_node`, not a passthrough to existing `rag.cli`), and it's not free — a smoke test
on a genuinely cross-topic question (`notebooks/agent_phase5.ipynb`) showed fanout costing ~3.6×
row 10's input tokens (4,844 vs 1,338) for the same question, latency roughly comparable (0.95s
vs 1.26s — row 10's rerank pass adds its own cost). That cost delta is real content for the
ablation table, not a bug to fix.

---

## 4. Synthesizer regenerates, it doesn't splice text

ROADMAP: "Synthesizer node: merge branch drafts, dedupe, preserve citations." Each branch cites
against its own local numbering (`[1]`..`[len(branch_chunks)]`); those numbers don't carry over
to a merged source list. Programmatically remapping them, or splicing branch draft text together
directly, risks exactly the failure mode `rag/generation.py` was built to design out from the
start (PHASE1_NOTES §3.1: a citation that looks right and points at the wrong source).

`rag.generation.synthesize()` is a second full generation pass instead: branch drafts are shown
as reference notes (their reasoning, explicitly labeled "do not reuse these numbers"), and the
model writes a fresh answer against a freshly-renumbered merged source list
(`format_context()`), parsed with the same `parse_citations()` every other ablation arm uses. A
synthesized answer is exactly as machine-verifiable as a single-call one — no special case.

`include_images=False` for synthesis (branch drafts already grounded any figures; re-attaching
every branch's images to one synthesis call would blow up prompt size for no measured benefit —
a simplification, stated rather than silently made).

---

## 5. Router fails open, never empty

If the router's LLM call errors, returns no valid categories, or returns categories outside
`config.INDEXED_CATEGORIES`, it falls back to **every** indexed category rather than crashing or
fanning out to zero branches. An empty route produces no answer at all — strictly worse than
over-including a category. Same "failures logged, never fatal" posture as ingestion
(PHASE1_NOTES §5, §9).

Sanity-checked, not just asserted: `router_eval.py` on a 15-question sample of
`goldset_curated.jsonl` gave hit_rate=0.667 against a chance baseline of 0.333 (uniform-random
over 3 categories), mean_route_len=1.73 — meaningfully narrower than "always return all 3"
(which would trivially score hit_rate=1.0 and is exactly what mean_route_len exists to catch).
Manually spot-checked single-topic questions too: "What is the Black-Scholes model used for in
options pricing?" → `['finance']` only; "How do transformer architectures affect risk modeling in
finance?" → `['finance', 'ai', 'cs.CL']`. The router is doing real discrimination, not defaulting.

---

## 6. Bug found: `Retriever`/`Generator`/`Reranker`'s lazy singletons are not thread-safe

Found running `rag/agent_cli.py` end-to-end for the first time, not by reasoning about it first —
same discovery pattern as every bug in PHASE1_NOTES §3. First run of a genuinely cross-topic
question raised `StorageLockedError` from inside `Retriever.client`, the exact error the code's
own comment describes as "usually a running notebook kernel" — but no notebook kernel was
running. Reproduced deterministically on a second attempt (both immediately after a `--show-context`
run that produced zero output before failing, meaning the exception fired before `render()` ever
printed anything).

**Root cause:** Phase 5's parallel fanout (`Send`-based, executed by LangGraph via a thread pool)
is the *first* caller anywhere in this codebase to share one `Retriever`/`Generator`/`Reranker`
instance across concurrent threads — every prior caller (`rag.cli`, `rag.eval`'s tier loops) is
strictly sequential. `Retriever.client`/`.embedder`/`.sparse_embedder`/`.reranker`,
`Generator.llm`, and `Reranker.model` are all lazy singletons with a plain
`if self._x is None: self._x = ...` pattern and no lock. Two `topic_subagent` branches racing on
first access both see `None`, both proceed to construct — for the embedded-mode Qdrant client
specifically, two concurrent `QdrantClient(path=...)` calls against the same storage directory
don't just duplicate work, the loser raises exactly the "storage locked by another instance"
error, because embedded Qdrant takes an exclusive single-writer lock per client instance, not
per process.

**Fix:** double-checked locking (`threading.Lock()` per instance, check-lock-recheck-construct)
added to all six lazy properties — `rag/retrieval.py` (`client`, `embedder`, `sparse_embedder`,
`reranker`), `rag/generation.py` (`Generator.llm`), `rag/rerank.py` (`Reranker.model`). Fast path
(already-initialized) stays lock-free; only the one-time construction is serialized. Verified by
re-running the same cross-topic question that failed before the fix — now completes and fans out
to all three branches correctly (`notebooks/agent_phase5.ipynb`, cell `fanout-build-run`).

**Why this didn't show up in Phase 1/3 testing:** nothing before Phase 5 ever called a shared
`Retriever`/`Generator` from more than one thread. This is the same class of lesson as
PHASE1_NOTES §3.6 (MuPDF thread-unsafety under a `ThreadPoolExecutor`) — a latent bug that is
invisible under sequential use and only surfaces the first time something genuinely concurrent
touches the same object.

---

## 7. Checkpointing: SqliteSaver, and a second bug it surfaced

**Why SqliteSaver over MemorySaver:** matches the project's existing local-file-persistence
pattern (embedded Qdrant is the same idea) and actually satisfies "checkpointing for replay"
across process restarts — `MemorySaver` loses everything the moment the process exits, which
defeats the stated purpose. `langgraph-checkpoint-sqlite` was not installed (confirmed via a
direct import check before adding it) and is now a real dependency.

**Bug found:** the default checkpoint serializer logged `Deserializing unregistered type
rag.retrieval.RetrievedChunk from checkpoint. This will be blocked in a future version` (and the
same for `rag.generation.Answer`) the first time a checkpoint was written in one process and read
back in a fresh one. `AgentState`'s checkpointed values include this codebase's own dataclasses,
not just JSON-safe primitives — LangGraph's `JsonPlusSerializer` supports them via msgpack today
but is moving toward `LANGGRAPH_STRICT_MSGPACK` blocking unregistered types by default. Since the
entire reason for choosing SqliteSaver was reliable replay across restarts, leaving this as a
warning to silently break on a future LangGraph upgrade wasn't acceptable.

**Fix:** `rag/graph.py`'s `_default_checkpointer()` constructs an explicit
`JsonPlusSerializer(allowed_msgpack_modules=[...])` naming `RetrievedChunk`, `Answer`, and
`BranchResult`, passed to `SqliteSaver(conn, serde=serde)`. Verified: replaying a checkpoint from
a fresh process now produces no warning (re-ran the same round-trip test that surfaced it).

---

## 8. Retrieval strategy inside a branch / row 10

**`topic_subagent_node`:** for each subtask (from the planner; `[question]` if bypassed), one
`retriever.search(subtask, category=branch_category, use_bm25=True, use_rerank=True)` call —
i.e. each subtask gets a full dense+BM25+rerank pass within the branch's category, independently.
Results across subtasks are merged and deduped by chunk id (keep the higher score) with no
further truncation — worst case (planner's own cap of 1-4 subtasks × `DEFAULT_TOP_K`=5) is ~20
chunks per branch, small enough not to need a second cap. **Known cost implication:** branch
context size scales with subtask count, uncapped — this is exactly what the ablation table's
$/query column is supposed to surface, not something to hide behind an artificial cap.

**`single_agent_node` (row 10):** every (subtask, category) pair retrieved *unreranked*
(`use_rerank=False`, full `candidate_k` pool each), pooled and deduped across the entire route,
then reranked **once** against the *original* question — not per-subtask — mirroring how
`Retriever.search()` itself reranks after fusion rather than per-source. Final generation call
also uses the original question and the base `SYSTEM_PROMPT` (no category specialization, since
the pooled context can span categories).

---

## 9. Known gaps / deferred, not silently dropped

1. **`rag/eval.py` integration is not done.** `run_tier1`/`run_tier23` are tightly coupled to
   `Retriever.search()`'s flat signature; plumbing the graph through them so Tier 2/3 sweeps can
   actually produce ablation rows 4/5/10 is its own small task, flagged during planning and left
   for a follow-up rather than attempted half-done here. `router_eval.py` and the
   `agent_cli.py --json` output already carry everything a future integration would need
   (route, subtasks, per-branch tokens/latency, final answer with usage).
2. **Router evaluated on n=15**, not the full 123-row curated goldset — a sanity check, not a
   final number for the ablation table. Re-run `python -m rag.router_eval` with no `-n` (or a
   larger one) before quoting a router accuracy figure anywhere final.
3. **Only `ai`/`cs.CL`/`finance` are indexed** (`config.INDEXED_CATEGORIES`, PHASE3_NOTES) — the
   router can never route to `astronomy`/`hep-th` until they're ingested. One-line config change
   when that happens; the router's prompt is built from this list, not hardcoded.
4. **Branch/pooled retrieval has no cross-subtask cap** (§8) — watch $/query on row 5 specifically
   if questions with many subtasks turn out to be common in the golden set.
5. **`SYNTHESIS`/planner/router all reuse `GEN_MODEL_ID`**, not a separate "Pro tier" model per
   ROADMAP's stack table — no Pro-tier Gemini ID is verified anywhere in this repo yet
   (`config/pricing.yaml`'s generator entry is still UNVERIFIED). Splitting synthesis onto a
   verified Pro-tier model later is a one-line change in `rag/graph.py`'s `synthesizer_node`,
   not a prerequisite for Phase 5.

---

## 10. Mechanics

```
python -m rag.agent_cli "question"                # full pipeline (planner+router+fanout+synthesis)
python -m rag.agent_cli "question" --no-fanout     # row 10 (pooled, no parallel branches)
python -m rag.agent_cli "question" --no-planner --no-fanout   # subtasks=[question], still router-scoped
python -m rag.router_eval [-n N]                   # router-only accuracy vs goldset_curated.jsonl
```

`notebooks/agent_phase5.ipynb` walks every piece (state shape, planner alone, router alone, full
fanout, row-10 comparison, checkpoint replay) cell by cell against the live index — run it to see
the pipeline work, not just read about it. Needs `nbconvert`/`nbclient` (added as project
dependencies) to execute non-interactively; both are optional for just reading the notebook in
Jupyter directly.

---

## 11. One-line answers for the defense

- **How do you know the parallel agents help?** Row 5 vs row 10, and row 10 is built to differ
  from row 5 *only* in fanout — same planner decomposition, same router category scope, just
  pooled-and-single-call instead of parallel-and-synthesized. See §3 for why that reading was
  chosen over a literal old-pipeline baseline.
- **Why does the synthesizer regenerate instead of merging text?** Reusing a branch's local
  citation numbers against a different merged source list is exactly the citation-hallucination
  shape this codebase designed out at Phase 1 (right answer, wrong source). Regenerating against
  a freshly-numbered list keeps every ablation arm's citations equally machine-verifiable. §4.
- **What did concurrency actually break, and how do you know it's fixed?** Two branches racing on
  a shared `Retriever`'s lazy Qdrant client, reproduced running `rag/agent_cli.py` end-to-end
  (not found by inspection). Fixed with double-checked locking on every lazy singleton the
  fanout touches; re-ran the exact question that failed and it now completes. §6.
- **Does checkpointing actually survive a restart?** Verified directly: wrote a checkpoint in one
  process, read it back in a separate process against the same `data/langgraph_checkpoints.sqlite`
  file, confirmed the state round-trips — and fixed a serializer warning that would have silently
  broken this on a future LangGraph upgrade. §7.
- **Is the router actually routing, or does it just return everything?** `mean_route_len`
  (router_eval.py) exists specifically to catch a lazy "always all 3" router — 1.73 on the n=15
  sample, well under 3, plus a hit_rate of 0.667 against a 0.333 chance floor. §5.
