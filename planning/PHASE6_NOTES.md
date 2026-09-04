# Phase 6 — Web search tool + escalation: decisions and defense notes

Working record for `rag/websearch.py`, `rag/web_eval.py`, the `use_web` additions to
`rag/agent_state.py`/`rag/graph.py`, the web-citation extensions to `rag/generation.py`, and
`rag/agent_cli.py --no-web`. Same purpose as PHASE1/3/4/5/5B_NOTES.md: what was decided, what
was measured, what's still open.

---

## 1. Where it sits

No new graph node. Web escalation is an inline check inside `synthesizer_node` and
`single_agent_node` — the check runs immediately before each path's one final generation call:

```
route_after_router (conditional)
  |                              |
Send(topic_subagent) x N    single_agent -- pool, rerank once --------\
  |                                                                    |
synthesizer -- merge, rerank'd scores already on the chunks ----------+
  |                                                                    |
  should_escalate(chunks)?  (rag/websearch.py)                        |
  |__ yes -> web_search(question) -> generate/synthesize WITH web_results
  |__ no  -> generate/synthesize WITHOUT web_results
                                                                       |
                                                                     END
```

**Why here and not a new conditional-edge node:** both `synthesizer_node` and `single_agent_node`
already hold the fully merged, fully reranked chunk set at the exact moment they're about to call
`generate()`/`synthesize()` — that's the only point in the graph where "the final answer's
corpus context" actually exists as one list with one top score. Splitting retrieval and
generation into separate nodes just to insert an escalation node between them would be a bigger
restructuring than a 6h phase (ROADMAP) calls for, for no behavioral difference.

**Why not inside `topic_subagent_node` (per branch):** each fanout branch only ever sees its own
category's chunks, so a per-branch check would escalate on a branch that individually scored low
even when a sibling branch's chunks already made the *merged* answer well-supported — the wrong
signal for a corpus-wide "is the whole answer under-supported" rule. Checking once, in
`synthesizer_node`, against the merged/deduped chunk set (which already carries each branch's own
reranked scores) is the router-independent, corpus-wide version of the rule ROADMAP asks for.

---

## 2. The escalation rule, and why the threshold isn't guessed

ROADMAP: "One escalation rule: reranker top-score below threshold → escalate. Threshold tuned on
the golden set, not guessed." `rag/websearch.py`'s `should_escalate(chunks)` is exactly that:
`chunks[0].score < config.WEB_ESCALATION_THRESHOLD` (empty `chunks` always escalates — score
treated as 0.0, and no corpus signal at all is the strongest case for going to the web).

**What the score actually is:** `chunks[0].score` after a `use_rerank=True` retrieval is the
`BAAI/bge-reranker-v2-m3` cross-encoder's relevance score for the single best candidate — not
the bi-encoder cosine/RRF score the initial dense+BM25 retrieval produced. `Reranker.rerank()`
(`rag/rerank.py`) runs the cross-encoder over `(query, chunk.text)` pairs and overwrites `.score`
with its output before re-sorting:

```python
scores = self.model.predict([(query, chunk.text) for chunk in chunks])
for chunk, score in zip(chunks, scores):
    chunk.rerank_score = float(score)
    chunk.score = float(score)   # score always reflects whichever stage ran last
```

That output is a sigmoid-bounded relevance score in roughly `[0, 1]`, not a raw, unbounded
logit or similarity value — confirmed by the measured distribution below (0.078–0.9999, never
negative, never far past 1). Checking `chunks[0].score` after reranking genuinely asks "how
confident is the reranker that its single best candidate answers this," which is exactly the
"reranker top-score" ROADMAP's rule names — not a leftover RRF or dense score from an earlier
stage.

**The tuning problem:** `goldset_curated.jsonl`'s 123 questions are all answerable from the
corpus by construction — `rag/goldset.py` generates each one from a real, already-indexed chunk.
There is no negative/unanswerable set yet (Phase 10 §④'s ~40-question adversarial set, including
"unanswerable-from-corpus," is a later phase). That means the golden set cannot teach this
threshold to recognize an unanswerable question directly — there's nothing in it to test against.

What it *can* do: show how low a **genuine, real match's** reranked top-1 score gets across the
full curated set. `rag/web_eval.py` runs `Retriever.search(question, use_bm25=True,
use_rerank=True)` (whole-corpus, uncategorized — escalation is router-independent) for every
curated question, and sets the threshold at a low percentile (5th, by default) of that
distribution. That means: only ~5% of genuinely-answerable curated questions would incorrectly
trigger web escalation, while a question genuinely outside the corpus — which the smoke tests in
PHASE5_NOTES.md already show scoring far lower (every cross-topic abstention case there scored
low enough to be an obvious outlier) — reliably escalates. A percentile of true-positive scores,
not a number picked by eye.

**Measured 2026-08-25**, `python -m rag.web_eval` (n=123, the full curated goldset, percentile=0.05):

```
{
  "n": 123,
  "percentile": 0.05,
  "threshold": 0.48619261384010315,
  "min": 0.07782477140426636,
  "max": 0.9999308586120605,
  "mean": 0.9220582750148889,
  "median": 0.9900396466255188,
  "would_escalate_at_threshold": 6,
  "false_escalation_rate": 0.04878048780487805
}
```

Most genuine matches score well above 0.9 (median 0.990) — the distribution is heavily
left-skewed, with a short tail of harder-but-still-answerable questions down toward 0.08.
`WEB_ESCALATION_THRESHOLD=0.4862` sits in that tail: only 6/123 curated questions (4.9%) would
incorrectly trigger escalation, while sitting well below where the bulk of genuine matches score.
`WEB_ESCALATION_THRESHOLD` in `rag/config.py` is set to this measured value, not a placeholder.

---

## 3. Web citations are a separate numbering space, never merged into the corpus's

ROADMAP: "Web results clearly attributed separately from corpus results in the answer." The
mechanism reuses Phase 1's whole citation-hallucination defense (PHASE1_NOTES §3①,
PHASE5_NOTES §4): the model is given numbered handles and asked to reproduce only those numbers,
never asked to reproduce an identifier. Web results get their **own** handle space — `[W1]`,
`[W2]`, parsed by a separate regex (`WEB_CITATION_RE`) from the corpus's `[1]`, `[2]`
(`CITATION_RE`) — rather than continuing the corpus's numbering or reusing the same bracket
syntax with a type flag buried in the payload.

Two regexes instead of one unified one specifically because `CITATION_RE` (`\[([0-9]...)\]`)
requires a digit immediately after `[`, so it structurally cannot match `[W1]` — the two citation
forms are unambiguous to the parser by construction, not by convention the model has to get right
every time. `Answer` carries `web_results`/`web_cited`/`web_uncited`/`web_invalid` as a fully
parallel set of fields to `chunks`/`cited`/`uncited`/`invalid` (rag/generation.py), and
`answer.sources()` / `answer.web_sources()` are two separate accessors — `rag/agent_cli.py`'s
`render()` prints them under two separate headings ("Sources" vs. "Web sources (separate from the
corpus above)"), so the separation ROADMAP asks for is enforced at the data-model level, not just
in how the CLI happens to print things.

The system prompt (`SYSTEM_PROMPT`/`SYNTHESIS_SYSTEM_PROMPT`, both extended with one new rule)
tells the model explicitly: prefer a corpus citation when both would support the same claim, use
a web citation only for what the corpus doesn't cover, and say so explicitly when a claim rests on
a web result instead of the corpus. This is instruction, not enforcement — nothing stops the model
from over-relying on web results the way nothing stops it from mis-citing a corpus chunk; it's the
same trust boundary Phase 1 already accepted for corpus citations, extended consistently.

---

## 4. Tool choice: Tavily over ddgs

Both `langchain-tavily` and `ddgs` were already project dependencies (`pyproject.toml`) before
this phase started, with `TAVILY_API_KEY` also already present in `.env` — someone anticipated
this exact phase. Tavily was picked as *the* single tool (ROADMAP: "single web search tool," not
"one of either"): it returns ranked, LLM-ready snippets from one call, which is exactly the shape
`rag/generation.py`'s citation model needs (title/url/content per result) — no second client, no
extra normalization step, and nothing to write the "no multi-query, no result reranking, no
credibility scoring" cut-list items *against*, since Tavily's own ranking is taken as-is and never
re-scored.

`web_search()` fails open, matching every other LLM-adjacent call in this codebase (planner,
router, contextualize — PHASE1_NOTES §5/§9's "failures logged, never fatal"): a missing key or a
live API error returns `[]`, degrading to the corpus-only answer that was already going to be
generated anyway, rather than crashing a request the corpus could have partially or fully
answered on its own.

---

## 5. Verified end-to-end (real models, live index)

`notebooks/agent_phase6.ipynb` picked its two demo questions deliberately rather than inventing
plausible-sounding ones: "What is the Settlement Modernisation Index (SMI) designed to capture?"
comes straight out of `goldset_curated.jsonl` (answerable by construction), and "What is the
Black-Scholes model used for in options pricing?" is **PHASE5B_NOTES.md §5's own worked
example** — the question that already-documented conversation showed abstaining end-to-end
because the 20-docs/category finance corpus doesn't cover it well.

Reranked top-1 scores confirm the gap is real, not assumed: **0.9972** for the SMI question vs.
**0.0048** for Black-Scholes (`WEB_ESCALATION_THRESHOLD=0.486`). Run through the full graph:

- SMI question: `final_answer.web_results` empty, fully corpus-cited answer (`[1]`, `[3]`).
- Black-Scholes question, `use_web=True` (default): escalation fires, `web_results` has 3 Tavily
  results, `cited=[]` / `web_cited=[1, 2, 3]` — the corpus contributed nothing usable and the
  answer is honestly 100% web-sourced, still correctly numbered in its own `[W1]`-`[W3]` space.
- Black-Scholes question, `use_web=False`: reverts to exactly PHASE5B_NOTES.md §5's original
  result — `INSUFFICIENT_CONTEXT`, same abstention as before Phase 6 existed.

That last comparison is the actual defense of this phase: the identical question that a prior,
already-documented phase left unanswerable now gets a real, separately-cited answer, and turning
`use_web` off reproduces the old behavior exactly — nothing about the corpus-only path changed.

---

## 6. Known gaps / deferred, not silently dropped

1. **No negative/unanswerable set to validate the threshold's true-negative behavior.** §2's
   percentile approach bounds the *false*-escalation rate on answerable questions but says
   nothing about the *catch* rate on genuinely unanswerable ones — that requires Phase 10 §④'s
   adversarial set, not built yet. Re-tune (or at minimum re-validate) `WEB_ESCALATION_THRESHOLD`
   once that set exists.
2. **`rag/eval.py` integration is not done**, same gap PHASE5_NOTES §9①  already flagged for the
   fanout/single-agent rows — row 6 ("+ web fallback") needs the same Tier 2/3 plumbing work,
   not attempted half-done here.
3. **Escalation is corpus-wide, not per-category.** A cross-topic question where one category is
   well-covered and another isn't only escalates if the *merged* top score is low — a
   category-scoped escalation policy was considered and rejected as out of scope for "one
   escalation rule" (ROADMAP's own wording), see §1's per-branch rejection.
4. **Web result reliability is unmeasured.** ROADMAP's cut list explicitly puts credibility
   scoring out of scope for this phase — Tavily's own ranking is trusted as-is, consistent with
   "no result reranking, no credibility scoring."

---

## 7. Mechanics

```
python -m rag.agent_cli "question"                 # web escalation on by default when scored low
python -m rag.agent_cli "question" --no-web         # never escalate, corpus-only answer
python -m rag.web_eval                              # re-tune WEB_ESCALATION_THRESHOLD (full goldset)
python -m rag.web_eval -n 30 --percentile 0.1        # smaller sample / different percentile
```

`notebooks/agent_phase6.ipynb` walks the escalation mechanism cell by cell against the live
index: `should_escalate()` on a well-covered vs. a documented weak-coverage question (§5), one
live `web_search()` call, then the full graph on both questions plus the `--no-web` ablation,
showing the final answer's corpus citations and web citations rendered separately.

---

## 8. One-line answers for the defense

- **How is the escalation threshold not just a guess?** `rag/web_eval.py` measures the reranked
  top-1 score on all 123 curated goldset questions (all answerable by construction) and sets the
  threshold at the 5th percentile of that distribution — see §2 for the actual measured numbers
  and why the golden set can bound the false-escalation rate but not the catch rate yet.
- **Why are web citations `[W1]` instead of continuing the corpus's `[1]`, `[2]`?** Two disjoint
  regexes, parsed independently, so a web citation can never be mistaken for (or silently
  renumbered into) a corpus one — the same citation-hallucination defense from Phase 1, extended
  to a second source type instead of re-litigated. §3.
- **Why no new graph node for escalation?** `synthesizer_node`/`single_agent_node` are already the
  one place each path's final generation call happens, with the fully merged/reranked chunk set
  already in hand — inserting a separate node would restructure retrieval/generation into two
  steps for no behavioral difference. §1.
- **Why Tavily over the other web-search dependency already in the project (`ddgs`)?** ROADMAP
  asks for *a single* tool; Tavily returns ranked, LLM-ready results in one call with no second
  client or normalization step needed, and its own ranking is taken as-is — nothing left to build
  the "no reranking, no credibility scoring" cut-list items against. §4.
- **Does this actually fix anything, or just add a code path?** PHASE5B_NOTES.md §5 already
  documented "What is the Black-Scholes model used for in options pricing?" abstaining
  end-to-end. Same question, same corpus, Phase 6 on: real Tavily-cited answer. Phase 6 off
  (`--no-web`): the exact original abstention, reproduced. §5.
