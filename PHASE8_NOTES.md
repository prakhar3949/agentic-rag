# Phase 8 — LangSmith observability: decisions and defense notes

Working record for the `rag/config.py` LangSmith wiring, `rag/eval.py`'s `--langsmith` flag
(`run_tier23_langsmith()`, `_langsmith_dataset()`), and the `.env`/`rag/config.py` fix that made
any of it possible. Same purpose as PHASE1/3/4/5/5B/6/7_NOTES.md: what was decided, what was
measured, what's still open.

---

## 1. A real bug found before any code could be trusted: wrong API region

`.env` already had a `LANGSMITH_API_key` before this phase started. Every call through it -
even a bare, read-only `client.list_projects()` - failed with **403 Forbidden**, not 401. That
distinction is the whole diagnosis: 401 means "I don't recognize these credentials," 403 means
"I recognize them, but they can't do this." Two different key *types* (a Personal Token, then a
freshly-generated Service Key) failed identically, which ruled out "just a bad/expired key."
A plan-tier theory was raised next (a LangSmith pricing screenshot showing several features
behind "Requires upgrade") and checked against LangSmith's actual current pricing structure
before accepting or rejecting it: the gated features (Deployment, Engine, Gateway Policies,
Insights, RBAC) all belong to LangGraph Platform's hosted-compute product line (billed in
LCU/LSU units), not to trace ingestion - the free Developer tier explicitly includes 5,000
traces/month. Ruled out.

An unauthenticated request to the same host returned 401 as expected, and the public `/info`
endpoint returned a clean 200 - so the network path itself was fine, and LangSmith's auth layer
was reachable and functioning. That left exactly one remaining explanation: **the account is
provisioned in LangSmith's APAC region**, whose keys don't authenticate against the default
(US) API host at all - they need `LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com`.
Setting that (with a second, confirmed-working key) fixed every call immediately, first try.

**Fix, in two files:**
- `.env`: added `LANGSMITH_ENDPOINT = "https://apac.api.smith.langchain.com"`, and a second key
  (`LANGSMITH_API_key2`) left alongside the original rather than silently overwriting it.
- `rag/config.py`'s alias-bridging loop: `LANGSMITH_API_key2` is checked **before**
  `LANGSMITH_API_key` (order is precedence, first match wins) - the working key takes priority
  automatically, with a comment explaining why both entries exist rather than one vanishing.

This is exactly the kind of finding this codebase's own convention exists to capture (PHASE1_NOTES
§3, §5: "found running end-to-end, not by reasoning about it") - a plausible-looking, real-format
API key that simply pointed at the wrong regional host, indistinguishable from "revoked" or
"wrong permissions" without actually differentiating 401 from 403 and checking each theory
against a real source rather than guessing.

---

## 2. Resolving ROADMAP Phase 0's open item: the actual current env vars

ROADMAP Phase 0 left this unchecked: "Confirm the current LangSmith env vars for the tracing
toggle *and* the sampling rate (they're separate settings)." Answered by reading the installed
`langsmith` SDK's own source (`langsmith/utils.py`'s `get_env_var()`, which checks
`LANGSMITH_<NAME>` first, then falls back to `LANGCHAIN_<NAME>`), not assumed from an older
convention:

| Setting | Current name | Legacy fallback |
|---|---|---|
| Tracing toggle | `LANGSMITH_TRACING` | `LANGCHAIN_TRACING_V2` |
| Project | `LANGSMITH_PROJECT` | `LANGCHAIN_PROJECT` |
| Sampling rate (0-1) | `LANGSMITH_TRACING_SAMPLING_RATE` | `LANGCHAIN_TRACING_SAMPLING_RATE` |
| Endpoint | `LANGSMITH_ENDPOINT` | `LANGCHAIN_ENDPOINT` |

All four are read directly from `os.environ` by the `langsmith`/`langchain_core` SDKs the first
time anything traced runs - and cached (`functools.lru_cache`) on that first read. `rag/config.py`
sets sensible defaults at **import time** (via `os.environ.setdefault(...)`, so an explicit `.env`
value still wins) specifically because every entry point imports `rag.config` before any LLM
call, guaranteeing the cache never sees a stale value.

---

## 3. Tracing defaults on; three projects; 10% sampling for interactive use

- **Tracing defaults to ON** (`LANGSMITH_TRACING=true` unless `.env` overrides it) - this phase's
  entire point is "trace the full graph," so leaving it opt-in would mean nobody gets the
  deliverable by default.
- **Three separate projects** (ROADMAP: "so retention is configured per project instead of
  globally"): `agentic-rag-dev` (default - CLI, chat, every notebook), `agentic-rag-tier2`,
  `agentic-rag-tier3`. `rag/eval.py`'s `main()` switches `LANGSMITH_PROJECT` to the matching name
  when `--langsmith` is passed on a Tier 2/3 run.
- **10% sampling by default** (`LANGSMITH_DEV_SAMPLING_RATE=0.1`, ROADMAP's own stated control),
  applied to the `dev` project only. `rag/eval.py` removes the sampling-rate env var entirely for
  `--langsmith` runs - every row of a sweep must be traced, not a random subset of it.
- **Retention is a LangSmith workspace/UI setting, not something this repo's code can set.**
  ROADMAP's own warning stands: extended retention on `tier3` must be turned on **before** that
  project's first trace (not retroactively available) - this is a one-time manual step for
  whoever runs the real Tier 3 sweep in Phase 12, documented here so it isn't rediscovered under
  time pressure.

---

## 4. Verified end-to-end: the full graph traces itself, no extra instrumentation

Ran a real question (`"What is the Settlement Modernisation Index (SMI) designed to capture?"`,
`--no-fanout`) through the live graph with tracing on, then read the resulting trace back via the
`Client` SDK rather than trusting "should work" reasoning. Result: **29 runs in one trace tree**,
with zero `@traceable` decorators or manual instrumentation added anywhere:

```
LangGraph (root)
├─ jailbreak                       outputs: jailbreak_latency_s
│  ├─ RunnableSequence → ChatGoogleGenerativeAI → PydanticOutputParser
├─ route_after_jailbreak
├─ contextualize                   outputs: {} (free bypass - turn 1, no history)
├─ scope                           outputs: scope_latency_s
│  ├─ RunnableSequence → ChatGoogleGenerativeAI → PydanticOutputParser
├─ route_after_scope
├─ clarify                         outputs: clarify_latency_s
│  ├─ ChatGoogleGenerativeAI → RunnableSequence → PydanticOutputParser
├─ route_after_clarify
├─ planner                         outputs: subtasks
│  ├─ ChatGoogleGenerativeAI → RunnableSequence → PydanticOutputParser
├─ router                          outputs: route
│  ├─ ChatGoogleGenerativeAI → RunnableSequence → PydanticOutputParser
├─ route_after_router
├─ single_agent                    outputs: final_answer
│  ├─ ChatGoogleGenerativeAI
└─ grounding                       outputs: final_answer, grounding_latency_s
```

**LangGraph's plain-Python-function nodes (each wrapped in a lambda by `build_graph()`) trace
themselves automatically** - LangGraph is built on LangChain's Runnable protocol underneath, so
every `add_node`/conditional-edge call becomes its own named run the moment tracing is on. This
was verified, not assumed: it would have been easy to guess this "should" work and be wrong about
plain-function nodes specifically (as opposed to Runnable subclasses) actually getting the same
treatment.

A second run with `use_web=True` on a weakly-covered question confirmed `tavily_search` also
shows up as its own named run inside the trace - Phase 6's web-escalation tool call auto-traces
exactly like an LLM call, since `langchain_tavily.TavilySearch` is itself a LangChain Runnable.

**"Custom span attributes" (ROADMAP: retrieval scores, chosen route, tokens, cost, per-stage
latency, guardrail verdicts) are already present for free**, not because of new instrumentation
but because of how this codebase's state design already worked from Phase 5 onward: every node
returns only the fields it changed, and LangChain's default tracer records a run's *outputs* as
exactly that returned dict. `route` and `subtasks` are visible on `router`/`planner`'s own runs;
every `*_latency_s` field is visible on its owning rail's run; `final_answer` (carrying
`usage` token counts, `grounding_score`, `blocked_reason`, `needs_clarification`) is visible on
whichever node produced it. The "One Question, Six Files" artifact's central claim - nodes
communicate only through the shared state dict - turns out to double as "the trace is already
informative for free" once tracing is turned on. **Cost is the one exception**: token counts are
visible, but converting them to a $ figure requires LangSmith's own model-pricing configuration
(a manual, one-time UI setting under workspace settings, not something this repo's code can set)
or the same `config/pricing.yaml`-based recomputation this project already uses for its CSVs -
not duplicated here.

---

## 5. Judge confirmed untraced by construction, verified after real calls

ROADMAP's #1 trace-volume control: "Judge client untraced." `rag/judge.py`'s `Judge` was already
built this way since Phase 4 (a raw `openai.OpenAI` client, never routed through
`langchain_openai`/LangChain's Runnable/tracer machinery) - turning global tracing on doesn't
change that, because LangSmith's automatic instrumentation only fires for LangChain-wrapped
clients or explicitly-wrapped ones (`langsmith.wrappers.wrap_openai`), not a bare `openai.OpenAI`
call. Verified directly rather than trusting the architecture argument alone: after a live graph
run (whose `grounding_node` calls `faithfulness()`, which calls the judge) plus a standalone
direct judge call, a scan of every run in the trace project found **zero** runs mentioning
`deepseek`/`fireworks` - the exact model family/provider name the judge uses. The distinct run
names present were entirely Gemini/LangGraph/parser/tool names.

---

## 6. Tier 2/3 → tracked LangSmith experiments, additive to the CSV

ROADMAP: "Push metric results as tracked experiment metrics (quality over time, not measured
once)." `rag/eval.py`'s docstring previously said this integration "is not wired up here - no
`LANGSMITH_API_KEY` is configured" - **stale even before this phase started** (the alias-bridge
in `rag/config.py` already resolved `LANGSMITH_API_KEY` from `.env`'s `LANGSMITH_API_key`), and
now doubly wrong once the regional-endpoint fix (§1) made the key actually work.

**Design: additive, never a replacement.** `run_tier23_langsmith()` is a new function alongside
the existing `run_tier23()`, not a rewrite of it - "the committed CSV is the artifact, not the
trace links" (ROADMAP's own words) means the CSV path had to stay byte-for-byte what it already
was. `--langsmith` on `python -m rag.eval --tier 2/3` runs BOTH: the existing loop (CSV, as
always) and, additionally, the same retrieval+generation+judge-metrics logic reshaped as a
`langsmith.evaluate()` target/evaluator call, landing as one experiment against a single reused
dataset (`config.LANGSMITH_DATASET_NAME = "agentic-rag-goldset"`, create-or-reuse, so repeated
sweeps stay comparable in LangSmith's own "Experiments" compare view instead of each becoming its
own disconnected dataset).

**Verified with a real n=3 run** (`python -m rag.eval --tier 2 --config dense+bm25+rerank -n 3
--langsmith`), first attempt, no signature fixes needed:
- CSV written normally (`results/eval/tier2_dense+bm25+rerank_20260828T123141.csv`) - untouched
  by the `--langsmith` addition.
- A real experiment URL printed: `.../datasets/.../compare?selectedSessions=...`.
- Read back via `Client.list_feedback()`: all 3 example runs carry all 4 metric scores
  (`context_precision`, `context_recall`, `faithfulness`, `response_relevancy`), e.g. one run
  scored `context_precision=0.5` while the other three metrics scored `1.0` on the same
  question - exactly the kind of per-question, per-metric granularity a CSV row already has, now
  also browsable and comparable across runs in LangSmith's UI.

**A real finding from that verification, not obvious in advance:** `evaluate()`'s experiment
lives in its own session (named via `experiment_prefix`, e.g. `tier2-dense+bm25+rerank-7bdd3f7a`)
- **separate from the ambient `LANGSMITH_PROJECT` project** the tier-routing (§3) sets. Querying
`agentic-rag-tier2` for feedback-bearing runs finds nothing; the actual scored runs live under the
experiment's own session name. The two mechanisms (project-level tracing, dataset-level
experiments) are independent LangSmith concepts that don't nest inside each other automatically -
worth knowing before assuming "wrong project" means "didn't work."

**A real cost-relevant finding, checked against LangSmith's own support docs rather than
assumed:** LangSmith **automatically upgrades any trace that receives feedback to extended
retention**, with no user-facing setting to disable it. `evaluate()`'s evaluator functions attach
feedback (via `create_feedback()` internally) to every example run - meaning **every
`--langsmith` Tier 2/3 row silently auto-upgrades to extended (400-day) retention**, regardless of
the project's own default. This directly contradicts ROADMAP §Budget's stated plan ("Leave `dev`
and `tier2` on base - they're disposable"): a `--langsmith` Tier 2 run can never actually stay on
base retention once it uses `evaluate()`'s evaluators.

**Quantified, not just flagged:** at `config/pricing.yaml`'s own rates (base 0.0005 LSU/trace,
extended 0.005 LSU/trace, 1 LSU = $1.00), an 80-row Tier 2 sweep costs $0.04 at base vs. $0.40
extended - a **$0.36 difference**, trivial against the project's ~$110-190 total budget. Accepted
as a bounded, documented cost rather than a design problem worth re-engineering around: the whole
point of `--langsmith` is the "tracked experiment, quality over time" UX (§6), which requires
evaluator feedback to exist at all. **`--langsmith` is opt-in** (plain `--tier 2 --config X`
without the flag never touches LangSmith or its retention at all) specifically so this tradeoff is
the caller's choice per run, not a cost silently paid on every Tier 2 iteration.

This also **corrects ROADMAP Phase 0's open assumption** that extended retention "likely" can't
be applied retroactively - checked against LangSmith's current docs rather than left as a guess:
retroactive application IS available, via an explicit checkbox that applies a new retention
setting to a workspace's *existing* traces (unchecked by default - new traces only). The
underlying sequencing risk ROADMAP was worried about (missing the window before Tier 3's real
sweep) is lower than assumed, but "verify before relying on it" (ROADMAP's own words) still
applies before Phase 12's actual sweep, since checkbox behavior can change.

---

## 7. Dashboard: LangSmith's native views, not a new Streamlit page

ROADMAP: "Dashboard. If LangSmith's trace-oriented views aren't enough, export runs via SDK into
the Streamlit page." The conditional in that sentence is doing real work: LangSmith's own project
view (trace list, latency/cost breakdowns) plus the dataset "compare" view §6 just verified
(side-by-side experiment metrics, exactly "quality over time") together cover everything ROADMAP
Phase 8 actually asks for. Building a Streamlit export now would duplicate infrastructure that
Phase 9 ("API + thin UI") already owns as its own line item - scoped out of this phase
deliberately, not overlooked. If LangSmith's views prove insufficient once real Tier 3 data
exists, that's a one-line addition to Phase 9's Streamlit page (`Client.list_runs()`/
`Client.list_feedback()`, both already exercised in §4-§6), not a new mechanism to design.

---

## 8. Known gaps / deferred, not silently dropped

1. **Cost isn't automatically computed in the trace UI.** Token counts are fully visible (§4);
   turning them into a $ figure needs either LangSmith's model-pricing UI setting (manual,
   one-time, not scriptable from here) or `config/pricing.yaml`'s existing recomputation path -
   not duplicated as a third mechanism.
2. **Raw Qdrant retrieval calls aren't separately traced** - only LangChain-wrapped calls
   (Gemini, Tavily) auto-instrument. This isn't a real information gap: chunk-level scores are
   already visible via each node's own output (`route`, `branch_results`) and via
   `rag/scorelog.py`'s existing persistent CSV, both predating this phase.
3. **`--langsmith` only covers the current non-agentic Tier 2/3 configs** (dense/dense+bm25/
   dense+bm25+rerank) - the agentic pipeline (planner/fanout/guardrails) still isn't wired into
   `rag/eval.py` at all, the same deferred gap flagged in PHASE5/6/7_NOTES. Extending
   `run_tier23_langsmith()`'s `target()` to route through `rag.graph.run_agent()` instead of raw
   `retriever.search()`/`generator.generate()` is the natural next step once that integration
   happens, not attempted half-done here.
4. **Retention on `tier3` is a manual step that must happen before Phase 12's real sweep** - not
   something this phase could set (§3), flagged here so it isn't rediscovered under time pressure
   when it's genuinely unrecoverable if missed.
5. **The pre-existing `x1` project** in the account (visible in `list_projects()` output, §1's
   diagnostic) predates this phase and wasn't touched - left as the user's own artifact.
6. **`--langsmith` Tier 2 runs cannot stay on base retention** (§6) - LangSmith's feedback-
   triggers-extended-retention behavior has no opt-out. Bounded at $0.36/80-row sweep, accepted,
   not re-engineered around; `--langsmith` stays opt-in so it's a per-run choice.

---

## 9. Mechanics

```
python -m rag.agent_cli "question"                          # traces to agentic-rag-dev, 10% sampled
python -m rag.eval --tier 2 --config dense+bm25+rerank       # CSV only, as before
python -m rag.eval --tier 2 --config dense+bm25+rerank --langsmith   # CSV + tracked experiment
python -m rag.eval --tier 3 --config dense+bm25+rerank --langsmith   # routes to agentic-rag-tier3
```

Verification for this phase was done via a one-off script reading traces back through the
`langsmith.Client` SDK (`list_runs`, `list_feedback`) rather than a committed notebook - Phase 8
is infrastructure/config verification, not a new agent capability to demonstrate interactively
cell-by-cell the way Phases 5-7's notebooks do.

---

## 10. One-line answers for the defense

- **How do you know tracing actually captures the agentic pipeline, not just top-level calls?**
  Read a real trace back via the SDK after a real run: 29 runs in one tree, every node and every
  conditional edge named individually, plus the nested LLM/parser calls inside each. §4.
- **Where did the LangSmith key trouble actually come from?** Regional API host mismatch (the
  account is provisioned in LangSmith's APAC region) - diagnosed by distinguishing a 401
  ("credentials not recognized") from the 403 every key produced ("recognized, but forbidden"),
  ruling out a plan/tier restriction against LangSmith's actual current pricing structure, and
  confirming the network path was clean via an unauthenticated call and the public `/info`
  endpoint. §1.
- **Does the judge staying untraced actually hold up, or is that just the intent?** Verified
  after real calls, not just architecturally: zero runs in the traced project mention
  deepseek/fireworks. §5.
- **Is "tracked experiment metrics" real, or just a CSV with extra steps?** Read back via
  `Client.list_feedback()`: each experiment run carries all four metric scores individually,
  browsable and comparable across sweeps in LangSmith's own compare view - not just printed to a
  terminal. §6.
- **Why build a dashboard feature into Streamlit now?** ROADMAP's own conditional says to, only
  if LangSmith's native views aren't enough - they already cover trace inspection and experiment
  comparison, and a custom export is squarely Phase 9's job once real Tier 3 data exists to
  export. §7.
- **Does `--langsmith` really keep Tier 2 "disposable" on base retention like ROADMAP planned?**
  No, and that's disclosed rather than hidden: LangSmith auto-upgrades any feedback-bearing trace
  to extended retention with no opt-out, and `evaluate()`'s evaluators attach feedback by design.
  Quantified at $0.36 extra for an 80-row Tier 2 sweep - accepted as trivial, and `--langsmith`
  stays opt-in so the tradeoff is chosen per run, not paid by default. §6.
