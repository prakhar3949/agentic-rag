# Phase 5b — Conversational query contextualization: decisions and defense notes

Short by design — a 1.5h phase, not a rewrite of `PHASE5_NOTES.md`. Working record for
`rag/contextualize.py`, the `history`/`use_context` additions to `rag/agent_state.py`, and
`rag/agent_cli.py --chat`.

---

## 1. Where it sits

```
START -> contextualize -> planner -> router -> ... (rest of the graph unchanged from Phase 5)
```

One new node, one new edge, **zero changes to any Phase 5 node**. `contextualize_node` rewrites
`state["question"]` in place before the planner ever sees it, so nothing downstream (planner,
router, topic subagents, single-agent baseline, synthesizer) needs to know a conversation is
happening at all — it just sees an already-standalone question, same as turn 1 of any run.

---

## 2. Bug found and fixed: reusing one `thread_id` across chat turns leaks `branch_results`

**First design** had `rag/agent_cli.py --chat` use one `thread_id` for the whole session
("session checkpoints stay inspectable together" seemed like a free bonus). **Verified wrong**
before shipping it, not after:

```python
SAME_THREAD = "shared-session-thread"
r1 = graph.invoke(initial_state("turn 1 question"), config={"configurable": {"thread_id": SAME_THREAD}})
print(len(r1["branch_results"]))   # 1 - correct
r2 = graph.invoke(initial_state("turn 2 question"), config={"configurable": {"thread_id": SAME_THREAD}})
print(len(r2["branch_results"]))   # 2 - WRONG, only 1 category was routed this turn
```

**Root cause:** `AgentState.branch_results` uses an `operator.add` reducer (Phase 5, so the
parallel `Send` fanout *within one turn* can accumulate one `BranchResult` per branch). LangGraph
resumes a thread's last checkpoint before applying new input, and merges that input through each
channel's reducer rather than replacing the channel outright. `question`/`route`/`subtasks`/
`final_answer` have no custom reducer (default: last write wins), so those overwrite cleanly
turn to turn — but `branch_results` doesn't overwrite, it **appends**, so turn 2's fresh branches
land on top of turn 1's still-present ones. The synthesizer would silently merge a stale prior
turn's draft into the current turn's answer.

**Fix:** each turn gets its own `thread_id` (`{session_id}-turn-{n}` in `rag/agent_cli.py`'s
`chat_loop`), not one shared id for the whole conversation. Turn ids still share a session prefix
so they're identifiable as one conversation when inspecting
`data/langgraph_checkpoints.sqlite` by hand — that part of the original idea survives, just
without sharing actual checkpoint *state*. Re-verified with the same script above using per-turn
ids: `branch_results` count stayed scoped to the current turn only.

**Why this matters beyond this one bug:** it's the general reason `history` itself was designed
as a plain caller-managed field rather than a reducer-accumulated one from the start (see §3) —
`operator.add` fields are exactly the ones that don't reset between separate invocations of the
same thread, and any *new* per-turn field added later needs the same question asked of it before
assuming thread reuse is free.

---

## 3. History is caller-managed, not graph-accumulated

Could have let LangGraph's checkpointer accumulate `history` automatically across turns on a
reused `thread_id` (the same mechanism that *caused* §2's bug, applied deliberately instead of
accidentally). Rejected for the same reason: it only works cleanly for fields that are meant to
grow forever within a thread, and `history` specifically needs a 3-turn cap
(ROADMAP §7 — "multi-turn memory beyond 3 turns, out of scope"), which is easiest to enforce as
a plain Python list slice (`history[-config.MAX_HISTORY_TURNS:]`) owned by the caller, not as
custom reducer logic that would have to reimplement bounded accumulation to stay in scope anyway.

Net effect: every `run_agent()` call remains a fully self-contained graph invocation, exactly
like Phase 5's original single-turn design. Only `rag/agent_cli.py --chat`'s REPL loop remembers
anything across turns, in an ordinary Python list.

---

## 4. History records the rewritten question, not the raw follow-up

`Turn.question` (rag/agent_state.py) stores what `contextualize_node` resolved the question to,
not what the user literally typed. A third follow-up ("and the first one?") needs to resolve
against something already unambiguous - chaining raw, pronoun-laden follow-ups would compound
ambiguity turn over turn instead of resolving it. The CLI still shows the user their own literal
input (it has the raw string locally before ever calling `run_agent()`); only the transcript
entry fed back into future rewrites uses the resolved form.

---

## 5. Verified end-to-end (real models, live index)

```
You: What is the Black-Scholes model used for in options pricing?
Route: ['finance']   [... abstained - corpus doesn't cover it, unrelated to this feature]

You: What about its assumptions?
  (interpreted as: What are the assumptions of the Black-Scholes model in options pricing?)
Route: ['finance']
```

"its" correctly resolved to "the Black-Scholes model" using turn 1's recorded question, and the
router stayed correctly scoped to `finance` on both turns. Both turns abstaining is a corpus-
coverage limitation already documented in PHASE3_NOTES.md/PHASE5_NOTES.md (20 docs/category,
not Phase 2's at-scale target) - unrelated to contextualization itself.

---

## 6. One-line answers for the defense

- **Why is this a separate node from the planner's decomposition?** Different problems: the
  planner splits one already-standalone question into retrievable pieces; contextualization
  resolves references to *prior turns* into one standalone question, before decomposition ever
  sees it. ROADMAP calls this out explicitly as a distinct problem.
- **Why does history live outside the graph's checkpointing instead of using it?** Tried the
  reducer-accumulation approach first, found it silently leaks `branch_results` across turns
  (§2) - the same mechanism that would make `history` convenient would also make a genuinely
  wrong bug free to reintroduce for any future turn-scoped field. Caller-managed avoids both.
- **Why cap at 3 turns?** ROADMAP §7's cut list states it explicitly - not an oversight to fix.
- **Why record the rewritten question in history instead of the raw one?** So a chain of
  follow-ups resolves against unambiguous prior turns instead of compounding pronoun references.
