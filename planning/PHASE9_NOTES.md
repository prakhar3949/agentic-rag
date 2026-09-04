# Phase 9 — API + thin UI: decisions and defense notes

Short by design, matching the phase's own 3h scope (PHASE5B_NOTES.md set this precedent: not
every phase needs a full working record). Covers `rag/api.py` and `streamlit_app.py`.

---

## 1. Shape

Two new files, one dependency each way:

```
streamlit_app.py --(HTTP POST /query)--> rag/api.py --(run_agent())--> the Phase 5-8 graph
```

`rag/api.py` (FastAPI): `/health`, `/query`. `Retriever`/`Generator`/`Judge`/the compiled graph
are built once at startup via FastAPI's `lifespan` context manager - not per request, which would
reload BGE-M3 and the reranker on every call. `streamlit_app.py` is a pure HTTP client of the
API, not a second copy of the pipeline - matches ROADMAP listing "FastAPI" and "Streamlit" as two
separate bullets, and means the UI has zero direct dependency on `rag/graph.py` or any model.

**Cut list honored exactly as ROADMAP states it:** no auth, no streaming (plain JSON
request/response, no SSE), no persisted history (every `/query` call gets a fresh
`uuid4()` checkpoint `thread_id` - nothing remembered between calls, matching Phase 5b's own
distinction that only `--chat` accumulates history, and this endpoint isn't chat).

---

## 2. Response shape: `answer.chunks` already holds the right list regardless of path

The "citations" and "retrieved chunks w/ scores" ROADMAP asks for map directly onto
`Answer.sources()` (cited only) and `Answer.chunks` (everything retrieved, cited or not) -
`rag/agent_cli.py`'s `render()` already displays exactly these two views. One shortcut this
phase relies on rather than re-derives: `synthesize()` (Phase 5's fanout path) and
`single_agent_node`'s `generate()` call both populate `Answer.chunks` with the complete,
correctly-renumbered source list either way - the API layer never needs to branch on
`use_fanout` to know where the source list lives.

**Figure thumbnails** are embedded as base64 data URIs directly in the JSON response
(`SourceOut.image_data_uri`), reusing `RetrievedChunk.image_path()` unchanged from Phase 1/5's
`build_message()` - not a second image-serving endpoint. One HTTP call per question stays the
whole interaction, matching the "no auth, no streaming" simplicity the cut list asks for.

---

## 3. Verified end-to-end (real server, real browser)

Started both servers for real (`uv run uvicorn rag.api:app`, `uv run streamlit run
streamlit_app.py`) and drove them, rather than trusting the code would work:

- `GET /health` → `{"status": "ok"}`
- `POST /query` on a well-covered question → full pipeline ran (`route: ["finance"]`),
  `grounding_score: 1.0`, four citations with real scores, matching `rag.agent_cli`'s own output
  for the same question (PHASE6_NOTES.md's SMI example).
- `POST /query` on an underspecified question → `needs_clarification: true`, `route: []`,
  confirming Phase 7's guardrail signals surface through the API's JSON shape, not just the CLI.
- `POST /query` on a figure-referencing question → one figure-modality chunk retrieved, a real
  63KB base64 `image_data_uri` resolved and returned - not a placeholder or an empty string.
- **The Streamlit page itself**, driven headlessly with Playwright against the live API (no
  project `run` skill existed yet for this repo, so the generic browser-driven pattern was used
  directly): typed a question, clicked "Ask," and the rendered page shows the answer, the
  route/model/latency/grounding metadata row, the numbered citations with scores, and the
  collapsed "all retrieved chunks" expander - screenshotted, not just loaded. Zero console errors.

Both test servers were stopped immediately after verification - embedded Qdrant is single-writer,
and leaving either running would lock out the next `rag.cli`/`rag.agent_cli`/notebook invocation
exactly the way a forgotten notebook kernel already does (PHASE1_NOTES/PHASE5_NOTES precedent).

---

## 4. Later addition: per-user conversation history (deliberate cut-list reversal)

Not part of Phase 9 as ROADMAP originally scoped it - the cut list explicitly names this out:
*"Rich UI (React, auth, streaming, history) — Zero hiring signal."* Added anyway, on request,
because a demo that remembers a follow-up is worth more than the ROADMAP's own reasoning weighed
it against - recorded here as a conscious reversal, not a silently-dropped decision.

**Shape:** `/query` takes a caller-declared `user_id` (a plain string - no password, so "no auth"
still holds; it's a partition key, not a login). `rag/api.py` keeps a process-local
`dict[str, list[Turn]]`, exactly mirroring `rag/agent_cli.py --chat`'s `chat_loop` - just keyed
per `user_id` instead of one REPL session. **In-memory only, on request** - a server restart
clears everyone's history, same as `chat_loop` losing its history when the terminal closes.
`/reset` clears one user's history on demand. `streamlit_app.py` grew a name/ID field, a running
(collapsed) transcript, and a "New conversation" button that calls `/reset`.

**A real bug found running the actual multi-turn scenario, not designed in from the start:**
alice's turn 1 asked about the SMI; turn 2 asked "What are its three phases?" -
`rag/contextualize.py` correctly rewrote this to *"What are the three phases of the Settlement
Modernisation Index (SMI)?"* - and `scope_node` then blocked that fully-resolved, obviously
on-topic question as `off_topic` anyway. The mechanics (history threading, per-user isolation,
per-turn `thread_id`s) all worked correctly on the first try; this was a genuine gap in Phase 7's
rails, newly exposed because nothing before this had ever run a guardrail rail against a real
multi-turn follow-up - Phase 7's own probe suite (`rag/guardrail_probe.py`) is entirely
single-turn by construction.

**Root cause:** `scope_node`/`clarify_node` only ever saw `state["question"]` - the standalone,
already-rewritten form - with no visibility into the conversation that motivated the rewrite.
`contextualize_node` did its narrow job (resolve the pronoun) correctly; the rewritten sentence
just happened to read slightly more like a generic finance-definition question than turn 1's
paper-flavored phrasing, enough to tip the classifier the wrong way when judged in isolation.

**Fix:** `format_transcript()` (the `Turn` → prompt-ready transcript formatter `contextualize.py`
already had) was promoted from `rag/contextualize.py` into `rag/agent_state.py`, next to `Turn`
itself - not duplicated the way `router.py`'s `_category_menu()` was for `scope_node`'s corpus
description (PHASE7_NOTES §4B), because here the SAME truncation lengths
(`MAX_HISTORY_TURNS`, the 300-char answer preview) genuinely need to stay in sync across both
consumers, not just coincidentally share a shape. `scope_node` and `clarify_node` now both build a
`_transcript_block()` from `state.get("history")` and include it in their prompts, instructed to
judge topic/specificity *in that context* - empty string (byte-identical prompt to before) on
turn 1 or when history is absent, so every existing single-turn test case is unaffected by
construction, not just by coincidence.

**Verified, not just reasoned about:**
- The exact failing scenario, re-run after the fix: alice's turn 2 now answers normally, citing
  real sources, instead of being blocked.
- `rag/guardrail_probe.py`'s full 17-case suite (all single-turn, `history` empty) re-run
  unchanged: **17/17 correct**, zero regression.
- A fresh, unrelated `user_id` asking the exact same raw follow-up ("What are its three phases?")
  still correctly gets flagged `off_topic` - no cross-user history leakage.

---

## 5. Mechanics

```
uv run uvicorn rag.api:app --reload          # or: make api
uv run streamlit run streamlit_app.py        # or: make ui
```

`streamlit_app.py` points at `AGENTIC_RAG_API_URL` (default `http://localhost:8000`) - set it if
the API runs on a different host/port. `POST /reset {"user_id": "..."}` clears one user's
in-memory conversation without restarting the server.

---

## 6. One-line answers for the defense

- **Why does Streamlit call the API instead of importing the graph directly?** Two ROADMAP
  bullets, two responsibilities - the UI has no model/retriever dependency at all, and the same
  API could serve a different frontend without change. §1.
- **How do figure thumbnails get from a chunk to the browser?** The same `image_path()` method
  Phase 1's `build_message()` already uses, base64-encoded straight into the JSON response - no
  new image-serving code path. §2.
- **Does the UI actually show guardrail/grounding behavior, or just happy-path answers?** Verified
  directly: an underspecified question returns `needs_clarification` through the API and the
  Streamlit page renders it as a distinct info banner, not a generic error. §3.
- **ROADMAP's cut list says no history - why does the demo remember conversations?** A deliberate,
  disclosed reversal, not an oversight - added on request, documented as an exception rather than
  silently dropping the cut-list line. §4.
- **How do you know per-user history doesn't leak between users, or break Phase 7's rails?**
  Verified, not assumed: a second, unrelated `user_id` asking the same raw follow-up still gets
  correctly blocked, and Phase 7's full 17-case single-turn probe suite still scores 17/17 after
  the change. §4.
