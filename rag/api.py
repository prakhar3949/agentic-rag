"""Phase 9's thin FastAPI layer over the Phase 5-8 agentic pipeline: `/health`, `/query`, `/reset`.

    uv run uvicorn rag.api:app --reload

No auth, no streaming (ROADMAP's explicit cut list) - a single stateless-per-call request/JSON
response shape. Persisted history is a deliberate, later addition on top of that cut, not part of
Phase 9 as originally scoped: `/query` takes a caller-declared `user_id` (a plain string, no
password - "no auth" still holds, this is a partition key, not a login) and keeps that user's
conversation `Turn` history in a process-local dict, exactly mirroring how
`rag/agent_cli.py --chat`'s `chat_loop` already keeps its own `history: list[Turn]` - just keyed
per user instead of per REPL session. **In-memory only, by design**: a server restart clears
everyone's history, same as `chat_loop` losing its history when the terminal closes. No locking
across concurrent requests for the same `user_id` either - acceptable for a demo UI, not something
this phase's scope calls for hardening.

`Retriever`/`Generator`/`Judge`/the compiled graph are constructed once at startup (FastAPI's
`lifespan`), matching every other entry point's "construct once, reuse" pattern (`rag/eval.py`'s
tier loops, notebooks, `rag/agent_cli.py`) - not once per request, which would reload
BGE-M3/the reranker on every call.
"""

import base64
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag import config
from rag.agent_state import Turn
from rag.generation import Answer, Generator
from rag.graph import build_graph, run_agent
from rag.judge import Judge
from rag.retrieval import CollectionMissingError, RetrievedChunk, Retriever, StorageLockedError

_state: dict = {}
# user_id -> conversation history, in-memory only (module docstring). Turn is the same dataclass
# rag/agent_cli.py's --chat REPL already uses - no new "what does a turn look like" concept.
_histories: dict[str, list[Turn]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = Retriever()
    generator = Generator()
    judge = Judge()
    graph = build_graph(retriever, generator, judge)
    _state.update(retriever=retriever, generator=generator, judge=judge, graph=graph)
    yield
    retriever.close()


app = FastAPI(title="Agentic RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    user_id: str
    question: str


class ResetRequest(BaseModel):
    user_id: str


class SourceOut(BaseModel):
    handle: int
    arxiv_id: str
    category: str
    section: str
    modality: str
    score: float
    provenance: str
    # Base64 data URI for figure-modality chunks only - embedded directly rather than served from
    # a second endpoint, so the Streamlit page needs exactly one HTTP call per question (matches
    # "no auth, no streaming" - keep the interaction shape as simple as the rest of this phase).
    image_data_uri: Optional[str] = None


class WebSourceOut(BaseModel):
    handle: int
    title: str
    url: str


class QueryResponse(BaseModel):
    turn: int              # 1-indexed position in this user_id's conversation
    question: str          # standalone form, after rag/contextualize.py - may differ from what
                            # the caller actually typed if this was a follow-up
    route: list[str]
    subtasks: list[str]
    answer: str
    blocked_reason: Optional[str] = None
    needs_clarification: bool = False
    abstained: bool = False
    grounding_score: Optional[float] = None
    sources: list[SourceOut]           # cited corpus sources only
    web_sources: list[WebSourceOut]    # cited web sources only (Phase 6 escalation)
    all_chunks: list[SourceOut]        # every retrieved chunk, cited or not, with its score
    model: str
    usage: dict
    latency_s: float


def _image_data_uri(chunk: RetrievedChunk) -> Optional[str]:
    path = chunk.image_path()
    if path is None:
        return None
    mime = path.suffix.lstrip(".").lower()
    mime = "jpeg" if mime == "jpg" else mime
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def _source_out(handle: int, chunk: RetrievedChunk) -> SourceOut:
    return SourceOut(
        handle=handle, arxiv_id=chunk.arxiv_id, category=chunk.category, section=chunk.section,
        modality=chunk.modality, score=chunk.score, provenance=chunk.provenance(),
        image_data_uri=_image_data_uri(chunk),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.user_id.strip():
        raise HTTPException(400, "user_id must not be empty")
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")

    history = _histories.setdefault(req.user_id, [])
    turn = len(history) + 1

    # Each turn gets its OWN checkpoint thread_id, not one shared per user - reusing a single
    # thread_id across turns was verified (PHASE5B_NOTES.md §2-3) to leak the PREVIOUS turn's
    # stale branch_results into the next turn's synthesizer via LangGraph's own reducer, since
    # resuming a thread merges new input into its prior checkpoint rather than replacing it.
    try:
        result = run_agent(
            req.question, _state["retriever"], _state["generator"], judge=_state["judge"],
            history=history[-config.MAX_HISTORY_TURNS:],
            thread_id=f"{req.user_id}-turn-{turn}", graph=_state["graph"],
        )
    except (CollectionMissingError, StorageLockedError) as exc:
        raise HTTPException(503, str(exc))

    answer: Answer = result["final_answer"]
    history.append(Turn(question=result["question"], answer=answer.text))

    return QueryResponse(
        turn=turn, question=result["question"], route=result["route"],
        subtasks=result["subtasks"],
        answer=answer.text, blocked_reason=answer.blocked_reason,
        needs_clarification=answer.needs_clarification, abstained=answer.abstained,
        grounding_score=answer.grounding_score,
        sources=[_source_out(h, c) for h, c in answer.sources()],
        web_sources=[
            WebSourceOut(handle=h, title=r.title, url=r.url) for h, r in answer.web_sources()
        ],
        all_chunks=[_source_out(i, c) for i, c in enumerate(answer.chunks, 1)],
        model=answer.model, usage=answer.usage, latency_s=answer.latency_s,
    )


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    """Starts a fresh conversation for this user_id - drops its in-memory history only, nothing
    else (no effect on other users, no effect on the checkpoint DB, which each turn's own
    thread_id already isolates)."""
    _histories.pop(req.user_id, None)
    return {"status": "cleared", "user_id": req.user_id}
