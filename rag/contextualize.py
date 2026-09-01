"""Phase 5b's conversational query contextualization node.

ROADMAP: "Rewrite follow-ups ('what about the second one?') into standalone queries from chat
history... distinct problem from decomposition." This runs BEFORE rag/planner.py's decomposer -
by the time the planner (or router, or retrieval) sees `state["question"]`, any pronoun/reference
to prior turns has already been resolved. Nothing downstream needs to know a conversation is
happening at all.
"""

from pydantic import BaseModel, Field

from rag.agent_state import AgentState, format_transcript
from rag.generation import Generator


class StandaloneQuery(BaseModel):
    standalone_question: str = Field(
        ...,
        description=(
            "The follow-up question, rewritten to be fully self-contained: resolve every "
            "pronoun and vague reference (e.g. 'it', 'that paper', 'the second one') using the "
            "conversation history, so the result can be understood with NO prior context. If the "
            "follow-up is already self-contained, return it unchanged. Do not answer the "
            "question - only rewrite it."
        ),
    )


CONTEXTUALIZE_PROMPT = (
    "Conversation so far:\n{transcript}\n\nFollow-up: {question}\n\n"
    "Rewrite the follow-up as a standalone question per the instructions."
)


def contextualize_node(state: AgentState, generator: Generator) -> dict:
    """Plain function, directly callable in isolation (notebooks/agent_phase5.ipynb) or wrapped
    in a closure by rag/graph.py's build_graph() - same shape as planner_node/router_node."""
    history = state.get("history") or []

    # Two free bypasses, no LLM call: an explicit off-switch, and turn 1 of any conversation -
    # there is nothing a rewrite could resolve against with zero prior turns (same "don't run a
    # call whose answer is knowable in advance" reasoning as planner_node's use_planner=False).
    if not state.get("use_context", True) or not history:
        return {}

    try:
        structured = generator.structured(StandaloneQuery)
        result: StandaloneQuery = structured.invoke(
            CONTEXTUALIZE_PROMPT.format(
                transcript=format_transcript(history), question=state["question"]
            )
        )
        rewritten = result.standalone_question.strip()
    except Exception as exc:
        # Fail open, not fatal - PHASE1_NOTES §5/§9's "failures logged, never fatal," same
        # posture as rag/planner.py and rag/router.py. Worse to crash a turn than to retrieve on
        # the raw (possibly ambiguous) follow-up.
        print(f"[contextualize] rewrite failed, falling back to the raw follow-up: {exc}")
        return {}

    return {"question": rewritten} if rewritten else {}
