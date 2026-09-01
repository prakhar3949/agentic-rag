"""Phase 5's router node and the Send-fanout routing function.

ROADMAP: "Router: query -> relevant topic categories (evaluate against arXiv labels - free
ground truth)." The evaluation half of that bullet is rag/router_eval.py, which scores this
node's output against results/goldset_curated.jsonl's `category` field.
"""

from typing import Union

from langgraph.types import Send
from pydantic import BaseModel, Field

from rag import config
from rag.agent_state import AgentState
from rag.generation import Generator


class RouteDecision(BaseModel):
    categories: list[str] = Field(
        ...,
        description=(
            "Subset of the allowed category list, most relevant first. Include every category "
            "whose corpus plausibly contains information relevant to the question - a question "
            "can legitimately span more than one category."
        ),
    )


def _category_menu() -> str:
    return "\n".join(
        f"- {cat}: {config.CATEGORY_DESCRIPTIONS.get(cat, cat)}"
        for cat in config.INDEXED_CATEGORIES
    )


ROUTER_PROMPT = (
    "Allowed categories:\n{menu}\n\n"
    "Question: {question}\n\n"
    "Which of the allowed categories' corpora are relevant to answering this question? Return "
    "only categories from the allowed list above, spelled exactly as shown."
)


def router_node(state: AgentState, generator: Generator) -> dict:
    """Plain function, directly callable in isolation (notebooks/agent_phase5.ipynb) or wrapped
    in a closure by rag/graph.py's build_graph()."""
    try:
        structured = generator.structured(RouteDecision)
        result: RouteDecision = structured.invoke(
            ROUTER_PROMPT.format(menu=_category_menu(), question=state["question"])
        )
        route = [c for c in result.categories if c in config.INDEXED_CATEGORIES]
    except Exception as exc:
        print(f"[router] routing failed, falling back to every indexed category: {exc}")
        route = []

    # Fail open: never return an empty route. An empty route fans out to zero branches and
    # produces no answer at all - strictly worse than over-including a category, and the same
    # "failures logged, never fatal" call as the planner's fallback (PHASE1_NOTES §5/§9).
    if not route:
        route = list(config.INDEXED_CATEGORIES)
        print(f"[router] no categories returned, using all {len(route)} indexed categories")
    return {"route": route}


def route_after_router(state: AgentState) -> Union[str, list[Send]]:
    """Conditional edge: fanout into parallel topic subagents, or the pooled single-agent path
    (row 10) that intentionally shares the same planner+router scope and differs only in whether
    the routed categories are handled by parallel branches or one merged call - see
    PHASE5_NOTES.md for why row 10 is built this way rather than as the plain Phase 1/3 baseline.
    """
    if not state.get("use_fanout", True):
        return "single_agent"
    return [
        Send("topic_subagent", {
            "question": state["question"],
            "subtasks": state["subtasks"],
            "category": category,
        })
        for category in state["route"]
    ]
