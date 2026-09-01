"""Phase 5's planner/decomposer node.

ROADMAP: "This is your query rewriting. Make it bypassable so it becomes an ablation arm."
`use_planner=False` skips the LLM call entirely rather than running it and discarding the result
- the bypassed arm (row 3) must cost nothing extra relative to today's baseline.
"""

from pydantic import BaseModel, Field

from rag.agent_state import AgentState
from rag.generation import Generator


class DecomposedQuery(BaseModel):
    subtasks: list[str] = Field(
        ...,
        description=(
            "1-4 standalone, independently-retrievable sub-questions that together cover the "
            "full original question. If the question already asks about exactly one thing, "
            "return exactly one subtask equal to the original question, unchanged."
        ),
    )


PLANNER_PROMPT = (
    "Decompose the following question into standalone sub-questions ONLY if it asks about more "
    "than one distinct thing. Each subtask must be answerable on its own, without needing the "
    "others for context. If the question is already atomic, return it unchanged as the single "
    "subtask - do not invent sub-questions that were not asked.\n\nQuestion: {question}"
)


def planner_node(state: AgentState, generator: Generator) -> dict:
    """Plain function, not a class - directly callable in isolation (see
    notebooks/agent_phase5.ipynb) or wrapped in a closure by rag/graph.py's build_graph()."""
    if not state.get("use_planner", True):
        return {"subtasks": [state["question"]]}

    try:
        structured = generator.structured(DecomposedQuery)
        result: DecomposedQuery = structured.invoke(
            PLANNER_PROMPT.format(question=state["question"])
        )
        subtasks = [s.strip() for s in result.subtasks if s and s.strip()]
    except Exception as exc:
        # Fail open, not fatal - PHASE1_NOTES §5/§9's "failures logged, never fatal." A planner
        # outage degrades to the atomic question rather than crashing the whole graph.
        print(f"[planner] decomposition failed, falling back to the original question: {exc}")
        subtasks = []

    return {"subtasks": subtasks or [state["question"]]}
