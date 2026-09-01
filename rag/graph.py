"""Phase 5's LangGraph wiring: planner -> router -> (parallel fanout | pooled single-agent) ->
final answer, checkpointed for replay. Phase 5b adds one more upstream step, contextualize,
which resolves conversational follow-ups before the planner ever sees them. Phase 6 adds no new
node - it's an inline escalation check inside synthesizer_node/single_agent_node (see
rag/websearch.py's should_escalate()), because both are already the one place each path's final
answer gets built. Phase 7 adds four guardrail nodes (rag/guardrails.py) - three input/dialogue
rails bracketing contextualize, one output rail after the final answer is built.

    START -> jailbreak -> route_after_jailbreak -> [contextualize | END]
    contextualize -> scope -> route_after_scope -> [clarify | END]
    clarify -> route_after_clarify -> [planner | END]
    planner -> router -> route_after_router (conditional)
                            |                        |
                  use_fanout=True          use_fanout=False
                            |                        |
              Send(topic_subagent) x N          single_agent -- may escalate to
                            |                        |          web_search() if
                      synthesizer -- may escalate ---+          reranker top score
                            \\______________________ /          < WEB_ESCALATION_
                                       |                        THRESHOLD
                                  grounding
                                       |
                                      END

Every node function here takes `(state, retriever, generator)` (or a subset) as plain, directly-
testable arguments - see notebooks/agent_phase5.ipynb for calling them in isolation. build_graph()
is the only place dependencies get closed over into the shape LangGraph expects.
"""

import sqlite3
from typing import Optional

from langgraph.graph import END, START, StateGraph

from rag import config
from rag.agent_state import AgentState, BranchResult, Turn
from rag.contextualize import contextualize_node
from rag.generation import Generator, specialist_system_prompt, synthesize
from rag.guardrails import (
    clarify_node,
    grounding_node,
    jailbreak_node,
    route_after_clarify,
    route_after_jailbreak,
    route_after_scope,
    scope_node,
)
from rag.judge import Judge
from rag.planner import planner_node
from rag.retrieval import RetrievedChunk, Retriever
from rag.router import route_after_router, router_node
from rag.websearch import should_escalate, web_search


def _merge_dedupe(chunk_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Union of several chunk lists, deduped by chunk id (keep the higher score), sorted by score.

    Cross-branch duplicate ids are structurally impossible (each branch is category-filtered and
    a chunk belongs to exactly one category) - this matters for the single_agent node, where the
    SAME category can appear via more than one subtask's retrieval and genuinely does need
    deduping.
    """
    best: dict[str, RetrievedChunk] = {}
    for chunks in chunk_lists:
        for chunk in chunks:
            existing = best.get(chunk.id)
            if existing is None or chunk.score > existing.score:
                best[chunk.id] = chunk
    return sorted(best.values(), key=lambda c: -c.score)


def topic_subagent_node(state: AgentState, retriever: Retriever, generator: Generator) -> dict:
    """One Send-fanout branch: retrieve every subtask within this branch's category, merge,
    draft an answer with a topic-specialist system prompt. ROADMAP: "same model, topic-specific
    system prompt, category-filtered retriever" - no new retrieval/generation primitives, just
    this composition of existing ones.
    """
    category = state["category"]
    subtasks = state["subtasks"] or [state["question"]]

    per_subtask = [
        retriever.search(
            subtask, category=category, top_k=config.DEFAULT_TOP_K,
            use_bm25=True, use_rerank=True,
        )
        for subtask in subtasks
    ]
    chunks = _merge_dedupe(per_subtask)

    draft = generator.generate(
        state["question"], chunks, system_prompt=specialist_system_prompt(category)
    )
    return {
        "branch_results": [
            BranchResult(category=category, subtasks=subtasks, chunks=chunks, draft=draft)
        ]
    }


def single_agent_node(state: AgentState, retriever: Retriever, generator: Generator) -> dict:
    """Row 10 (ablation baseline): keeps the planner's subtasks and the router's category scope,
    but pools every (subtask, category) retrieval into one candidate set, reranks it ONCE against
    the original question (mirroring how Retriever.search() reranks after fusion, not per-source),
    and answers with a single generation call - no parallel branches, no synthesizer. This isolates
    fanout as the only variable between this node and topic_subagent_node/synthesizer_node
    (see PHASE5_NOTES.md's row-10 design decision).
    """
    subtasks = state["subtasks"] or [state["question"]]
    candidate_k = config.RETRIEVAL_CANDIDATES

    per_subtask_category = [
        retriever.search(
            subtask, category=category, top_k=candidate_k,
            use_bm25=True, use_rerank=False, candidate_k=candidate_k,
        )
        for subtask in subtasks
        for category in state["route"]
    ]
    pooled = _merge_dedupe(per_subtask_category)
    final_chunks = retriever.reranker.rerank(state["question"], pooled, config.DEFAULT_TOP_K)

    web_results = []
    if state.get("use_web", True) and should_escalate(final_chunks):
        web_results = web_search(state["question"])

    answer = generator.generate(state["question"], final_chunks, web_results=web_results)
    return {"final_answer": answer}


def synthesizer_node(state: AgentState, generator: Generator) -> dict:
    """Merges every branch's chunks + drafts into one final answer via rag.generation.synthesize
    - see that function's docstring for why this regenerates rather than splicing draft text.

    Phase 6's escalation check runs here rather than inside each topic_subagent branch: this is
    the one place fanout's final answer is actually built, so it's the one place that needs to
    decide whether the merged corpus context was thin enough to go to the web - checking it three
    times (once per branch, against each branch's own narrower top score) would escalate on a
    branch that individually scored low even when a sibling branch's chunks made the merged
    answer perfectly well-supported.
    """
    branch_results: list[BranchResult] = state.get("branch_results", [])
    merged_chunks = _merge_dedupe([br.chunks for br in branch_results])
    branch_drafts = [(br.category, br.draft.text) for br in branch_results if br.draft]

    web_results = []
    if state.get("use_web", True) and should_escalate(merged_chunks):
        web_results = web_search(state["question"])

    answer = synthesize(
        generator, state["question"], branch_drafts, merged_chunks, web_results=web_results
    )
    return {"final_answer": answer}


def _default_checkpointer():
    """SqliteSaver over config.CHECKPOINT_PATH - see PHASE5_NOTES.md for why SqliteSaver over
    MemorySaver (replay across process restarts, not just within one process)."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    # AgentState's checkpointed values include our own dataclasses (RetrievedChunk, Answer,
    # BranchResult), not just plain JSON-safe types. The default serializer logged "deserializing
    # unregistered type ... will be blocked in a future version" for exactly these three the first
    # time a checkpoint was written and re-read in a fresh process - the whole point of choosing
    # SqliteSaver was that replay keeps working across restarts, so silently letting a future
    # LangGraph version break it (LANGGRAPH_STRICT_MSGPACK becoming the default) isn't acceptable.
    # Registering them explicitly here fixes that instead of leaving it as a warning to rediscover.
    serde = JsonPlusSerializer(allowed_msgpack_modules=[
        ("rag.retrieval", "RetrievedChunk"),
        ("rag.generation", "Answer"),
        ("rag.agent_state", "BranchResult"),
        ("rag.agent_state", "Turn"),  # Phase 5b: AgentState.history entries
        ("rag.websearch", "WebResult"),  # Phase 6: nested inside Answer.web_results
    ])

    config.CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.CHECKPOINT_PATH), check_same_thread=False)
    saver = SqliteSaver(conn, serde=serde)
    saver.setup()
    return saver


def build_graph(
    retriever: Retriever, generator: Generator, judge: Optional[Judge] = None, checkpointer=None
):
    """Closes `retriever`/`generator`/`judge` into each node - matches rag/eval.py's run_tier23
    style of constructing Retriever()/Generator() once and reusing them, rather than threading
    dependencies through LangGraph's RunnableConfig machinery. `judge` defaults internally (Phase
    7's grounding_node is the first node here to need one) - same `X = X or _default_X()` shape
    already used for `checkpointer` below, so notebooks/agent_phase5.ipynb and
    notebooks/agent_phase6.ipynb's existing `build_graph(retriever, generator)` calls keep working
    unmodified."""
    judge = judge or Judge()
    graph = StateGraph(AgentState)
    graph.add_node("jailbreak", lambda state: jailbreak_node(state, generator))
    graph.add_node("contextualize", lambda state: contextualize_node(state, generator))
    graph.add_node("scope", lambda state: scope_node(state, generator))
    graph.add_node("clarify", lambda state: clarify_node(state, generator))
    graph.add_node("planner", lambda state: planner_node(state, generator))
    graph.add_node("router", lambda state: router_node(state, generator))
    graph.add_node("topic_subagent", lambda state: topic_subagent_node(state, retriever, generator))
    graph.add_node("single_agent", lambda state: single_agent_node(state, retriever, generator))
    graph.add_node("synthesizer", lambda state: synthesizer_node(state, generator))
    graph.add_node("grounding", lambda state: grounding_node(state, judge))

    graph.add_edge(START, "jailbreak")
    graph.add_conditional_edges("jailbreak", route_after_jailbreak, ["contextualize", END])
    graph.add_edge("contextualize", "scope")
    graph.add_conditional_edges("scope", route_after_scope, ["clarify", END])
    graph.add_conditional_edges("clarify", route_after_clarify, ["planner", END])
    graph.add_edge("planner", "router")
    graph.add_conditional_edges("router", route_after_router, ["topic_subagent", "single_agent"])
    graph.add_edge("topic_subagent", "synthesizer")
    graph.add_edge("synthesizer", "grounding")
    graph.add_edge("single_agent", "grounding")
    graph.add_edge("grounding", END)

    return graph.compile(checkpointer=checkpointer or _default_checkpointer())


def initial_state(
    question: str,
    use_planner: bool = True,
    use_fanout: bool = True,
    history: list[Turn] = (),
    use_context: bool = True,
    use_web: bool = True,
    use_jailbreak: bool = True,
    use_scope: bool = True,
    use_clarify: bool = True,
    use_grounding: bool = True,
) -> AgentState:
    return {
        "question": question,
        "history": list(history),
        "use_planner": use_planner,
        "use_fanout": use_fanout,
        "use_context": use_context,
        "use_web": use_web,
        "use_jailbreak": use_jailbreak,
        "use_scope": use_scope,
        "use_clarify": use_clarify,
        "use_grounding": use_grounding,
        "jailbreak_latency_s": None,
        "scope_latency_s": None,
        "clarify_latency_s": None,
        "grounding_latency_s": None,
        "subtasks": [],
        "route": [],
        "branch_results": [],
        "category": None,
        "final_answer": None,
    }


def run_agent(
    question: str,
    retriever: Retriever,
    generator: Generator,
    use_planner: bool = True,
    use_fanout: bool = True,
    history: list[Turn] = (),
    use_context: bool = True,
    use_web: bool = True,
    use_jailbreak: bool = True,
    use_scope: bool = True,
    use_clarify: bool = True,
    use_grounding: bool = True,
    judge: Optional[Judge] = None,
    thread_id: str = "default",
    graph=None,
) -> AgentState:
    """Non-CLI programmatic entry point. Pass a prebuilt `graph` (build_graph() once) to reuse
    the checkpointer connection across multiple questions instead of reopening it each call.
    `judge` is only used when `graph` isn't supplied (forwarded into build_graph()) - a prebuilt
    graph already has its own judge closed in.

    `history` is caller-managed (rag/agent_cli.py's --chat REPL keeps the running list) - each
    call is still a fully self-contained graph invocation, exactly like Phase 5's single-turn
    design; only the caller remembers anything across turns. See PHASE5B_NOTES.md for why this
    isn't accumulated via the checkpointer/thread_id instead.
    """
    graph = graph or build_graph(retriever, generator, judge)
    return graph.invoke(
        initial_state(
            question, use_planner, use_fanout, history, use_context, use_web,
            use_jailbreak, use_scope, use_clarify, use_grounding,
        ),
        config={"configurable": {"thread_id": thread_id}},
    )
