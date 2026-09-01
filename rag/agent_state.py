"""Phase 5's LangGraph state schema.

TypedDict, not a dataclass - this is the graph's state schema, and LangGraph merges per-node
return values into it by key using each key's reducer (plain replace by default, `operator.add`
for `branch_results` below). A dataclass would work too, but TypedDict is what every LangGraph
example builds against, and there's no reason to diverge from the idiomatic shape here.
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Optional, TypedDict

from rag import config
from rag.generation import Answer
from rag.retrieval import RetrievedChunk


@dataclass
class Turn:
    """One past conversation turn - Phase 5b's unit of chat history.

    `question` is the STANDALONE (already-contextualized) form, not necessarily what the user
    literally typed - see rag/contextualize.py's docstring for why a later follow-up should
    resolve against an unambiguous prior turn rather than a chain of raw pronoun references.
    """

    question: str
    answer: str


# Answers are truncated in a transcript, not the question - matches the truncation already used
# for chunk/draft previews elsewhere (rag/cli.py, rag/agent_cli.py's --show-context): enough to
# resolve a reference or judge continuity, not the full cited answer, which would dominate a
# small prompt.
_ANSWER_PREVIEW_CHARS = 300


def format_transcript(history: list[Turn]) -> str:
    """Recent turns as a short "Q: ...\\nA: ..." block for a prompt - truncated to
    config.MAX_HISTORY_TURNS turns and to _ANSWER_PREVIEW_CHARS per answer.

    Lives here (next to Turn) rather than in rag/contextualize.py, where it originated (Phase
    5b) - Phase 9's per-user chat history exposed a second, genuinely new consumer
    (rag/guardrails.py's scope_node/clarify_node, judging a follow-up's topic/specificity against
    conversation context, not just resolving its pronouns), and unlike the router.py
    _category_menu()-style duplications elsewhere in this codebase, letting the two truncation
    lengths drift independently here would be a real correctness risk, not just a cosmetic
    DRY violation - both call sites need the SAME notion of "recent enough to matter."
    """
    recent = history[-config.MAX_HISTORY_TURNS:]
    blocks = []
    for turn in recent:
        answer = " ".join(turn.answer.split())[:_ANSWER_PREVIEW_CHARS]
        blocks.append(f"Q: {turn.question}\nA: {answer}")
    return "\n\n".join(blocks)


@dataclass
class BranchResult:
    """One topic subagent's output - what the Send-fanout for `category` produced.

    `subtasks` is carried per-branch (even though every branch currently sees the same list from
    the planner) rather than read back off the parent state, so a branch result is self-describing
    on its own - useful the moment `--show-context`-style diagnostics or PHASE5_NOTES.md want to
    print "this branch, given these subtasks, retrieved these chunks" without cross-referencing
    the rest of AgentState.
    """

    category: str
    subtasks: list[str]
    chunks: list[RetrievedChunk] = field(default_factory=list)
    draft: Optional[Answer] = None


class AgentState(TypedDict):
    question: str

    # Phase 5b: caller-supplied conversation history, most-recent-last. Plain field, NOT an
    # operator.add reducer - unlike branch_results (which must accumulate WITHIN one turn's Send
    # fanout), history must NOT accumulate ACROSS separate run_agent() calls on a reused
    # thread_id, or it would grow forever with no per-turn reset. The caller (rag/agent_cli.py's
    # --chat REPL) owns the running list and passes the relevant slice in fresh each turn, exactly
    # like `question` itself. rag/contextualize.py truncates to the last config.MAX_HISTORY_TURNS
    # before use (ROADMAP §7: "multi-turn memory beyond 3 turns - out of scope").
    history: list[Turn]

    # Ablation flags (ROADMAP §4 rows 4/5/10) - runtime input, not derived. False bypasses the
    # corresponding LLM call entirely rather than running it and discarding the result, so the
    # bypassed arm costs nothing extra (rag/planner.py, rag/router.py).
    use_planner: bool
    use_fanout: bool

    # Phase 5b ablation/debug flag, same shape as use_planner/use_fanout - False skips
    # rag/contextualize.py's LLM call even when history is present.
    use_context: bool

    # Phase 6 ablation/debug flag, same shape as the others - False skips rag/websearch.py's
    # should_escalate() check entirely (zero-cost bypass, same as every other flag here), so a
    # corpus-only run never makes a Tavily call even when the reranker score would trigger one.
    use_web: bool

    # Phase 7 ablation/debug flags, same shape as the others - each False skips its rail's LLM
    # call entirely (rag/guardrails.py). Named use_scope (not use_topic) because "topic" already
    # means something else in this graph (topic_subagent, router.py's category routing).
    use_jailbreak: bool
    use_scope: bool
    use_clarify: bool
    use_grounding: bool

    # Phase 7: per-rail latency, seconds. Flat fields, not a shared dict - each is written by
    # exactly one node (same reasoning as `route`/`subtasks` below), so no operator.add reducer
    # is needed. None when the rail was bypassed (use_X=False) or never reached (an earlier rail
    # already halted the graph - see route_after_jailbreak/route_after_scope/route_after_clarify
    # in rag/guardrails.py).
    jailbreak_latency_s: Optional[float]
    scope_latency_s: Optional[float]
    clarify_latency_s: Optional[float]
    grounding_latency_s: Optional[float]

    # Planner output. [question] unchanged when use_planner=False.
    subtasks: list[str]

    # Router output: subset of config.INDEXED_CATEGORIES. Populated whether or not fanout is on -
    # row 10 (single-agent baseline) still uses the router's category scope, only the fanout
    # structure differs (see PHASE5_NOTES.md's row-10 design decision).
    route: list[str]

    # Send-fanout reducer: each parallel topic_subagent invocation returns {"branch_results":
    # [one BranchResult]}, and operator.add concatenates them onto this list across the parallel
    # invocations. Empty when use_fanout=False - single_agent_node doesn't populate this key.
    branch_results: Annotated[list[BranchResult], operator.add]

    # Per-branch input only. Not set by any node in the main path - route_after_router's Send()
    # calls set this per fanout invocation, so topic_subagent_node knows which category it owns.
    # Declared here (not just passed loose in the Send payload) because LangGraph validates a
    # Send's state dict against the graph's own schema/channels.
    category: Optional[str]

    # Terminal output. Written by whichever path actually ran: synthesizer_node/single_agent_node
    # on a normal completion, or - Phase 7 - jailbreak_node/scope_node/clarify_node with a
    # synthetic Answer (blocked_reason set, or needs_clarification=True) if a rail halted the
    # graph early. rag/guardrails.py's route_after_jailbreak/route_after_scope/route_after_clarify
    # each treat "final_answer is not None" as the sole halt signal - initial_state() sets it to
    # None and nothing upstream of a terminal node ever sets it otherwise.
    final_answer: Optional[Answer]
