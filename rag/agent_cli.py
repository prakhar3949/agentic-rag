"""Question in, agentic (planner + router + fanout/synthesis) cited answer out.

    python -m rag.agent_cli "How do transformer architectures affect risk modeling in finance?"
    python -m rag.agent_cli --chat

Mirrors rag/cli.py's flag conventions. `--no-planner`/`--no-fanout` step back through Phase 5's
ablation arms (ROADMAP §4 rows 4/5/10) the same way rag/cli.py's `--no-bm25`/`--no-rerank` step
back through rows 1-3. `--chat` (Phase 5b) drops into a multi-turn REPL where follow-ups get
rewritten against the running conversation before anything else in the graph sees them. `--no-web`
(Phase 6) disables web escalation - by default, a reranker top score below
`config.WEB_ESCALATION_THRESHOLD` triggers one Tavily call, cited separately from the corpus.
`--no-jailbreak`/`--no-scope`/`--no-clarify`/`--no-grounding` (Phase 7, or `--no-guardrails` for
all four at once) disable the input/dialogue/output rails - see PHASE7_NOTES.md.
"""

import argparse
import dataclasses
import json
import sys
import uuid

from rag import config
from rag.agent_state import Turn
from rag.generation import Generator
from rag.graph import build_graph, run_agent
from rag.judge import Judge
from rag.retrieval import CollectionMissingError, Retriever, StorageLockedError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.agent_cli",
        description="Planner -> router -> parallel topic fanout -> synthesizer, cited answer - "
                    "the full Phase 5 agentic pipeline by default.",
    )
    parser.add_argument("question", nargs="?", default=None,
                        help="the question to answer (omit with --chat)")
    parser.add_argument("--chat", action="store_true",
                        help="multi-turn REPL - follow-ups are rewritten against the running "
                             "conversation before the planner/router see them (ROADMAP Phase 5b)")
    parser.add_argument("--no-planner", action="store_true",
                        help="skip decomposition - subtasks=[question] (ROADMAP row 3/10 arm)")
    parser.add_argument("--no-fanout", action="store_true",
                        help="skip parallel branches - one pooled retrieval + generation call "
                             "over the router's category scope instead (ablation row 10)")
    parser.add_argument("--no-context", action="store_true",
                        help="skip conversational rewriting in --chat mode - follow-ups are sent "
                             "to the planner/router exactly as typed")
    parser.add_argument("--no-web", action="store_true",
                        help="skip web escalation (ROADMAP Phase 6) - never call the web search "
                             "tool even when the reranker's top score is below threshold")
    parser.add_argument("--no-jailbreak", action="store_true",
                        help="skip the jailbreak input rail (ROADMAP Phase 7)")
    parser.add_argument("--no-scope", action="store_true",
                        help="skip the off-topic/sensitive input rail (ROADMAP Phase 7)")
    parser.add_argument("--no-clarify", action="store_true",
                        help="skip the clarification dialogue rail (ROADMAP Phase 7)")
    parser.add_argument("--no-grounding", action="store_true",
                        help="skip the output grounding check (ROADMAP Phase 7)")
    parser.add_argument("--no-guardrails", action="store_true",
                        help="shorthand for all four --no-jailbreak/--no-scope/--no-clarify/"
                             "--no-grounding at once (ablation row 7 'off' arm)")
    parser.add_argument("--thread-id", default=None,
                        help="checkpoint thread id, for replaying a specific run's state "
                             "(default: a fresh random id, printed on every run; in --chat mode "
                             "it's used as a session prefix, and each turn gets its own "
                             "'<id>-turn-N' checkpoint - see PHASE5B_NOTES.md for why)")
    parser.add_argument("--show-context", action="store_true",
                        help="print the route, subtasks, and every branch's retrieved chunks "
                             "before the final answer")
    parser.add_argument("--json", action="store_true",
                        help="emit route/subtasks/branch_results/final_answer as JSON")
    return parser


def render(result: dict, show_context: bool) -> None:
    answer = result["final_answer"]

    # Phase 7: a rail may have halted the graph before the router ever ran, in which case
    # route/subtasks are still their initial_state() defaults ([]) and printing them would be
    # noise, not information.
    if answer.blocked_reason:
        print(f"Blocked ({answer.blocked_reason}):\n\n{answer.text}\n")
        return
    if answer.needs_clarification:
        print(f"Clarification needed:\n\n{answer.text}\n")
        return

    print(f"Route: {result['route']}")
    print(f"Subtasks: {result['subtasks']}")

    if show_context:
        for branch in result["branch_results"]:
            print(f"\n[{branch.category}] {len(branch.chunks)} chunk(s)")
            for chunk in branch.chunks:
                print(f"    {chunk.score:.4f}  {chunk.provenance()}")
            if branch.draft:
                print(f"    draft: {' '.join(branch.draft.text.split())[:200]}")

    print("\n" + answer.text.strip() + "\n")

    if answer.sources():
        print("Sources\n" + "-" * 70)
        for handle, chunk in answer.sources():
            print(f"  [{handle}] {chunk.provenance()}  score={chunk.score:.4f}")

    if answer.web_results:
        print(f"\n  Web escalated ({len(answer.web_results)} result(s) - reranker top score was "
              f"below threshold)")
    if answer.web_sources():
        print("\nWeb sources (separate from the corpus above)\n" + "-" * 70)
        for handle, result in answer.web_sources():
            print(f"  [W{handle}] {result.title}  {result.url}")

    if answer.invalid:
        print(f"\n  WARNING: answer cites {answer.invalid}, but only "
              f"{len(answer.chunks)} sources were provided - broken citation(s).")
    if answer.web_invalid:
        print(f"\n  WARNING: answer cites web handles {answer.web_invalid}, but only "
              f"{len(answer.web_results)} web result(s) were provided - broken citation(s).")
    if answer.uncited:
        print(f"\n  {len(answer.uncited)}/{len(answer.chunks)} retrieved chunks went uncited: "
              f"{answer.uncited}")
    if answer.abstained:
        print("\n  Model abstained: the retrieved context did not support an answer.")
    if answer.grounding_checked:
        score = f"{answer.grounding_score:.2f}" if answer.grounding_score is not None else "n/a"
        print(f"\n  Grounding check: {score} (fraction of claims entailed by the sources)")

    usage = answer.usage or {}
    tokens = (f"{usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out"
              if usage else "tokens unavailable")
    print(f"\n  {answer.model} | {tokens} | {answer.latency_s:.1f}s")


def chat_loop(
    retriever: Retriever, generator: Generator, graph, session_id: str,
    use_planner: bool, use_fanout: bool, use_context: bool, use_web: bool,
    use_jailbreak: bool, use_scope: bool, use_clarify: bool, use_grounding: bool,
    show_context: bool,
) -> None:
    """Phase 5b's multi-turn REPL. One `history` list, owned entirely here - each run_agent()
    call is still a fully self-contained graph invocation (see PHASE5B_NOTES.md for why history
    is caller-managed rather than accumulated via the checkpointer/thread_id).

    Each turn gets its OWN thread_id (`{session_id}-turn-N`), not one shared id for the whole
    session - `branch_results` uses an operator.add reducer (Phase 5, for the Send fanout WITHIN
    one turn), and reusing a single thread_id across turns was verified to leak the PREVIOUS
    turn's stale branches into the next one's synthesizer, since LangGraph resumes a thread's
    prior checkpoint and merges new input into it via each channel's reducer rather than
    replacing it. Turn ids share the session_id prefix so they're still identifiable as one
    conversation when inspecting data/langgraph_checkpoints.sqlite by hand.
    """
    print("Chat mode - blank line or 'exit' to quit.\n")
    history: list[Turn] = []
    turn = 0

    while True:
        try:
            raw_question = input("You: ").strip()
        except EOFError:
            break
        if not raw_question or raw_question.lower() in ("exit", "quit"):
            break
        turn += 1

        result = run_agent(
            raw_question, retriever, generator,
            use_planner=use_planner, use_fanout=use_fanout,
            history=history[-config.MAX_HISTORY_TURNS:], use_context=use_context, use_web=use_web,
            use_jailbreak=use_jailbreak, use_scope=use_scope, use_clarify=use_clarify,
            use_grounding=use_grounding,
            thread_id=f"{session_id}-turn-{turn}", graph=graph,
        )

        rewritten = result["question"]
        if rewritten != raw_question:
            print(f"  (interpreted as: {rewritten})")

        print()
        render(result, show_context=show_context)
        print()

        history.append(Turn(question=rewritten, answer=result["final_answer"].text))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.chat and args.question is None:
        parser.error("question is required unless --chat is set")
    thread_id = args.thread_id or str(uuid.uuid4())

    # --no-guardrails is shorthand for all four rail flags at once (ablation row 7's "off" arm) -
    # an individual --no-X still works standalone, so either form disables that rail.
    use_jailbreak = not (args.no_jailbreak or args.no_guardrails)
    use_scope = not (args.no_scope or args.no_guardrails)
    use_clarify = not (args.no_clarify or args.no_guardrails)
    use_grounding = not (args.no_grounding or args.no_guardrails)

    try:
        with Retriever() as retriever:
            generator = Generator()
            judge = Judge()
            graph = build_graph(retriever, generator, judge)

            if args.chat:
                chat_loop(
                    retriever, generator, graph, thread_id,
                    use_planner=not args.no_planner, use_fanout=not args.no_fanout,
                    use_context=not args.no_context, use_web=not args.no_web,
                    use_jailbreak=use_jailbreak, use_scope=use_scope, use_clarify=use_clarify,
                    use_grounding=use_grounding,
                    show_context=args.show_context,
                )
                return 0

            result = run_agent(
                args.question, retriever, generator,
                use_planner=not args.no_planner, use_fanout=not args.no_fanout,
                use_web=not args.no_web,
                use_jailbreak=use_jailbreak, use_scope=use_scope, use_clarify=use_clarify,
                use_grounding=use_grounding,
                thread_id=thread_id, graph=graph,
            )
    except (CollectionMissingError, StorageLockedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "route": result["route"],
            "subtasks": result["subtasks"],
            "thread_id": thread_id,
            # dataclasses.asdict() recurses through nested dataclass fields on its own (chunks,
            # and the branch's draft Answer - itself holding chunks), no manual unpacking needed.
            "branch_results": [dataclasses.asdict(b) for b in result["branch_results"]],
            "final_answer": dataclasses.asdict(result["final_answer"]),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"Q: {args.question}   (thread_id={thread_id})\n")
        render(result, show_context=args.show_context)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
