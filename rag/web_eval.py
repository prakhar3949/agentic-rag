"""Tunes Phase 6's WEB_ESCALATION_THRESHOLD against goldset_curated.jsonl's reranked top-1 scores.

    python -m rag.web_eval [-n N] [--percentile P]

ROADMAP: "Threshold tuned on the golden set, not guessed." Every goldset question is answerable
from the indexed corpus by construction (rag/goldset.py generates each one from a real, already-
indexed chunk) - there is no negative/unanswerable set yet (that's Phase 10 §④, a later phase).
So this can't teach the threshold to recognize an unanswerable question directly. What it CAN do:
measure how low a genuine match's reranked top-1 score gets across the curated set, and set the
threshold at a low percentile of that distribution - comfortably below typical answerable scores,
so only a small, known fraction of genuinely-answerable questions would incorrectly trigger web
escalation, while a question truly outside the corpus (which scores far lower than any of these)
reliably escalates.

Mirrors rag/router_eval.py's shape: same goldset loader/sampler, same "write a JSON report,
print the summary" ending.
"""

import argparse
import json
import statistics
from pathlib import Path

from rag import config
from rag.retrieval import Retriever
from rag.router_eval import load_goldset, sample_goldset


def top1_scores(retriever: Retriever, questions: list[str]) -> list[float]:
    """Whole-corpus (no category filter) reranked top-1 score per question - escalation is a
    router-independent, corpus-wide signal, not scoped to whatever category the router happens
    to pick."""
    scores = []
    for question in questions:
        chunks = retriever.search(question, use_bm25=True, use_rerank=True)
        scores.append(chunks[0].score if chunks else 0.0)
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.web_eval")
    parser.add_argument("-n", "--n", type=int, default=None,
                        help="questions to sample (default: the whole curated goldset)")
    parser.add_argument("--percentile", type=float, default=0.05,
                        help="threshold = this percentile of top-1 scores (default: 0.05, i.e. "
                             "only ~5%% of answerable questions would incorrectly escalate)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goldset", default=str(config.GOLDSET_CURATED_PATH))
    args = parser.parse_args(argv)

    examples = load_goldset(Path(args.goldset))
    if args.n:
        examples = sample_goldset(examples, args.n, seed=args.seed)
    questions = [ex["question"] for ex in examples]
    print(f"Web escalation threshold tuning | n={len(questions)} | percentile={args.percentile}")

    with Retriever() as retriever:
        scores = top1_scores(retriever, questions)

    scores_sorted = sorted(scores)
    idx = max(0, min(len(scores_sorted) - 1, round(args.percentile * (len(scores_sorted) - 1))))
    threshold = scores_sorted[idx]
    would_escalate = sum(s < threshold for s in scores)

    summary = {
        "n": len(scores),
        "percentile": args.percentile,
        "threshold": threshold,
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "would_escalate_at_threshold": would_escalate,
        "false_escalation_rate": would_escalate / len(scores),
    }

    config.WEB_ESCALATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.WEB_ESCALATION_REPORT_PATH.write_text(
        json.dumps({"summary": summary, "scores": scores}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {config.WEB_ESCALATION_REPORT_PATH}")
    print(json.dumps(summary, indent=2))
    print(f"\nSet WEB_ESCALATION_THRESHOLD={threshold!r} in rag/config.py "
          f"(or the WEB_ESCALATION_THRESHOLD env var) to apply it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
