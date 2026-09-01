"""Interactive hand-review pass over the golden QA set (ROADMAP §Phase 4: "Hand-review >=50 of
these before trusting the set" - not automatable; `rag.goldset` only generates candidates).

    python -m rag.goldset_review

Walks each row in `results/goldset.jsonl` next to the source passage it was generated from
(the judge only ever saw that one passage, so review needs it too) and asks for a verdict:

    [a]ccept   - question + answer are fine as-is
    [e]dit     - fix the question and/or answer text, then accept
    [r]eject   - unusable (not answerable from the passage, hallucinated answer, trivial, etc.)
    [s]kip     - decide later, leave unreviewed
    [q]uit     - save progress and exit

Progress is resumable: rows that already have a verdict are skipped on the next run unless
--redo is passed. Verdicts are written to results/goldset_review.jsonl, keyed by the goldset
row's `id` - goldset.jsonl itself is never modified, so the judge's raw output and the human
verdict stay two separate, comparable things.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag import config


@dataclass
class Review:
    id: str
    verdict: str  # accept | edit | reject
    question: str
    reference_answer: str
    notes: str
    reviewed_at: str


class Quit(Exception):
    pass


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_reviews(path: Path, reviews: dict[str, Review]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in reviews.values():
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def ask(label: str, current: str) -> str:
    print(f"  current {label}: {current}")
    new = input(f"  new {label} (Enter to keep): ").strip()
    return new or current


def review_one(example: dict, chunk_text: str, index: int, total: int) -> Review | None:
    print("\n" + "=" * 78)
    print(f"[{index}/{total}]  {example['category']}/{example['modality']}  "
          f"{example['arxiv_id']}  section={example['section']}")
    print(f"topic: {example['topic']}")
    print("-" * 78)
    print("passage:")
    print(" ", " ".join(chunk_text.split())[:1200])
    print("-" * 78)
    print(f"Q: {example['question']}")
    print(f"A: {example['reference_answer']}")

    now = lambda: datetime.now(timezone.utc).isoformat()

    while True:
        choice = input("\n  [a]ccept / [e]dit / [r]eject / [s]kip / [q]uit > ").strip().lower()

        if choice in ("a", "accept"):
            notes = input("  notes (optional): ").strip()
            return Review(example["id"], "accept", example["question"],
                          example["reference_answer"], notes, now())

        if choice in ("e", "edit"):
            question = ask("question", example["question"])
            answer = ask("answer", example["reference_answer"])
            notes = input("  notes (optional): ").strip()
            return Review(example["id"], "edit", question, answer, notes, now())

        if choice in ("r", "reject"):
            reason = input("  reason: ").strip()
            return Review(example["id"], "reject", example["question"],
                          example["reference_answer"], reason, now())

        if choice in ("s", "skip"):
            return None

        if choice in ("q", "quit"):
            raise Quit

        print("  unrecognized - a/e/r/s/q")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.goldset_review")
    parser.add_argument("--goldset", default=str(config.GOLDSET_PATH))
    parser.add_argument("--output", default=str(config.GOLDSET_REVIEW_PATH))
    parser.add_argument("--redo", action="store_true",
                         help="re-review rows that already have a verdict")
    parser.add_argument("--category", default=None, help="restrict to one arXiv category")
    parser.add_argument("--modality", default=None, choices=["text", "figure", "table"])
    args = parser.parse_args(argv)

    goldset_path = Path(args.goldset)
    output_path = Path(args.output)

    examples = load_jsonl(goldset_path)
    if not examples:
        print(f"no goldset rows found at {goldset_path}")
        return 1
    chunks_by_id = {c["id"]: c["text"] for c in load_jsonl(config.CHUNKS_PATH)}

    existing = {r["id"]: Review(**r) for r in load_jsonl(output_path)}

    todo = [e for e in examples
            if (args.redo or e["id"] not in existing)
            and (args.category is None or e["category"] == args.category)
            and (args.modality is None or e["modality"] == args.modality)]

    if not todo:
        print(f"nothing to review - {len(existing)}/{len(examples)} rows already have a verdict "
              f"(pass --redo to re-review)")
        return 0

    print(f"{len(todo)} row(s) to review ({len(existing)}/{len(examples)} already reviewed) "
          f"-> writing to {output_path}")

    reviewed_this_run = 0
    try:
        for i, example in enumerate(todo, 1):
            chunk_text = chunks_by_id.get(example["gold_chunk_id"], "(source chunk not found)")
            result = review_one(example, chunk_text, i, len(todo))
            if result is not None:
                existing[result.id] = result
                reviewed_this_run += 1
                save_reviews(output_path, existing)  # persist after every decision
    except Quit:
        print("\nstopped - progress saved")
    except KeyboardInterrupt:
        print("\ninterrupted - progress saved")

    counts: dict[str, int] = {}
    for r in existing.values():
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    print(f"\n{reviewed_this_run} reviewed this run. totals: "
          f"{counts.get('accept', 0)} accepted, {counts.get('edit', 0)} edited, "
          f"{counts.get('reject', 0)} rejected, "
          f"{len(examples) - len(existing)} still unreviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
