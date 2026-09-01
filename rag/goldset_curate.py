"""Apply hand-review verdicts to the golden QA set (ROADMAP §Phase 4) so a rejected or
never-reviewed question can't quietly stay in the eval set.

    python -m rag.goldset_curate

Reads results/goldset.jsonl (rag.goldset's output) + results/goldset_review.jsonl
(rag.goldset_review's output) and writes results/goldset_curated.jsonl:

    accept  -> row kept unchanged
    edit    -> row kept, question/reference_answer replaced with the reviewer's corrected text
    reject  -> dropped
    (none)  -> dropped (never reviewed, or explicitly skipped - goldset_review.jsonl can't tell
               the two apart, and an un-vetted row has no business in an eval set either way)

goldset.jsonl and goldset_review.jsonl are both read-only here. Point `rag.eval --goldset` at
the curated file once it exists - eval.py still defaults to the raw goldset and never picks this
up implicitly.
"""

import argparse
import json
from pathlib import Path

from rag import config
from rag.goldset_review import load_jsonl


def curate(goldset: list[dict], reviews: dict[str, dict]) -> tuple[list[dict], dict[str, int]]:
    kept: list[dict] = []
    counts = {"accept": 0, "edit": 0, "reject": 0, "unreviewed": 0}
    for example in goldset:
        review = reviews.get(example["id"])
        if review is None:
            counts["unreviewed"] += 1
            continue
        if review["verdict"] == "reject":
            counts["reject"] += 1
            continue
        if review["verdict"] == "edit":
            example = {**example, "question": review["question"],
                       "reference_answer": review["reference_answer"]}
            counts["edit"] += 1
        else:  # accept
            counts["accept"] += 1
        kept.append(example)
    return kept, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.goldset_curate")
    parser.add_argument("--goldset", default=str(config.GOLDSET_PATH))
    parser.add_argument("--review", default=str(config.GOLDSET_REVIEW_PATH))
    parser.add_argument("--output", default=str(config.GOLDSET_CURATED_PATH))
    args = parser.parse_args(argv)

    goldset = load_jsonl(Path(args.goldset))
    if not goldset:
        print(f"no goldset rows found at {args.goldset}")
        return 1
    reviews = {r["id"]: r for r in load_jsonl(Path(args.review))}

    kept, counts = curate(goldset, reviews)
    if not kept:
        print(f"0/{len(goldset)} rows survive curation ({counts}) - nothing written")
        return 1

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(kept)}/{len(goldset)} rows to {output_path} "
          f"({counts['accept']} accepted, {counts['edit']} edited, "
          f"{counts['reject']} rejected, {counts['unreviewed']} unreviewed/skipped - "
          f"all dropped except accept/edit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
