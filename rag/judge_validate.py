"""Cohen's kappa validation of the LLM judge (ROADMAP §Phase 4: "hand-label 50 examples, compute
Cohen's kappa between judge and labels" - load-bearing, answers the defense question "how do you
know your judge is trustworthy?"). See PHASE4_NOTES.md for the full why.

This validates something different from `rag.goldset_review`: that tool checked whether the
GOLDSET's questions/reference answers are correct. This tool checks whether the JUDGE's verdicts
about a system's retrieval/generation are correct - two separate failure points, both need
checking, and a clean goldset scored by an unreliable judge still produces a garbage ablation
table.

Three subcommands:

    python -m rag.judge_validate generate   # run the curated goldset through retrieval +
                                             # generation + judge, pool every atomic claim/chunk
                                             # verdict (real, paid: 1 generation + 3 judge calls
                                             # per question)
    python -m rag.judge_validate label       # hand-label a sample of that pool, blind to the
                                             # judge's verdict (resumable, like goldset_review)
    python -m rag.judge_validate report      # compute Cohen's kappa (hand-implemented, no
                                             # framework - consistent with metrics_retrieval.py)
                                             # between your labels and the judge's

Only `generate` costs money. `label` and `report` are free/local and can be re-run any time.

Scope: only the three judge metrics that decompose into binary per-claim/per-chunk verdicts
(context_precision, context_recall, faithfulness) are pooled. response_relevancy is a single
continuous 0.0-1.0 score with nothing decomposed to label item-by-item - see PHASE4_NOTES.md for
why that one is out of scope for a kappa-style agreement check.
"""

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag import config
from rag.generation import Generator
from rag.goldset_review import Quit, load_jsonl
from rag.judge import Judge
from rag.metrics_llm import context_precision, context_recall, faithfulness
from rag.retrieval import Retriever

_METRICS = ("context_precision", "context_recall", "faithfulness")

_PROMPTS = {
    "context_precision": (
        "chunk", "Does this chunk contribute information used in the reference answer?"
    ),
    "context_recall": ("claim", "Is this claim supported by the retrieved context?"),
    "faithfulness": ("claim", "Is this claim supported by the retrieved context?"),
}


@dataclass
class PoolItem:
    item_id: str
    question_id: str
    category: str          # goldset row's arXiv category - lets report() slice reliability by domain
    metric: str
    question: str
    unit_text: str        # the claim (recall/faithfulness) or chunk (precision) being judged
    evidence_text: str     # what unit_text is checked against
    judge_verdict: bool
    judge_reason: str


@dataclass
class HumanLabel:
    item_id: str
    human_verdict: bool
    notes: str
    labeled_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cohens_kappa(rater_a: list[bool], rater_b: list[bool]) -> float | None:
    """Chance-corrected agreement between two binary raters (Cohen, 1960). Hand-implemented
    rather than pulled from a library, same reasoning as rag/metrics_retrieval.py's hit@k/MRR/
    NDCG: the formula is small enough to own, and owning it means it can be read out loud.
    Worked example checked against by hand in PHASE4_NOTES.md."""
    n = len(rater_a)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(rater_a, rater_b)) / n
    a_rate = sum(rater_a) / n
    b_rate = sum(rater_b) / n
    pe = a_rate * b_rate + (1 - a_rate) * (1 - b_rate)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


# --- generate ---------------------------------------------------------------------------------

def _pool_from_example(
    ex: dict, retriever: Retriever, generator: Generator, judge: Judge, scope: set[str]
) -> tuple[list[PoolItem], int]:
    """Returns (pool items, count of judge calls that returned no verdict after retries - a
    persistent transient failure or unparseable output, per rag/judge.py). A failed call simply
    contributes no items for that metric on this question; it never becomes a fabricated verdict.

    `scope` restricts which of the 3 metrics get computed - e.g. re-validating only
    context_precision after a rubric change shouldn't re-spend judge calls (or a generation call)
    on context_recall/faithfulness, which weren't touched."""
    final = retriever.search(ex["question"], top_k=config.DEFAULT_TOP_K, use_bm25=True, use_rerank=True)
    chunk_texts = [c.text for c in final]
    context_text = "\n\n".join(chunk_texts)

    items = []
    failures = 0

    if "context_precision" in scope:
        cp = context_precision(judge, ex["question"], ex["reference_answer"], chunk_texts)
        if cp.judge_response and cp.judge_response.parsed is None:
            failures += 1
        for v in cp.claims:
            handle = v.get("handle")
            unit = chunk_texts[handle - 1] if isinstance(handle, int) and 1 <= handle <= len(chunk_texts) else ""
            items.append(PoolItem(str(uuid.uuid4()), ex["id"], ex["category"], "context_precision",
                                   ex["question"], unit, ex["reference_answer"],
                                   bool(v.get("relevant")), v.get("reason", "")))

    if "context_recall" in scope:
        cr = context_recall(judge, ex["question"], ex["reference_answer"], context_text)
        if cr.judge_response and cr.judge_response.parsed is None:
            failures += 1
        for c in cr.claims:
            items.append(PoolItem(str(uuid.uuid4()), ex["id"], ex["category"], "context_recall",
                                   ex["question"], c.get("claim", ""), context_text,
                                   bool(c.get("supported")), c.get("reason", "")))

    if "faithfulness" in scope:
        # Only metric of the three that needs a generated answer - skipped entirely (no Gemini
        # call) when faithfulness isn't in scope.
        answer = generator.generate(ex["question"], final, include_images=False)
        fa = faithfulness(judge, answer.text, context_text)
        if fa.judge_response and fa.judge_response.parsed is None:
            failures += 1
        for c in fa.claims:
            items.append(PoolItem(str(uuid.uuid4()), ex["id"], ex["category"], "faithfulness",
                                   ex["question"], c.get("claim", ""), context_text,
                                   bool(c.get("supported")), c.get("reason", "")))

    return items, failures


def cmd_generate(args: argparse.Namespace) -> int:
    examples = load_jsonl(Path(args.goldset))
    if args.n:
        examples = examples[: args.n]
    if not examples:
        print(f"no goldset rows found at {args.goldset}")
        return 1

    scope = {args.metric} if args.metric else set(_METRICS)
    output_path = Path(args.output)
    existing_pool = load_jsonl(output_path)

    present: dict[str, set[str]] = {}
    for p in existing_pool:
        present.setdefault(p["question_id"], set()).add(p["metric"])

    if args.redo:
        # Regenerate in-scope metrics for every requested question - drop their existing in-scope
        # rows first so this doesn't just duplicate them (today's --redo doesn't drop anything).
        # Rows for out-of-scope metrics, and rows on questions not requested this run, are untouched.
        to_process = examples
        drop_qids = {ex["id"] for ex in examples}
        kept_rows = [p for p in existing_pool
                     if not (p["metric"] in scope and p["question_id"] in drop_qids)]
    else:
        to_process = [ex for ex in examples if not scope <= present.get(ex["id"], set())]
        kept_rows = existing_pool
        skipped = len(examples) - len(to_process)
        if skipped:
            print(f"skipping {skipped} question(s) already pooled for {sorted(scope)} in "
                  f"{output_path} (pass --redo to regenerate)")

    if not to_process:
        print("nothing left to pool - every question already has pool items for this scope")
        return 0

    if args.redo:
        dropped_ids = {p["item_id"] for p in existing_pool} - {p["item_id"] for p in kept_rows}
        if dropped_ids:
            stale = sum(1 for lab in load_jsonl(Path(args.labels)) if lab["item_id"] in dropped_ids)
            if stale:
                print(f"note: {stale} existing human label(s) in {args.labels} reference item_ids "
                      f"being regenerated - they'll show as 'not found in pool' in the next report "
                      f"until the new items are re-labeled")

    # Existing pool is rewritten first (kept_rows only), then new rows are appended + flushed
    # after every question - a transient failure or an interrupted run partway through must not
    # lose already-paid-for work (this bit us: a run crashed on question 30/47 and the file mode
    # in place at the time discarded 1-29 with it).
    with open(output_path, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()

        pooled_items = 0
        total_failures = 0
        with Retriever() as retriever:
            generator = Generator()
            judge = Judge()
            for i, ex in enumerate(to_process, 1):
                items, failures = _pool_from_example(ex, retriever, generator, judge, scope)
                for item in items:
                    f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
                f.flush()
                pooled_items += len(items)
                total_failures += failures
                if i % 10 == 0:
                    print(f"  {i}/{len(to_process)} questions processed, {pooled_items} items pooled so far")

    print(f"wrote {pooled_items} item(s) from {len(to_process)} question(s) -> {output_path} "
          f"({len(kept_rows)} pre-existing row(s) preserved)")
    if total_failures:
        print(f"  {total_failures} judge call(s) returned no verdict after retries (transient "
              f"failure or unparseable output) - contributed no items for that metric/question, "
              f"not scored as a fabricated verdict")
    return 0


# --- label ------------------------------------------------------------------------------------

def _stratified_by_metric(items: list[dict], target_total: int, seed: int) -> list[dict]:
    """Round-robin across the 3 metrics so a 50-item label sample isn't dominated by whichever
    metric happens to decompose into the most claims (faithfulness/recall vs. one entry per
    retrieved chunk for precision) - same pattern as rag.goldset's stratified_sample."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item["metric"], []).append(item)
    for g in groups.values():
        rng.shuffle(g)

    pools = {k: iter(v) for k, v in groups.items()}
    sampled: list[dict] = []
    while len(sampled) < target_total and pools:
        for key in list(pools):
            try:
                sampled.append(next(pools[key]))
            except StopIteration:
                del pools[key]
                continue
            if len(sampled) >= target_total:
                break
    return sampled


def _save_labels(path: Path, labels: dict[str, HumanLabel]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for label in labels.values():
            f.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")


def label_one(item: dict, index: int, total: int) -> HumanLabel | None:
    unit_kind, ask = _PROMPTS[item["metric"]]
    evidence_label = "reference answer" if item["metric"] == "context_precision" else "retrieved context"

    print("\n" + "=" * 78)
    print(f"[{index}/{total}]  metric={item['metric']}")
    print(f"question: {item['question']}")
    print("-" * 78)
    print(f"evidence ({evidence_label}):")
    print(" ", " ".join(item["evidence_text"].split())[:1000])
    print("-" * 78)
    print(f"{unit_kind}: {item['unit_text']}")
    print(f"\n{ask}")

    while True:
        choice = input("\n  [y]es / [n]o / [s]kip / [q]uit > ").strip().lower()
        if choice in ("y", "yes"):
            notes = input("  notes (optional): ").strip()
            return HumanLabel(item["item_id"], True, notes, _now())
        if choice in ("n", "no"):
            notes = input("  notes (optional): ").strip()
            return HumanLabel(item["item_id"], False, notes, _now())
        if choice in ("s", "skip"):
            return None
        if choice in ("q", "quit"):
            raise Quit
        print("  unrecognized - y/n/s/q")


def cmd_label(args: argparse.Namespace) -> int:
    pool = load_jsonl(Path(args.pool))
    if not pool:
        print(f"no pool found at {args.pool} - run `python -m rag.judge_validate generate` first")
        return 1
    if args.metric:
        pool = [p for p in pool if p["metric"] == args.metric]
        if not pool:
            print(f"no pool items for metric={args.metric} in {args.pool}")
            return 1

    output_path = Path(args.output)
    existing = {r["item_id"]: HumanLabel(**r) for r in load_jsonl(output_path)}

    remaining = [p for p in pool if args.redo or p["item_id"] not in existing]
    if not remaining:
        print(f"nothing to label - {len(existing)}/{len(pool)} items already labeled "
              f"(pass --redo to relabel)")
        return 0

    todo = _stratified_by_metric(remaining, args.n, seed=args.seed)
    print(f"{len(todo)} item(s) to label ({len(existing)}/{len(pool)} already labeled) "
          f"-> {output_path}")

    labeled_this_run = 0
    try:
        for i, item in enumerate(todo, 1):
            result = label_one(item, i, len(todo))
            if result is not None:
                existing[result.item_id] = result
                labeled_this_run += 1
                _save_labels(output_path, existing)
    except Quit:
        print("\nstopped - progress saved")
    except KeyboardInterrupt:
        print("\ninterrupted - progress saved")

    print(f"\n{labeled_this_run} labeled this run. {len(existing)} total labeled.")
    return 0


# --- report -----------------------------------------------------------------------------------

def _kappa_report(judge_labels: list[bool], human_labels: list[bool]) -> dict:
    n = len(judge_labels)
    pairs = list(zip(judge_labels, human_labels))
    return {
        "n": n,
        "raw_agreement": sum(j == h for j, h in pairs) / n if n else None,
        "cohens_kappa": cohens_kappa(judge_labels, human_labels),
        "confusion": {
            "judge_true_human_true": sum(j and h for j, h in pairs),
            "judge_true_human_false": sum(j and not h for j, h in pairs),
            "judge_false_human_true": sum((not j) and h for j, h in pairs),
            "judge_false_human_false": sum((not j) and (not h) for j, h in pairs),
        },
    }


def cmd_report(args: argparse.Namespace) -> int:
    pool = {p["item_id"]: p for p in load_jsonl(Path(args.pool))}
    labels = load_jsonl(Path(args.labels))
    if not labels:
        print(f"no labels found at {args.labels} - run `python -m rag.judge_validate label` first")
        return 1

    pairs = []  # (metric, category, judge_verdict, human_verdict)
    missing = 0
    for lab in labels:
        item = pool.get(lab["item_id"])
        if item is None:
            missing += 1
            continue
        # .get(): pool rows written before category was added to PoolItem won't have the key -
        # falls back to "unknown" rather than crashing on a pool built by an older run.
        pairs.append((item["metric"], item.get("category", "unknown"),
                      item["judge_verdict"], lab["human_verdict"]))

    if not pairs:
        print("no labeled item matched the pool - stale labels file relative to the pool?")
        return 1

    report = {"overall": _kappa_report([j for _, _, j, _ in pairs], [h for _, _, _, h in pairs])}
    for metric in _METRICS:
        sub = [(j, h) for m, _, j, h in pairs if m == metric]
        if sub:
            report[metric] = _kappa_report([j for j, _ in sub], [h for _, h in sub])

    by_category = {}
    for category in sorted({c for _, c, _, _ in pairs}):
        sub = [(j, h) for _, c, j, h in pairs if c == category]
        if sub:
            by_category[category] = _kappa_report([j for j, _ in sub], [h for _, h in sub])
    if by_category:
        report["by_category"] = by_category

    print(json.dumps(report, indent=2))
    if missing:
        print(f"\n(warning: {missing} labeled item id(s) not found in the pool - "
              f"labels file may be stale relative to a re-generated pool)")

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote report -> {output_path}")
    return 0


# --- CLI ---------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.judge_validate")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="pool judge verdicts over the curated goldset (paid)")
    p_gen.add_argument("--goldset", default=str(config.GOLDSET_CURATED_PATH))
    p_gen.add_argument("--output", default=str(config.JUDGE_POOL_PATH))
    p_gen.add_argument("--labels", default=str(config.JUDGE_LABELS_PATH),
                        help="read-only, used only to warn about labels a --redo would orphan")
    p_gen.add_argument("-n", type=int, default=None,
                        help="limit to the first N goldset questions (default: all)")
    p_gen.add_argument("--metric", choices=list(_METRICS), default=None,
                        help="only pool this one metric (default: all three)")
    p_gen.add_argument("--redo", action="store_true",
                        help="regenerate in-scope metrics even for questions already pooled")

    p_label = sub.add_parser("label", help="hand-label a sample of the pool, blind to the judge")
    p_label.add_argument("--pool", default=str(config.JUDGE_POOL_PATH))
    p_label.add_argument("--output", default=str(config.JUDGE_LABELS_PATH))
    p_label.add_argument("-n", type=int, default=50, help="target items to label (default: 50)")
    p_label.add_argument("--seed", type=int, default=0)
    p_label.add_argument("--metric", choices=list(_METRICS), default=None,
                          help="only sample from this one metric (default: all three)")
    p_label.add_argument("--redo", action="store_true", help="relabel items that already have a label")

    p_report = sub.add_parser("report", help="compute Cohen's kappa: your labels vs. the judge")
    p_report.add_argument("--pool", default=str(config.JUDGE_POOL_PATH))
    p_report.add_argument("--labels", default=str(config.JUDGE_LABELS_PATH))
    p_report.add_argument("--output", default=str(config.JUDGE_REPORT_PATH))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "label":
        return cmd_label(args)
    if args.command == "report":
        return cmd_report(args)
    return 1  # unreachable, subparsers are required


if __name__ == "__main__":
    raise SystemExit(main())
