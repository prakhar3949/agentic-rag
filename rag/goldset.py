"""Golden set generation (ROADMAP §Phase 4): one judge call per sampled chunk, asking for a
question answerable *only* from that passage. The source chunk is the gold label by
construction - no separate human labelling pass is needed to know which chunk a question came
from, which is what makes ~200 pairs affordable. Coverage is stratified by (category, modality)
so text-only and figure/table-requiring questions are both represented, not just whatever the
corpus happens to have the most of.

This generates candidates. It does not replace ROADMAP's explicit next step: hand-review >=50 of
them and say so in the README - synthetic-only golden sets are the standard criticism, and no
script can do that review.
"""

import argparse
import json
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from rag import config
from rag.judge import Judge

_JSON_ONLY = "Respond with a single JSON object only - no prose, no markdown fences."

_GOLDSET_SYSTEM = f"""You write one question-answer pair from a single passage of a research \
paper, for a retrieval evaluation set. Rules:

1. The question must be answerable using ONLY the passage given - no outside knowledge, no other \
part of the paper.
2. The answer must be grounded entirely in the passage - quote or closely paraphrase it, do not \
add anything the passage does not state.
3. Do NOT include the paper's title in the question, and do not reference "this paper" or "the \
passage" - phrase it as a standalone question a reader might actually ask.
4. If the passage is a figure or table description, the question should ask about what it shows, \
not about the paper in general.

{_JSON_ONLY}
Schema: {{"question": <string>, "answer": <string>}}"""


@dataclass
class GoldExample:
    id: str
    question: str
    reference_answer: str
    gold_chunk_id: str
    arxiv_id: str
    category: str
    modality: str
    topic: str
    section: str


def load_chunks(path=config.CHUNKS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_existing_chunk_ids(path: Path) -> set[str]:
    """gold_chunk_ids already present in an existing goldset output file, so growing the set
    (append mode, ROADMAP: "run make goldset N=200 ... to grow it") samples new chunks instead
    of re-picking ones a prior run already turned into a question."""
    if not path.exists():
        return set()
    return {
        json.loads(line)["gold_chunk_id"]
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }


def stratified_sample(
    chunks: list[dict], target_total: int, seed: int = 0
) -> list[dict]:
    """Round-robin sample across (category, modality) strata so small strata (figures, tables)
    are not drowned out by the corpus's text-chunk majority. Each stratum is sampled without
    replacement, capped by its own size."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict]] = {}
    for c in chunks:
        strata.setdefault((c["category"], c["modality"]), []).append(c)
    for group in strata.values():
        rng.shuffle(group)

    sampled: list[dict] = []
    pools = {key: iter(group) for key, group in strata.items()}
    keys = list(pools.keys())
    while len(sampled) < target_total and pools:
        for key in list(keys):
            if key not in pools:
                continue
            try:
                sampled.append(next(pools[key]))
            except StopIteration:
                del pools[key]
                continue
            if len(sampled) >= target_total:
                break
    return sampled


def generate_example(judge: Judge, chunk: dict) -> GoldExample | None:
    """One chunk -> one gold example, or None if the judge call fails to parse or keeps leaking
    the paper's topic string after a single corrective retry."""
    user = f"Passage:\n{chunk['text']}"
    resp = judge.call_json(_GOLDSET_SYSTEM, user)
    if resp.parsed is None:
        return None

    question = resp.parsed.get("question", "")
    answer = resp.parsed.get("answer", "")
    topic = chunk.get("topic", "")

    if topic and topic.lower() in question.lower():
        # Retry once with an explicit correction rather than discarding immediately - the model
        # usually complies when told exactly what it did wrong.
        retry_user = (
            f"{user}\n\nYour previous question mentioned the paper's title "
            f'("{topic}"). Rewrite it as a standalone question with no title or paper reference.'
        )
        resp = judge.call_json(_GOLDSET_SYSTEM, retry_user)
        if resp.parsed is None:
            return None
        question = resp.parsed.get("question", "")
        answer = resp.parsed.get("answer", "")
        if topic and topic.lower() in question.lower():
            return None

    if not question or not answer:
        return None

    return GoldExample(
        id=str(uuid.uuid4()),
        question=question,
        reference_answer=answer,
        gold_chunk_id=chunk["id"],
        arxiv_id=chunk["arxiv_id"],
        category=chunk["category"],
        modality=chunk["modality"],
        topic=topic,
        section=chunk["section"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.goldset")
    parser.add_argument("-n", "--target", type=int, default=200,
                        help="target number of QA pairs (default: 200)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", default=str(config.GOLDSET_PATH))
    args = parser.parse_args(argv)
    output_path = Path(args.output)

    chunks = load_chunks()
    already_used = load_existing_chunk_ids(output_path)
    if already_used:
        chunks = [c for c in chunks if c["id"] not in already_used]
        print(f"excluding {len(already_used)} chunk(s) already in {output_path}")
    if not chunks:
        print("no chunks left to sample from - every chunk already has a goldset question")
        return 1

    sample = stratified_sample(chunks, args.target, seed=args.seed)
    print(f"sampled {len(sample)}/{len(chunks)} chunks across "
          f"{len({(c['category'], c['modality']) for c in chunks})} (category, modality) strata")

    judge = Judge()
    written = 0
    rejected = 0
    t0 = time.perf_counter()
    with open(output_path, "a", encoding="utf-8") as f:
        for i, chunk in enumerate(sample, 1):
            example = generate_example(judge, chunk)
            if example is None:
                rejected += 1
                continue
            f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
            written += 1
            if i % 20 == 0:
                print(f"  {i}/{len(sample)} processed ({written} written, {rejected} rejected)")

    elapsed = time.perf_counter() - t0
    print(f"wrote {written} examples ({rejected} rejected/failed) in {elapsed:.1f}s -> {args.output}")
    print("NEXT STEP (not automatable): hand-review >=50 of these before trusting the set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
