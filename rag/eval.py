"""Tiered eval loop (ROADMAP §Phase 4): `python -m rag.eval --tier {1,2,3} --config NAME`.

| Tier | Questions | Metrics                          | Cost          |
|------|-----------|-----------------------------------|---------------|
| 1    | 30        | retrieval only (hit@k/MRR/NDCG)   | $0, seconds   |
| 2    | 80        | + all four judge metrics          | low           |
| 3    | 240       | + all four judge metrics          | budget it     |

Tier 1 needs no judge and no generator - it is the fast/free loop meant to absorb most iterations
while tuning the retriever. Tiers 2-3 add generation and the four LLM-judge metrics per question,
and log one CSV row per question to `results/eval/` so the ablation table can be rebuilt from
disk without re-running anything.

LangSmith Dataset + Evaluator tracking (ROADMAP's other Tier 2-3 requirement, Phase 8) is now
wired up via `--langsmith` on a Tier 2/3 run: additive, alongside the CSV, never instead of it -
`run_tier23_langsmith()` runs the same retrieval+generation+judge-metrics logic through
`langsmith.evaluate()` so results land as a comparable experiment against one reused dataset
(`config.LANGSMITH_DATASET_NAME`), not a fresh, incomparable one per run. The CSV remains the
ground truth either way (ROADMAP §Phase 8: "the committed CSV is the artifact, not the trace
links") - if `--langsmith` fails or is omitted, the CSV output is unaffected.
"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from rag import config
from rag.generation import Generator
from rag.judge import Judge
from rag.metrics_llm import context_precision, context_recall, faithfulness, response_relevancy
from rag.metrics_retrieval import aggregate, score_query
from rag.retrieval import Retriever

# Ablation rows 1-3 (ROADMAP §4). Row 8 (long-context, no rerank, stuffed) is reachable via the
# same Retriever.search() flags but isn't a named tier-comparison arm here.
CONFIGS: dict[str, dict] = {
    "dense": {"use_bm25": False, "use_rerank": False},
    "dense+bm25": {"use_bm25": True, "use_rerank": False},
    "dense+bm25+rerank": {"use_bm25": True, "use_rerank": True},
}

TIER_DEFAULT_N = {1: 30, 2: 80, 3: 240}


def load_goldset(path: Path = config.GOLDSET_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sample_goldset(examples: list[dict], n: int, seed: int = 0) -> list[dict]:
    if n >= len(examples):
        if n > len(examples):
            print(f"warning: requested {n} examples but goldset only has {len(examples)} - "
                  f"using all of them")
        return list(examples)
    return random.Random(seed).sample(examples, n)


@dataclass
class Tier1Row:
    question_id: str
    config_name: str
    gold_chunk_id: str
    hit_at_50: bool
    mrr_50: float
    ndcg_50: float
    hit_at_5: bool
    mrr_5: float
    ndcg_5: float


def run_tier1(
    examples: list[dict], retriever: Retriever, cfg_name: str, candidate_k: int = config.RETRIEVAL_CANDIDATES
) -> list[Tier1Row]:
    """Retrieval-only: recall@50 (pre-rerank ceiling) and recall@5 (post-rerank actual), so a bad
    final answer is attributable to "never found it" vs "found it, buried by reranking"
    (ROADMAP §3② retrieval ceiling decomposition)."""
    cfg = CONFIGS[cfg_name]
    rows = []
    for ex in examples:
        pool = retriever.search(
            ex["question"], top_k=candidate_k, use_bm25=cfg["use_bm25"], use_rerank=False,
            candidate_k=candidate_k,
        )
        if cfg["use_rerank"]:
            final = retriever.search(
                ex["question"], top_k=config.DEFAULT_TOP_K, use_bm25=cfg["use_bm25"],
                use_rerank=True, candidate_k=candidate_k,
            )
        else:
            final = pool[:config.DEFAULT_TOP_K]

        gold = {ex["gold_chunk_id"]}
        s50 = score_query([c.id for c in pool], gold, k=candidate_k)
        s5 = score_query([c.id for c in final], gold, k=config.DEFAULT_TOP_K)
        rows.append(Tier1Row(
            question_id=ex["id"], config_name=cfg_name, gold_chunk_id=ex["gold_chunk_id"],
            hit_at_50=s50.hit_at_k, mrr_50=s50.reciprocal_rank, ndcg_50=s50.ndcg,
            hit_at_5=s5.hit_at_k, mrr_5=s5.reciprocal_rank, ndcg_5=s5.ndcg,
        ))
    return rows


@dataclass
class Tier23Row:
    question_id: str
    config_name: str
    tier: int
    question: str
    gold_chunk_id: str
    hit_at_50: bool
    hit_at_5: bool
    context_precision: Optional[float]
    context_recall: Optional[float]
    faithfulness: Optional[float]
    response_relevancy: Optional[float]
    abstained: bool
    gen_input_tokens: int
    gen_output_tokens: int
    judge_input_tokens: int
    judge_output_tokens: int
    latency_s: float


def run_tier23(
    examples: list[dict], retriever: Retriever, generator: Generator, judge: Judge,
    cfg_name: str, tier: int, candidate_k: int = config.RETRIEVAL_CANDIDATES,
) -> list[Tier23Row]:
    cfg = CONFIGS[cfg_name]
    rows = []
    for ex in examples:
        t0 = time.perf_counter()
        pool = retriever.search(
            ex["question"], top_k=candidate_k, use_bm25=cfg["use_bm25"], use_rerank=False,
            candidate_k=candidate_k,
        )
        final = retriever.search(
            ex["question"], top_k=config.DEFAULT_TOP_K, use_bm25=cfg["use_bm25"],
            use_rerank=cfg["use_rerank"], candidate_k=candidate_k,
        )
        gold = {ex["gold_chunk_id"]}
        hit50 = score_query([c.id for c in pool], gold, k=candidate_k).hit_at_k
        hit5 = score_query([c.id for c in final], gold, k=config.DEFAULT_TOP_K).hit_at_k

        answer = generator.generate(ex["question"], final, include_images=False)
        context_text = "\n\n".join(c.text for c in final)
        chunk_texts = [c.text for c in final]

        cp = context_precision(judge, ex["question"], ex["reference_answer"], chunk_texts)
        cr = context_recall(judge, ex["question"], ex["reference_answer"], context_text)
        fa = faithfulness(judge, answer.text, context_text)
        rr = response_relevancy(judge, ex["question"], answer.text)

        judge_in = sum(
            m.judge_response.usage.get("prompt_tokens", 0)
            for m in (cp, cr, fa, rr) if m.judge_response
        )
        judge_out = sum(
            m.judge_response.usage.get("completion_tokens", 0)
            for m in (cp, cr, fa, rr) if m.judge_response
        )
        gen_usage = answer.usage or {}

        rows.append(Tier23Row(
            question_id=ex["id"], config_name=cfg_name, tier=tier, question=ex["question"],
            gold_chunk_id=ex["gold_chunk_id"], hit_at_50=hit50, hit_at_5=hit5,
            context_precision=cp.score, context_recall=cr.score,
            faithfulness=fa.score, response_relevancy=rr.score,
            abstained=answer.abstained,
            gen_input_tokens=gen_usage.get("input_tokens", 0),
            gen_output_tokens=gen_usage.get("output_tokens", 0),
            judge_input_tokens=judge_in, judge_output_tokens=judge_out,
            latency_s=time.perf_counter() - t0,
        ))
    return rows


def _langsmith_dataset(examples: list[dict]) -> str:
    """Create-or-reuse config.LANGSMITH_DATASET_NAME - one dataset shared across every eval run,
    not a fresh one per invocation. LangSmith's `evaluate()` compares experiments run against the
    SAME dataset over time; a new dataset each run would defeat "quality over time, not measured
    once" (ROADMAP §Phase 8) before it started."""
    from langsmith import Client

    client = Client()
    try:
        dataset = client.read_dataset(dataset_name=config.LANGSMITH_DATASET_NAME)
    except Exception:
        dataset = client.create_dataset(
            config.LANGSMITH_DATASET_NAME,
            description="goldset_curated.jsonl questions - reused across every rag.eval "
                        "--langsmith run so experiments are comparable over time.",
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": ex["question"]},
                    "outputs": {
                        "reference_answer": ex["reference_answer"],
                        "gold_chunk_id": ex["gold_chunk_id"],
                    },
                    "metadata": {"id": ex["id"], "category": ex.get("category")},
                }
                for ex in examples
            ],
        )
    return dataset.name


def run_tier23_langsmith(
    examples: list[dict], retriever: Retriever, generator: Generator, judge: Judge,
    cfg_name: str, tier: int, candidate_k: int = config.RETRIEVAL_CANDIDATES,
) -> None:
    """Additive LangSmith tracking for Tier 2/3 (`--langsmith`): the SAME retrieval+generation+
    judge-metrics logic as run_tier23(), reshaped into `langsmith.evaluate()`'s target/evaluator
    functions so results land as one comparable "experiment" against `_langsmith_dataset()`'s
    dataset, viewable side-by-side against prior runs in LangSmith's UI - never a replacement for
    run_tier23()'s CSV, which stays the ground-truth artifact regardless (module docstring).

    Examples not present in the dataset yet (e.g. this sample is a random subset of a larger
    goldset) are added on the fly - `evaluate()` needs every question it's given to already exist
    as a dataset example.
    """
    from langsmith import evaluate

    cfg = CONFIGS[cfg_name]
    by_question = {ex["question"]: ex for ex in examples}

    def target(inputs: dict) -> dict:
        ex = by_question[inputs["question"]]
        final = retriever.search(
            ex["question"], top_k=config.DEFAULT_TOP_K, use_bm25=cfg["use_bm25"],
            use_rerank=cfg["use_rerank"], candidate_k=candidate_k,
        )
        answer = generator.generate(ex["question"], final, include_images=False)
        return {
            "answer": answer.text,
            "context": "\n\n".join(c.text for c in final),
            "chunk_texts": [c.text for c in final],
        }

    def eval_context_precision(run, example) -> dict:
        result = context_precision(
            judge, example.inputs["question"], example.outputs["reference_answer"],
            run.outputs["chunk_texts"],
        )
        return {"key": "context_precision", "score": result.score}

    def eval_context_recall(run, example) -> dict:
        result = context_recall(
            judge, example.inputs["question"], example.outputs["reference_answer"],
            run.outputs["context"],
        )
        return {"key": "context_recall", "score": result.score}

    def eval_faithfulness(run, example) -> dict:
        result = faithfulness(judge, run.outputs["answer"], run.outputs["context"])
        return {"key": "faithfulness", "score": result.score}

    def eval_response_relevancy(run, example) -> dict:
        result = response_relevancy(judge, example.inputs["question"], run.outputs["answer"])
        return {"key": "response_relevancy", "score": result.score}

    dataset_name = _langsmith_dataset(examples)
    evaluate(
        target, data=dataset_name,
        evaluators=[eval_context_precision, eval_context_recall, eval_faithfulness,
                    eval_response_relevancy],
        experiment_prefix=f"tier{tier}-{cfg_name}",
        metadata={"tier": tier, "config": cfg_name},
    )


def _write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _summarize_tier1(rows: list[Tier1Row]) -> dict:
    from rag.metrics_retrieval import RetrievalScore
    at50 = [RetrievalScore(r.hit_at_50, r.mrr_50, r.ndcg_50, None) for r in rows]
    at5 = [RetrievalScore(r.hit_at_5, r.mrr_5, r.ndcg_5, None) for r in rows]
    return {"recall@50": aggregate(at50), "recall@5": aggregate(at5)}


def _mean_or_null(values: list[Optional[float]]) -> tuple[Optional[float], int]:
    present = [v for v in values if v is not None]
    nulls = len(values) - len(present)
    mean = sum(present) / len(present) if present else None
    return mean, nulls


def _summarize_tier23(rows: list[Tier23Row]) -> dict:
    summary = {
        "n": len(rows),
        "recall@50": sum(r.hit_at_50 for r in rows) / len(rows),
        "recall@5": sum(r.hit_at_5 for r in rows) / len(rows),
        "abstention_rate": sum(r.abstained for r in rows) / len(rows),
    }
    for metric_name, attr in (
        ("context_precision", "context_precision"), ("context_recall", "context_recall"),
        ("faithfulness", "faithfulness"), ("response_relevancy", "response_relevancy"),
    ):
        mean, nulls = _mean_or_null([getattr(r, attr) for r in rows])
        summary[metric_name] = mean
        summary[f"{metric_name}_nulls"] = nulls
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.eval")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--config", choices=list(CONFIGS), required=True)
    parser.add_argument("-n", "--n", type=int, default=None,
                        help="questions to sample (default: tier's roadmap default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goldset", default=str(config.GOLDSET_PATH))
    parser.add_argument("--langsmith", action="store_true",
                        help="Tier 2/3 only: ALSO push results to LangSmith as a tracked "
                             "experiment (ROADMAP Phase 8) - additive, the CSV is written either "
                             "way. Routes to the tier2/tier3 LangSmith project and traces every "
                             "row (no interactive-dev sampling)")
    args = parser.parse_args(argv)

    if args.langsmith and args.tier in (2, 3):
        os.environ["LANGSMITH_PROJECT"] = (
            config.LANGSMITH_PROJECT_TIER2 if args.tier == 2 else config.LANGSMITH_PROJECT_TIER3
        )
        os.environ.pop("LANGSMITH_TRACING_SAMPLING_RATE", None)

    n = args.n or TIER_DEFAULT_N[args.tier]
    examples = sample_goldset(load_goldset(Path(args.goldset)), n, seed=args.seed)
    print(f"Tier {args.tier} | config={args.config} | n={len(examples)}")

    with Retriever() as retriever:
        if args.tier == 1:
            rows = run_tier1(examples, retriever, args.config)
            summary = _summarize_tier1(rows)
        else:
            generator, judge = Generator(), Judge()
            rows = run_tier23(examples, retriever, generator, judge, args.config, args.tier)
            summary = _summarize_tier23(rows)
            if args.langsmith:
                print(f"\npushing to LangSmith project '{os.environ['LANGSMITH_PROJECT']}'...")
                run_tier23_langsmith(examples, retriever, generator, judge, args.config, args.tier)

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = config.EVAL_RESULTS_DIR / f"tier{args.tier}_{args.config}_{timestamp}.csv"
    if rows:
        _write_csv(rows, out_path)
        print(f"wrote {len(rows)} rows -> {out_path}")

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
