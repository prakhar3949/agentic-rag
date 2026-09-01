"""Non-LLM retrieval metrics: hit@k, MRR, NDCG. Plain ranking arithmetic, no judge calls.

This is Tier 1 (ROADMAP §Phase 4): free, instant, and what most eval iterations should run
against while tuning the retriever/chunking. Each function takes the ranked list of retrieved
chunk IDs and the set of gold-relevant chunk IDs for one query - the golden set (`rag/goldset.py`)
generates one gold chunk per question by construction, but every function here accepts a set so a
future multi-hop question with several gold chunks needs no rewrite.
"""

import math
from dataclasses import dataclass


@dataclass
class RetrievalScore:
    """One query's scores. `rank` is the 1-indexed position of the first relevant hit, or None."""

    hit_at_k: bool
    reciprocal_rank: float
    ndcg: float
    rank: int | None


def score_query(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> RetrievalScore:
    top_k = retrieved_ids[:k]
    rank = next((i for i, cid in enumerate(top_k, 1) if cid in relevant_ids), None)
    return RetrievalScore(
        hit_at_k=rank is not None,
        reciprocal_rank=(1.0 / rank) if rank is not None else 0.0,
        ndcg=_ndcg(top_k, relevant_ids),
        rank=rank,
    )


def _ndcg(top_k: list[str], relevant_ids: set[str]) -> float:
    """Binary relevance NDCG. DCG discounts a hit by its rank; IDCG is the DCG of the best
    possible ordering (all relevant IDs first) - dividing by it bounds NDCG to [0, 1] regardless
    of how many relevant IDs exist or how large k is."""
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(top_k, 1) if cid in relevant_ids)
    ideal_hits = min(len(relevant_ids), len(top_k))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(scores: list[RetrievalScore]) -> dict:
    """Mean each metric over a batch of queries - one row of the ablation table's
    recall@50 / recall@5 columns come from this, run once per (config, tier)."""
    n = len(scores)
    if n == 0:
        return {"hit_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0, "n": 0}
    return {
        "hit_at_k": sum(s.hit_at_k for s in scores) / n,
        "mrr": sum(s.reciprocal_rank for s in scores) / n,
        "ndcg": sum(s.ndcg for s in scores) / n,
        "n": n,
    }
