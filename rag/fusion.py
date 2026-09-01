"""Reciprocal rank fusion - owned math, not Qdrant's built-in FusionQuery.

Qdrant can fuse prefetch branches server-side, but that path has no tunable `k`, and ROADMAP
Phase 3 requires `k` exposed as config. RRF is a few lines of arithmetic anyway: owning it means
"why RRF over weighted score fusion, and why k=60" is answerable by pointing at this function,
the same reasoning that dropped Ragas for hand-written judge metrics (ROADMAP §7).
"""

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[str]], k: int
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists by rank alone - no raw scores, which sit on incomparable scales
    (cosine similarity vs. BM25 term weight). Each list contributes 1/(k + rank) per ID it
    contains, ranks are 1-indexed, and an ID absent from a list simply contributes nothing from
    it. Returns (id, fused_score) sorted by descending fused score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
