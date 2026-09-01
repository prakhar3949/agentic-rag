"""Append-only CSV of every retrieval's per-chunk scores.

The concrete mechanism behind ROADMAP Phase 3's "every stage returns scores + provenance so
Phase 10 can attribute failures" - and the raw material for the Phase 4 eval harness and the
Phase 12 ablation table (recall@50 pre-rerank, recall@5 post-rerank, per-config comparisons)
without re-running retrieval later to reconstruct them.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from rag import config

if TYPE_CHECKING:
    from rag.retrieval import RetrievedChunk

FIELDS = [
    "timestamp", "query", "config_name", "result_rank", "chunk_id", "arxiv_id", "category",
    "modality", "section", "dense_rank", "dense_score", "bm25_rank", "bm25_score", "rrf_score",
    "rerank_score", "final_score",
]


def log_query(
    query: str,
    config_name: str,
    chunks: list["RetrievedChunk"],
    path: Path = config.SCORE_LOG_PATH,
) -> None:
    """Append one row per retrieved chunk. Writes the header only if the file doesn't exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    timestamp = datetime.now(timezone.utc).isoformat()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        for rank, chunk in enumerate(chunks, 1):
            writer.writerow({
                "timestamp": timestamp,
                "query": query,
                "config_name": config_name,
                "result_rank": rank,
                "chunk_id": chunk.id,
                "arxiv_id": chunk.arxiv_id,
                "category": chunk.category,
                "modality": chunk.modality,
                "section": chunk.section,
                "dense_rank": chunk.dense_rank,
                "dense_score": chunk.dense_score,
                "bm25_rank": chunk.bm25_rank,
                "bm25_score": chunk.bm25_score,
                "rrf_score": chunk.rrf_score,
                "rerank_score": chunk.rerank_score,
                "final_score": chunk.score,
            })
