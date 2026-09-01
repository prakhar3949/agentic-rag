"""Cross-encoder reranking: top-N fused candidates -> top-k, precision over the bi-encoder pass.

A cross-encoder scores (query, doc) jointly with full cross-attention, rather than comparing
independently-computed embeddings - more expensive per pair, which is exactly why it runs on the
already-narrowed candidate pool instead of the whole collection.
"""

import threading
from typing import TYPE_CHECKING

from rag import config

if TYPE_CHECKING:
    from rag.retrieval import RetrievedChunk


class Reranker:
    """Lazily loads the cross-encoder. Same rationale as Retriever.embedder: sentence-transformers
    pulls in torch, which costs seconds - `python -m rag.cli --help` should not pay for that."""

    def __init__(self, model_id: str = config.RERANK_MODEL_ID):
        self.model_id = model_id
        self._model = None
        # Same double-checked-locking reason as rag/retrieval.py's Retriever: a shared Reranker
        # (retriever.reranker) can be hit by Phase 5's parallel topic_subagent branches, each
        # calling .rerank() -> this lazy property, before any of them has initialized it.
        self._init_lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._init_lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    # Explicit, not left at the library default. PHASE1_NOTES §10.2 found BGE-M3's
                    # embedder truncates past its max length silently - no error, a valid-looking
                    # output that just doesn't cover the back of the text. Same risk if implicit.
                    self._model = CrossEncoder(self.model_id, max_length=1024)
        return self._model

    def rerank(
        self, query: str, chunks: list["RetrievedChunk"], top_k: int
    ) -> list["RetrievedChunk"]:
        if not chunks:
            return []
        scores = self.model.predict([(query, chunk.text) for chunk in chunks])
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
            chunk.score = float(score)  # score always reflects whichever stage ran last
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])
        return [chunks[i] for i in order[:top_k]]
