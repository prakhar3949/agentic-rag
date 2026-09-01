"""Ensemble retrieval over the Qdrant collection built by notebooks/embed_index_phase1.ipynb.

Dense (BGE-M3) and BM25 (Qdrant native sparse vector) run as two separate `query_points` calls,
fused by rank alone via `rag.fusion.reciprocal_rank_fusion`, then optionally reranked by
`rag.rerank.Reranker`'s cross-encoder. `use_bm25=False, use_rerank=False` reproduces ablation row
1's dense-only floor exactly - that combination is what Phase 1 shipped and everything else in
this file has to beat.
"""

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient, models

from rag import config
from rag.fusion import reciprocal_rank_fusion
from rag.rerank import Reranker


class CollectionMissingError(RuntimeError):
    pass


class StorageLockedError(RuntimeError):
    pass


@dataclass
class RetrievedChunk:
    """One hit, carrying the provenance the citation layer needs.

    Everything here comes straight off the Qdrant payload - retrieval never re-reads the JSONL.
    Qdrant is the only store in this system; there is no second lookup by ID somewhere else.
    """

    id: str
    score: float
    text: str
    arxiv_id: str
    category: str
    topic: str
    section: str
    modality: str
    page: Optional[int]
    figure_ref: Optional[str]
    section_source: str

    # Per-stage provenance, populated only for stages that actually ran. `score` above keeps its
    # existing meaning - whichever stage ran last (rerank > RRF > raw dense) - so every existing
    # call site (cli.py, JSON serialization) needs no changes. These feed Phase 10's failure
    # taxonomy and the recall@50-pre-rerank vs recall@5-post-rerank decomposition (ROADMAP §3②).
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    @classmethod
    def from_point(cls, point: models.ScoredPoint, **provenance) -> "RetrievedChunk":
        return cls.from_payload(str(point.id), point.payload or {}, point.score, **provenance)

    @classmethod
    def from_payload(
        cls, chunk_id: str, payload: dict, score: float, **provenance
    ) -> "RetrievedChunk":
        p = payload
        return cls(
            id=chunk_id,
            score=score,
            text=p.get("text", ""),
            arxiv_id=p.get("arxiv_id", ""),
            category=p.get("category", ""),
            topic=p.get("topic", ""),
            section=p.get("section", ""),
            modality=p.get("modality", ""),
            page=p.get("page"),
            figure_ref=p.get("figure_ref"),
            section_source=p.get("section_source", ""),
            **provenance,
        )

    def image_path(self) -> Optional[Path]:
        """The original image bytes for a figure chunk, or None.

        ROADMAP §1's multimodal claim has two halves: figures are RETRIEVED by their VLM caption,
        and the ORIGINAL image is handed to the generator at answer time. This resolves the
        second half - `data/phase1/images/<arxiv_id>/<figure_ref>.<ext>`, written by
        `save_extractions` during ingestion.

        Globbed rather than read out of documents.jsonl so retrieval stays dependent on Qdrant
        alone. Tables also carry a figure_ref (the table_id) and have no image file, which is why
        modality is checked first rather than relying on the glob coming back empty.
        """
        if self.modality != "figure" or not self.figure_ref:
            return None
        matches = sorted((config.IMAGES_DIR / self.arxiv_id).glob(f"{self.figure_ref}.*"))
        return matches[0] if matches else None

    def provenance(self) -> str:
        """Human-readable source line. What the CLI prints under 'Sources'."""
        where = self.section
        if self.page is not None:
            where += f", p{self.page + 1}"  # pages are 0-indexed internally, 1-indexed for humans
        ref = f" {self.figure_ref}" if self.figure_ref else ""
        return f"{self.arxiv_id} <{self.category}> ({self.modality}: {where}){ref}"


class Retriever:
    """Embedder + Qdrant client. The model load is the slow part, so it is lazy and reused."""

    def __init__(self, collection: str = config.COLLECTION):
        self.collection = collection
        self._embedder = None
        self._sparse_embedder = None
        self._client = None
        self._reranker = None
        # Phase 5 (rag/graph.py) is the first caller to share one Retriever across concurrent
        # threads - LangGraph's Send-based fanout runs topic_subagent branches in parallel, each
        # calling .search() on the SAME instance. Every lazy property below was a plain
        # check-then-set with no lock, which is fine sequentially but is a genuine race the moment
        # two branches touch an uninitialized property at once - for `client` specifically, two
        # concurrent QdrantClient(path=...) calls against the same embedded-mode storage directory
        # don't just duplicate work, they raise StorageLockedError (found running rag/agent_cli.py
        # end-to-end: two branches, one lock, one loser). Double-checked locking below closes that
        # without giving up the fast (lock-free) path once a property is already initialized.
        self._init_lock = threading.Lock()

    # --- lazy resources ---------------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            with self._init_lock:
                if self._embedder is None:
                    # Imported here, not at module scope: sentence-transformers pulls in torch,
                    # which costs seconds. `python -m rag.cli --help` should not pay for that.
                    from sentence_transformers import SentenceTransformer

                    self._embedder = SentenceTransformer(config.EMBED_MODEL_ID)
        return self._embedder

    @property
    def sparse_embedder(self):
        if self._sparse_embedder is None:
            with self._init_lock:
                if self._sparse_embedder is None:
                    from fastembed import SparseTextEmbedding

                    self._sparse_embedder = SparseTextEmbedding(model_name=config.SPARSE_MODEL_ID)
        return self._sparse_embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            with self._init_lock:
                if self._reranker is None:
                    self._reranker = Reranker()
        return self._reranker

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            with self._init_lock:
                if self._client is None:
                    self._init_client()
        return self._client

    def _init_client(self) -> None:
        if config.QDRANT_MODE == "local":
            try:
                self._client = QdrantClient(path=str(config.QDRANT_PATH))
            except (RuntimeError, sqlite3.OperationalError) as exc:
                # Embedded Qdrant is single-writer and takes an exclusive lock on its storage
                # directory. It reports that contention TWO different ways depending on which
                # process got there first:
                #
                #   RuntimeError("Storage folder ... already accessed by another instance")
                #       - the .lock file was already claimed
                #   sqlite3.OperationalError("database is locked")
                #       - the .lock file was free but SQLite itself was busy, which is the
                #         race window when two processes open the store at the same moment
                #
                # Both are the same situation and neither means the index is damaged, but the
                # second one is a bare SQLite error with no mention of Qdrant, which reads
                # like corruption. Catching only RuntimeError leaks it as a raw traceback.
                raise StorageLockedError(
                    f"Qdrant storage at {config.QDRANT_PATH} is locked by another process "
                    f"(usually a running notebook kernel). Shut that kernel down, or run "
                    f"`retriever.close()` / `client.close()` in it, then retry. The index "
                    f"itself is fine.\n  original: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            self._client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

        if not self._client.collection_exists(self.collection):
            raise CollectionMissingError(
                f"Qdrant collection '{self.collection}' does not exist. Run "
                f"notebooks/embed_index_phase1.ipynb to build the index first."
            )

    # --- query ------------------------------------------------------------------------
    def embed_query(self, query: str) -> list[float]:
        """BGE-M3 takes the raw query.

        No instruction prefix: BGE-M3 is trained without one, unlike `bge-large-en-v1.5` which
        needs "Represent this sentence for searching relevant passages:". Adding it here hurts.
        Normalised to match the indexed vectors, so cosine is a dot product on both sides.
        """
        return self.embedder.encode(query, normalize_embeddings=True).tolist()

    def embed_query_sparse(self, query: str) -> models.SparseVector:
        """fastembed's query-side BM25 vector: term presence, not the document-side term-weight
        form (`.embed()`) used at index time - Qdrant's `Modifier.IDF` applies corpus IDF to the
        document side only, which is the standard fastembed/Qdrant BM25 hybrid-search split."""
        embedding = next(iter(self.sparse_embedder.query_embed([query])))
        return models.SparseVector(
            indices=list(embedding.indices), values=list(embedding.values)
        )

    def _build_filter(
        self, category: Optional[str], modality: Optional[str]
    ) -> Optional[models.Filter]:
        conditions = []
        if category:
            conditions.append(
                models.FieldCondition(key="category", match=models.MatchValue(value=category))
            )
        if modality:
            conditions.append(
                models.FieldCondition(key="modality", match=models.MatchValue(value=modality))
            )
        return models.Filter(must=conditions) if conditions else None

    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        category: Optional[str] = None,
        modality: Optional[str] = None,
        use_bm25: bool = True,
        use_rerank: bool = True,
        candidate_k: int = config.RETRIEVAL_CANDIDATES,
        rrf_k: int = config.RRF_K,
    ) -> list[RetrievedChunk]:
        """Ablation rows fall out of two independent booleans:

        row 1 (dense only)          use_bm25=False, use_rerank=False - byte-for-byte the Phase 1
                                     path: fetches exactly top_k, no fusion, no rerank.
        row 2 (+ BM25, RRF)         use_bm25=True,  use_rerank=False - each branch fetches
                                     candidate_k, fused by rank, truncated to top_k.
        row 3 (+ cross-encoder)     use_bm25=True,  use_rerank=True  - candidate_k fused
                                     candidates reranked down to top_k.
        row 8 (long-context, no rerank, stuffed) - call with
                                     use_bm25=True, use_rerank=False, top_k=candidate_k.
        """
        query_filter = self._build_filter(category, modality)
        fetch_k = candidate_k if (use_bm25 or use_rerank) else top_k

        if use_bm25:
            dense_points = self.client.query_points(
                collection_name=self.collection,
                query=self.embed_query(query),
                using="dense",
                limit=fetch_k,
                query_filter=query_filter,
                with_payload=True,
            ).points
            sparse_points = self.client.query_points(
                collection_name=self.collection,
                query=self.embed_query_sparse(query),
                using=config.SPARSE_VECTOR_NAME,
                limit=fetch_k,
                query_filter=query_filter,
                with_payload=True,
            ).points

            dense_rank = {str(p.id): (i, p.score) for i, p in enumerate(dense_points, 1)}
            bm25_rank = {str(p.id): (i, p.score) for i, p in enumerate(sparse_points, 1)}
            payload_by_id = {str(p.id): (p.payload or {}) for p in (*dense_points, *sparse_points)}

            fused = reciprocal_rank_fusion(
                [[str(p.id) for p in dense_points], [str(p.id) for p in sparse_points]],
                k=rrf_k,
            )[:fetch_k]

            chunks = []
            for chunk_id, rrf_score in fused:
                d_rank, d_score = dense_rank.get(chunk_id, (None, None))
                b_rank, b_score = bm25_rank.get(chunk_id, (None, None))
                chunks.append(RetrievedChunk.from_payload(
                    chunk_id, payload_by_id[chunk_id], score=rrf_score,
                    dense_rank=d_rank, dense_score=d_score,
                    bm25_rank=b_rank, bm25_score=b_score, rrf_score=rrf_score,
                ))
        else:
            points = self.client.query_points(
                collection_name=self.collection,
                query=self.embed_query(query),
                using="dense",
                limit=fetch_k,
                query_filter=query_filter,
                with_payload=True,
            ).points
            chunks = [
                RetrievedChunk.from_point(p, dense_rank=i, dense_score=p.score)
                for i, p in enumerate(points, 1)
            ]

        if use_rerank:
            chunks = self.reranker.rerank(query, chunks, top_k)
        else:
            chunks = chunks[:top_k]

        return chunks

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
