# Phase 3 — Ensemble retrieval + reranking: defense notes

Conceptual notes for `rag/retrieval.py` (RRF fusion + cross-encoder rerank), written before
implementation for defense prep. See ROADMAP.md §5 for the pre-written defense questions this
supports directly: "Why RRF over weighted score fusion?"

---

## 1. Pipeline shape

```
query
  ├─→ dense retriever (BGE-M3 + Qdrant, cosine)  → ranked list A (top-50)
  └─→ sparse retriever (BM25)                    → ranked list B (top-50)
           ↓
      RRF fusion (rank-based merge of A and B)
           ↓
      one fused ranked list, top-50
           ↓
      cross-encoder reranker (bge-reranker-v2-m3)
           ↓
      top-5 → generator
```

Two stages, two different jobs:

- **RRF fusion optimizes recall** — cast a wide net cheaply by combining two independently-cheap
  rankings (semantic + lexical) so a doc strong in *either* signal survives into the top-50.
- **Cross-encoder rerank optimizes precision** — of the docs already found, rank the truly
  relevant ones highest.

This split is exactly why the ablation table separates `recall@50` (pre-rerank) from `recall@5`
(post-rerank): each number attributes failure to a different stage. A bad final answer is either
"we never found it" (fusion failure) or "we found it but reranking buried it" (rerank failure).

---

## 2. RRF fusion — mechanics

RRF merges ranked lists **by rank position only**, never by raw score:

```
RRF(d) = Σ  1 / (k + rank_i(d))
```

summed over each retriever `i` that returned document `d`, where `rank_i(d)` is `d`'s position
in that retriever's list (1st, 2nd, ...) and `k` is a damping constant (commonly 60).

**Why rank, not score:** BM25 scores (unbounded, term-frequency-based) and cosine similarity
(bounded [-1, 1]) live on incomparable scales. Weighted score fusion requires inventing a
normalization scheme and a weight hyperparameter with no principled way to justify either value.
RRF sidesteps this: a doc ranked #1 in BM25 and #3 in dense outranks one that's #1 in dense but
absent from BM25 entirely, with no tunable fusion weight to defend.

**Important nuance — this is not a comparison/validation step.** RRF's fused output *replaces*
the two input lists for everything downstream; the originals aren't kept around to check the
fusion against at inference time. Per-stage scores are still logged for provenance (feeds the
Phase 10 failure taxonomy), but that's diagnostic bookkeeping for the ablation table — not
something the live pipeline does per-query.

---

## 3. Cross-encoder vs. bi-encoder — what actually changes

**Bi-encoder (BGE-M3, used for the dense retrieval stage):** query and document are embedded
*separately* into vectors; relevance is a similarity function (cosine) between two independently
-produced points. This separateness is what makes it scale — every document's vector is
precomputed once and stored in Qdrant; only the query is embedded at request time. Search over
the whole corpus is cheap.

**Cross-encoder (bge-reranker-v2-m3, used for reranking):** query and *one specific* candidate
document are fed together as a single input, through a transformer with full cross-attention —
every query token attends to every document token and vice versa. Output is one relevance score
for that pair. This catches interactions a bi-encoder's "two vectors, one dot product" misses:
shared vocabulary that answers a different question, subtle query-document relationships only
visible once the model sees both texts at once.

**The cost is exactly why it's positioned where it is:** a cross-encoder score doesn't exist
until you have both the query and the doc, so there's no vector to precompute or index. Running
it over the full 10K-doc corpus is infeasible; running it over the RRF-fused top-50 is cheap.
That's the whole reason the pipeline looks like recall-stage → precision-stage rather than
either stage alone.

---

## 4. One-line answers for the defense

- **Why RRF over weighted score fusion?** Scores are incomparable across BM25/cosine; ranks
  aren't. No unjustifiable normalization hyperparameter.
- **What does RRF actually output?** One fused top-50 ranking, replacing both inputs — not a
  comparison against the originals.
- **Why cross-encoder over just using the bi-encoder's similarity score?** Joint cross-attention
  over (query, doc) catches relevance signals a separately-encoded similarity score can't; it's
  restricted to the top-50 shortlist because it can't be indexed/precomputed.
- **How do you know fusion vs. rerank each earned its place?** `recall@50` isolates fusion's
  contribution; `recall@5` isolates the reranker's. Ablation rows 1/2/3 (ROADMAP §4) make each
  addition a measurable delta.
