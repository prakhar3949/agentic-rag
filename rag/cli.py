"""Question in, cited answer out.

    python -m rag.cli "How do multi-agent systems adapt their communication topology?"

Phase 1 shipped this as dense-only, one generation call, cited answer. Phase 3 added BM25 fusion
and cross-encoder reranking underneath it - `--no-bm25`/`--no-rerank` step back through the
ablation arms without touching generation at all.
"""

import argparse
import dataclasses
import json
import sys

from rag.generation import Generator
from rag.retrieval import CollectionMissingError, Retriever, StorageLockedError
from rag.scorelog import log_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.cli",
        description="One question in, one generation call, cited answer - "
                    "dense + BM25 + reranking by default.",
    )
    parser.add_argument("question", help="the question to answer")
    parser.add_argument("-k", "--top-k", type=int, default=None,
                        help="chunks to retrieve (default: config.DEFAULT_TOP_K)")
    parser.add_argument("--category", default=None,
                        help="restrict to one arXiv category, e.g. ai, astronomy, hep-th")
    parser.add_argument("--modality", default=None, choices=["text", "figure", "table"],
                        help="restrict to one modality")
    parser.add_argument("--no-images", action="store_true",
                        help="skip attaching original figure bytes (text-over-captions only)")
    parser.add_argument("--no-bm25", action="store_true",
                        help="disable BM25/RRF fusion - dense only (ablation row 1)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="disable cross-encoder reranking")
    parser.add_argument("--candidates", type=int, default=None,
                        help="pre-rerank/pre-fusion candidate pool size "
                             "(default: config.RETRIEVAL_CANDIDATES)")
    parser.add_argument("--no-log", action="store_true",
                        help="skip appending this query's scores to the retrieval score log")
    parser.add_argument("--show-context", action="store_true",
                        help="print every retrieved chunk before the answer")
    parser.add_argument("--json", action="store_true",
                        help="emit the full Answer as JSON - the eval-harness surface")
    return parser


def render(answer, show_context: bool) -> None:
    if show_context:
        print("Retrieved\n" + "-" * 70)
        for handle, chunk in enumerate(answer.chunks, 1):
            print(f"[{handle}] {chunk.score:.4f}  {chunk.provenance()}")
            stages = (
                f"dense={chunk.dense_score:.4f}" if chunk.dense_score is not None else "dense=-",
                f"bm25={chunk.bm25_score:.4f}" if chunk.bm25_score is not None else "bm25=-",
                f"rrf={chunk.rrf_score:.4f}" if chunk.rrf_score is not None else "rrf=-",
                f"rerank={chunk.rerank_score:.4f}" if chunk.rerank_score is not None else "rerank=-",
            )
            print(f"     [{' '.join(stages)}]")
            print(f"     {' '.join(chunk.text.split())[:300]}")
        print()

    print(answer.text.strip() + "\n")

    if answer.sources():
        print("Sources\n" + "-" * 70)
        for handle, chunk in answer.sources():
            print(f"  [{handle}] {chunk.provenance()}  score={chunk.score:.4f}")
            print(f"       {chunk.topic[:90]}")

    # Diagnostics, always shown. Every one of these lines is a Phase 10 failure-mode counter
    # that costs nothing to collect now and cannot be reconstructed later.
    if answer.invalid:
        print(f"\n  WARNING: answer cites {answer.invalid}, but only "
              f"{len(answer.chunks)} sources were provided - broken citation(s).")
    if answer.uncited:
        print(f"\n  {len(answer.uncited)}/{len(answer.chunks)} retrieved chunks went uncited: "
              f"{answer.uncited}   (retrieved-and-ignored, or retrieval noise)")
    if answer.abstained:
        print("\n  Model abstained: the retrieved context did not support an answer.")

    usage = answer.usage or {}
    tokens = (f"{usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out"
              if usage else "tokens unavailable")
    print(f"\n  {answer.model} | {tokens} | {answer.images_attached} image(s) attached "
          f"| {answer.latency_s:.1f}s")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        with Retriever() as retriever:
            kwargs = {
                "category": args.category,
                "modality": args.modality,
                "use_bm25": not args.no_bm25,
                "use_rerank": not args.no_rerank,
            }
            if args.top_k is not None:
                kwargs["top_k"] = args.top_k
            if args.candidates is not None:
                kwargs["candidate_k"] = args.candidates
            chunks = retriever.search(args.question, **kwargs)

            if not args.no_log:
                stages = ["dense"]
                if not args.no_bm25:
                    stages.append("bm25")
                if not args.no_rerank:
                    stages.append("rerank")
                log_query(args.question, "+".join(stages), chunks)

            answer = Generator().generate(
                args.question, chunks, include_images=not args.no_images
            )
    except (CollectionMissingError, StorageLockedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = dataclasses.asdict(answer)
        payload["chunks"] = [{**c, "image_path": str(answer.chunks[i].image_path() or "")}
                             for i, c in enumerate(payload["chunks"])]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Q: {args.question}\n")
        render(answer, show_context=args.show_context)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
