"""Phase 1 retrieval + generation.

Deliberately importable rather than notebook-resident: the CLI and the notebook run the SAME
code path, so what gets eyeballed in `notebooks/generate_phase1.ipynb` is what ships. A helper
copy-pasted into a notebook drifts from the one in the CLI within a day, and then the numbers
in the write-up describe neither.
"""

from rag.config import COLLECTION, DEFAULT_TOP_K, GEN_MODEL_ID
from rag.generation import Answer, Generator
from rag.retrieval import RetrievedChunk, Retriever

__all__ = [
    "Answer",
    "COLLECTION",
    "DEFAULT_TOP_K",
    "GEN_MODEL_ID",
    "Generator",
    "RetrievedChunk",
    "Retriever",
]
