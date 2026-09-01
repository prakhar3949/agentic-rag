"""One generation call over retrieved chunks, with citations that resolve.

The word doing the work in "one generation call" is *one*. No planner, no decomposition, no
fanout, no reranker, no web escalation - those are Phases 3-6, and each has to beat this. This
is ablation row 1, and a floor is only useful if it is honestly a floor.
"""

import base64
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from rag import config
from rag.retrieval import RetrievedChunk
from rag.websearch import WebResult

# The model cites HANDLES - [1], [2] - never identifiers. It is given numbers, it returns
# numbers, and the code maps numbers back to arxiv_ids it already knows. A model asked to
# reproduce "2607.28503v1" from context will eventually emit "2607.28530v1": a citation that
# looks perfect, resolves to nothing, and is invisible to every retrieval metric. That is
# PHASE1_NOTES §3① "citation hallucination - right answer, wrong source", designed out rather
# than measured later.
#
# INSUFFICIENT_CONTEXT is a machine-detectable abstention sentinel. Phase 10④ measures the
# abstention rate on ~40 unanswerable questions, and doing that by phrase-matching prose ("I'm
# not sure", "the context does not appear to") is a losing game. Ask for a token instead.
SYSTEM_PROMPT = """You answer questions about research papers using ONLY the numbered sources provided.

Rules:
1. Every factual claim must carry a citation to the source it came from, written in square \
brackets: [1], [3]. For several sources, write [1][3].
2. Cite ONLY the numbers shown. Never write an arXiv ID, author name, or paper title as a \
citation - the numbers are the only valid citation format.
3. If the sources do not contain enough information to answer, reply with exactly \
INSUFFICIENT_CONTEXT on the first line, then one sentence naming what is missing. Do not fall \
back on your own knowledge of the literature.
4. Do not speculate beyond what the sources state. Where sources disagree, say so and cite both.
5. Be concise. Three to six sentences unless the question needs more.
6. Sources prefixed W (e.g. [W1], [W2]) come from a live web search, not the corpus - use them \
only for what the numbered corpus sources above don't cover, prefer a corpus citation when both \
support the same claim, and say explicitly when a claim rests on a web result instead of the corpus.

Some sources are figures. Their text is a generated description of the figure, and the figure \
image itself is attached - use both."""


def specialist_system_prompt(category: str) -> str:
    """SYSTEM_PROMPT plus one line naming the category scope - Phase 5's topic subagents.

    Same rules, same citation mechanism, same abstention sentinel as every other ablation arm.
    The only difference from the baseline prompt is telling the model *why* its source list looks
    the way it does (all one category) - without this a specialist would have no way to tell a
    genuinely thin source list from a retrieval bug.
    """
    description = config.CATEGORY_DESCRIPTIONS.get(category, category)
    return (
        f"{SYSTEM_PROMPT}\n\nYou are the {category} ({description}) specialist. Every source "
        f"below was retrieved from the {category} corpus only - other topic areas are handled by "
        f"separate specialists and are deliberately not visible to you."
    )


SYNTHESIS_SYSTEM_PROMPT = """You merge draft answers written by several topic specialists into ONE final answer.

Each specialist saw only its own topic's sources and cited them by its own local numbering. You \
are given their draft text as reference notes, plus a fresh, combined, renumbered source list \
covering everything every specialist saw. Write a new answer from that combined source list \
directly - do not reuse a specialist's citation numbers, they do not match the list you were given.

Rules:
1. Every factual claim must carry a citation to the source it came from, written in square \
brackets: [1], [3]. For several sources, write [1][3].
2. Cite ONLY the numbers in the combined source list below. Never write an arXiv ID, author \
name, or paper title as a citation.
3. Resolve overlaps and disagreements between the specialists' drafts explicitly - if two \
specialists' notes conflict, say so and cite both sides.
4. If, combining every specialist's notes and sources, the question still cannot be fully \
answered, reply with exactly INSUFFICIENT_CONTEXT on the first line, then one sentence naming \
what is missing.
5. Do not speculate beyond what the combined sources state.
6. Be concise. Three to eight sentences unless the question needs more.
7. Sources prefixed W (e.g. [W1], [W2]) come from a live web search, not the corpus - use them \
only for what the numbered corpus sources don't cover, prefer a corpus citation when both \
support the same claim, and say explicitly when a claim rests on a web result instead of the corpus."""


@dataclass
class Answer:
    """A generated answer plus everything needed to audit it."""

    question: str
    text: str
    chunks: list[RetrievedChunk]
    cited: list[int] = field(default_factory=list)          # handles the model used, 1-indexed
    uncited: list[int] = field(default_factory=list)        # retrieved, given, never referenced
    invalid: list[int] = field(default_factory=list)        # handles that do not exist
    abstained: bool = False
    images_attached: int = 0
    model: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0

    # Phase 6: web search results, present only on an escalated answer (rag/websearch.py's
    # should_escalate()). A separate numbering space ([W1], [W2] - see WEB_CITATION_RE below) so
    # a web citation can never be mistaken for a corpus one, mirroring `cited`/`uncited`/`invalid`
    # above but against `web_results` instead of `chunks`.
    web_results: list[WebResult] = field(default_factory=list)
    web_cited: list[int] = field(default_factory=list)
    web_uncited: list[int] = field(default_factory=list)
    web_invalid: list[int] = field(default_factory=list)

    # Phase 7: guardrails. `blocked_reason` is None on every normal answer (including every
    # Answer built before Phase 7 existed - pure addition, no restructuring); a short-circuiting
    # input rail (rag/guardrails.py) sets it to "jailbreak" / "jailbreak_error" / "off_topic" /
    # "sensitive" on a synthetic Answer built with chunks=[] instead of ever calling generate().
    # `needs_clarification` is the dialogue rail's equivalent signal - `text` holds the clarifying
    # question itself, not an answer. `grounding_score`/`grounding_checked` come from the output
    # rail (rag/guardrails.py's grounding_node, reusing rag/metrics_llm.py's faithfulness()
    # verbatim) - `grounding_checked=False` distinguishes "the rail didn't run" (bypassed, or the
    # answer abstained so there was nothing to ground-check) from "it ran but the judge failed to
    # parse" (grounding_checked=True, grounding_score=None) - the same null-not-zero discipline
    # Phase 4's judge already uses.
    blocked_reason: Optional[str] = None
    needs_clarification: bool = False
    grounding_score: Optional[float] = None
    grounding_checked: bool = False

    def sources(self) -> list[tuple[int, RetrievedChunk]]:
        """(handle, chunk) for each corpus source the answer actually cited."""
        return [(h, self.chunks[h - 1]) for h in self.cited]

    def web_sources(self) -> list[tuple[int, WebResult]]:
        """(handle, result) for each web source the answer actually cited."""
        return [(h, self.web_results[h - 1]) for h in self.web_cited]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the chunks and stamp each with its provenance.

    The provenance header is in the prompt as well as in the payload because the generator has
    to be able to say "both papers report X" without being handed the arxiv_ids as citable
    strings - it sees where a source came from, and still cites by number.
    """
    blocks = []
    for handle, chunk in enumerate(chunks, 1):
        header = f"[{handle}] {chunk.arxiv_id} <{chunk.category}> - {chunk.topic}"
        where = f"    section: {chunk.section} | type: {chunk.modality}"
        if chunk.page is not None:
            where += f" | page {chunk.page + 1}"
        if chunk.modality == "figure":
            where += " | the image for this source is attached"
        blocks.append(f"{header}\n{where}\n{chunk.text}")
    return "\n\n".join(blocks)


def format_web_context(web_results: list[WebResult]) -> str:
    """Number web results with a W prefix - a disjoint numbering space from format_context()'s
    corpus handles, so [1] and [W1] can never resolve to each other by accident."""
    blocks = []
    for handle, result in enumerate(web_results, 1):
        blocks.append(f"[W{handle}] {result.title}\n    {result.url}\n{result.snippet}")
    return "\n\n".join(blocks)


def build_message(
    question: str,
    chunks: list[RetrievedChunk],
    web_results: list[WebResult] = (),
    include_images: bool = True,
) -> tuple[HumanMessage, int]:
    """The user turn: context, question, and the original bytes for any figure source.

    Returns (message, images_attached). Each image is preceded by a text part naming its handle,
    because a bare sequence of images gives the model no way to tell which source each belongs to
    - and a figure it cannot attribute is a figure it will cite wrongly. Web results carry no
    images - only corpus figure chunks do.
    """
    text = f"Sources:\n\n{format_context(chunks)}\n\n"
    if web_results:
        text += (
            f"Web search results (separate from the corpus above - cite as [W1], [W2]):\n\n"
            f"{format_web_context(web_results)}\n\n"
        )
    text += f"Question: {question}"
    parts: list[dict] = [{"type": "text", "text": text}]

    attached = 0
    if include_images:
        for handle, chunk in enumerate(chunks, 1):
            path = chunk.image_path()
            if chunk.modality == "figure" and path is None:
                # Loud, not silent. A figure chunk whose bytes cannot be found means retrieval
                # matched a caption for an image the generator will never see - the multimodal
                # path degrading to text-over-captions without saying so.
                print(f"[generation] figure source [{handle}] {chunk.figure_ref}: "
                      f"no image file under {config.IMAGES_DIR / chunk.arxiv_id}")
                continue
            if path is None:
                continue
            mime = path.suffix.lstrip(".").lower()
            mime = "jpeg" if mime == "jpg" else mime
            b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
            parts.append({"type": "text", "text": f"Image for source [{handle}]:"})
            parts.append({"type": "image_url", "image_url": f"data:image/{mime};base64,{b64}"})
            attached += 1

    return HumanMessage(content=parts), attached


CITATION_RE = re.compile(r"\[([0-9][0-9,\s]*)\]")

# A W immediately after '[' - CITATION_RE above requires a digit there, so the two never match
# the same bracket. Two independent numbering spaces, never one pattern with a shared parser.
WEB_CITATION_RE = re.compile(r"\[W([0-9][0-9,\s]*)\]", re.IGNORECASE)


def _parse_handles(pattern: re.Pattern, text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Shared by parse_citations/parse_web_citations: extract handles matched by `pattern`, split
    into (valid, invalid), each sorted and deduped.

    Invalid handles are returned rather than dropped. A model that cites [7] when it was given
    five sources has produced a broken link, and the difference between catching that here and
    rendering it in a UI is the difference between a bug report and a defense question.
    """
    seen: list[int] = []
    for group in pattern.findall(text):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit() and int(part) not in seen:
                seen.append(int(part))
    valid = sorted(h for h in seen if 1 <= h <= n_sources)
    invalid = sorted(h for h in seen if not 1 <= h <= n_sources)
    return valid, invalid


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Extract cited corpus handles ([1], [3]) - see _parse_handles."""
    return _parse_handles(CITATION_RE, text, n_sources)


def parse_web_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Extract cited web handles ([W1], [W2]) - see _parse_handles."""
    return _parse_handles(WEB_CITATION_RE, text, n_sources)


def message_text(response) -> str:
    """`.content` is a list of content blocks on langchain v1, a plain str on older versions.

    Calling .strip() on the list form is what silently killed every VLM caption in the first
    ingestion run (PHASE1_NOTES §3.1). Same normalisation, same reason.
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content).strip()


class Generator:
    def __init__(
        self, model_id: str = config.GEN_MODEL_ID, temperature: float = config.GEN_TEMPERATURE
    ):
        self.model_id = model_id
        self.temperature = temperature
        self._llm = None
        # Same double-checked-locking reason as rag/retrieval.py's Retriever: Phase 5's parallel
        # topic_subagent branches share one Generator and can all hit this lazy property before
        # any of them has set it.
        self._init_lock = threading.Lock()

    @property
    def llm(self):
        if self._llm is None:
            with self._init_lock:
                if self._llm is None:
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    self._llm = ChatGoogleGenerativeAI(
                        model=self.model_id, temperature=self.temperature
                    )
        return self._llm

    def structured(self, schema):
        """The same lazily-constructed `llm`, wrapped for one-shot structured output.

        Used by rag/planner.py and rag/router.py so model construction (id, temperature, API
        wiring) stays in exactly one place instead of each node building its own client.
        """
        return self.llm.with_structured_output(schema)

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        include_images: bool = True,
        system_prompt: str = SYSTEM_PROMPT,
        web_results: list[WebResult] = (),
    ) -> Answer:
        if not chunks and not web_results:
            # Nothing retrieved and no web escalation is itself an abstention, and it costs no
            # tokens to say so.
            return Answer(
                question=question,
                text="INSUFFICIENT_CONTEXT\nRetrieval returned no chunks for this question.",
                chunks=[],
                abstained=True,
                model=self.model_id,
            )

        message, attached = build_message(
            question, chunks, web_results=web_results, include_images=include_images
        )

        start = time.perf_counter()
        response = self.llm.invoke([SystemMessage(content=system_prompt), message])
        latency = time.perf_counter() - start

        text = message_text(response)
        cited, invalid = parse_citations(text, len(chunks))
        web_cited, web_invalid = parse_web_citations(text, len(web_results))

        return Answer(
            question=question,
            text=text,
            chunks=chunks,
            cited=cited,
            # Retrieved, handed to the model, and never referenced. This is the observable
            # signal for the "retrieved and ignored by generator" failure mode (ROADMAP §3①) -
            # recorded from run one so the Phase 10 taxonomy has something to count.
            uncited=[h for h in range(1, len(chunks) + 1) if h not in cited],
            invalid=invalid,
            abstained=text.strip().upper().startswith("INSUFFICIENT_CONTEXT"),
            images_attached=attached,
            model=self.model_id,
            # Token counts stored per run so $/query can be recomputed from config/pricing.yaml
            # without re-running anything (ROADMAP §Phase 4).
            usage=getattr(response, "usage_metadata", None) or {},
            latency_s=latency,
            web_results=list(web_results),
            web_cited=web_cited,
            web_uncited=[h for h in range(1, len(web_results) + 1) if h not in web_cited],
            web_invalid=web_invalid,
        )


def synthesize(
    generator: "Generator",
    question: str,
    branch_drafts: list[tuple[str, str]],
    merged_chunks: list[RetrievedChunk],
    web_results: list[WebResult] = (),
) -> Answer:
    """Phase 5's synthesizer: merges per-category branch drafts into one final cited answer.
    Phase 6 extends it with optional web results, escalated by rag/graph.py's synthesizer_node
    when `rag.websearch.should_escalate(merged_chunks)` fires - cited separately as [W1], [W2],
    never mixed into the corpus's own [1], [2] numbering (see WEB_CITATION_RE above).

    Deliberately NOT text-splicing. Each branch cited its own draft against its own local handle
    numbering (1..len(branch_chunks)), so those numbers do not carry over to a merged source list
    - reusing them here would silently produce exactly the "citation looks right, points at the
    wrong source" failure mode this codebase designed out from the start (PHASE1_NOTES §3.1).
    Instead this is a second full generation pass: the branch drafts are shown as reference notes
    (their *reasoning*, not their citation numbers), and the model is asked to write a fresh
    answer against `merged_chunks`, renumbered from scratch via the same `format_context()` /
    `parse_citations()` machinery every other ablation arm uses - so a synthesized answer is
    exactly as machine-verifiable as a single-call one.

    `branch_drafts` is (category, draft_text) pairs, not full `Answer`/`BranchResult` objects -
    keeps this module independent of rag/agent_state.py (avoids a circular import; graph.py does
    the unpacking).
    """
    if not merged_chunks and not web_results:
        return Answer(
            question=question,
            text="INSUFFICIENT_CONTEXT\nNo branch produced any retrieved sources.",
            chunks=[],
            abstained=True,
            model=generator.model_id,
        )

    notes = "\n\n".join(
        f"--- {category} specialist's draft (its own citation numbers, do not reuse) ---\n{draft}"
        for category, draft in branch_drafts
    )
    user_text = (
        f"Specialist drafts:\n\n{notes}\n\n"
        f"Combined source list (renumbered - cite THESE numbers only):\n\n"
        f"{format_context(merged_chunks)}\n\n"
    )
    if web_results:
        user_text += (
            f"Web search results (separate from the corpus above - cite as [W1], [W2]):\n\n"
            f"{format_web_context(web_results)}\n\n"
        )
    user_text += f"Question: {question}"

    start = time.perf_counter()
    response = generator.llm.invoke([
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ])
    latency = time.perf_counter() - start

    text = message_text(response)
    cited, invalid = parse_citations(text, len(merged_chunks))
    web_cited, web_invalid = parse_web_citations(text, len(web_results))

    return Answer(
        question=question,
        text=text,
        chunks=merged_chunks,
        cited=cited,
        uncited=[h for h in range(1, len(merged_chunks) + 1) if h not in cited],
        invalid=invalid,
        abstained=text.strip().upper().startswith("INSUFFICIENT_CONTEXT"),
        images_attached=0,  # synthesis is text-only - see rag/generation.py's synthesize() docstring
        model=generator.model_id,
        usage=getattr(response, "usage_metadata", None) or {},
        latency_s=latency,
        web_results=list(web_results),
        web_cited=web_cited,
        web_uncited=[h for h in range(1, len(web_results) + 1) if h not in web_cited],
        web_invalid=web_invalid,
    )
