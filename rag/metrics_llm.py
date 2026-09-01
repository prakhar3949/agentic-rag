"""The four purpose-written LLM-judge metrics (ROADMAP §Phase 4): context precision, context
recall, faithfulness, response relevancy. Each is a judge prompt, not an algorithm - owning them
means the prompt, the judge version, and the parsing are all things that can be read out loud
(ROADMAP §7: this is exactly why Ragas was dropped rather than pinned harder).

Every metric returns its per-claim (or per-chunk) intermediate verdicts alongside the score - a
bare 0.87 cannot be attributed to anything, and the claim-level verdicts are what make the
Phase 10 failure taxonomy possible. They cost nothing extra: the judge call already produces them.

A metric's `score` is `None`, never `0.0`, when the judge response fails to parse after retries
(`rag/judge.py`'s `JudgeResponse.parsed is None`) or when there is nothing to score (e.g. zero
claims decomposed from an empty answer). A parse failure recorded as zero would be
indistinguishable from a genuinely bad answer and would silently drag a config's mean down.
"""

from dataclasses import dataclass, field
from typing import Optional

from rag.judge import Judge, JudgeResponse

_JSON_ONLY = "Respond with a single JSON object only - no prose, no markdown fences."


@dataclass
class LLMMetricResult:
    metric: str
    score: Optional[float]
    claims: list[dict] = field(default_factory=list)   # per-claim/per-chunk intermediate verdicts
    judge_response: Optional[JudgeResponse] = None      # usage/latency, for cost tracking


def _failed(metric: str, judge_response: JudgeResponse) -> LLMMetricResult:
    return LLMMetricResult(metric=metric, score=None, claims=[], judge_response=judge_response)


# --- 1. Context precision ------------------------------------------------------------------

_CONTEXT_PRECISION_SYSTEM = f"""You judge whether retrieved passages are relevant to constructing \
a reference answer. For each numbered passage, decide if it contributes information used in the \
reference answer.

A passage is relevant only if it actually supplies specific information stated in the reference \
answer - a fact, number, name, or claim the answer contains. Sharing vocabulary or discussing a \
related concept is NOT enough: a passage that uses the same terms (e.g. the same method or metric \
name) but does not provide the particular fact the reference answer states about it is NOT \
relevant. When in doubt, ask "could this specific passage alone have supplied this part of the \
answer" - not "is this passage about the same general topic." {_JSON_ONLY}

Schema: {{"chunk_verdicts": [{{"handle": <int>, "relevant": <bool>, "reason": <string>}}]}}
One entry per passage, in the order given."""


def context_precision(
    judge: Judge, question: str, reference_answer: str, chunks: list[str]
) -> LLMMetricResult:
    """Rank-weighted relevance of each retrieved chunk against the reference answer.

    Score = mean, over every chunk judged relevant, of precision@rank at that chunk's position -
    a relevant chunk that shows up early contributes more than one buried at the bottom. 0.0 if
    no chunk is judged relevant (the standard convention, matching Ragas' definition)."""
    passages = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, 1))
    user = f"Question: {question}\n\nReference answer: {reference_answer}\n\nPassages:\n{passages}"
    resp = judge.call_json(_CONTEXT_PRECISION_SYSTEM, user)
    if resp.parsed is None:
        return _failed("context_precision", resp)

    verdicts = resp.parsed.get("chunk_verdicts", [])
    if not verdicts:
        return LLMMetricResult("context_precision", None, [], resp)

    relevant_flags = [bool(v.get("relevant")) for v in verdicts]
    n_relevant = sum(relevant_flags)
    if n_relevant == 0:
        return LLMMetricResult("context_precision", 0.0, verdicts, resp)

    precisions_at_k = []
    hits = 0
    for k, is_relevant in enumerate(relevant_flags, 1):
        if is_relevant:
            hits += 1
            precisions_at_k.append(hits / k)
    score = sum(precisions_at_k) / n_relevant
    return LLMMetricResult("context_precision", score, verdicts, resp)


# --- 2. Context recall ----------------------------------------------------------------------

_CONTEXT_RECALL_SYSTEM = f"""You decompose a reference answer into atomic factual claims, then \
judge whether each claim is supported by the retrieved context. A claim is supported only if the \
context states it or directly implies it - not if it merely seems plausible. {_JSON_ONLY}

Schema: {{"claims": [{{"claim": <string>, "supported": <bool>, "reason": <string>}}]}}"""


def context_recall(
    judge: Judge, question: str, reference_answer: str, context_text: str
) -> LLMMetricResult:
    """Fraction of the reference answer's claims supported by the retrieved context - isolates
    whether the retriever found enough to construct the correct answer, independent of what the
    generator actually did with it."""
    user = (
        f"Question: {question}\n\nReference answer: {reference_answer}\n\n"
        f"Retrieved context:\n{context_text}"
    )
    resp = judge.call_json(_CONTEXT_RECALL_SYSTEM, user)
    if resp.parsed is None:
        return _failed("context_recall", resp)

    claims = resp.parsed.get("claims", [])
    if not claims:
        return LLMMetricResult("context_recall", None, [], resp)

    supported = sum(1 for c in claims if c.get("supported"))
    return LLMMetricResult("context_recall", supported / len(claims), claims, resp)


# --- 3. Faithfulness -------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = f"""You decompose a generated answer into atomic factual claims, then \
judge whether each claim is entailed by the retrieved context - supported explicitly or by \
direct implication, not by outside knowledge. {_JSON_ONLY}

Schema: {{"claims": [{{"claim": <string>, "supported": <bool>, "reason": <string>}}]}}"""


def faithfulness(judge: Judge, generated_answer: str, context_text: str) -> LLMMetricResult:
    """Fraction of the GENERATED answer's claims that are grounded in the retrieved context -
    isolates whether the generator stayed faithful to what it was given, independent of whether
    what it was given was any good. Row 9's oracle-context arm (ROADMAP §3③) is the generation
    ceiling: faithfulness measured there with perfect context is the true generator-quality floor."""
    user = f"Generated answer: {generated_answer}\n\nRetrieved context:\n{context_text}"
    resp = judge.call_json(_FAITHFULNESS_SYSTEM, user)
    if resp.parsed is None:
        return _failed("faithfulness", resp)

    claims = resp.parsed.get("claims", [])
    if not claims:
        return LLMMetricResult("faithfulness", None, [], resp)

    supported = sum(1 for c in claims if c.get("supported"))
    return LLMMetricResult("faithfulness", supported / len(claims), claims, resp)


# --- 4. Response relevancy -------------------------------------------------------------------

_RESPONSE_RELEVANCY_SYSTEM = f"""You judge whether a generated answer directly addresses the \
question that was actually asked - relevance and completeness of the response to the question, \
NOT factual correctness (a confidently wrong answer to the right question still scores high \
here; a correct answer to a different question scores low). Score from 0.0 (entirely off-topic \
or non-responsive) to 1.0 (fully addresses the question asked). {_JSON_ONLY}

Schema: {{"score": <float 0.0-1.0>, "reason": <string>}}"""


def response_relevancy(judge: Judge, question: str, generated_answer: str) -> LLMMetricResult:
    """Does the answer address the question actually asked - deliberately not decomposed into
    claims (there is nothing to decompose; it is one judgment about the answer as a whole), but
    the reason is still returned as the metric's intermediate output."""
    user = f"Question: {question}\n\nGenerated answer: {generated_answer}"
    resp = judge.call_json(_RESPONSE_RELEVANCY_SYSTEM, user)
    if resp.parsed is None:
        return _failed("response_relevancy", resp)

    score = resp.parsed.get("score")
    if not isinstance(score, (int, float)):
        return LLMMetricResult("response_relevancy", None, [resp.parsed], resp)
    return LLMMetricResult(
        "response_relevancy", float(score), [{"reason": resp.parsed.get("reason", "")}], resp
    )
