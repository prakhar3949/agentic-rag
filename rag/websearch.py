"""Phase 6's single web search tool + escalation rule.

ROADMAP: "Single web search tool... No multi-query, no result reranking, no credibility scoring."
Tavily via `langchain-tavily` - already a project dependency with a key in `.env`, chosen over
`ddgs` (also a dependency, unused) because Tavily is built for exactly this: one call returns
ranked, LLM-ready snippets, no second client to write reranking or scoring logic against that
ROADMAP explicitly puts out of scope.
"""

from dataclasses import dataclass

from rag import config
from rag.retrieval import RetrievedChunk


@dataclass
class WebResult:
    """One web hit - title/url/snippet, not a RetrievedChunk. No arxiv_id/category/score-per-stage
    provenance to carry, because a web result was never in the index. `rag/generation.py` cites
    these separately (`[W1]`, `[W2]`) precisely so they're never confused with a corpus chunk."""

    title: str
    url: str
    snippet: str


def should_escalate(chunks: list[RetrievedChunk]) -> bool:
    """ROADMAP's one escalation rule: reranker top-score below threshold -> escalate.

    Checks `chunks[0].score` - only meaningful when `chunks` came back through a reranking pass
    (`rag/rerank.py`'s `Reranker.rerank()` overwrites `.score` with the cross-encoder score, "score
    always reflects whichever stage ran last"). Empty `chunks` escalates unconditionally (score
    treated as 0.0, below any sane threshold) - no corpus signal at all is the strongest case for
    going to the web, not a reason to skip the check.
    """
    top_score = chunks[0].score if chunks else 0.0
    return top_score < config.WEB_ESCALATION_THRESHOLD


def web_search(query: str, max_results: int = config.WEB_SEARCH_MAX_RESULTS) -> list[WebResult]:
    """One Tavily call, ranked results taken as-is - no multi-query, no reranking, no credibility
    scoring (ROADMAP §Phase 6's explicit cut list).

    Fails open, not fatal: same "failures logged, never fatal" posture as the planner/router/
    contextualize nodes (PHASE1_NOTES §5/§9). A missing key or a live API outage degrades to "no
    web results" - the corpus-only answer underneath is still valid - rather than crashing a
    request the corpus could otherwise have partially answered.
    """
    if not config.TAVILY_API_KEY:
        print("[websearch] TAVILY_API_KEY not set, continuing without web results")
        return []

    try:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=max_results, tavily_api_key=config.TAVILY_API_KEY)
        response = tool.invoke({"query": query})
        raw = response.get("results", []) if isinstance(response, dict) else response
    except Exception as exc:
        print(f"[websearch] search failed, continuing without web results: {exc}")
        return []

    return [
        WebResult(
            title=r.get("title", "") or "", url=r.get("url", "") or "",
            snippet=r.get("content", "") or "",
        )
        for r in raw
    ]
