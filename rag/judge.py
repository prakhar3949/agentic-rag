"""The eval judge client - DeepSeek-V4-Flash-0731 via Fireworks, raw OpenAI-compatible calls.

Deliberately not routed through `langchain_google_genai` or any LangChain chat model: the judge
must stay untraced (ROADMAP §Phase 8 - a traced judge roughly doubles a Tier 3 sweep's trace
count) and separate from the generator's client construction. A plain `openai.OpenAI` pointed at
Fireworks' endpoint is both simpler and satisfies "untraced" by construction - there is no
LangSmith instrumentation to accidentally inherit.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional

from rag import config


@dataclass
class JudgeResponse:
    """A single judge call, parsed or not - the caller decides what a `None` parsed value means.

    `parsed` is `None` after every retry fails to produce valid JSON. It is never coerced to a
    default score: a parse failure recorded as 0.0 would be indistinguishable from a genuinely
    bad answer and would silently drag a config's mean down (ROADMAP §Phase 4).
    """

    parsed: Optional[dict]
    raw_text: str
    retries: int
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0


class Judge:
    def __init__(
        self,
        model_id: str = config.JUDGE_MODEL_ID,
        temperature: float = config.JUDGE_TEMPERATURE,
        max_retries: int = config.JUDGE_MAX_RETRIES,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            # Imported here, not at module scope, matching the lazy-import pattern used
            # throughout rag/ (Retriever.embedder, Reranker.model) - a script that only needs
            # config values should not pay for the import.
            from openai import OpenAI

            self._client = OpenAI(api_key=config.JUDGE_API_KEY, base_url=config.JUDGE_BASE_URL)
        return self._client

    def call_json(self, system_prompt: str, user_prompt: str) -> JudgeResponse:
        """One judge call, JSON-mode, retried on unparseable output *and* on a transient
        server-side failure (Fireworks 5xx/429/connection drop - "service overloaded" has been
        observed in practice).

        A retry re-issues the identical request - at temperature=0 this mostly re-samples the
        same deterministic output, but Fireworks' JSON mode has occasionally been observed to
        drop a closing brace on a truncated generation, and retrying is cheaper than special-
        casing that here. Transient errors get a short backoff first (retrying instantly into an
        overloaded service just repeats the failure); parse retries don't need one.
        """
        from openai import APIConnectionError, InternalServerError, RateLimitError

        last_text = ""
        last_usage: dict = {}
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
            except (APIConnectionError, InternalServerError, RateLimitError):
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                continue
            last_text = response.choices[0].message.content or ""
            last_usage = _usage_dict(response.usage)
            try:
                parsed = json.loads(last_text)
            except json.JSONDecodeError:
                continue
            return JudgeResponse(
                parsed=parsed, raw_text=last_text, retries=attempt,
                usage=last_usage, latency_s=time.perf_counter() - t0,
            )
        # Exhausted retries, whether from bad JSON or a persistent transient failure - `parsed`
        # stays `None` either way, never coerced to a score (ROADMAP §Phase 4: null, not 0.0).
        return JudgeResponse(
            parsed=None, raw_text=last_text, retries=self.max_retries,
            usage=last_usage, latency_s=time.perf_counter() - t0,
        )


def _usage_dict(usage) -> dict:
    if usage is None:
        return {}
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
