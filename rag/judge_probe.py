"""Adversarial probe for the judge's context_recall/faithfulness rubric (PHASE4_NOTES.md).

`rag.judge_validate`'s real 47-question sample produced a judge_verdict of "supported" on 100% of
context_recall/faithfulness items - 0 negatives, ever. That's not evidence the judge is reliable;
it's an absence of the one case (can it say "no") that would tell us either way. Cohen's kappa
against an all-yes judge is structurally forced toward 0 regardless of how good the individual
calls were (see PHASE4_NOTES.md - it's a chance-correction artifact, not necessarily a real
finding), and the sample never tested negative-case recall at all.

This is a different kind of check than `rag.judge_validate`: ground truth here isn't a blind human
label, it's KNOWN BY CONSTRUCTION - each case was deliberately built to be either unsupported (10
cases, spanning different failure modes a real generator could plausibly produce) or supported (6
control cases, so an all-"no" run would be just as visible a failure as the original all-"yes"
run). Reports plain accuracy against a fixed answer key, not kappa.

Claims are written as either a single atomic fact, or (deliberately, for the *_hallucinated_addition
and *_perturbed_number cases) several facts in one sentence mixing a true part with a false one.
Either way they run through the SAME rubric/prompt path production uses (rag/metrics_llm.py), not
a new mechanism - context_recall/faithfulness's own claim-decomposition step may legitimately
split a compound sentence into several atomic claims, and `run_case()` scores that correctly
(every decomposed claim must be judged supported for the case to count as "supported overall" -
see its docstring), not as a rewrite-your-claims problem. Context passages are real corpus text
(data/phase1/chunks.jsonl), not synthetic strings.

**2026-08-20: extended from 10 to 26 cases, `ai`-only to `ai`/`cs.CL`/`finance`.** The original
10 only ever demonstrated negative-case recall on AI/ML content - the goldset and corpus grew to
span 3 categories, and nothing established the judge's "can it say no" capability generalizes
beyond the domain it was first tested on. First pass added 6 cases (2 negative + 1 control per
new category, from the Salience Bias and SEC 8-K event-extraction papers); second pass added 10
more, bringing `cs.CL`/`finance` to full parity with `ai`'s 8-mode failure taxonomy (`off_topic`,
`mismatched_context`, `perturbed_number`, `hallucinated_entity`, `unstated_fact`,
`negation_flip`, `control_verbatim`, `control_paraphrase`) using 4 further real papers per domain
(pump.fun graduation rates, TabNet fraud detection, LLM Theory-of-Mind/consciousness alignment,
FinBERT-vs-LLM financial-news sentiment) - deliberately different source papers each pass, so the
probe isn't just re-testing familiarity with 2-3 documents. See `ProbeCase.domain` for the
per-case arXiv category (distinct from `.category`, which is the failure-mode label -
unfortunate naming collision, kept because renaming the existing field would touch every case
above for no functional gain).

    python -m rag.judge_probe          # run all 26 cases (26 judge calls, no generation, cheap)
    python -m rag.judge_probe --show   # also print each case's full judge reasoning
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rag import config
from rag.judge import Judge
from rag.metrics_llm import context_recall, faithfulness

_AGNN_EER_TABLE = (
    "|  | C (SR ) peak peak |  | C (SR ) pred pred |  | C /C pred train |  |\n"
    "| --- | --- | --- | --- | --- | --- | --- |\n"
    "|  | EER | EW | EER | EW | EER | EW |\n"
    "| DGNN AGNN H yb rid H yb rid LCA | 4 00 (1 00) . . 6 48 (0 78) . . 6 00 (1 00) . . "
    "6 00 (1 00) . . | 4 00 (1 00) . . 10 79 (0 33) . . 8 00 (1 00) . . 8 00 (1 00) . . | "
    "4 00 (1 00) . . 48 02 (0 00) . . 8 27 (1 00) . . 11 25 (1 00) . . | 5 90 (1 00) . . "
    "18 61 (0 10) . . 8 92 (1 00) . . 10 93 (1 00) . . | 0 70 . 8 39 . 1 44 . 1 97 . | 1 03 . 3 25"
)
_VOTING_RULES_REFS = (
    "2021. [GJS+25]\nSushmita Gupta, Pallavi Jain, Souvik Saha, Saket Saurabh, and Anannya Upasana.\n"
    "More efforts towards fixed-parameter approximability of multiwinner rules. In Proceedings of "
    "the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025), pages "
    "3891-3899, 2025.\n[JST20]\nPallavi Jain, Krzysztof Sornat, and Nimrod Talmon. Participatory "
    "budgeting with project interactions. In Proceedings of the 29th International Joint "
    "Conference on Artificial Intelligence (IJCAI 2020), pages 386-392, 2020."
)
_SYSTEM_PROMPT_AUDIT_CONCLUSION = (
    "of third-party auditing for AI system prompts and could inspire new research and industry "
    "standards on building trustworthy and transparent AI systems."
)
_TASK_TYPES_PASSAGE = (
    "web split undergoes the human vetting and peer cross-checking described in S3.2. The "
    "remaining machine-generated instructions feed exclusively into the OS-Shepherd training "
    "corpus. Furthermore, because the platform installs no applications, its coverage is defined "
    "by the diversity of its tasks rather than an application roster (see Table 7 for the other "
    "three platforms). Figure 12 profiles the roughly 32K filtered instructions submitted for "
    "collection: the twenty most frequent task types, about 65% of the pool, span article "
    "reading, fact lookups, academic-paper, shopping, and library tasks."
)

# --- 2026-08-20 additions: cs.CL and finance domains, added after the 3-category corpus/goldset
# expansion - the original 10 cases above are 100% `ai`-sourced, so "the judge can say no" was
# only ever demonstrated on AI/ML content. Real corpus text from 2607.08346v1 (SEC 8-K event
# extraction) and 2607.21826v1 (crypto bubble detection), both `finance`; 2607.28478v1 (Salience
# Bias), `cs.CL`.

_SEC_8K_ABSTRACT = (
    "score. Applying the system to 292,984 filings from 2022 to 2026\n"
    "yields 601,088 grounded event tags, which we release. Over 5,125\n"
    "stratified tags, an LLM judge finds precision rises monotonically\n"
    "with the quality score, from 12% to 96%, while unsupported tags\n"
    "fall from 8% to near zero. Ablation shows the score is calibrated\n"
    "only when assigned in a dedicated second pass. An event study on\n"
    "unsigned abnormal returns confirms, without any language model,\n"
    "that the taxonomy separates economically distinct events sharing\n"
    "an item code."
)
_CRYPTO_BUBBLE_METHODOLOGY = (
    "Abstract\nThe growth of peer-to-peer exchanges and the blockchain technology has led to a "
    "proliferation of cryptocurrencies and to a massive increase in the number of investors who "
    "actually negotiate digital money. Cryptocurrencies trade at prices mainly driven by investor "
    "sentiment, becoming a potential source of financial bubbles and instabilities.\n"
    "In this work, we apply quantitative models to the study of Bitcoin and Ether, two of the "
    "most famous cryptocurrencies. Our bubble detection methodology combines the Log Periodic "
    "Power Law (LPPL) model, originally created by Johansen, Ledoit and Sornette (JLS), and the "
    "statistical model developed by Phillips, Shi, and Yu (PSY)."
)
_SALIENCE_BIAS_PASSAGE = (
    "Abstract\nAs large language models (LLMs) continue to advance in complex reasoning tasks, "
    "they have learned to heavily prioritize explicit conditions provided in the input. However, "
    "in everyday commonsense reasoning, this mechanism exposes a critical vulnerability which we "
    "term Salience Bias: models become easily hijacked by useless explicit distractors (e.g., "
    "numerical values), leading them to ignore the implicit physical or commonsense prerequisites "
    "of a task. ... Evaluating 12 state-of-the-art LLMs, we find that all mainstream models "
    "suffer significantly from salience bias, with severity scaling with distractor density and "
    "detecting the trap often decoupled from actually avoiding it.\n\n"
    "(iii) We conduct a comprehensive evaluation of 12 LLMs, revealing that salience bias is "
    "pervasive, correlated with capability, and structured along distractor density and model "
    "provenance."
)

# --- 2026-08-20, second batch: 10 more cases bringing cs.CL/finance to parity with ai's 8
# failure-mode taxonomy (each domain only had 3/8 modes covered after the first 6-case batch).
# Different source papers from the first batch, on purpose - reusing the same 2-3 papers for
# every case would make the probe trivially guessable rather than a real generalization test.

_PUMPFUN_CONCLUSION = (
    "Conclusion\nWe have presented the first Kaplan-Meier and Cox proportional-hazards survival "
    "analysis of pump.fun token graduation, on a 832,941-launch sample with a 0.198% pooled "
    "graduation rate (Wilson 95% CI [0.189%, 0.208%]) - a 3.18x decline from the September-"
    "October 2025 Marino et al. (2026) baseline. Social-presence indicators are highly "
    "predictive: a Telegram-channel indicator alone produces an 8.94x lift, and a full "
    "social-channel stack 17.4x. The top market-cap quartile in our sample graduates at 0.634%, "
    "almost exactly matching the Marino et al. (2026) pooled rate, suggesting that the "
    "cross-regime decline is substantially attributable to a compositional shift toward "
    "zero-self-buy launches."
)
_TABNET_RESULTS = (
    "6.3 Important results from experiment\n"
    "a. TabNet: An Outstanding Example\n"
    "The most effective model that was examined was TabNet, which achieved\n"
    "- A ROC-AUC value of 0.9739 - 97.39 percent accuracy in classification\n"
    "- Minimal instances of both false positives and negatives\n"
    "- The model met the needs for transparency and regulatory alignment in high-risk domains "
    "like fraud detection with its built-in interpretability (via feature masks) and sparse "
    "attention mechanism, which allowed it to dynamically prioritize key features during "
    "training.\n"
    "b. Conventional DNN, LSTM, GRU, and CNN1D Are Not Sufficient for Tabular Fraud Data\n"
    "- Every other model showed deficient performance:\n"
    "  o LSTM and GRU: A little better than randomness, but high misclassification.\n"
    "  o CNN1D: Designed for spatial/temporal data; inappropriate for tabular input.\n"
    "  o DNN: collapsed predictions; ROC-AUC 0.5"
)
_TOM_CONSCIOUSNESS_PASSAGE = (
    "Large Language Models (LLMs) increasingly occupy social roles such as coaches, tutors, and "
    "romantic partners. A central alignment objective in this context is preventing models from "
    "attributing consciousness, emotions and other aspects of mindedness to themselves. ... "
    "The IDAQ direction shows a significant negative shift, whereas the subject-matched control "
    "shows no significant shift. This confirms that the safety-IDAQ entanglement is driven by "
    "mental-state attribution specifically, not by the subjects (e.g., robots, animals) "
    "themselves."
)
_WORLD_ENGLISHES_PASSAGE = (
    "Englishes are reproduced and contested. Addressing these inequities requires that "
    "linguists, technologists, and institutions recognize language ideologies as a matter that "
    "must be considered in the design of AI systems. The challenge ahead is to ensure that "
    "future AI systems do not simply reproduce the linguistic hierarchies of the past but "
    "contribute to a more inclusive and pluralistic understanding of Englishes in the world."
)
_FINANCIAL_NEWS_NLP_ABSTRACT = (
    "This observation motivates our central hypothesis: financial news contains multiple "
    "information dimensions that are partially orthogonal to surface sentiment, and these "
    "dimensions carry independent predictive value for stock price movements. ... Across 41,618 "
    "samples, FinBERT and LLaMA disagree on sentiment polarity in 53.5% of cases, with "
    "disagreement rates ranging from 39.4% (merger events) to 67.4% (uncategorized events)... "
    "All six dimensions contribute meaningfully to prediction (importance range: 14-21%), with "
    "no single dominant feature, confirming that the framework captures distinct information "
    "channels."
)


@dataclass
class ProbeCase:
    id: str
    metric: str       # "context_recall" | "faithfulness"
    category: str     # failure-mode label (off_topic, perturbed_number, ...), NOT arXiv category
    domain: str        # arXiv category the source passage is drawn from (ai | cs.CL | finance)
    question: str
    claim: str
    context: str
    expected_supported: bool


PROBE_CASES: list[ProbeCase] = [
    # --- negatives: should be scored "not supported" ---------------------------------------
    ProbeCase(
        "off_topic", "context_recall", "off_topic", "ai",
        "What does the crystal entropy proxy enforce?",
        "The crystal entropy proxy enforces thermodynamic stability.",
        _VOTING_RULES_REFS,  # entirely unrelated topic (election-rule references)
        expected_supported=False,
    ),
    ProbeCase(
        "mismatched_context", "context_recall", "mismatched_context", "ai",
        "What is the EER for the AGNN method?",
        "The EER for the AGNN method is 8.39.",
        _SYSTEM_PROMPT_AUDIT_CONCLUSION,  # true claim, but paired with the wrong passage
        expected_supported=False,
    ),
    ProbeCase(
        "perturbed_number", "faithfulness", "perturbed_number", "ai",
        "What is the EER for the AGNN method?",
        "The EER for the AGNN method is 99.99.",
        _AGNN_EER_TABLE,  # right passage, but the real value (8.39) doesn't match the claim
        expected_supported=False,
    ),
    ProbeCase(
        "hallucinated_addition", "context_recall", "hallucinated_entity", "ai",
        "What proportion of the pool do the frequent task types span, and what categories?",
        "The twenty most frequent task types span approximately 65% of the pool, covering "
        "shopping, banking, and travel booking.",
        _TASK_TYPES_PASSAGE,  # 65%/shopping is real; "banking"/"travel booking" are not stated
        expected_supported=False,
    ),
    ProbeCase(
        "unstated_fact", "context_recall", "unstated_fact", "ai",
        "What does the system prompt auditing framework do?",
        "The AISPA framework was evaluated on over 10,000 real-world system prompts.",
        _SYSTEM_PROMPT_AUDIT_CONCLUSION,  # plausible-sounding number, never stated
        expected_supported=False,
    ),
    ProbeCase(
        "negation_flip", "faithfulness", "negation_flip", "ai",
        "What proportion of the pool do the frequent task types span?",
        "The twenty most frequent task types span less than 10% of the pool.",
        _TASK_TYPES_PASSAGE,  # direct numeric contradiction of the stated ~65%
        expected_supported=False,
    ),
    ProbeCase(
        "finance_perturbed_number", "faithfulness", "perturbed_number", "finance",
        "How many grounded event tags did the system produce?",
        "The system was applied to 292,984 filings and produced 750,000 grounded event tags.",
        _SEC_8K_ABSTRACT,  # right filing count, wrong tag count (real: 601,088)
        expected_supported=False,
    ),
    ProbeCase(
        "finance_mismatched_context", "context_recall", "mismatched_context", "finance",
        "How does judge-assessed precision change with the quality score?",
        "Judge-assessed precision rises monotonically from 12% at the lowest quality score to "
        "96% at the top score.",
        _CRYPTO_BUBBLE_METHODOLOGY,  # true claim, paired with an unrelated finance passage
        expected_supported=False,
    ),
    ProbeCase(
        "cs_cl_hallucinated_addition", "context_recall", "hallucinated_entity", "cs.CL",
        "What did the evaluation find about salience bias across the tested LLMs?",
        "Salience bias is pervasive across LLMs, correlated with capability, and occurs only in "
        "mathematical reasoning tasks.",
        _SALIENCE_BIAS_PASSAGE,  # pervasive/correlated-with-capability real; "math-only" is not
        # - the passage explicitly frames this as a commonsense-reasoning phenomenon, contrasted
        # with math/coding tasks where "the provided conditions are always useful and necessary"
        expected_supported=False,
    ),
    ProbeCase(
        "cs_cl_negation_flip", "faithfulness", "negation_flip", "cs.CL",
        "How common is salience bias among large language models?",
        "Salience bias is rare among large language models and shows no relationship to "
        "distractor density.",
        _SALIENCE_BIAS_PASSAGE,  # direct contradiction of "pervasive" / "severity scaling with
        # distractor density"
        expected_supported=False,
    ),
    ProbeCase(
        "finance_off_topic", "context_recall", "off_topic", "finance",
        "What did the stablecoin regulation study find about cross-exchange trading?",
        "Gateway regulation redirects stablecoin trading across exchanges while leaving "
        "aggregates essentially unchanged.",
        _PUMPFUN_CONCLUSION,  # true claim (from 2607.09514v1), paired with an unrelated
        # finance passage (pump.fun graduation rates) - no MiCA/USDT/USDC content at all
        expected_supported=False,
    ),
    ProbeCase(
        "finance_hallucinated_addition", "faithfulness", "hallucinated_entity", "finance",
        "What did the pump.fun graduation study find, across which chains?",
        "Applied to a 832,941-launch sample, the pooled graduation rate was 0.198%, and the "
        "study also found that Ethereum-based tokens graduated at twice the rate of "
        "Solana-based tokens.",
        _PUMPFUN_CONCLUSION,  # the 832,941/0.198% figures are real; the Ethereum comparison is
        # fabricated - the paper is entirely about Solana/pump.fun, no other chain studied
        expected_supported=False,
    ),
    ProbeCase(
        "finance_unstated_fact", "context_recall", "unstated_fact", "finance",
        "Where was the TabNet fraud detection model deployed?",
        "The TabNet fraud detection model was deployed in production at three major Indian "
        "banks.",
        _TABNET_RESULTS,  # plausible-sounding deployment claim; this is a research study on a
        # Kaggle dataset, never stated as a production deployment
        expected_supported=False,
    ),
    ProbeCase(
        "finance_negation_flip", "faithfulness", "negation_flip", "finance",
        "How did TabNet's fraud-detection accuracy compare to the other models tested?",
        "TabNet achieved poor accuracy, with a ROC-AUC well below 0.5, performing worse than "
        "conventional DNN models.",
        _TABNET_RESULTS,  # direct contradiction: TabNet scored 0.9739, DNN collapsed to 0.5
        expected_supported=False,
    ),
    ProbeCase(
        "cs_cl_off_topic", "context_recall", "off_topic", "cs.CL",
        "What does the IDAQ shift confirm about safety-IDAQ entanglement?",
        "The IDAQ direction shows a significant negative shift, confirming that safety-IDAQ "
        "entanglement is driven by mental-state attribution.",
        _WORLD_ENGLISHES_PASSAGE,  # true claim (from 2607.28607v1), paired with an unrelated
        # cs.CL passage (AI and language ideologies in World Englishes)
        expected_supported=False,
    ),
    ProbeCase(
        "cs_cl_mismatched_context", "context_recall", "mismatched_context", "cs.CL",
        "What importance range did the six extracted dimensions show for stock-movement "
        "prediction?",
        "All six extracted dimensions contribute meaningfully to stock-movement prediction, "
        "with an importance range of 14 to 21 percent.",
        _TOM_CONSCIOUSNESS_PASSAGE,  # true claim (from 2607.28496v1), paired with an unrelated
        # cs.CL passage (LLM consciousness/Theory-of-Mind alignment)
        expected_supported=False,
    ),
    ProbeCase(
        "cs_cl_perturbed_number", "faithfulness", "perturbed_number", "cs.CL",
        "How often do FinBERT and LLaMA disagree on sentiment, and how important are the "
        "extracted dimensions?",
        "FinBERT and LLaMA disagree on sentiment polarity in 53.5% of cases, and all six "
        "extracted dimensions show an importance range of 40 to 60 percent.",
        _FINANCIAL_NEWS_NLP_ABSTRACT,  # 53.5% is real; the real importance range is 14-21%,
        # not 40-60%
        expected_supported=False,
    ),
    ProbeCase(
        "cs_cl_unstated_fact", "context_recall", "unstated_fact", "cs.CL",
        "Which specific commercial models were used in the consciousness-steering experiments?",
        "The consciousness-steering experiments were conducted on GPT-4 and Claude 3.5, the two "
        "most widely deployed commercial models.",
        _TOM_CONSCIOUSNESS_PASSAGE,  # the passage discusses conditions (instruction-tuned,
        # safety-ablated, consciousness-steered), never these specific commercial model names
        expected_supported=False,
    ),
    # --- controls: should be scored "supported" ---------------------------------------------
    ProbeCase(
        "control_verbatim_number", "context_recall", "control_verbatim", "ai",
        "What is the EER for the AGNN method?",
        "The EER for the AGNN method is 8.39.",
        _AGNN_EER_TABLE,  # same claim as perturbed_number, but the correct passage this time
        expected_supported=True,
    ),
    ProbeCase(
        "control_paraphrase", "context_recall", "control_paraphrase", "ai",
        "What might third-party auditing of AI system prompts inspire?",
        "Auditing AI system prompts from outside the organization could lead to new norms and "
        "standards for making AI systems more trustworthy and transparent.",
        _SYSTEM_PROMPT_AUDIT_CONCLUSION,  # same fact, reworded rather than verbatim
        expected_supported=True,
    ),
    ProbeCase(
        "control_verbatim_pct", "faithfulness", "control_verbatim", "ai",
        "What proportion of the pool do the frequent task types span?",
        "The twenty most frequent task types span about 65% of the pool.",
        _TASK_TYPES_PASSAGE,
        expected_supported=True,
    ),
    ProbeCase(
        "control_reference_list", "context_recall", "control_verbatim", "ai",
        "Which authors wrote the IJCAI 2020 paper on pages 386-392 cited in the references?",
        "Pallavi Jain, Krzysztof Sornat, and Nimrod Talmon wrote a paper cited in the references.",
        _VOTING_RULES_REFS,
        expected_supported=True,
    ),
    ProbeCase(
        "finance_control_verbatim", "faithfulness", "control_verbatim", "finance",
        "How many filings were processed and how many event tags did that produce?",
        "Applying the system to 292,984 filings produced 601,088 grounded event tags.",
        _SEC_8K_ABSTRACT,
        expected_supported=True,
    ),
    ProbeCase(
        "cs_cl_control_paraphrase", "context_recall", "control_paraphrase", "cs.CL",
        "How many LLMs were evaluated for salience bias, and how widespread was the effect?",
        "The study evaluated 12 large language models and found that salience bias affects "
        "nearly all of them, with its severity varying depending on how many distracting "
        "details were present.",
        _SALIENCE_BIAS_PASSAGE,  # safe paraphrase - avoids asserting a capability *direction*
        # the passage doesn't actually state ("correlated with capability" has no stated sign)
        expected_supported=True,
    ),
    ProbeCase(
        "finance_control_paraphrase", "faithfulness", "control_paraphrase", "finance",
        "How did TabNet's fraud-detection performance compare to the other models tested?",
        "TabNet outperformed the other neural network architectures tested, reaching over 97% "
        "classification accuracy while keeping both false positives and false negatives low.",
        _TABNET_RESULTS,  # paraphrase of "97.39 percent accuracy... minimal instances of both
        # false positives and negatives" plus the explicit contrast with the other models
        expected_supported=True,
    ),
    ProbeCase(
        "cs_cl_control_verbatim", "context_recall", "control_verbatim", "cs.CL",
        "What is the central hypothesis about financial news and stock price movements?",
        "Financial news contains multiple information dimensions that are partially orthogonal "
        "to surface sentiment, and these dimensions carry independent predictive value for "
        "stock price movements.",
        _FINANCIAL_NEWS_NLP_ABSTRACT,  # verbatim sentence from the source
        expected_supported=True,
    ),
]


@dataclass
class ProbeResult:
    case: ProbeCase
    actual_supported: bool | None   # None if unscoreable (bad decomposition or judge failure)
    n_claims: int
    judge_reason: str
    correct: bool | None


def run_case(judge: Judge, case: ProbeCase) -> ProbeResult:
    """A compound claim (several facts in one sentence, e.g. the *_hallucinated_addition and
    *_perturbed_number cases deliberately mixing a true part with a false one) legitimately
    decomposes into more than one atomic claim - that's the judge doing its job, not a scoring
    problem. `actual_supported` generalizes to "were ALL decomposed claims judged supported" -
    for N=1 this is identical to the single-claim check, so nothing changes for the simple cases;
    for N>1 it correctly scores "correct" only when the judge accepted every true part and
    rejected every false part, exactly matching what `expected_supported` means for a mixed
    claim. Only a genuine judge/parse failure (zero claims) is left unscoreable."""
    if case.metric == "context_recall":
        result = context_recall(judge, case.question, case.claim, case.context)
    else:
        result = faithfulness(judge, case.claim, case.context)

    if not result.claims:
        return ProbeResult(case, None, 0, "(judge call failed or decomposed zero claims)", None)

    sub_verdicts = [bool(c.get("supported")) for c in result.claims]
    actual = all(sub_verdicts)
    reasons = "; ".join(c.get("reason", "") for c in result.claims)
    if len(result.claims) > 1:
        reasons = f"[{len(result.claims)} sub-claims, verdicts={sub_verdicts}] {reasons}"
    return ProbeResult(case, actual, len(result.claims), reasons, actual == case.expected_supported)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.judge_probe")
    parser.add_argument("--show", action="store_true", help="print full judge reasoning per case")
    parser.add_argument("--output", default=str(config.JUDGE_PROBE_REPORT_PATH))
    args = parser.parse_args(argv)

    judge = Judge()
    results = [run_case(judge, case) for case in PROBE_CASES]

    scored = [r for r in results if r.correct is not None]
    negatives = [r for r in scored if not r.case.expected_supported]
    positives = [r for r in scored if r.case.expected_supported]
    correct = sum(r.correct for r in scored)
    neg_correct = sum(r.correct for r in negatives)
    pos_correct = sum(r.correct for r in positives)

    print(f"{correct}/{len(scored)} correct overall "
          f"({len(results) - len(scored)} unscoreable - decomposition mismatch or judge failure)")
    print(f"  negatives (judge should say NOT supported): {neg_correct}/{len(negatives)} correct")
    print(f"  positives (judge should say supported):     {pos_correct}/{len(positives)} correct")
    by_domain: dict[str, dict] = {}
    for domain in sorted({r.case.domain for r in scored}):
        sub = [r for r in scored if r.case.domain == domain]
        n_ok = sum(1 for r in sub if r.correct)
        by_domain[domain] = {"n": len(sub), "correct": n_ok,
                              "accuracy": n_ok / len(sub) if sub else None}
        print(f"  domain={domain:10} {n_ok}/{len(sub)} correct")
    print()
    for r in results:
        status = "?????" if r.correct is None else ("OK   " if r.correct else "WRONG")
        print(f"[{status}] {r.case.id:26} ({r.case.category:20}) domain={r.case.domain:8} "
              f"expected={r.case.expected_supported!s:5} actual={r.actual_supported!s:5}")
        if args.show or r.correct is False or r.correct is None:
            print(f"          reason: {r.judge_reason[:300]}")

    report = {
        "n_cases": len(PROBE_CASES),
        "n_scored": len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "negative_recall": neg_correct / len(negatives) if negatives else None,
        "positive_recall": pos_correct / len(positives) if positives else None,
        "by_domain": by_domain,
        "cases": [
            {
                "id": r.case.id, "metric": r.case.metric, "category": r.case.category,
                "domain": r.case.domain, "expected_supported": r.case.expected_supported,
                "actual_supported": r.actual_supported, "n_claims_decomposed": r.n_claims,
                "correct": r.correct, "judge_reason": r.judge_reason,
            }
            for r in results
        ],
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
