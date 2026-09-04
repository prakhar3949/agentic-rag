# Phase 4 — Eval harness: defense notes

Conceptual notes for the golden set and judge-validation tooling, written for defense prep. See
ROADMAP.md §4/§5 for the ablation table and pre-written defense questions these support directly,
in particular *"How do you know your LLM judge is trustworthy?"*

---

## 1. Two separate validation problems, easy to conflate

Everything in Phase 4 rests on two independent artifacts, and each one needs its own check:

| Artifact | What could be wrong with it | Checked by |
|---|---|---|
| The golden set (`goldset.jsonl`) | A question isn't really answerable from its source chunk; a reference answer contains something the passage doesn't state | `rag.goldset_review` (hand-review, done: 60 reviewed, 44 accept / 3 edit / 13 reject → `goldset_curated.jsonl`, 47 rows) |
| The judge (`rag/judge.py`, DeepSeek-V4-Flash) | It scores a system's retrieval/generation incorrectly against a *correct* reference — over-lenient, over-strict, or systematically wrong about what "supported" means | `rag.judge_validate` (this doc) |

A clean golden set scored by an unreliable judge still produces a garbage ablation table. Hand-
reviewing the goldset (which is done) says nothing about whether the judge itself can be trusted
— that requires an independent check against human judgment on the judge's own output, which is
what `rag.judge_validate` does.

---

## 2. Why Cohen's kappa specifically, not raw agreement %

Raw "% of items where judge and human agree" is inflated whenever one label is common. If 90% of
claims in the corpus are genuinely faithful, a judge that lazily answers `supported=True` on
*everything* still scores 90% raw agreement with a careful human — despite carrying zero signal.

Cohen's kappa corrects for exactly this by subtracting out the agreement you'd expect from chance
alone, given each rater's own marginal rate of saying "yes":

```
po = observed agreement                     (fraction of items where judge == human)
pe = P(both say yes by chance) + P(both say no by chance)
   = judge_yes_rate * human_yes_rate + judge_no_rate * human_no_rate
kappa = (po - pe) / (1 - pe)
```

`kappa = 0` means no better than chance; `kappa = 1` means perfect agreement. Rough interpretation
bands (Landis & Koch 1977): `<0.2` slight, `0.21–0.4` fair, `0.41–0.6` moderate, `0.61–0.8`
substantial, `>0.8` almost perfect.

**Worked example, hand-checked against `cohens_kappa()` in `rag/judge_validate.py`:**

20 items. Judge says "yes" on 12, human says "yes" on 10, and they agree on 18/20.

```
po = 18/20 = 0.90
pe = (12/20)(10/20) + (8/20)(10/20) = 0.30 + 0.20 = 0.50
kappa = (0.90 - 0.50) / (1 - 0.50) = 0.80        # substantial-to-almost-perfect agreement
```

`cohens_kappa([True]*12+[False]*8, human_labels)` returns exactly `0.8` for this input — confirmed
by running it, same "own the math, verify it by hand" approach as `rag/metrics_retrieval.py`'s
hit@k/MRR/NDCG (no `scipy`/`sklearn` dependency for one formula). Edge cases also checked: empty
input returns `None` (not a crash or a misleading 0), and `pe >= 1.0` (both raters constant and
equal) returns `1.0` rather than dividing by zero.

---

## 3. What's actually being validated — and what's deliberately out of scope

`rag/metrics_llm.py`'s four metrics don't all produce the same *shape* of judgment:

- **context_precision** — one `relevant: bool` verdict per retrieved chunk.
- **context_recall** — one `supported: bool` verdict per atomic claim decomposed from the
  *reference* answer.
- **faithfulness** — one `supported: bool` verdict per atomic claim decomposed from the
  *generated* answer.
- **response_relevancy** — a single continuous `0.0–1.0` score for the whole answer, nothing
  decomposed.

Kappa needs two raters producing the *same discrete categories* on the *same items*. The first
three already do exactly that — the judge call already emits a binary per-claim/per-chunk verdict
as its intermediate output (`LLMMetricResult.claims`, per `metrics_llm.py`'s own docstring: "they
cost nothing extra, the judge call already produces them"). `response_relevancy` doesn't decompose
into anything item-level to independently re-judge; forcing it into a binary bucket (e.g.
threshold at 0.5) would validate a threshold I invented, not the judge's actual scoring behavior.
**Decision: validate context_precision, context_recall, and faithfulness via kappa; leave
response_relevancy unvalidated by this method** — document this scope choice rather than fake a
number for it.

---

## 4. Why the human labels the pool *blind* to the judge's verdict

`rag.judge_validate label` never shows the judge's `judge_verdict`/`judge_reason` while asking for
a human call — only the claim/chunk text and its evidence (retrieved context, or the reference
answer for context_precision). If the judge's answer were visible first, a tired reviewer's
fastest path is to rubber-stamp it, which would inflate agreement without the judge actually being
right. The judge's verdict is only joined back in at `report` time, after all labels are recorded.

This is the same principle as the goldset review's structure — human judgment is only meaningful
if it's actually independent, not a confirmation click.

---

## 5. Applying the rubric while labeling — two different judgments, not three

The three metrics split into two categories of question, not three independent ones. Knowing
which one you're answering matters more than it sounds — mixing them up adds pure labeler noise
that has nothing to do with the judge's actual reliability.

**`context_recall` and `faithfulness` — literally the same rubric.** Both ask the identical
question, with the identical standard (`metrics_llm.py` lines 81-83 / 110-114 — text is near-
duplicated on purpose, same rubric): *the evidence must state the claim or directly imply it —
not "seems plausible," not "I know this is true from reading elsewhere in the paper."* Pure
textual entailment, evidence-only, no outside knowledge. The only difference between the two
metrics is *where the claim came from* (`context_recall`: decomposed from the goldset's reference
answer; `faithfulness`: decomposed from whatever the pipeline actually generated) — that changes
what the metric *measures* downstream, not how an individual item gets labeled. Same test, same
strictness, every time.

**`context_precision` is a genuinely different kind of judgment.** The question is "does this
chunk contribute information used in the reference answer?" — a *relevance/contribution* call,
not literal entailment. Three things flip relative to the other two:

| | context_recall / faithfulness | context_precision |
|---|---|---|
| unit judged | one decomposed atomic claim | one whole retrieved chunk |
| evidence shown | retrieved context | reference answer |
| standard | strict textual entailment ("does the evidence say this") | looser relevance/utility ("was this passage useful for this answer") |

A chunk can be relevant/contributory without containing an exact quotable sentence; conversely a
chunk that's topically adjacent but contains none of the answer's actual facts should be marked
not relevant. Don't apply the strict "must be stated verbatim" bar here — that's the other two
metrics' rubric, not this one's.

**Worked example — the same strict standard, in practice:** a `context_recall` item showed the
claim *"The pathfinders include Hybrid Pathfinder"* against a garbled OCR'd heatmap table
containing the literal string `"(c) Hybrid Pathfinder"` as a sub-panel label. Textually present →
counts as stated by the rubric, even though the source is a mangled table dump, not prose. The
test is "does the text contain this," not "is this a clean, readable sentence saying it."

---

## 6. Rubric fix: before/after on context_precision

§5 diagnosed why `context_precision` was the weak metric: the judge sometimes called a chunk
"relevant" purely on shared vocabulary (e.g. *"mentions spectral consistency and thermodynamic
entropy minimization"*) rather than genuine contribution. `_CONTEXT_PRECISION_SYSTEM`
(`rag/metrics_llm.py`) was tightened with an explicit anti-keyword-overlap instruction and an
operational test — *"could this specific passage alone have supplied this part of the answer"* —
then only `context_precision` was re-validated
(`rag.judge_validate generate`/`label --metric context_precision --redo`), leaving
`context_recall`/`faithfulness` below untouched since they weren't part of the fix.

| | before | after |
|---|---|---|
| n | 16 | 15 |
| raw agreement | 56.3% | 60.0% |
| Cohen's κ | 0.125 (slight) | **0.308 (fair)** |
| judge_true_human_false (false positives) | 2 | **0** |
| judge_false_human_true (false negatives) | 5 | 6 |

The fix worked on exactly what it targeted: false positives (the judge over-crediting keyword
overlap) dropped to **zero**. But the judge's error profile didn't just shrink — it inverted. It's
now producing false negatives instead: marking chunks "not relevant" that the human considered
relevant (`judge_yes_rate` fell from 5/16 to 4/15, while `human_yes_rate` stayed roughly flat at
~10/15–16). κ improved 2.5×, but "fair" (0.21–0.4 band) is still short of "moderate" —tightening
the rubric traded one failure mode for a smaller, different one, not for a solved metric. Recorded
honestly rather than stopping at the first improvement.

(n dropped from 16 to 15 in "after" for a mechanical reason, not new data loss: `--redo` replaced
the old `context_precision` pool rows, which orphans any human label still pointing at a dropped
`item_id` — `rag.judge_validate report` correctly excludes those rather than silently mismatching
a label to the wrong item. See `rag/judge_validate.py`'s stale-label warning.)

---

## 7. Adversarial probe: does the judge have negative-case recall?

`rag.judge_validate report`'s real numbers, from the first 47-question sample:

| | n | raw agreement | Cohen's κ |
|---|---|---|---|
| overall | 47 | 70.2% | 0.262 (fair) |
| context_precision | 16 | 56.3% | 0.125 (slight) |
| context_recall | 16 | 75.0% | 0.0 |
| faithfulness | 15 | 80.0% | 0.0 |

The two κ=0.0 rows have a specific, checkable cause — not a vague "the judge is unreliable." Their
confusion matrices both showed `judge_false_human_true=0` and `judge_false_human_false=0`: the
judge said "supported" on **100% of the 31 items**, no exceptions. When one rater has zero
variance, `pe` converges toward `po` by construction (§2's formula: `judge_yes_rate=1.0` makes
`pe = 1.0 × human_yes_rate`, which tracks `po` almost exactly) and kappa is forced toward 0
regardless of whether the individual calls were actually good. **That result cannot distinguish
"the judge is a lazy yes-man" from "this sample never contained a genuinely false claim."**

`rag/judge_probe.py` resolves the ambiguity directly instead of guessing: 10 hand-built
(claim, context) pairs with ground truth known **by construction**, not blind human labels — 6
built to be genuinely unsupported (off-topic context, right topic/wrong number, a true claim with
a fabricated detail stitched on, a specific fact never stated, a flat numeric contradiction), 4
built as supported controls (including one using the *exact same claim text* as a negative case,
just paired with the correct passage — a clean minimal pair). Real corpus passages throughout,
run through the production `context_recall`/`faithfulness` functions directly — same rubric, same
prompt path, not a separate mechanism.

**Result: 10/10 correct**, once scored at the right granularity. The script's own scoring (rigid
"exactly 1 decomposed claim") flagged 3 as unscoreable because the judge split them into multiple
atomic sub-claims instead of one; re-running those 3 and inspecting the sub-claims directly showed
every one was correct:

- `unstated_fact` → single claim, correctly `False`.
- `control_paraphrase` → 2 sub-claims, both correctly `True` (confirms the rubric's "or directly
  implies it" clause actually works — a paraphrase, not a verbatim match, was still credited).
- `hallucinated_addition` (a claim mixing a real fact with two fabricated ones — "...65% of the
  pool, covering shopping, **banking, and travel booking**," where banking/travel booking are not
  in the source) → 4 sub-claims: `65%`=`True`, `shopping`=`True`, `banking`=`False`,
  `travel booking`=`False`. The judge didn't blanket-accept or blanket-reject a mixed claim — it
  isolated exactly the fabricated pieces. This is the realistic shape a generator hallucination
  actually takes, and it's a stronger result than a simple binary pass/fail would have shown.

**Conclusion: the κ=0.0 rows are an absence-of-evidence artifact, not a red flag.** The judge does
have negative-case recall — it was simply never asked a question where "no" was the right answer
in the original sample. `context_precision` was the real, separately-diagnosed weak spot — §6
covers its rubric fix and the before/after numbers (0.125 → 0.308), a different failure mode
(relevance judgment, not entailment) and untouched by this probe.

One reproducibility caveat worth recording: re-running the same 3 cases a second time
(`temperature=0`) produced different claim-decomposition granularity than the first pass (verdicts
matched both times; claim *counts* didn't). Worth knowing before assuming byte-identical
repeatability from a pinned temperature alone — a known characteristic of MoE-routed models, not a
bug in this code.

---

## 8. Mechanics — three subcommands, only one costs money

```
python -m rag.judge_validate generate   # retrieval + generation + 3 judge calls per question
                                         # over goldset_curated.jsonl -> judge_pool.jsonl (PAID)
python -m rag.judge_validate label      # blind hand-labeling, resumable -> judge_validation.jsonl
python -m rag.judge_validate report     # join pool + labels, compute kappa -> judge_validation_report.json
python -m rag.judge_probe               # 10-case known-ground-truth adversarial check (§7),
                                         # 10 judge calls, no generation -> judge_probe_report.json
```

`generate` picks the `dense+bm25+rerank` config (the strongest arm) rather than looping over all
three ablation configs — the thing under test here is the *judge's* reliability, which doesn't
change with the retriever config; using one representative, realistic pipeline output is enough.

Target sample size for `label` is 50 (ROADMAP's stated bar), stratified round-robin across the
three metrics so context_precision (one item per retrieved chunk, usually the largest pool) can't
crowd out context_recall/faithfulness — same stratification pattern as `rag.goldset`'s
`stratified_sample`, just over metric name instead of (category, modality).

---

## 9. One-line answers for the defense

- **How do you know your LLM judge is trustworthy?** Two independent checks, not one. First,
  blind hand-labeling against real pipeline items: overall κ=0.247 (46 items, fair), with
  `context_recall`/`faithfulness` both landing at κ=0.0 for a specific, diagnosed reason (§7) and
  `context_precision` — the genuine weak spot — improved from κ=0.125 to **κ=0.308** after a
  targeted rubric fix (§6), still "fair," not fully solved. Second, a 10-case adversarial probe
  with known-by-construction ground truth (§7), built specifically to test what the blind sample
  couldn't: 10/10 correct, including correctly isolating a fabricated detail stitched into an
  otherwise-true claim.
- **Isn't using the same model to generate and judge biased?** Different model families (Gemini
  generator, DeepSeek judge) rules out literal self-evaluation — but that's an architecture
  argument, not proof the judge is any *good* at the task. Kappa and the adversarial probe are the
  empirical checks that architecture alone doesn't give you.
- **Why kappa instead of raw agreement?** Raw agreement is inflated by class imbalance — a judge
  that always says "yes" scores high raw agreement whenever "yes" is the common case. Kappa
  subtracts out exactly that chance component.
- **Your context_recall/faithfulness kappa was 0.0 — doesn't that mean the judge is unreliable?**
  No — the judge said "supported" on 100% of that sample, so kappa had zero variance to work with
  and was mathematically forced toward 0 regardless of call quality (§7). A follow-up adversarial
  probe with known-negative cases confirmed the judge does correctly say "not supported" when it
  should; the 0.0 was an absence-of-evidence artifact in the sample, not a reliability finding.
- **Why isn't response_relevancy kappa-validated too?** It's a single continuous score per answer,
  not decomposed into discrete items — bucketing it for kappa would validate an invented
  threshold, not the judge's real behavior. Documented as an explicit scope decision, not an
  oversight.
- **Why blind labeling?** Seeing the judge's verdict first biases a reviewer toward confirming it;
  the check is only meaningful if the human's call is genuinely independent.
- **How did you keep your own labeling consistent with what the judge was actually asked?**
  Applied each metric's own rubric, not one generic "is this right" gut check — strict textual
  entailment for context_recall/faithfulness (evidence must state or directly imply the claim),
  looser relevance/contribution for context_precision (was this chunk useful for the answer).
  Mixing the two would add labeler noise indistinguishable from genuine judge disagreement.

---

## 10. How the golden set is actually generated — `rag/goldset.py`

`python -m rag.goldset -n 90` sounds like "90 random chunks, 90 questions." Neither half of that
is quite right, and both details matter for reading the goldset honestly.

**Selection is stratified round-robin, not uniform random.** After excluding chunks that already
have a question (`load_existing_chunk_ids()`, keyed by `gold_chunk_id` in the existing output
file — this is also what makes re-running the command additive instead of wastefully
re-generating questions for chunks that already have one), `stratified_sample()` buckets every
remaining chunk by `(category, modality)` — `(ai, text)`, `(finance, table)`, `(cs.CL, figure)`,
etc. Each bucket is shuffled once (seeded, so reproducible), then the sampler draws one chunk
from each bucket in turn, cycling through every bucket repeatedly until it has `n` chunks or every
bucket is empty. **Why this matters concretely:** as of 2026-08-19, 60 `ai` chunks already have
questions, leaving `ai`≈482 / `cs.CL`≈289 / `finance`≈388 unused chunks in the pool. Plain random
sampling over that combined pool would landslide toward `ai` purely because it's the largest
remaining bucket. Round-robin-by-bucket means the smaller `finance`/`cs.CL` buckets get pulled
from just as often as `ai` does, right up until a bucket empties out — which is the entire reason
the coverage requirement (ROADMAP §Phase 4: "text-only vs figure-requiring questions tagged") is
achievable at all instead of being swamped by whichever category/modality the corpus happens to
have the most of.

**Each selected chunk maps to at most one question — never zero-or-more in the other direction.**
One judge call per chunk, asked to write one question + one reference answer *answerable only
from that single passage*. Chunks are never combined, and a chunk never yields two questions.

**Requesting `n` gets you *up to* `n` rows, not exactly `n`.** A chunk is dropped entirely, not
retried until success, if its judge call fails to parse, or if the generated question still
leaks the source paper's `topic` string after one corrective retry (`generate_example()`
returns `None` for both cases). Historically the reject rate has been low (0/30 on the very
first run), but it is not guaranteed zero — the honest framing for `-n 90` is "90 chunks get one
attempt each," not "90 questions will be written."

**Cost model:** one primary judge call per selected chunk, plus a second call only for chunks
whose first attempt leaked the title (rare, and self-correcting on retry) — same per-call cost
as every prior goldset batch, just scaled by `n`.
