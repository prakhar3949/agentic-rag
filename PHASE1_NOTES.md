# Phase 1 — Ingestion: decisions, bugs, and defense notes

Working record for `notebooks/ingest_phase1.ipynb` and `notebooks/embed_index_phase1.ipynb`.
Written for defense prep: what was decided, what broke, what the evidence was, and what is
still open.

---

## 1. Scope

Deliberately **not** a full-document parser. Per PDF:

- **front** = abstract + introduction, **back** = summary + conclusion. Nothing else.
- figures and tables falling in those spans
- metadata: `topic`, `category`, `num_pages`, `authors`

Output is a flat `list[Document]` of chunks with the Phase 2 metadata schema:
`arxiv_id`, `category`, `topic`, `section`, `modality`, `page`, `figure_ref`, `section_source`.

---

## 2. `category` vs `topic` — the metadata distinction

Asked: *"Topic should come from the paper's heading; category should be the directory."*

| field | source | cardinality | role |
|---|---|---|---|
| `category` | parent directory (`arxiv_papers/<cat>/`) | 5 values, closed | **filter key**; router ground truth |
| `topic` | paper's own title, parsed from page 1 | ~1 per doc, free text | citations, display |

**Defense point:** these are not interchangeable. Retrieval filters and the router are scored
against `category` precisely because it is exact, closed-vocabulary, and free ground truth
(ROADMAP §1). `topic` is free text with no controlled vocabulary — usable as payload, useless
as a filter clause. A `topic == "..."` filter matches zero documents or exactly one, which is a
primary-key lookup, not retrieval.

Title detection = largest font in the top half of page 1, falling back to the PDF `/Title`
field, then the first substantial line. Layout is tried **before** `/Title` because LaTeX
toolchains routinely leave `main`, `paper`, or a filename in that field. `topic_source` records
which path ran.

---

## 3. Bugs found, in order — the most defensible material

Each of these was found by instrumenting, not by reasoning. That pattern is the story.

### 3.1 Every VLM caption silently discarded
`response.content` is a **list of content blocks** on langchain v1, not a `str`. Calling
`.content.strip()` raised `AttributeError`, the handler set `caption = None`, and
`build_chunks` skipped every figure.

**Impact:** the pipeline reported success, produced 170 plausible chunks, and contained
**zero figures** — while having called and been billed for every VLM request.
**Fix:** `message_text()` normalises both shapes.
**Lesson:** the run looked healthy. Only the modality breakdown exposed it.

### 3.2 Word spacing destroyed
LaTeX justifies by positioning each word individually rather than emitting space characters, so
PyMuPDF returns one span per word with no whitespace. `"".join(...)` produced
`"Cansupply-chainAImovebeyondisolateddecisionmod-"`.

**Impact:** run-together text embeds terribly — it matches no normally-worded query.
**Fix:** `line_text()` inserts a space where the horizontal gap between spans exceeds
`SPAN_SPACE_GAP_RATIO × font_size`. Gap-based, not unconditional: a font switch mid-word
(italic term, ligature) also starts a new span but abuts, so `SCOPE` does not become `S COPE`.
Also stopped filtering out whitespace-only spans, which were real spaces.

### 3.3 Section cuts at page granularity
Text was extracted by whole page, so the front span dragged in whatever followed the
introduction on its last page — typically **Related Work**.

**Why this is the dangerous one:** Related Work describes *other papers'* claims, but the chunk
carries this paper's `arxiv_id`. The generator then cites a rival's result as this paper's
finding — a confidently wrong, correctly-cited answer. Invisible to recall@k.
**Fix:** cut at the exact heading **line**. Page spans are now derived from kept lines and used
only to scope image/table search.

### 3.4 The introduction dropped entirely, while reporting success
Template put the section number on its own line: `"1"` then `"Introduction"`. `FRONT_CONTINUE`
saw a bare `"1"`, didn't match, and cut there.

**Impact:** 4 chunks instead of ~40; `sections_found=True` throughout.
**Fix:** `_merge_orphan_numbers()` rejoins them before pattern matching.
**Related fix:** the span's heading *level* must rise as it absorbs continuation headings.
Abstract is 10pt, Introduction 12pt — pinning the level at Abstract's size made every 10pt bold
paragraph lead-in inside the intro look like a terminating heading.

### 3.5 arXiv stamp read as a section heading
`arXiv:2607.28503v1 [cs.AI] 28 Jul 2026` is rotated 90° down the left margin of page 1. In
content order it lands mid-document, is short, and has its own size — so it registered as a
heading and truncated the front span.
**Fix:** `_is_marginalia()` — text pattern **and** a writing-direction test (`dir != (1,0)`),
because the anchored `^arxiv` regex only caught it at line start.

### 3.6 MuPDF is not thread-safe
Same file, same code, two different answers:

```
single-doc:  chunks=13 {'text': 13}                 -> 0 tables
batch:       chunks=15 {'text': 13, 'table': 2}     -> 2 tables
```

The only difference was `ThreadPoolExecutor`. Concurrent access to MuPDF's shared state does not
crash; it silently returns wrong results.
**Fix:** `PDF_LOCK` serialises every `fitz` call. Both network calls (section fallback,
captioning) sit **outside** the lock, so the parallelism that matters is preserved — parsing is
milliseconds, VLM round trips are seconds.
**Defense point:** cost is near zero, determinism is total. `max_workers` now widens VLM
concurrency only; scaling parsing would need processes, not threads.

### 3.7 `Table.to_markdown()` broken — 100% table loss
Detection worked the whole time. PyMuPDF's own `to_markdown()` raised
`ValueError: max() iterable argument is empty` on **every** table in the corpus.

```
[tables] to_markdown FAILED 2607.28488v1 p6 t0: ValueError: max() iterable argument is empty
[tables] to_markdown FAILED 2607.28503v1 p10 t0: ...
```

**Why it took several rounds:** my `except Exception: continue` made a library crash
indistinguishable from "this page has no tables." We chased threading and page-scoping first —
both real observations, neither the cause.
**Fix:** `table_to_markdown()` builds from `table.extract()` (the raw cell grid), pads ragged
rows, synthesises a header, escapes `|`.
**Lesson worth stating out loud:** a 100% failure rate looked plausible because 0–2 tables per
paper is expected when you only ingest abstract/intro/conclusion. Counters must count *causes*
(`tables_found` vs `tables_rendered`), not totals.

### 3.8 Cross-listed papers downloaded into more than one category folder

Found 2026-08-19, expanding from `ai/` to `ai/` + `cs.CL/` + `finance/`: `save_extractions()`
raised `duplicate chunk id` on `2607.28498v1` — a genuine collision, not a hash coincidence.
Checked all five category folders for repeated `arxiv_id` stems and confirmed by MD5: **7 PDFs
are byte-identical duplicates across two category folders** — 6 shared between `ai`/`cs.CL`
(unsurprising; those fields overlap heavily on arXiv), 1 between `ai`/`hep-th`.

**Impact:** the original Phase 1 bulk-fetch downloaded a paper into every category folder it was
cross-listed under, with no dedup. `category` is supposed to be the closed-vocabulary, one-value
ground truth the router is scored against (§2) — a paper physically present under two categories
silently breaks that assumption, and would have shown up as a router "failure" that was actually
a data problem, not a routing problem.
**Fix:** dedup by `arxiv_id` before batching, first-seen-in-category-order wins (`ai` kept its
existing 6, `cs.CL` dropped 6, `hep-th` would drop 1 whenever it's added). Logged, not silent —
prints which arxiv_id was kept under which category and which duplicate was skipped.
**Lesson:** this was sitting in the corpus since Phase 1's very first bulk fetch. Nothing caught
it while only one category was ever indexed, because a duplicate needs a *second* category
present to collide with. Single-category testing cannot surface a cross-category bug.

---

## 4. Section detection — the strategy ladder

Final design, cheapest and most reliable first. `section_source` records which fired and rides
on every chunk.

| tier | strategy | why it is where it is |
|---|---|---|
| 1 | `toc` — PDF outline (bookmarks) | the author's own section list, exact titles; sidesteps rendering entirely |
| 2 | `headings` — font size + bold | no config needed; body size measured per document |
| 3 | `llm` — model picks boundaries | handles templates the font rules cannot see |
| 4 | `page_fallback` — first 3 / last 2 pages | last resort, flagged |

**Font heuristics detail.** Body size = character-weighted most common size (weighted by
characters so a paper with many short headings cannot vote its heading size into first place).
Font size also gives heading *level* for free: a span ends at the next heading **the same size
or larger**, so `2.1 Background` sits inside the intro but `2 Related Work` terminates it.

**Bookmarks detail.** Only level-1 entries are considered. Titles are matched to rendered lines
via `_normalize_heading`, which strips numbering and punctuation — `I. INTRODUCTION`,
`1 Introduction`, and `Introduction` all reduce to `introduction`. Matching moves forward only,
so a body sentence containing "Introduction" cannot pull a boundary backwards. Known gap:
matched by title, not by bookmark destination coordinates; an author overriding
`\section[short]{long}` will not match and falls through to tier 2.

### The LLM fallback — why it is defensible

Triggered by `2607.28623v1`: IEEE template, small-caps headings. `I. INTRODUCTION` renders as a
10pt `I` plus an 8pt `NTRODUCTION`, so the line is neither larger than body text nor bold —
`is_heading` found 3 "headings" in a 906-line paper, all of them title/author lines.

Design choices that matter under questioning:

- **The catalogue is short lines, not detected headings.** Filtering by `is_heading` would hand
  the model an empty list on exactly the documents that need help.
- **The model returns indices, never text.** It cannot fabricate content; the worst case is a
  wrong line.
- **Every index is validated** against the offered candidates (`_valid_index`). A bad answer
  degrades to `page_fallback` rather than producing a plausible-looking wrong span.
- **Cost:** ~150 short lines ≈ a few hundred tokens, flash-lite, on ~5% of the corpus. 95% stays
  fully deterministic and free.

One-line summary for the defense: *"Deterministic parsing, with an LLM escalation for the small
fraction heuristics miss, validated against the parse so it cannot hallucinate boundaries."*

---

## 5. Other design decisions

**Threads, not processes** — captioning is network-I/O bound, so the GIL is not the constraint;
a thread releases it while waiting on a socket. Processes would cost ~50–100 ms spawn plus
pickling image bytes to parallelise something never CPU-bound. (Separately: thread-safety of a
C library is a different problem from the GIL — hence `PDF_LOCK`.)

**Figures and tables are one chunk each** — splitting a three-sentence caption or a small table
only fragments it.

**Figures indexed by VLM caption**, not by the paper's own `Figure 3: ...` text, which stays in
the body and is chunked separately. The two are not linked — revisit if figure-requiring
questions underperform (§3⑥).

**Failures logged, never fatal** — one bad image or PDF must not sink a document or a batch. That
log is the Phase 10 ingestion-failure taxonomy input.

**`pymupdf-layout` adopted** for table detection. **LICENSE: Polyform Noncommercial / Artifex
commercial** — disclose in the README. Note this partially reverses the ROADMAP §7 "layout-aware
parsing" cut; the defensible framing is *"I cut layout-aware pipelines (Docling/unstructured),
then adopted PyMuPDF's layout model for table detection specifically"* — one import, not an
integration.

---

## 6. Chunking

Fixed-window with overlap (`chunk_size=1000`, `chunk_overlap=150`) via
`RecursiveCharacterTextSplitter` — the first arm of the ROADMAP §Phase 2 comparison. The
`section` metadata field is what makes the section-aware arm possible later.

Overlap exists because splits are arbitrary: without it, a fact near a boundary ends up half in
chunk N and half in N+1 and neither answers the question — the "chunk boundary split the answer"
failure in §3①.

**Note for the ablation:** because only abstract/intro/conclusion are ingested, chunks are
already section-scoped — a chunk never straddles the intro/conclusion boundary. So the
fixed-window arm is really *"fixed-window within a section,"* a slightly stronger baseline than
pure fixed-window over full text. Worth stating rather than letting a reviewer find it.

---

## 7. Telemetry — what to report

Every chunk carries `section_source`. Track across the full slice:

- `section_source` distribution: `toc` / `headings` / `llm` / `page_fallback`
- `topic_source` distribution: `layout` / `pdf_field` / `first_line` / `none`
- figures captioned vs figures extracted
- tables **detected** vs tables **rendered** (the §3.7 lesson)
- papers possessing a PDF outline at all

**Then slice Phase 4 retrieval metrics by `section_source`.** That answers empirically whether
fallback documents underperform, instead of speculating. If they do not, the remaining gap needs
no work and you have measured justification for skipping it.

---

## 8. Known gaps / still open

1. **Tables and images are still page-scoped** while text is line-scoped. A table below the cut
   on the intro's last page is still ingested. Fix = scope by vertical region, not page number.
2. **`sections_found` only means the front span was located.** A 200-character "introduction"
   counts as success. Track span character counts before claiming a parse rate.
3. **Two-column reading order** is handled by banding around full-width blocks — verify on
   `astronomy` and `hep-th`, which use different templates from `cs.CL`.
4. **`FRONT_START` accepts only Arabic numerals**, so `I. INTRODUCTION` fails the regex even when
   detected. Adding Roman numerals would likely convert most `llm`-tier documents to `headings`
   and cut LLM calls. Do it only if the distribution says it matters.
5. **No retry on VLM calls.** Transient 429/503 permanently costs a figure. `llm.with_retry()`
   is the one-line fix — deferred until real failure rates are observed.
6. **`except Exception` is broad** in captioning: a rate limit and a corrupt image are handled
   identically. Matters at 10K documents, not at 100.
7. **`pymupdf-layout` licensing** must be disclosed.
8. **`config/pricing.yaml` is still placeholders** — `gemini-3.1-flash-lite` unverified, all
   rates 0.00 (ROADMAP Phase 0).

---

## 9. Process lessons — the honest meta-point

Three of the seven bugs were **silent**: the pipeline reported success and produced
plausible-looking output while dropping an entire modality. In each case the fix was
instrumentation, not cleverness:

- the modality breakdown exposed the caption bug
- printing the exception exposed the `to_markdown` bug
- comparing serial vs threaded runs exposed the thread-safety bug

The corresponding anti-pattern, which cost several rounds here: **`except Exception: continue`
makes a crash indistinguishable from an empty result.** Every silent handler in an ingestion
pipeline is a place where a 100% failure rate can look like a normal day.

---

## 10. Embedding + indexing — decisions and defense points

Second half of the Phase 1 slice, `notebooks/embed_index_phase1.ipynb`. Input is `data/phase1/`
as written by ingestion above; output is a queryable Qdrant collection.

### 10.1 Deterministic chunk IDs make re-runs idempotent, not additive

Point IDs are the UUIDv5s minted at ingestion (`chunk_id()`, §2 above) from
`arxiv_id|ordinal|modality|figure_ref` — chunk *text* is deliberately not part of the key.
`client.upsert()` with these IDs overwrites a chunk with itself on re-run instead of duplicating
it; verified `collection count == chunk count` after every upsert, plus a round-trip check that
retrieves a few points back and asserts payload text and vector match the source arrays.

**Why text is excluded from the key:** a re-parse that nudges a chunk boundary by a word should
overwrite the old chunk, not duplicate it alongside its own previous version.
**The trade-off this creates:** the key leans on `ordinal`, so anything that changes how many
chunks a document produces renumbers every ordinal after the change and orphans the old points
under those positions. Originally scoped as a *deliberate chunking-strategy change* (window size,
overlap) — **narrower than reality**, confirmed 2026-08-19: re-parsing `ai/` unchanged (same code,
same 20 PDFs) produced 542 chunks instead of 544, because VLM captioning isn't perfectly
deterministic even at fixed settings — one image flipped from captioned to `NOT_A_FIGURE` between
runs. That two-chunk drop shifted every later ordinal in the affected documents, which changed
their chunk IDs, which orphaned the old ones in Qdrant. **The trigger is routine LLM
non-determinism, not just an intentional re-chunk** — a distinction worth having ready, since
"why would ordinals ever change if the code didn't change" is the obvious follow-up question.
Figures/tables key on `figure_ref` instead and survive a re-chunk cleanly.
**Defense point:** re-ingestion is a safe upsert *only when chunk composition is unchanged*; a
chunking-strategy change **or an ordinary re-parse that ends up with a different chunk count**
needs the collection dropped first, and that asymmetry is by design, not an oversight.

### 10.2 Truncation is silent — audited before embedding, not after

BGE-M3's `max_seq_length` is 8,192 tokens. Past that limit `sentence-transformers` truncates
silently: no error, no warning, a perfectly valid-looking unit vector that simply doesn't cover
the back of the chunk. Same failure shape as the ingestion bugs in §3 — the run reports success
and looks healthy.

Audited token lengths directly against `embedder.tokenizer`, not char lengths — a markdown table
runs ~1.76 chars/token versus ~3.6 for prose, because every `|` and digit becomes its own token,
so a short-looking table chunk can cost nearly as many tokens as a much longer paragraph:

```
tokens: min=33 median=247 p95=346 max=790 (limit 8192)
chunks over the limit: 0
```

Worst case uses 9.6% of the window. Figures and tables bypass the splitter by design (§5 above)
so they are the only chunks that could exceed `CHUNK_SIZE`; this measures the real risk instead
of assuming it away.

### 10.3 Dense-only is a deliberate floor, not the finished system

BGE-M3 emits dense, sparse (lexical, ~250K-dim vocabulary, learned term weights — not classic
BM25, but the same idea), and a ColBERT-style multi-vector output from one forward pass. Phase 1
keeps dense only and discards the rest, so row 1 of the ablation table is a real floor for dense
retrieval alone, not an already-hybrid number. BM25/sparse lands in Phase 3 as an *additional*
named vector — the `dense` name was chosen at collection creation for exactly this reason:
Qdrant fixes the vector schema at creation, so an unnamed default vector today would force a
collection rebuild later instead of a clean addition.

### 10.4 What a retrieval call returns

`client.query_points(...)` returns a `QueryResponse`; the field that matters is `.points`, a
ranked `list[ScoredPoint]`. Each point carries `id` (round-trips to the ingested chunk ID),
`score` (cosine similarity — the collection uses `Distance.COSINE` over pre-normalized vectors,
so cosine and dot product coincide, keeping scores on one scale through the reranker and any
later threshold work), and `payload` (chunk text plus the full metadata schema). Qdrant is the
only store in this system — there is no second lookup by ID into another database, which is why
`text` always rides in the payload alongside the filter fields.

### 10.6 `delete_collection()` does not purge embedded/local storage

Found 2026-08-19, recovering from the §10.1 ordinal-drift orphans: called
`client.delete_collection(COLLECTION)` then `client.create_collection(...)` with the same name,
expecting a clean slate. Point count after re-upserting exactly 1,219 chunks came back **1,227** —
the identical wrong number as *before* the drop. Scrolled the collection and diffed IDs against
`chunks.jsonl` directly: **8 points survived a full drop-and-recreate cycle**, all orphaned `ai`
figure/table chunks from the pre-expansion 544-point state.

**Impact:** in embedded/local mode (`QdrantClient(path=...)`, storage is a directory, not a
server), `delete_collection()` reporting success does not mean the on-disk segment files for that
collection name were actually removed — a same-named `create_collection()` right after can still
see stale data. This is invisible unless you verify the *actual* stored ID set against source,
not just `client.count()` after upsert (which matched the wrong total both times, consistently,
looking like a stable if incorrect state rather than an error).
**Fix:** delete the `data/qdrant/` directory itself before recreating, not just the API call.
Safe because it's a fully derived index — nothing lives there that isn't reconstructible from
`data/phase1/chunks.jsonl` plus the cached `embeddings.npy`/`bm25_indices.json`/`bm25_values.json`.
**Defense point:** "drop the collection" (§10.1, ROADMAP line 350) means the directory, not the
API call, in embedded mode — worth stating explicitly since the API call *looks* like it worked
and silently doesn't.

### 10.7 Known gaps carried from this notebook

- **Exact search, not HNSW; payload indexes declared, not built.** Embedded-mode Qdrant does
  neither — `create_payload_index` is a documented no-op locally, and `hnsw_config` is accepted
  and ignored. Recall measured here is a ceiling, not the deployed system's number, and no
  latency figure from this notebook describes the server-mode system. Flipping
  `QDRANT_MODE=server` activates both — do that before quoting either figure.
- **`ai/`, `cs.CL/`, `finance/` indexed (2026-08-19) — 20 docs/category, not Phase 2's ~2,000/category
  "at scale" target.** `astronomy`/`hep-th` have not been through the ingestion parser yet, so
  their `section_source` distribution is unmeasured. Deterministic IDs mean re-running ingestion
  plus this notebook upserts cleanly over existing points **only when chunk composition per
  document is unchanged** — see §10.1's non-determinism caveat and §10.6's drop-collection
  mechanics before assuming a re-run is free.
- **Figure chunks are VLM captions, not images.** Retrieval matches caption text; the original
  bytes live in `data/phase1/images/` and are resolved by `figure_ref` at generation time.
