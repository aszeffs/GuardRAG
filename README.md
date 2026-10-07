# GuardRAG

A secure RAG assistant over public Philippine government service documents (BIR, PhilHealth, SSS, LGU citizen charters), with retrieval evals, Ragas scoring and promptfoo red-team tests that gate CI.

> Status: in progress.

## Planned stack
- FastAPI + PostgreSQL/pgvector, Docker Compose
- Hybrid search: pgvector cosine + Postgres full-text, fused with Reciprocal Rank Fusion
- Groq LLM with structured, cited answers (`answer`, `citations[]`, `confidence`) and an "I don't know" fallback
- Evals: hand-written golden set (recall@5, MRR), Ragas (faithfulness, answer relevancy, context precision)
- Red-team: promptfoo suite (prompt injection, jailbreak, PII, off-topic)
- GitHub Actions: eval and red-team gates, plus CodeQL, Gitleaks, Trivy, SBOM, signed images

## Run

```sh
docker compose up -d --build
docker compose run --rm api python -m guardrag.ingest
curl 127.0.0.1:8000/health
```

The ingest downloads the 35 documents in `guardrag/ingest/sources.yaml` into `data/raw/` once, then extracts, chunks, embeds and stores them. Re-running it skips every document whose content hash is unchanged (`--refresh` re-downloads). `/health` reports the Passage count.

### Chunking

Passages are capped at 126 tokens of the embedder's own tokenizer: its 128-token window minus `[CLS]` and `[SEP]` ([ADR 0004](docs/adr/0004-small-passages-for-multilingual-embedder.md)). A Passage never crosses a section, so it always has one heading and page to cite. Within a section, a Passage ends at the last sentence or line end in its second half when there is one, so it is rarely cut mid-sentence.

Consecutive Passages in a section overlap by up to **20 tokens**, about 15% of a Passage. That is roughly one requirement line or one clause in a Citizen's Charter table ("Valid ID (1 original, 1 photocopy)"), so an item split at a boundary still appears whole in one Passage and can be retrieved and cited intact. More overlap would duplicate text across the top-5 results and inflate the index for little gain. With such small Passages, every overlapping token takes room from new text. On the Corpus as of October 2026, this gives 5,950 Passages with a median of 118 tokens.

### Grounding and Confidence

Grounding is checked in code after the model replies, not left to the prompt ([ADR 0002](docs/adr/0002-grounding-enforced-in-code.md)). A Citation survives only if it names a Passage retrieved for this request; the rest are dropped, and a Passage cited twice is listed once. If the model marks the request out of scope, the reply is an `out_of_scope` Refusal with a fixed, polite decline. If no Citation survives or the answer is blank, including when nothing was retrieved or the model's reply was malformed, the reply is an `out_of_corpus` Refusal: "I don't know", naming the Agency of the best retrieved Passage as the one most likely to help. Every Refusal has `confidence: none` and no Citations, and every other reply has at least one Citation.

A cited answer's Confidence comes from the evidence, never from the model:

- **`high`**: the answer cites the top-ranked retrieved Passage, and none of the model's Citations had to be dropped.
- **`low`**: anything else. Either the answer rests only on lower-ranked Passages, or the model also cited something it was never given, so part of the answer may be unsupported.

The rules use rank rather than score because scores are comparable only within one retriever mode: cosine similarity, `ts_rank_cd` and RRF are on different scales.

### Input guards

Every question is screened before retrieval or any model call, in this order:

1. **Rate limit.** Each client IP gets at most `RATE_LIMIT_REQUESTS` requests per `RATE_LIMIT_WINDOW_SECONDS` (20 per 60 by default). Over the limit, the reply is a 429 with `Retry-After`. The limiter is in memory, which is enough for a single-process, local-only deployment. Behind a reverse proxy every client would share the proxy's IP, and so one limit.
2. **Size and format.** A question that is blank, longer than 1,000 characters, or contains control characters (anything besides tabs and newlines) gets a 422. The 422 body says what was wrong but does not echo the question.
3. **Requests for someone else's personal data**, such as "What is the TIN of Juan Dela Cruz?", "Ano ang address ng kapitbahay ko?" or "Whose phone number is 0917…?", get an `out_of_scope` Refusal (see [Personal data](#personal-data)).
4. **Injection heuristics.** Patterns for well-known injection and jailbreak phrasings in English, Filipino and Taglish, such as "ignore previous instructions", "kalimutan mo ang naunang utos", "i-ignore mo yung instructions", role-play setups, and chat-template markers. Invisible characters, full-width letters and Cyrillic or Greek lookalikes are folded away first. A match gets an `out_of_scope` Refusal. The patterns are kept narrow so that ordinary questions such as "Can my employer ignore the guidelines on overtime pay?" get through (`NEAR_MISSES` in `tests/test_guards.py`). The cost is that paraphrases and spaced-out letters ("i.g.n.o.r.e") can slip past them.
5. **Llama Prompt Guard** (`meta-llama/llama-prompt-guard-2-86m` on Groq). A score of 0.5 or more gets the same Refusal. If the classifier errors, times out (5 s, no retries) or hits Groq's limit of 30 requests a minute, the request continues on the heuristics alone and a warning is logged ([ADR 0003](docs/adr/0003-injection-guard-fails-open.md)). The question itself is never logged.

The two layers cover each other's gaps. In a live check on 2026-10-06, Prompt Guard flagged none of the 49 Golden Set questions, but on its own it missed 12 of the 29 attacks in `tests/test_guards.py`, including 6 of the 7 Filipino ones. It also scored the ordinary question "Do I still need to follow the old instructions on the form?" 0.998, so it gets refused. No threshold can fix that, and it is a known limitation.

### Personal data

GuardRAG never repeats or stores a citizen's own identifiers or contact details, and it won't look up anyone else's.

- **Answers.** TINs, SSS numbers, PhilHealth numbers, Pag-IBIG MIDs, PhilSys numbers, PH mobile and landline numbers, and emails are replaced with `[redacted ID number]`, `[redacted phone number]` or `[redacted email]`. The one exception is a value that also appears in the Passages the answer was drafted from. The Corpus lists many Agency hotlines and health-centre emails, and an answer should be able to give them.
- **Logs.** Every log record is redacted as it is created, whatever the logger (including tracebacks), with no exceptions. The question is never logged anyway.
- **Requests for someone else's data.** Patterns catch a named person ("TIN ni Juan", "Maria Santos's SSS number", "the address of Atty. Garcia"), a described one ("my neighbor's", "ng ibang tao"), and reverse lookups ("Whose number is…", "Kanino ang…"). Asking about your own data, a family member's or an employee's, or an Agency's contact details is fine. Questions asking *how* to get something ("How do I get the TIN of someone who died?") are left to the answer model, which is told to decline private information about a person.

Redaction follows the formats people actually write: grouped numbers as each Agency prints them (123-456-789-000, 34-1234567-8, 1234-5678-9012) and bare runs of 9 to 16 digits. Fees, dates, form numbers and circulars ("₱1,500.00", "2026-10-07", "BIR Form 1902", "RR No. 8-2018") are left alone (`NEAR_MISSES` in `tests/test_pii.py`). The cost is that an identifier with unusual grouping, or a bare run under 9 digits, gets through.

## Retrieval eval

The Golden Set (`evals/golden_set.yaml`) has 49 questions. It was **LLM-drafted from the source documents and reviewed by the author**. 43 are answerable and 6 are Out-of-Corpus Questions (Pag-IBIG, DFA, LTO, Makati City, PSA). 11 (22%) are in Filipino or Taglish. Some are trick questions, such as a bare form number, a paraphrase that avoids the document's own wording, or an Agency the Corpus doesn't cover. Two questions span two Agencies. Each answerable question names the document and section that answers it, not Passage ids, so the Golden Set survives re-chunking.

```sh
python -m guardrag.evals.retrieval              # all three retrievers; writes evals/results/retrieval.json
python -m guardrag.evals.retrieval --gate hybrid --min-recall 0.65
```

recall@5 is the share of a question's expected sections found in the top 5, averaged over the answerable questions. MRR is 1 / rank of the first Passage from an expected section, or 0 if none is in the top 5. Before scoring, the command checks that every expected section exists in the Corpus and stops if one doesn't, so a typo or an extraction change can't pass as a retrieval miss. On the Corpus as of October 2026 ([results](evals/results/retrieval.json)):

| Retriever | recall@5 | MRR | English (34) | Filipino (5) | Taglish (4) |
|---|---|---|---|---|---|
| vector | 0.570 | 0.412 | 0.632 / 0.433 | 0.400 / 0.400 | 0.250 / 0.250 |
| keyword | 0.558 | 0.371 | 0.618 / 0.396 | 0.200 / 0.200 | 0.500 / 0.375 |
| **hybrid (RRF)** | **0.721** | **0.494** | 0.794 / 0.544 | 0.400 / 0.250 | 0.500 / 0.375 |

The language columns show recall@5 / MRR over the answerable questions in that language. Hybrid search beats either retriever alone, and it lifts recall@5 over vector-only by 15 points. Filipino and Taglish questions are still the weak spot: the Corpus is English, and the 384-dimension multilingual embedder only partly bridges the gap. With only 5 Filipino and 4 Taglish answerable questions, those columns are indicative only.

A hit means a Passage came from the right section, not that it contains the answer. Most Service Document sections are short, but some are large. For example, RA 9994 "Section 3" has 72 Passages, and only one of them states the 20% discount. A few documents have no usable sections at all (the PhilHealth benefits page, and the Pasig PDAO and PESO charters), so for those any Passage of the document counts. These scores are therefore an upper bound on Passage-level recall. The answer evals (#12) measure whether the cited Passage actually supports the answer.

Every PR runs the eval in CI (`retrieval-eval`) on a freshly seeded Corpus, at no LLM cost. Raw downloads and the embedder are cached between runs. The job fails if hybrid recall@5 falls below **0.65**, about three questions below today's score.

## Tests

```sh
docker compose up -d db
pip install -e ".[dev]"
pytest -rs
```

Tests marked `db` need Postgres. Locally they are skipped when it isn't running, but CI always runs them. Retrieval tests run against a separate `guardrag_test` database, which is recreated each run and seeded with a small fixed Corpus (`tests/seed.py`). Tests marked `pending("#N")` describe a ticket that hasn't landed yet. They report as skipped while the code they exercise raises `NotImplementedError`, then run for real once it doesn't.

## Security pipeline

Every pull request into `main` runs CodeQL (Python), Gitleaks over the full history, dependency review, and a container build that is smoke-tested against pgvector and scanned with Trivy. All of these are required checks on `main`. Actions are pinned to commit SHAs and images to digests, and each workflow starts read-only, widening permissions only in the job that needs them.

A merge to `main` publishes the scanned image to `ghcr.io/aszeffs/guardrag`, tagged with its commit, and attaches Sigstore-signed SLSA build provenance and an SPDX SBOM to its digest. To verify an image built from `<commit>`:

```sh
gh attestation verify oci://ghcr.io/aszeffs/guardrag:<commit> \
  --repo aszeffs/GuardRAG \
  --signer-workflow aszeffs/GuardRAG/.github/workflows/container.yml \
  --source-ref refs/heads/main \
  --source-digest <commit>
```

`gh` resolves the tag to a digest and checks the attestations against that digest. The image carries no separate cosign signature; the signed attestations are what bind it to this workflow and commit.

Trivy exceptions are recorded in `.trivyignore.yaml`, each with a reason and an expiry; see `docs/trivy-exceptions.md`.
