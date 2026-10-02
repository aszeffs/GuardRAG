# GuardRAG

A question-answering assistant for citizens asking how to get things done with Philippine government agencies, answering only from a fixed set of public agency documents and citing where each answer came from.

## Language

### Corpus

**Corpus**:
The fixed set of public government documents GuardRAG is allowed to answer from.
_Avoid_: knowledge base, dataset

**Service Document**:
An agency document describing how to obtain a government service: requirements, steps, fees, processing time (e.g. a Citizen's Charter entry, an agency FAQ page).
_Avoid_: guide, procedure doc

**Statute**:
A law (Republic Act) included in the corpus so questions about what the law says have a source.
_Avoid_: legal doc, regulation

**Agency**:
The government body that issues a document or delivers a service (BIR, PhilHealth, SSS, Pag-IBIG, an LGU).
_Avoid_: office, department

### Questions and answers

**Out-of-Corpus Question**:
An on-topic question about government services that the corpus does not cover.
_Avoid_: unanswerable question

**Out-of-Scope Request**:
A request that is not about Philippine government services, or that asks for something harmful or private.
_Avoid_: off-topic question, unanswerable question

**Refusal**:
A response that gives no substantive answer. It is either an "I don't know" for an Out-of-Corpus Question or a decline for an Out-of-Scope Request; the two are counted separately.
_Avoid_: fallback, rejection

**Citation**:
A pointer from an answer to the exact corpus passage that supports it, shown as document title, section or page, link, and As-of Date. Only passages actually given to the model can be cited; an answer left with no valid Citation becomes a Refusal.
_Avoid_: reference, source link

**Confidence**:
A label (`high`, `low`, `none`) describing how strongly the retrieved evidence and Citations support an answer; `none` always means a Refusal. It is never the model's own opinion of itself.
_Avoid_: score, certainty

**As-of Date**:
The date a document's content is known to be valid: its printed effective date when there is one, otherwise the date it was fetched into the corpus.
_Avoid_: last updated, timestamp

**Answer Language**:
The language a reply is written in, which matches the language of the question (English, Filipino, or Taglish); requirement names, form numbers and fees keep their original wording.

### Evaluation

**Golden Set**:
The curated list of test questions, each tagged as answerable (with the passages that should be retrieved) or as an Out-of-Corpus Question, used to measure retrieval and answers.
_Avoid_: test set, benchmark, ground truth

**Red-Team Attack**:
A single adversarial input (prompt injection, jailbreak, PII extraction, off-topic push) that GuardRAG must block or refuse.
_Avoid_: attack prompt, adversarial test
