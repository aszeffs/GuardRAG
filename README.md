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
