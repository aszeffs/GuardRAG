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
