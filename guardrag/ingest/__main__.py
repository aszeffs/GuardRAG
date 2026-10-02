"""Build the corpus: python -m guardrag.ingest [--refresh] [--extract-only]"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from guardrag.chunking import chunk_document
from guardrag.config import get_settings
from guardrag.db import connect
from guardrag.domain import SourceDocument
from guardrag.embedder import FastEmbedEmbedder
from guardrag.ingest.extract import extract_html, extract_pdf
from guardrag.ingest.fetch import fetch
from guardrag.ingest.manifest import load_manifest
from guardrag.ingest.store import document_unchanged, replace_document

log = logging.getLogger("guardrag.ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download every source")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="write extracted sections to data/extracted/ for inspection; no DB, no chunking",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    data = Path(settings.data_dir)
    entries = load_manifest()

    embedder = FastEmbedEmbedder(settings.embedding_model)
    conn_cm = None if args.extract_only else connect()
    conn = conn_cm.__enter__() if conn_cm else None
    try:
        for entry in entries:
            fetched = fetch(entry.url, data / "raw", refresh=args.refresh)
            sections = extract_pdf(fetched.path) if fetched.is_pdf else extract_html(fetched.path)
            doc = SourceDocument(
                url=entry.url,
                title=entry.title,
                agency=entry.agency,
                kind=entry.kind,
                fetched_at=fetched.fetched_at,
                effective_date=entry.effective_date,
                sections=sections,
            )
            if not sections:
                log.warning("no text extracted: %s", entry.url)
                continue

            if args.extract_only:
                out = data / "extracted" / f"{fetched.path.stem}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                payload = json.dumps(asdict(doc), default=str, indent=1, ensure_ascii=False)
                out.write_text(payload, encoding="utf-8")
                log.info("%-70s %4d sections", doc.title[:70], len(sections))
                continue

            if document_unchanged(conn, entry.url, fetched.sha256):
                log.info("unchanged  %s", doc.title)
                continue
            passages = chunk_document(doc, embedder.count_tokens)
            embeddings = embedder.embed([p.text for p in passages])
            replace_document(conn, doc, fetched.sha256, passages, embeddings)
            conn.commit()
            log.info("%-70s %4d passages", doc.title[:70], len(passages))
    finally:
        if conn_cm:
            conn_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
