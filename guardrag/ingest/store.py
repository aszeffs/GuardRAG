import psycopg

from guardrag.domain import Passage, SourceDocument


def document_unchanged(conn: psycopg.Connection, url: str, sha256: str) -> bool:
    row = conn.execute(
        """SELECT 1 FROM documents d
           WHERE d.url = %s AND d.content_sha256 = %s
             AND EXISTS (SELECT 1 FROM passages p WHERE p.document_id = d.id)""",
        (url, sha256),
    ).fetchone()
    return row is not None


def replace_document(
    conn: psycopg.Connection,
    doc: SourceDocument,
    sha256: str,
    passages: list[Passage],
    embeddings: list[list[float]],
) -> None:
    """Upsert the document row and replace all of its passages, in one transaction."""
    with conn.transaction():
        doc_id = conn.execute(
            """INSERT INTO documents
                 (url, title, agency, kind, effective_date, fetched_at, content_sha256)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (url) DO UPDATE SET
                 title = EXCLUDED.title, agency = EXCLUDED.agency, kind = EXCLUDED.kind,
                 effective_date = EXCLUDED.effective_date, fetched_at = EXCLUDED.fetched_at,
                 content_sha256 = EXCLUDED.content_sha256
               RETURNING id""",
            (doc.url, doc.title, doc.agency, doc.kind, doc.effective_date, doc.fetched_at, sha256),
        ).fetchone()[0]
        conn.execute("DELETE FROM passages WHERE document_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO passages (document_id, ordinal, section, page, text, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                [
                    (doc_id, p.ordinal, p.section, p.page, p.text, emb)
                    for p, emb in zip(passages, embeddings, strict=True)
                ],
            )
