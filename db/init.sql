CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id             bigserial PRIMARY KEY,
    url            text NOT NULL UNIQUE,
    title          text NOT NULL,
    agency         text NOT NULL,
    kind           text NOT NULL CHECK (kind IN ('service', 'statute')),
    effective_date date,
    fetched_at     date NOT NULL,
    content_sha256 text NOT NULL
);

CREATE TABLE IF NOT EXISTS passages (
    id          bigserial PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal     int NOT NULL,
    section     text,
    page        int,
    text        text NOT NULL,
    embedding   vector(384) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(section, '') || ' ' || text)
                ) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS passages_embedding_idx ON passages USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS passages_tsv_idx ON passages USING gin (tsv);
