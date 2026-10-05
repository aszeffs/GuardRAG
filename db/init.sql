CREATE EXTENSION IF NOT EXISTS vector;

-- Writes every peso amount one way, so "₱1,500.00", "PHP 1500" and "1500 pesos" all become the
-- single search term 'p1500'. Used on both Passages and questions.
CREATE OR REPLACE FUNCTION normalize_pesos(t text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE RETURN
    regexp_replace(
        regexp_replace(
            regexp_replace(
                regexp_replace(t, '(\d),(?=\d{3}(?!\d))', '\1', 'g'),  -- thousands separators
                '(\d)\.00(?!\d)', '\1', 'g'),                         -- zero centavos
            '\m(\d+)\s*pesos?\M', 'P\1', 'gi'),                       -- "100 pesos"
        '(₱|\mphp\.?|\mp)\s*(?=\d)', 'P', 'gi');                      -- "₱100", "PHP 100"

-- Words that carry no meaning in a Filipino or Taglish question. The 'english' configuration
-- keeps them as search terms. Removed from questions only: a Passage term no question asks for
-- never matches.
CREATE OR REPLACE FUNCTION drop_filipino_filler(t text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE RETURN
    regexp_replace(t,
        '\m(ang|ng|nang|sa|mga|na|pa|po|ho|ba|si|ni|kay|ay|at|o|ko|mo|ka|ako|ikaw|siya|kami|'
        'tayo|sila|nila|namin|natin|niya|ninyo|ito|iyan|iyon|yung|dito|doon|din|rin|lang|lamang|'
        'naman|kung|kapag|para|dahil|pero|paano|saan|ano|anong|kailan|bakit|sino|alin|gaano|'
        'magkano|ilan|mag|nag|pag|ma|naka|maka|makapag|magpa)\M',
        ' ', 'gi');

-- The question's terms, OR-combined: plainto_tsquery AND-s them, which finds nothing for most
-- natural-language questions. Only stop words or filler give an empty tsquery, which matches
-- nothing.
CREATE OR REPLACE FUNCTION keyword_query(question text) RETURNS tsquery
LANGUAGE sql IMMUTABLE PARALLEL SAFE RETURN
    replace(
        plainto_tsquery('english', drop_filipino_filler(normalize_pesos(question)))::text,
        ' & ', ' | '
    )::tsquery;

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
                    to_tsvector('english', normalize_pesos(coalesce(section, '') || ' ' || text))
                ) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS passages_embedding_idx ON passages USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS passages_tsv_idx ON passages USING gin (tsv);
