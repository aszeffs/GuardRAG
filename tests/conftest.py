import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import make_conninfo
from seed import seed

from guardrag.config import get_settings
from guardrag.embedder import FastEmbedEmbedder

ROOT = Path(__file__).parents[1]
TEST_DB = "guardrag_test"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `db` tests when Postgres is unreachable locally. In CI they must run, so never skip."""
    db_items = [item for item in items if "db" in item.keywords]
    if os.environ.get("CI") or not db_items:
        return
    try:
        psycopg.connect(get_settings().database_url, connect_timeout=2).close()
    except psycopg.OperationalError:
        skip = pytest.mark.skip(reason="Postgres not reachable (docker compose up db)")
        for item in db_items:
            item.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Report a `pending` test as skipped while the code it exercises raises NotImplementedError.

    Once the ticket lands the test runs for real, so it can only pass or fail.
    """
    outcome = yield
    marker = item.get_closest_marker("pending")
    if marker and call.excinfo and call.excinfo.errisinstance(NotImplementedError):
        report = outcome.get_result()
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1], f"pending {marker.args[0]}")


@pytest.fixture(scope="session")
def embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder(get_settings().embedding_model)


@pytest.fixture(scope="session")
def seeded_db_url(embedder: FastEmbedEmbedder) -> Iterator[str]:
    """A fresh `guardrag_test` database holding the schema and the seed Corpus (tests/seed.py).

    Kept apart from the main database so a locally ingested Corpus is neither read nor touched.
    """
    main_url = get_settings().database_url
    with psycopg.connect(main_url, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    url = make_conninfo(main_url, dbname=TEST_DB)
    with psycopg.connect(url) as conn:
        conn.execute((ROOT / "db" / "init.sql").read_text(encoding="utf-8"))
        conn.commit()
        seed(conn, embedder)
    yield url
