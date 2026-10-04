import os

import psycopg
import pytest

from guardrag.config import get_settings


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `db` tests when Postgres is unreachable locally. In CI they must run, so never skip."""
    if os.environ.get("CI") or not any("db" in item.keywords for item in items):
        return
    try:
        psycopg.connect(get_settings().database_url, connect_timeout=2).close()
    except psycopg.OperationalError:
        skip = pytest.mark.skip(reason="Postgres not reachable (docker compose up db)")
        for item in items:
            if "db" in item.keywords:
                item.add_marker(skip)
