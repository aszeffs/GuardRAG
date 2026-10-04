import os

import pytest
from fastapi.testclient import TestClient

from guardrag.api import app


@pytest.mark.db
def test_health_answers_against_an_empty_database() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # CI starts from a freshly applied schema; a local database may already be seeded.
    assert body["passages"] == 0 if os.environ.get("CI") else body["passages"] >= 0
