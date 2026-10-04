import os

import pytest
from fastapi.testclient import TestClient

from guardrag.api import app


@pytest.mark.db
def test_health_reports_passage_count() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    if os.environ.get("CI"):
        # CI starts from a freshly applied schema; a local database may already be seeded.
        assert body["passages"] == 0
