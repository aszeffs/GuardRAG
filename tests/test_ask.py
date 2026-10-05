"""The `/ask` shell (seam 1 in issue #1): request and response shape, and how the dependencies
are wired. Grounding is replaced here so these tests don't wait on #8; tests/test_grounding.py
covers the real one."""

from datetime import date

import pytest
from fakes import FakeClassifier, ScriptedLLM, StaticRetriever
from fastapi.testclient import TestClient
from seed import LOAN_APPLICATION, LOAN_ELIGIBILITY, TIN_FOR_EMPLOYEES, retrieved

from guardrag.answer import AskResponse, Citation, DraftAnswer
from guardrag.api import create_app, create_production_app
from guardrag.config import Settings
from guardrag.llm import ANSWER_MODEL, GroqLLM
from guardrag.retrieval import HybridRetriever

RESULTS = [
    retrieved(LOAN_APPLICATION, 0.9),
    retrieved(LOAN_ELIGIBILITY, 0.7),
    retrieved(TIN_FOR_EMPLOYEES, 0.2),
]
GROUNDED = AskResponse(
    answer="Apply online through the My.SSS member portal.",
    citations=[
        Citation(
            passage_id=202,
            document_title="SSS Salary Loan",
            section="Salary Loan: How to apply",
            page=None,
            url="https://www.sss.gov.ph/test/salary-loan/",
            as_of=date(2026, 10, 1),
        )
    ],
    confidence="high",
    refusal=None,
)


class RecordingGrounder:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, draft, retrieved):
        self.calls.append((draft, list(retrieved)))
        return GROUNDED


@pytest.fixture
def parts():
    draft = DraftAnswer(answer="Use My.SSS.", citations=[202])
    return {
        "retriever": StaticRetriever(RESULTS),
        "llm": ScriptedLLM([draft]),
        "classifier": FakeClassifier(),
        "grounder": RecordingGrounder(),
    }


def ask(parts, question: str, k: int = 2):
    return TestClient(create_app(**parts, k=k)).post("/ask", json={"question": question})


def test_ask_returns_the_grounded_response(parts) -> None:
    response = ask(parts, "How do I apply for an SSS salary loan?")

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Apply online through the My.SSS member portal.",
        "citations": [
            {
                "passage_id": 202,
                "document_title": "SSS Salary Loan",
                "section": "Salary Loan: How to apply",
                "page": None,
                "url": "https://www.sss.gov.ph/test/salary-loan/",
                "as_of": "2026-10-01",
            }
        ],
        "confidence": "high",
        "refusal": None,
    }


def test_the_draft_and_grounding_see_the_top_k_retrieved_passages(parts) -> None:
    ask(parts, "How do I apply for an SSS salary loan?", k=2)

    assert parts["retriever"].queries == [("How do I apply for an SSS salary loan?", 2)]
    assert parts["llm"].calls == [("How do I apply for an SSS salary loan?", RESULTS[:2])]
    [(draft, given)] = parts["grounder"].calls
    assert draft == parts["llm"].drafts[0]
    assert given == RESULTS[:2]


@pytest.mark.parametrize("body", [{}, {"question": ""}, {"question": 42}, {"q": "hi"}])
def test_a_request_without_a_question_is_rejected(parts, body) -> None:
    response = TestClient(create_app(**parts)).post("/ask", json=body)

    assert response.status_code == 422
    assert parts["llm"].calls == []


def test_production_app_wires_groq_and_the_configured_retriever() -> None:
    app = create_production_app(Settings(groq_api_key="gsk_test", retriever="hybrid"))

    assert isinstance(app.state.llm, GroqLLM)
    assert app.state.llm.api_key == "gsk_test"
    assert app.state.llm.model == ANSWER_MODEL == "openai/gpt-oss-120b"
    assert isinstance(app.state.retriever, HybridRetriever)


def test_groq_client_is_built_from_the_api_key() -> None:
    client = GroqLLM("gsk_test")._client

    assert type(client).__module__.startswith("groq")
    assert client.api_key == "gsk_test"
