"""Acceptance tests for grounding (#8), through `POST /ask` with the real `ground`.

Each test scripts the draft the model returns and fixes what was retrieved, then checks the
response a citizen gets. Skipped while `ground` raises NotImplementedError.
"""

import pytest
from fakes import FakeClassifier, ScriptedLLM, StaticRetriever
from fastapi.testclient import TestClient
from seed import (
    CERTIFICATE_REPLACEMENT,
    LOAN_APPLICATION,
    ONE_TIME_TAXPAYER,
    TIN_FOR_EMPLOYEES,
    retrieved,
)

from guardrag.answer import DraftAnswer
from guardrag.api import create_app

pytestmark = pytest.mark.pending("#8")

# What the retriever returns for a BIR question: best first. LOAN_APPLICATION is in the Corpus
# but not among these, so a Citation to it was never given to the model.
BIR_RESULTS = [
    retrieved(TIN_FOR_EMPLOYEES, 0.82),
    retrieved(ONE_TIME_TAXPAYER, 0.74),
    retrieved(CERTIFICATE_REPLACEMENT, 0.31),
]
QUESTION = "What do I need to get a TIN as a new employee?"


def ask(draft: DraftAnswer, results=BIR_RESULTS) -> dict:
    app = create_app(
        retriever=StaticRetriever(results),
        llm=ScriptedLLM([draft]),
        classifier=FakeClassifier(),
        k=len(results),
    )
    response = TestClient(app).post("/ask", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    return response.json()


def cited_ids(body: dict) -> list[int]:
    return [c["passage_id"] for c in body["citations"]]


def test_a_cited_answer_carries_full_citations() -> None:
    body = ask(DraftAnswer(answer="File BIR Form 1902 with one valid ID.", citations=[101]))

    assert body["refusal"] is None
    assert body["confidence"] in ("high", "low")
    assert "BIR Form 1902" in body["answer"]
    assert body["citations"] == [
        {
            "passage_id": 101,
            "document_title": "BIR Revenue District Office Citizen's Charter",
            "section": "Issuance of TIN to Local Employees",
            "page": 12,
            "url": "https://www.bir.gov.ph/test/rdo-citizens-charter.pdf",
            "as_of": "2025-03-01",
        }
    ]


def test_citations_to_passages_not_retrieved_are_dropped() -> None:
    # 999 exists nowhere; 202 is in the Corpus but was not retrieved for this request.
    draft = DraftAnswer(answer="File BIR Form 1902.", citations=[999, 101, LOAN_APPLICATION.id])

    body = ask(draft)

    assert cited_ids(body) == [101]
    assert body["refusal"] is None


def test_each_passage_is_cited_once() -> None:
    body = ask(DraftAnswer(answer="File BIR Form 1902.", citations=[101, 101, 102]))

    assert sorted(cited_ids(body)) == [101, 102]


@pytest.mark.parametrize("citations", [[], [999], [LOAN_APPLICATION.id]], ids=str)
def test_no_valid_citation_becomes_an_out_of_corpus_refusal(citations) -> None:
    body = ask(DraftAnswer(answer="You need a barangay clearance.", citations=citations))

    assert body["refusal"] == "out_of_corpus"
    assert body["confidence"] == "none"
    assert body["citations"] == []
    assert "barangay clearance" not in body["answer"]


def test_an_out_of_corpus_refusal_points_to_the_likely_agency() -> None:
    body = ask(DraftAnswer(answer="", citations=[]))

    assert body["refusal"] == "out_of_corpus"
    assert "BIR" in body["answer"]


def test_nothing_retrieved_is_an_out_of_corpus_refusal() -> None:
    body = ask(DraftAnswer(answer="Probably a form.", citations=[101]), results=[])

    assert body["refusal"] == "out_of_corpus"
    assert body["confidence"] == "none"
    assert body["citations"] == []


def test_an_out_of_scope_draft_is_declined() -> None:
    draft = DraftAnswer(answer="Roses are red...", citations=[101], out_of_scope=True)

    body = ask(draft)

    assert body["refusal"] == "out_of_scope"
    assert body["confidence"] == "none"
    assert body["citations"] == []
    assert body["answer"] and "Roses are red" not in body["answer"]


def test_confidence_follows_the_evidence() -> None:
    """Citing the top retrieved Passages is high; citing only the weakest one is low.

    The exact rules are the author's and are written down in the README; adjust the scores above
    if the rules need a different case to tell the two apart.
    """
    strong = ask(DraftAnswer(answer="File BIR Form 1902.", citations=[101, 102]))
    weak = ask(DraftAnswer(answer="Submit an affidavit of loss.", citations=[103]))

    assert strong["confidence"] == "high"
    assert weak["confidence"] == "low"


DRAFTS = [
    DraftAnswer(answer="a", citations=[101]),
    DraftAnswer(answer="b", citations=[103, 102]),
    DraftAnswer(answer="c", citations=[]),
    DraftAnswer(answer="d", citations=[999, 202]),
    DraftAnswer(answer="e", citations=[101], out_of_scope=True),
    DraftAnswer(answer="f", citations=[], out_of_scope=True),
]


@pytest.mark.parametrize("draft", DRAFTS, ids=lambda d: d.answer)
def test_every_answer_is_cited_and_every_refusal_has_no_confidence(draft) -> None:
    body = ask(draft)

    if body["refusal"] is None:
        assert body["citations"]
        assert body["confidence"] in ("high", "low")
    else:
        assert body["refusal"] in ("out_of_corpus", "out_of_scope")
        assert body["confidence"] == "none"
        assert body["citations"] == []
