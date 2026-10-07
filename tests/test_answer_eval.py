"""The answer eval (#12): Golden Set questions through `POST /ask`, answers scored by a judge.

The pipeline runs on fakes and the judge is scripted, so these tests make no model calls. Ragas
itself sits behind the `Judge` protocol and is exercised by the CI job, not here.
"""

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import httpx
import pytest
from fakes import FakeClassifier, ScriptedLLM, ScriptedRetriever
from seed import LOAN_ELIGIBILITY, LOAN_INTEREST, TIN_FOR_EMPLOYEES, SeedPassage, retrieved

from guardrag.answer import DraftAnswer
from guardrag.evals.answers import PR_SUBSET_TAG, evaluate, passes_gate, summarize, with_tag
from guardrag.evals.clients import retrying_groq
from guardrag.evals.golden import GoldenQuestion, Location, Target, load_golden_set
from guardrag.llm import GroqLLM
from guardrag.prompt import passage_block


@dataclass
class FakeJudge:
    """Returns preset scores per answer and records what it was asked to judge."""

    scores: dict[str, dict[str, float]]
    calls: list[tuple[str, str, list[str]]] = field(default_factory=list)

    def score(self, question: str, answer: str, passages: Sequence[str]) -> dict[str, float]:
        self.calls.append((question, answer, list(passages)))
        return self.scores[answer]


def answerable(id: str, passage: SeedPassage, language: str = "en") -> GoldenQuestion:
    return GoldenQuestion(
        id=id,
        question=f"question {id}",
        language=language,
        kind="answerable",
        expected=(Target((Location(passage.document.title, passage.section),)),),
    )


def out_of_corpus(id: str) -> GoldenQuestion:
    return GoldenQuestion(id=id, question=f"question {id}", language="en", kind="out_of_corpus")


def run(questions, rankings, drafts, judge, triggers=()):
    return evaluate(
        questions,
        retriever=ScriptedRetriever(rankings),
        llm=ScriptedLLM(drafts),
        classifier=FakeClassifier(triggers),
        judge=judge,
        k=5,
    )


def test_an_answer_is_judged_against_the_passages_as_the_model_was_shown_them() -> None:
    """With their document and section: a Passage cut mid-sentence means little without them."""
    q = answerable("loan", LOAN_INTEREST)
    passages = [retrieved(LOAN_INTEREST, 0.9), retrieved(LOAN_ELIGIBILITY, 0.5)]
    judge = FakeJudge({"Ten percent a year.": {"faithfulness": 1.0}})

    [outcome] = run(
        [q],
        {q.question: passages},
        [DraftAnswer(answer="Ten percent a year.", citations=[LOAN_INTEREST.id])],
        judge,
    )

    assert judge.calls == [
        (q.question, "Ten percent a year.", [passage_block(p) for p in passages])
    ]
    assert (outcome.refusal, outcome.confidence, outcome.scores) == (
        None,
        "high",
        {"faithfulness": 1.0},
    )


def test_a_refusal_is_not_judged() -> None:
    q = answerable("loan", LOAN_INTEREST)
    judge = FakeJudge({})

    [outcome] = run(
        [q], {q.question: [retrieved(LOAN_INTEREST, 0.9)]}, [DraftAnswer(answer="?")], judge
    )

    assert judge.calls == []
    assert (outcome.refusal, outcome.scores) == ("out_of_corpus", {})


def test_a_question_blocked_before_retrieval_is_not_given_the_previous_passages() -> None:
    first, blocked = (
        answerable("first", LOAN_INTEREST),
        answerable("ignore previous", LOAN_INTEREST),
    )
    judge = FakeJudge({"Ten percent.": {"faithfulness": 1.0}})

    outcomes = run(
        [first, blocked],
        {first.question: [retrieved(LOAN_INTEREST, 0.9)]},
        [DraftAnswer(answer="Ten percent.", citations=[LOAN_INTEREST.id])],
        judge,
        triggers=("ignore previous",),
    )

    assert [o.refusal for o in outcomes] == [None, "out_of_scope"]
    assert outcomes[1].passages == []


def test_the_summary_averages_each_metric_over_the_judged_answers() -> None:
    a, b = answerable("a", LOAN_INTEREST), answerable("b", TIN_FOR_EMPLOYEES, language="fil")
    refused = answerable("refused", LOAN_ELIGIBILITY)
    judge = FakeJudge(
        {
            "A.": {"faithfulness": 1.0, "answer_relevancy": 0.8},
            "B.": {"faithfulness": 0.5, "answer_relevancy": 0.6},
        }
    )

    summary = summarize(
        run(
            [a, b, refused],
            {
                a.question: [retrieved(LOAN_INTEREST, 0.9)],
                b.question: [retrieved(TIN_FOR_EMPLOYEES, 0.9)],
            },
            [
                DraftAnswer(answer="A.", citations=[LOAN_INTEREST.id]),
                DraftAnswer(answer="B.", citations=[TIN_FOR_EMPLOYEES.id]),
                DraftAnswer(answer="Not sure."),
            ],
            judge,
        )
    )

    assert summary.judged == 2
    assert summary.means == {"faithfulness": 0.75, "answer_relevancy": pytest.approx(0.7)}
    assert summary.refused_answerable == ["refused"]


def test_a_score_the_judge_could_not_give_is_left_out_of_the_mean_and_listed() -> None:
    a, b = answerable("a", LOAN_INTEREST), answerable("b", TIN_FOR_EMPLOYEES)
    judge = FakeJudge({"A.": {"faithfulness": 0.9}, "B.": {"faithfulness": math.nan}})

    summary = summarize(
        run(
            [a, b],
            {
                a.question: [retrieved(LOAN_INTEREST, 0.9)],
                b.question: [retrieved(TIN_FOR_EMPLOYEES, 0.9)],
            },
            [
                DraftAnswer(answer="A.", citations=[LOAN_INTEREST.id]),
                DraftAnswer(answer="B.", citations=[TIN_FOR_EMPLOYEES.id]),
            ],
            judge,
        )
    )

    assert summary.means == {"faithfulness": 0.9}
    assert summary.unscored == {"faithfulness": ["b"]}


def test_the_out_of_corpus_refusal_rate_counts_only_out_of_corpus_refusals() -> None:
    refused, answered, declined = out_of_corpus("r"), out_of_corpus("a"), out_of_corpus("d")
    judge = FakeJudge({"Go to Pag-IBIG.": {"faithfulness": 0.0}})

    summary = summarize(
        run(
            [refused, answered, declined],
            {answered.question: [retrieved(LOAN_INTEREST, 0.9)]},
            [
                DraftAnswer(answer="I don't know."),
                DraftAnswer(answer="Go to Pag-IBIG.", citations=[LOAN_INTEREST.id]),
                DraftAnswer(answer="No.", out_of_scope=True),
            ],
            judge,
        )
    )

    assert (summary.out_of_corpus, summary.out_of_corpus_refused) == (3, 1)
    assert summary.out_of_corpus_refusal_rate == pytest.approx(1 / 3)
    assert summary.out_of_corpus_missed == ["a", "d"]


def test_with_no_out_of_corpus_questions_there_is_no_refusal_rate() -> None:
    assert summarize([]).out_of_corpus_refusal_rate is None


@pytest.mark.parametrize(
    ("means", "scored", "passes"),
    [
        ({"faithfulness": 0.85}, 6, True),
        ({"faithfulness": 0.849}, 12, False),
        ({"answer_relevancy": 1.0}, 0, False),
        ({"faithfulness": 0.9}, 5, False),
    ],
    ids=["at the threshold", "below it", "nothing judged", "too few of 12 scored"],
)
def test_the_gate_needs_faithfulness_at_the_threshold_over_half_the_questions(
    means, scored, passes
) -> None:
    """Refusals and unscorable answers leave the mean, so few scores must not pass it alone."""
    summary = replace(summarize([]), answerable=12, faithfulness_scored=scored, means=means)

    assert passes_gate(summary, min_faithfulness=0.85) is passes


def test_with_tag_keeps_the_questions_with_a_tag() -> None:
    tagged = GoldenQuestion("t", "q", "en", "out_of_corpus", tags=(PR_SUBSET_TAG,))
    other = GoldenQuestion("o", "q", "en", "out_of_corpus")

    assert with_tag([tagged, other], PR_SUBSET_TAG) == [tagged]
    assert with_tag([tagged, other], None) == [tagged, other]


def test_the_pr_subset_is_12_answerable_questions_with_some_in_filipino_or_taglish() -> None:
    subset = with_tag(load_golden_set(), PR_SUBSET_TAG)

    assert len(subset) == 12
    assert all(q.kind == "answerable" for q in subset)
    assert any(q.language != "en" for q in subset)


@dataclass
class RateLimitedOnce:
    """A Groq endpoint that answers 429 to the first request and succeeds after that."""

    calls: int = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls == 1:
            return httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {}})
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"answer": "Ok."}'},
                    }
                ],
            },
        )


def test_the_eval_survives_a_rate_limited_answer_model_call() -> None:
    endpoint = RateLimitedOnce()
    client = retrying_groq("key", http_client=httpx.Client(transport=httpx.MockTransport(endpoint)))

    draft = GroqLLM("key", client=client).draft("q", [retrieved(LOAN_INTEREST, 0.9)])

    assert draft.answer == "Ok."
    assert endpoint.calls == 2


def test_the_judge_survives_a_rate_limited_call() -> None:
    pytest.importorskip("openai", reason="needs the eval extra: pip install -e '.[eval]'")
    from guardrag.evals.clients import judge_client

    endpoint = RateLimitedOnce()
    client = judge_client(
        "key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint))
    )

    async def ask():
        return await client.chat.completions.create(model="m", messages=[])

    assert asyncio.run(ask()).choices[0].message.content == '{"answer": "Ok."}'
    assert endpoint.calls == 2


def test_the_judge_checks_a_long_answers_claims_a_few_at_a_time() -> None:
    """One verdict per claim, with reasons, outgrows the judge's 1,000-token reply limit."""
    pytest.importorskip("ragas", reason="needs the eval extra: pip install -e '.[eval]'")
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.collections.faithfulness.util import (
        NLIStatementOutput,
        StatementFaithfulnessAnswer,
        StatementGeneratorOutput,
    )

    from guardrag.evals.judge import CLAIMS_PER_REQUEST, BatchedFaithfulness

    claims = [f"claim {i}" for i in range(2 * CLAIMS_PER_REQUEST + 1)]
    batches = []

    class ScriptedJudgeLLM(InstructorBaseRagasLLM):
        def generate(self, prompt: str, model):
            raise NotImplementedError

        async def agenerate(self, prompt: str, model):
            if model is StatementGeneratorOutput:
                return StatementGeneratorOutput(statements=claims)
            batch = [c for c in claims if f'"{c}"' in prompt]
            batches.append(batch)
            return NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(statement=c, reason="", verdict=c != "claim 0")
                    for c in batch
                ]
            )

    metric = BatchedFaithfulness(llm=ScriptedJudgeLLM())
    result = asyncio.run(metric.ascore(user_input="q", response="a", retrieved_contexts=["c"]))

    assert [len(b) for b in batches] == [CLAIMS_PER_REQUEST, CLAIMS_PER_REQUEST, 1]
    assert result.value == pytest.approx(1 - 1 / len(claims))
