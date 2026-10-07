"""Answer eval over the Golden Set: python -m guardrag.evals.answers [--help]

Each question goes through `POST /ask` with the production pipeline: guards, retrieval, the answer
model, grounding and redaction. A judge model then scores every answer that isn't a Refusal
against the Passages the answer model was given, using Ragas faithfulness, answer relevancy and
context precision. Refusals are not judged. For an answerable question, a Refusal is listed. For
an Out-of-Corpus Question, an `out_of_corpus` Refusal is the right reply, and the share of them
is the Out-of-Corpus refusal rate.
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol, get_args

from fastapi.testclient import TestClient

from guardrag.answer import AskResponse, Confidence, RefusalKind
from guardrag.api import create_app
from guardrag.config import Settings, get_settings
from guardrag.embedder import FastEmbedEmbedder
from guardrag.evals.clients import retrying_groq
from guardrag.evals.golden import GoldenQuestion, Language, QuestionKind, load_golden_set
from guardrag.guards import InjectionClassifier, PromptGuard
from guardrag.llm import LLM, GroqLLM
from guardrag.prompt import passage_block
from guardrag.retrieval import RetrievedPassage, Retriever, build_retriever

RESULTS_PATH = Path(__file__).parents[2] / "evals" / "results" / "answers.json"
Metric = Literal["faithfulness", "answer_relevancy", "context_precision"]
METRICS: tuple[Metric, ...] = get_args(Metric)
# Golden Set tag for the questions every PR runs; the full set runs nightly.
PR_SUBSET_TAG = "pr"
OUT_OF_CORPUS_TARGET = 0.9
# Refusals and answers the judge couldn't score are left out of the faithfulness mean, so the gate
# also needs a score for at least this share of the answerable questions.
MIN_SCORED_SHARE = 0.5


class Judge(Protocol):
    def score(self, question: str, answer: str, passages: Sequence[str]) -> dict[Metric, float]:
        """Scores in [0, 1] by metric name, NaN for a score the judge could not give."""
        ...


@dataclass(frozen=True)
class Outcome:
    """One Golden Set question, as `/ask` answered it and the judge scored it."""

    id: str
    language: Language
    kind: QuestionKind
    answer: str
    refusal: RefusalKind | None
    confidence: Confidence
    # The Passages as the answer model was shown them; empty when a guard stopped the question.
    passages: list[str]
    scores: dict[Metric, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Summary:
    answerable: int
    judged: int
    faithfulness_scored: int
    # Mean of each metric over the answers it could score.
    means: dict[Metric, float]
    # Ids of judged answers with no score for a metric, by metric.
    unscored: dict[Metric, list[str]]
    refused_answerable: list[str]
    out_of_corpus: int
    out_of_corpus_refused: int
    out_of_corpus_refusal_rate: float | None
    # Ids of Out-of-Corpus Questions that got an answer or an `out_of_scope` decline.
    out_of_corpus_missed: list[str]


class _RecordingRetriever:
    """Keeps the Passages of the latest search, which are what the answer model is given."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.latest_results: list[RetrievedPassage] = []

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        self.latest_results = self.inner.search(query, k)
        return self.latest_results


def evaluate(
    questions: Sequence[GoldenQuestion],
    *,
    retriever: Retriever,
    llm: LLM,
    classifier: InjectionClassifier,
    judge: Judge,
    k: int,
) -> list[Outcome]:
    recorder = _RecordingRetriever(retriever)
    client = TestClient(create_app(retriever=recorder, llm=llm, classifier=classifier, k=k))
    outcomes = []
    for q in questions:
        recorder.latest_results = []
        reply = client.post("/ask", json={"question": q.question})
        reply.raise_for_status()
        response = AskResponse.model_validate(reply.json())
        passages = [passage_block(p) for p in recorder.latest_results]
        scores = {} if response.refusal else judge.score(q.question, response.answer, passages)
        outcomes.append(
            Outcome(
                id=q.id,
                language=q.language,
                kind=q.kind,
                answer=response.answer,
                refusal=response.refusal,
                confidence=response.confidence,
                passages=passages,
                scores=scores,
            )
        )
        # A full run takes a long time under Groq's rate limits, so show where it is.
        shown = ", ".join(f"{m} {v:.2f}" for m, v in scores.items()) or response.refusal
        print(f"[{len(outcomes)}/{len(questions)}] {q.id}: {shown}", file=sys.stderr, flush=True)
    return outcomes


def summarize(outcomes: Sequence[Outcome]) -> Summary:
    judged = [o for o in outcomes if o.refusal is None]
    means, unscored = {}, {}
    for metric in METRICS:
        scored = [o for o in judged if metric in o.scores]
        values = [o.scores[metric] for o in scored if not math.isnan(o.scores[metric])]
        if values:
            means[metric] = sum(values) / len(values)
        missing = [o.id for o in scored if math.isnan(o.scores[metric])]
        if missing:
            unscored[metric] = missing
    ooc = [o for o in outcomes if o.kind == "out_of_corpus"]
    refused = [o for o in ooc if o.refusal == "out_of_corpus"]
    return Summary(
        answerable=sum(o.kind == "answerable" for o in outcomes),
        judged=len(judged),
        faithfulness_scored=sum(
            "faithfulness" in o.scores and not math.isnan(o.scores["faithfulness"]) for o in judged
        ),
        means=means,
        unscored=unscored,
        refused_answerable=[o.id for o in outcomes if o.kind == "answerable" and o.refusal],
        out_of_corpus=len(ooc),
        out_of_corpus_refused=len(refused),
        out_of_corpus_refusal_rate=len(refused) / len(ooc) if ooc else None,
        out_of_corpus_missed=[o.id for o in ooc if o.refusal != "out_of_corpus"],
    )


def passes_gate(summary: Summary, min_faithfulness: float) -> bool:
    """Mean faithfulness at or above `min_faithfulness`, over enough of the questions."""
    enough = summary.faithfulness_scored >= MIN_SCORED_SHARE * summary.answerable
    return enough and summary.means.get("faithfulness", -1.0) >= min_faithfulness


def with_tag(questions: Sequence[GoldenQuestion], tag: str | None) -> list[GoldenQuestion]:
    """The questions carrying `tag`, or all of them for None."""
    return [q for q in questions if tag is None or tag in q.tags]


def _report(
    outcomes: Sequence[Outcome], summary: Summary, settings: Settings, subset: str | None
) -> dict:
    return {
        "answer_model": settings.answer_model,
        "judge_model": settings.judge_model,
        "retriever": settings.retriever,
        "k": settings.retrieval_k,
        "subset": subset or "all",
        "questions": len(outcomes),
        "summary": _rounded(asdict(summary)),
        "outcomes": [
            _rounded({k: v for k, v in asdict(o).items() if k != "passages"}) for o in outcomes
        ],
    }


def _rounded(value):
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    if isinstance(value, dict):
        return {key: _rounded(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def _print_summary(summary: Summary) -> None:
    print(f"judged answers: {summary.judged} (of {summary.answerable} answerable questions)")
    for metric, mean in summary.means.items():
        print(f"{metric:<18} {mean:.3f}")
    for metric, ids in summary.unscored.items():
        print(f"{metric} not scored: {', '.join(ids)}")
    if summary.refused_answerable:
        print(f"answerable but refused: {', '.join(summary.refused_answerable)}")
    rate = summary.out_of_corpus_refusal_rate
    if rate is not None:
        verdict = "meets" if rate >= OUT_OF_CORPUS_TARGET else "below"
        print(
            f"Out-of-Corpus refusal rate {rate:.3f} ({summary.out_of_corpus_refused}"
            f"/{summary.out_of_corpus}), {verdict} the {OUT_OF_CORPUS_TARGET:.0%} target"
        )
        if summary.out_of_corpus_missed:
            print(f"not refused as Out-of-Corpus: {', '.join(summary.out_of_corpus_missed)}")


def main(argv: Sequence[str] | None = None) -> int:
    # Needs the eval extra, which the tests of this module don't.
    from guardrag.evals.judge import RagasJudge

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--subset", choices=[PR_SUBSET_TAG], help="only the questions with this Golden Set tag"
    )
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--out", type=Path, default=RESULTS_PATH, help="results JSON to write")
    parser.add_argument("--min-faithfulness", type=float, help="fail if mean faithfulness is lower")
    args = parser.parse_args(argv)
    if args.min_faithfulness is not None and "faithfulness" not in args.metrics:
        parser.error("--min-faithfulness needs the faithfulness metric")

    settings = get_settings()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set", file=sys.stderr)
        return 2
    questions = with_tag(load_golden_set(), args.subset)
    embedder = FastEmbedEmbedder(settings.embedding_model)
    groq = retrying_groq(settings.groq_api_key)
    outcomes = evaluate(
        questions,
        retriever=build_retriever(settings.retriever, embedder),
        llm=GroqLLM(settings.groq_api_key, settings.answer_model, client=groq),
        classifier=PromptGuard(settings.groq_api_key, settings.prompt_guard_model, client=groq),
        judge=RagasJudge(settings.groq_api_key, embedder, args.metrics, settings.judge_model),
        k=settings.retrieval_k,
    )

    summary = summarize(outcomes)
    _print_summary(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = _report(outcomes, summary, settings, args.subset)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    if args.min_faithfulness is not None and not passes_gate(summary, args.min_faithfulness):
        faithfulness = summary.means.get("faithfulness")
        shown = "no score" if faithfulness is None else f"{faithfulness:.3f}"
        print(
            f"FAIL: faithfulness {shown}, scored on {summary.faithfulness_scored} of"
            f" {summary.answerable} answerable questions; the gate needs >= {args.min_faithfulness}"
            f" on at least {MIN_SCORED_SHARE:.0%} of them",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
