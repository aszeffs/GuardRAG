"""Retrieval eval over the Golden Set: python -m guardrag.evals.retrieval [--help]

Scores each Retriever on the answerable questions with recall@k (the share of a question's Targets
found in the top k, averaged) and MRR (1 / rank of the first Passage meeting any Target, 0 if none
is in the top k). Out-of-Corpus questions have nothing to retrieve, so they are not scored here.
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import get_args

from guardrag.config import RetrieverMode, get_settings
from guardrag.db import connect
from guardrag.embedder import FastEmbedEmbedder
from guardrag.evals.golden import GoldenQuestion, load_golden_set
from guardrag.retrieval import Retriever, build_retriever

RESULTS_PATH = Path(__file__).parents[2] / "evals" / "results" / "retrieval.json"


@dataclass(frozen=True)
class Scores:
    questions: int
    recall: float
    mrr: float


@dataclass(frozen=True)
class RetrieverScores:
    overall: Scores
    by_language: dict[str, Scores]
    # Ids of answerable questions with at least one Target missing from the top k.
    misses: list[str] = field(default_factory=list)


def evaluate(questions: Sequence[GoldenQuestion], retriever: Retriever, k: int) -> RetrieverScores:
    per_language: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    misses = []
    for q in questions:
        if q.kind != "answerable":
            continue
        ranked = retriever.search(q.question, k)[:k]
        found = [any(t.met_by(p) for p in ranked) for t in q.expected]
        first = next(
            (rank for rank, p in enumerate(ranked, 1) if any(t.met_by(p) for t in q.expected)),
            None,
        )
        per_language[q.language].append((sum(found) / len(found), 1 / first if first else 0.0))
        if not all(found):
            misses.append(q.id)
    return RetrieverScores(
        overall=_mean([pair for pairs in per_language.values() for pair in pairs]),
        by_language={language: _mean(pairs) for language, pairs in sorted(per_language.items())},
        misses=misses,
    )


def passes_gate(scores: RetrieverScores, min_recall: float) -> bool:
    return scores.overall.recall >= min_recall


def _mean(pairs: list[tuple[float, float]]) -> Scores:
    n = len(pairs)
    return Scores(
        questions=n,
        recall=sum(r for r, _ in pairs) / n if n else 0.0,
        mrr=sum(rr for _, rr in pairs) / n if n else 0.0,
    )


def missing_locations(questions: Sequence[GoldenQuestion], db_url: str | None = None) -> set:
    """Locations no Passage in the Corpus holds: a typo, or an extraction change to fix first."""
    with connect(db_url) as conn:
        rows = conn.execute(
            "SELECT DISTINCT d.title, p.section FROM passages p JOIN documents d"
            " ON d.id = p.document_id"
        ).fetchall()
    sections = set(rows)
    titles = {title for title, _ in rows}
    wanted = {loc for q in questions for t in q.expected for loc in t.locations}
    return {
        loc
        for loc in wanted
        if (loc.section is None and loc.document not in titles)
        or (loc.section is not None and (loc.document, loc.section) not in sections)
    }


def _report(questions: Sequence[GoldenQuestion], k: int, results: dict) -> dict:
    languages = defaultdict(int)
    for q in questions:
        languages[q.language] += 1
    return {
        "k": k,
        "golden_set": {
            "questions": len(questions),
            "answerable": sum(q.kind == "answerable" for q in questions),
            "out_of_corpus": sum(q.kind == "out_of_corpus" for q in questions),
            "by_language": dict(sorted(languages.items())),
        },
        "retrievers": {mode: _rounded(asdict(scores)) for mode, scores in results.items()},
    }


def _rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _rounded(v) for key, v in value.items()}
    return value


def _print_table(results: dict[str, RetrieverScores], k: int) -> None:
    languages = sorted({lang for s in results.values() for lang in s.by_language})
    header = f"{'retriever':<10} {'n':>3} {f'recall@{k}':>9} {'MRR':>6}"
    header += "".join(f" {f'{lang} R@{k}':>11} {f'{lang} MRR':>9}" for lang in languages)
    print(header)
    for mode, s in results.items():
        line = f"{mode:<10} {s.overall.questions:>3} {s.overall.recall:>9.3f} {s.overall.mrr:>6.3f}"
        for lang in languages:
            ls = s.by_language.get(lang)
            line += f" {ls.recall:>11.3f} {ls.mrr:>9.3f}" if ls else f" {'-':>11} {'-':>9}"
        print(line)
    for mode, s in results.items():
        if s.misses:
            print(f"{mode} misses: {', '.join(s.misses)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--retrievers", nargs="+", choices=get_args(RetrieverMode), default=get_args(RetrieverMode)
    )
    parser.add_argument("--out", type=Path, default=RESULTS_PATH, help="results JSON to write")
    parser.add_argument("--gate", choices=get_args(RetrieverMode), help="retriever to gate on")
    parser.add_argument("--min-recall", type=float, help="fail if the gated recall@k is lower")
    args = parser.parse_args(argv)
    if (args.gate is None) != (args.min_recall is None):
        parser.error("--gate and --min-recall go together")

    questions = load_golden_set()
    missing = missing_locations(questions)
    if missing:
        for loc in sorted(missing, key=lambda loc: (loc.document, loc.section or "")):
            print(f"not in the Corpus: {loc.document!r} / {loc.section!r}", file=sys.stderr)
        return 2

    settings = get_settings()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    modes = list(dict.fromkeys([*args.retrievers, *([args.gate] if args.gate else [])]))
    results = {mode: evaluate(questions, build_retriever(mode, embedder), args.k) for mode in modes}

    _print_table(results, args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = _report(questions, args.k, results)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    if args.gate and not passes_gate(results[args.gate], args.min_recall):
        recall = results[args.gate].overall.recall
        fail = f"FAIL: {args.gate} recall@{args.k} {recall:.3f} < {args.min_recall}"
        print(fail, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
