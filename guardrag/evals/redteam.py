"""Red-team tally: python -m guardrag.evals.redteam PROMPTFOO_RESULTS [--help]

Reads the JSON that `promptfoo eval -o` writes for the suite in evals/redteam/ (#13) and counts
the Red-Team Attacks GuardRAG blocked, overall and per category. Whether one attack was blocked is
decided by evals/redteam/checks.js; this only tallies, writes the results file and gates.

An attack whose request failed (a 500 when Groq is rate limited, the API down) was never answered,
so it is neither blocked nor through: any such error fails the run instead of being counted.
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

RESULTS_PATH = Path(__file__).parents[2] / "evals" / "results" / "redteam.json"


@dataclass(frozen=True)
class AttackOutcome:
    id: str
    category: str
    language: str
    blocked: bool
    # Why checks.js passed or failed it, or the request's error when there was no reply.
    reason: str
    question: str
    reply: object = None
    errored: bool = False


@dataclass(frozen=True)
class Tally:
    blocked: int
    total: int


def load_outcomes(path: Path) -> list[AttackOutcome]:
    results = json.loads(path.read_text(encoding="utf-8"))["results"]["results"]
    return [_outcome(r) for r in results]


def _outcome(result: dict) -> AttackOutcome:
    test = result["testCase"]
    grading = result.get("gradingResult")
    response = result.get("response") or {}
    common = {
        "id": test["description"],
        "category": test["metadata"]["category"],
        "language": test["metadata"]["language"],
        "question": test["vars"]["question"],
    }
    if grading is None:
        status = ((response.get("metadata") or {}).get("http") or {}).get("status")
        error = result.get("error") or "no reply"
        reason = f"HTTP {status}: {error}" if status else error
        return AttackOutcome(**common, blocked=False, reason=reason, errored=True)
    # The check's own reason, not promptfoo's "All assertions passed".
    reasons = [c.get("reason", "") for c in grading.get("componentResults") or []]
    return AttackOutcome(
        **common,
        blocked=bool(grading["pass"]),
        reason="; ".join(r for r in reasons if r) or grading.get("reason", ""),
        reply=response.get("output"),
    )


def tally(outcomes: Sequence[AttackOutcome], key: str) -> dict[str, Tally]:
    counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    for o in outcomes:
        counts[getattr(o, key)][0] += o.blocked
        counts[getattr(o, key)][1] += 1
    return {k: Tally(*v) for k, v in sorted(counts.items())}


def report(outcomes: Sequence[AttackOutcome]) -> dict:
    through = [o for o in outcomes if not o.blocked and not o.errored]
    return {
        "blocked": sum(o.blocked for o in outcomes),
        "total": len(outcomes),
        "by_category": {k: asdict(v) for k, v in tally(outcomes, "category").items()},
        "by_language": {k: asdict(v) for k, v in tally(outcomes, "language").items()},
        "got_through": [
            {k: getattr(o, k) for k in ("id", "category", "question", "reason", "reply")}
            for o in through
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("promptfoo_results", type=Path, help="the JSON from `promptfoo eval -o`")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH, help="results JSON to write")
    parser.add_argument("--min-blocked", type=int, help="fail if fewer attacks were blocked")
    args = parser.parse_args(argv)

    outcomes = load_outcomes(args.promptfoo_results)
    errored = [o for o in outcomes if o.errored]
    if errored:
        for o in errored:
            print(f"no reply to {o.id}: {o.reason}", file=sys.stderr)
        print(f"FAIL: {len(errored)} of {len(outcomes)} attacks got no reply", file=sys.stderr)
        return 2

    summary = report(outcomes)
    _print_summary(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    if args.min_blocked is not None and summary["blocked"] < args.min_blocked:
        print(
            f"FAIL: {summary['blocked']}/{summary['total']} blocked; the gate needs"
            f" {args.min_blocked}",
            file=sys.stderr,
        )
        return 1
    return 0


def _print_summary(summary: dict) -> None:
    print(f"blocked {summary['blocked']}/{summary['total']}")
    for name, t in summary["by_category"].items():
        print(f"  {name:<10} {t['blocked']}/{t['total']}")
    for o in summary["got_through"]:
        print(f"got through: {o['id']} ({o['reason']})")


if __name__ == "__main__":
    sys.exit(main())
