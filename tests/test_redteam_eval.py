"""The red-team tally (#13): counts what evals/redteam/checks.js decided, from promptfoo's JSON.

The check itself is tested in evals/redteam/checks.test.js (node --test).
"""

import json
from pathlib import Path

import pytest
import yaml

from guardrag.evals.redteam import load_outcomes, main, report

ROOT = Path(__file__).parents[1]
ATTACKS = ROOT / "evals" / "redteam" / "attacks.yaml"

OUT_OF_SCOPE = {
    "answer": "Sorry.",
    "citations": [],
    "confidence": "none",
    "refusal": "out_of_scope",
}


def graded(id: str, category: str, passed: bool, reason: str = "", language: str = "en") -> dict:
    """One result as `promptfoo eval -o` writes it, cut down to the fields the tally reads."""
    return {
        "testCase": {
            "description": id,
            "metadata": {"category": category, "language": language},
            "vars": {"question": f"attack {id}"},
        },
        "gradingResult": {
            "pass": passed,
            "reason": "All assertions passed" if passed else reason,
            "componentResults": [{"pass": passed, "reason": reason}],
        },
        "response": {"output": OUT_OF_SCOPE},
        "success": passed,
    }


def errored(id: str, category: str) -> dict:
    return {
        "testCase": {
            "description": id,
            "metadata": {"category": category, "language": "en"},
            "vars": {"question": f"attack {id}"},
        },
        "gradingResult": None,
        "error": "No output",
        "response": {"raw": "Internal Server Error", "metadata": {"http": {"status": 500}}},
        "success": False,
    }


def write(tmp_path: Path, *results: dict) -> Path:
    path = tmp_path / "promptfoo.json"
    path.write_text(json.dumps({"results": {"results": list(results)}}), encoding="utf-8")
    return path


def test_blocked_attacks_are_counted_overall_and_per_category(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        graded("a", "injection", True, "out_of_scope Refusal"),
        graded("b", "injection", False, 'answer contains "banana"'),
        graded("c", "pii", True, "out_of_scope Refusal", language="fil"),
    )

    summary = report(load_outcomes(path))

    assert (summary["blocked"], summary["total"]) == (2, 3)
    assert summary["by_category"] == {
        "injection": {"blocked": 1, "total": 2},
        "pii": {"blocked": 1, "total": 1},
    }
    assert summary["by_language"]["fil"] == {"blocked": 1, "total": 1}


def test_an_attack_that_got_through_is_listed_with_the_checks_reason(tmp_path: Path) -> None:
    path = write(tmp_path, graded("b", "injection", False, 'answer contains "banana"'))

    [through] = report(load_outcomes(path))["got_through"]

    assert through["id"] == "b"
    assert through["reason"] == 'answer contains "banana"'
    assert through["reply"] == OUT_OF_SCOPE


def test_the_gate_fails_below_the_threshold_only(tmp_path: Path) -> None:
    path = write(tmp_path, graded("a", "injection", True), graded("b", "pii", False))
    out = tmp_path / "redteam.json"

    assert main([str(path), "--out", str(out), "--min-blocked", "1"]) == 0
    assert main([str(path), "--out", str(out), "--min-blocked", "2"]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["blocked"] == 1


def test_an_attack_with_no_reply_fails_the_run_instead_of_being_counted(
    tmp_path: Path, capsys
) -> None:
    path = write(tmp_path, graded("a", "injection", True), errored("b", "pii"))
    out = tmp_path / "redteam.json"

    assert main([str(path), "--out", str(out)]) == 2
    assert not out.exists()
    assert "no reply to b: HTTP 500" in capsys.readouterr().err


@pytest.fixture(scope="module")
def attacks() -> list[dict]:
    return yaml.safe_load(ATTACKS.read_text(encoding="utf-8"))


class TestTheSuite:
    """The checked-in attacks meet the shape agreed in #13."""

    def test_it_has_about_30_attacks_with_unique_ids(self, attacks) -> None:
        ids = [a["description"] for a in attacks]
        assert 28 <= len(ids) <= 32
        assert len(set(ids)) == len(ids)

    def test_it_covers_the_four_categories_with_filipino_and_taglish(self, attacks) -> None:
        categories = {a["metadata"]["category"] for a in attacks}
        languages = [a["metadata"]["language"] for a in attacks]
        assert categories == {"injection", "jailbreak", "pii", "off_topic"}
        assert languages.count("fil") + languages.count("taglish") >= 6

    def test_every_attack_has_a_pass_condition_and_one_question(self, attacks) -> None:
        for a in attacks:
            assert a["metadata"]["expect"] in {"out_of_scope", "refusal", "safe"}, a
            # A list in `vars` would make promptfoo run one test per item.
            assert set(a["vars"]) == {"question"}, a
