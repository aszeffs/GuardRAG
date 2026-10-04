"""The Trivy exception gate: every exception names a finding, says why, and expires."""

from datetime import date
from pathlib import Path

import pytest
from check_trivy_exceptions import MAX_EXCEPTION_DAYS, check, main

TODAY = date(2026, 10, 4)
REASON = "libssl3 in the base image; no fixed build is published yet."


def entry(**fields: str) -> str:
    lines = [f"    {name}: {value}" for name, value in fields.items()]
    lines[0] = "  - " + lines[0].lstrip()
    return "vulnerabilities:\n" + "\n".join(lines) + "\n"


def test_no_exceptions_is_healthy() -> None:
    assert check("", today=TODAY) == []
    assert check("# nothing recorded\nvulnerabilities: []\n", today=TODAY) == []


def test_a_justified_exception_with_a_near_expiry_passes() -> None:
    text = entry(id="CVE-2026-0001", statement=f'"{REASON}"', expired_at="2026-11-11")

    assert check(text, today=TODAY) == []


def test_a_folded_statement_counts_as_a_statement() -> None:
    text = (
        "misconfigurations:\n"
        "  - id: DS-0026\n"
        "    statement: >-\n"
        "      Health is checked by Compose and the CI smoke test,\n"
        "      not by the image itself.\n"
        "    expired_at: 2026-12-01\n"
    )

    assert check(text, today=TODAY) == []


def test_a_bare_id_is_refused() -> None:
    problems = check("vulnerabilities:\n  - id: CVE-2026-0001\n", today=TODAY)

    assert any("statement" in p for p in problems)
    assert any("expired_at" in p for p in problems)


def test_a_token_statement_is_refused() -> None:
    text = entry(id="CVE-2026-0001", statement="wontfix", expired_at="2026-11-11")

    assert any("statement" in p for p in check(text, today=TODAY))


def test_an_expired_exception_is_refused() -> None:
    text = entry(id="CVE-2026-0001", statement=f'"{REASON}"', expired_at="2026-10-03")

    assert any("expired on 2026-10-03" in p for p in check(text, today=TODAY))


def test_an_exception_may_not_outlive_the_review_window() -> None:
    limit = date.fromordinal(TODAY.toordinal() + MAX_EXCEPTION_DAYS).isoformat()
    too_far = date.fromordinal(TODAY.toordinal() + MAX_EXCEPTION_DAYS + 1).isoformat()

    assert check(entry(id="X", statement=f'"{REASON}"', expired_at=limit), today=TODAY) == []
    assert check(entry(id="X", statement=f'"{REASON}"', expired_at=too_far), today=TODAY)


def test_an_impossible_date_is_refused_not_rolled_over() -> None:
    text = entry(id="CVE-2026-0001", statement=f'"{REASON}"', expired_at="2026-13-40")

    assert check(text, today=TODAY)


def test_a_date_that_is_not_iso_is_refused() -> None:
    text = entry(id="CVE-2026-0001", statement=f'"{REASON}"', expired_at='"next year"')

    assert any("YYYY-MM-DD" in p for p in check(text, today=TODAY))


def test_unknown_sections_and_fields_are_refused() -> None:
    assert check("vulns:\n  - id: CVE-2026-0001\n", today=TODAY)
    text = entry(id="X", statement=f'"{REASON}"', expired_at="2026-11-11", expires="2026-11-11")
    assert any("expires" in p for p in check(text, today=TODAY))


def test_a_field_given_twice_is_refused() -> None:
    # YAML keeps the last value, which would quietly replace a reviewed expiry.
    text = entry(id="X", statement=f'"{REASON}"', expired_at="2026-11-11") + (
        "    expired_at: 2027-01-01\n"
    )

    assert any("twice" in p for p in check(text, today=TODAY))


def test_cli_gates_on_problems(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("vulnerabilities: []\n", encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text("vulnerabilities:\n  - id: CVE-2026-0001\n", encoding="utf-8")

    assert main(["check", str(good)]) == 0
    assert main(["check", str(bad)]) == 1
    assert "CVE-2026-0001" in capsys.readouterr().out


def test_the_repository_exceptions_pass(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check"]) == 0, capsys.readouterr().out


def test_an_unreadable_key_is_reported_not_raised() -> None:
    assert check("vulnerabilities:\n  - ? [a]\n    : b\n", today=TODAY)
