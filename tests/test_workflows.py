"""Supply-chain policy for the workflows and the image, checked on every run.

Actions are pinned to commit SHAs, images to digests, and no workflow grants more
than read access at the top: each job that needs a write scope widens it itself.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

# `owner/repo[/path]@<40-hex sha> # v1.2.3`: the comment is what Dependabot bumps
# with the SHA, and what a reviewer reads instead of the SHA.
PINNED_ACTION = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40} # v\d+(\.\d+)*$")
PINNED_IMAGE = re.compile(r"^\S+:\S+@sha256:[0-9a-f]{64}$")


def uses_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [m.group(1).strip() for m in re.finditer(r"^\s*(?:- )?uses:\s*(.+)$", text, re.M)]


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_security_workflows_exist() -> None:
    names = {path.name for path in WORKFLOWS}

    assert {
        "ci.yml",
        "codeql.yml",
        "secret-scan.yml",
        "dependency-review.yml",
        "container.yml",
    } <= names


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit(path: Path) -> None:
    unpinned = [line for line in uses_lines(path) if not PINNED_ACTION.match(line)]

    assert uses_lines(path), "no `uses:` lines found; the pattern has stopped matching"
    assert unpinned == []


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_level_permissions_are_read_only(path: Path) -> None:
    assert load(path).get("permissions") == {"contents": "read"}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_service_images_are_pinned_by_digest(path: Path) -> None:
    for job in load(path)["jobs"].values():
        for service in job.get("services", {}).values():
            assert PINNED_IMAGE.match(service["image"]), service["image"]


def test_base_image_is_pinned_by_digest() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    bases = re.findall(r"^FROM\s+(\S+)", dockerfile, re.M)

    assert bases
    assert all(PINNED_IMAGE.match(base) for base in bases), bases


def test_only_main_publishes_signs_and_attests() -> None:
    jobs = load(ROOT / ".github" / "workflows" / "container.yml")["jobs"]
    writers = {
        name: job
        for name, job in jobs.items()
        if any(
            job.get("permissions", {}).get(scope) == "write" for scope in ("packages", "id-token")
        )
    }

    assert set(writers) == {"publish", "attest"}
    for job in writers.values():
        assert job["if"] == "github.event_name == 'push' && github.ref == 'refs/heads/main'"


def test_the_answer_eval_skips_every_llm_step_without_the_groq_key() -> None:
    """Fork and Dependabot PRs get no secrets: the job must pass with a notice, not fail.

    Only setup and the key-free checks of the eval extra run without the key.
    """
    job = load(ROOT / ".github" / "workflows" / "answer-eval.yml")["jobs"]["answer-eval"]
    check, *rest = job["steps"]
    ungated = [step for step in rest if "steps.key.outputs.present" not in step.get("if", "")]

    assert check["id"] == "key"
    assert len(ungated) < len(rest)
    assert [step.get("uses", "").split("@")[0] or step["run"] for step in ungated] == [
        "actions/checkout",
        "actions/setup-python",
        'pip install -e ".[dev,eval]"',
        'python -c "import guardrag.evals.judge"',
        "pytest -rs tests/test_answer_eval.py",
    ]
