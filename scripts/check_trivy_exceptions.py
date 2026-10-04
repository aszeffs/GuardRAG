"""Check that every Trivy exception carries a justification and an expiry.

Trivy honours `- id: CVE-2026-0001` on its own. That entry is the gate switched off
for one finding, forever, by someone who left no record of why. The threshold stays
at HIGH and the gate stays on; the price of stepping around a finding is saying why,
and saying when the decision gets revisited.

The file is read strictly: an unknown section or field, or a field given twice, is a
problem rather than something to skip. A check that ignores what it does not
recognise reports success while an unjustified exception sits underneath it.

Usage: python scripts/check_trivy_exceptions.py [path]   (default: .trivyignore.yaml)
"""

import re
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

EXCEPTIONS_FILE = Path(__file__).parents[1] / ".trivyignore.yaml"

# Long enough that an unfixable base-image CVE need not be re-justified every
# sprint, short enough that no exception outlives the reasoning behind it.
MAX_EXCEPTION_DAYS = 180

# Enough to be a reason. "wontfix" and "TODO" are what gets written when the
# field is merely mandatory.
MIN_STATEMENT_LENGTH = 20

SECTIONS = ("vulnerabilities", "misconfigurations", "secrets", "licenses")
FIELDS = ("id", "statement", "expired_at", "paths")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class _StrictLoader(yaml.SafeLoader):
    """A SafeLoader that refuses duplicate keys instead of keeping the last one."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode) -> dict:
    seen: set[object] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"`{key}` is given twice", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def check(text: str, *, today: date) -> list[str]:
    """Return every problem with the exceptions in `text`; empty means the file is fine."""
    try:
        document = yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - a SafeLoader subclass
    except (yaml.YAMLError, ValueError) as error:
        # ValueError: PyYAML turns 2026-13-40 into a date and fails doing it.
        return [f"cannot read the file: {error}"]

    if document is None:
        return []
    if not isinstance(document, dict):
        return ["the file must be a mapping of sections to lists of exceptions."]

    problems = []
    for section, entries in document.items():
        if section not in SECTIONS:
            problems.append(f"`{section}` is not a section Trivy reads (expected: {SECTIONS}).")
            continue
        if entries is None:
            continue
        if not isinstance(entries, list):
            problems.append(f"`{section}` must be a list of exceptions.")
            continue
        for index, exception in enumerate(entries, start=1):
            problems.extend(_inspect(exception, f"{section}[{index}]", today))
    return problems


def _inspect(exception: object, where: str, today: date) -> list[str]:
    if not isinstance(exception, dict):
        return [f"{where}: an exception must be a mapping of fields."]

    problems = [
        f"{where}: `{name}` is not a field Trivy reads here (expected: {FIELDS})."
        for name in exception
        if name not in FIELDS
    ]

    finding = exception.get("id")
    if not isinstance(finding, str) or not finding.strip():
        problems.append(f"{where}: has no `id`, so it names no finding.")
    else:
        where = f"`{finding}`"

    statement = exception.get("statement")
    if not isinstance(statement, str) or len(statement.strip()) < MIN_STATEMENT_LENGTH:
        problems.append(
            f"{where}: needs a `statement` of at least {MIN_STATEMENT_LENGTH} characters "
            "saying why the finding cannot be fixed."
        )

    expiry = _as_date(exception.get("expired_at"))
    if "expired_at" not in exception:
        problems.append(f"{where}: needs an `expired_at` date, so the decision is revisited.")
    elif expiry is None:
        problems.append(f"{where}: `expired_at` must be a YYYY-MM-DD date.")
    elif expiry < today:
        problems.append(
            f"{where}: the exception expired on {expiry.isoformat()}. Remove it if the "
            "finding is gone, or renew it with a fresh justification."
        )
    elif (expiry - today).days > MAX_EXCEPTION_DAYS:
        problems.append(
            f"{where}: `expired_at: {expiry.isoformat()}` is {(expiry - today).days} days out; "
            f"no exception may run longer than {MAX_EXCEPTION_DAYS} days without review."
        )
    return problems


def _as_date(value: object) -> date | None:
    # Unquoted, YAML has already made it a date; quoted, it is still a string. A
    # timestamp with a time of day is not the format asked for.
    if isinstance(value, datetime):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str) and ISO_DATE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def report(name: str, problems: list[str]) -> str:
    lines = ["## Trivy exceptions", "", f"File: `{name}`", ""]
    if problems:
        lines.append("Every exception must name a finding, say why it cannot be fixed, and expire.")
        lines.append("")
        lines += [f"- {problem}" for problem in problems]
        lines += ["", "See `docs/trivy-exceptions.md`."]
    else:
        lines.append("Every recorded exception is justified and expires in time. Ok.")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else EXCEPTIONS_FILE
    # A missing file means no exceptions, which is the healthy state.
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    problems = check(text, today=date.today())
    sys.stdout.write(report(path.name, problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
