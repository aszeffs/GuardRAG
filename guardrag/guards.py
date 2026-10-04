"""Screening that happens before any model call. Input limits, heuristics, Prompt Guard and rate
limiting arrive with issue #10; this module holds the classifier seam they plug into."""

from typing import Protocol


class InjectionClassifier(Protocol):
    def is_injection(self, text: str) -> bool:
        """True if `text` looks like prompt injection or a jailbreak.

        May raise when the classifier is unavailable; callers then continue on heuristics alone
        and log a warning (ADR 0003, fail open).
        """
        ...


class NoClassifier:
    """Flags nothing. Stands in for Llama Prompt Guard until #10 wires it up."""

    def is_injection(self, text: str) -> bool:
        return False
