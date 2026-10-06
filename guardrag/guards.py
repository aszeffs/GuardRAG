"""Screening that happens before any model call (#10): input limits, injection heuristics, Llama
Prompt Guard, and per-client rate limiting."""

import logging
import math
import re
import threading
import time
import unicodedata
from collections import defaultdict, deque
from collections.abc import Callable
from functools import cached_property
from typing import Any, Protocol

from guardrag.config import PROMPT_GUARD_MODEL

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000
PROMPT_GUARD_THRESHOLD = 0.5  # Prompt Guard's score is the probability of an attack
MAX_TRACKED_CLIENTS = 1000  # past this, the rate limiter forgets clients idle for a window


def check_question(question: str) -> str:
    """Reject a blank question or one holding control characters (other than tab and newlines).

    The length cap is MAX_QUESTION_CHARS, enforced on AskRequest itself.
    """
    if not question.strip():
        raise ValueError("question is blank")
    if any(unicodedata.category(c) == "Cc" and c not in "\t\r\n" for c in question):
        raise ValueError("question contains control characters")
    return question


class InjectionClassifier(Protocol):
    def is_injection(self, text: str) -> bool:
        """True if `text` looks like prompt injection or a jailbreak.

        May raise when the classifier is unavailable; callers then continue on heuristics alone
        and log a warning (ADR 0003, fail open).
        """
        ...


def screen_for_injection(question: str, classifier: InjectionClassifier) -> bool:
    """Heuristics first, then the classifier. A classifier failure fails open (ADR 0003)."""
    if looks_like_injection(question):
        return True
    try:
        return classifier.is_injection(question)
    except Exception as e:
        # The question is not logged: it may hold the asker's personal data (#11).
        logger.warning("Injection classifier unavailable (%r); screening on heuristics only", e)
        return False


# Words that may sit between an "ignore" verb and its object, in English, Filipino and Taglish.
_FILLER = (
    r"(?:all|any|of|the|your|these|those|my|everything|previous|prior|above|earlier|preceding"
    r"|original|initial|system|developer|safety|mo|mong|nyo|ninyo|muna|na|nang|lahat|ng|ang|yung"
    r"|iyong|iyo|mga|sa|naunang|nakaraang|dating|ibinigay|binigay)"
)
_IGNORE = (
    r"(?:ignore|i ?ignore|disregard|i ?disregard|forget|override|bypass|kalimutan|balewalain"
    r"|isantabi|(?:huwag|wag) (?:mo |mong |nyo |nyong )?(?:nang )?(?:pansinin|sundin))"
)
# No "rules" or "guidelines": citizens ask whether an official can ignore those.
_ORDERS = (
    r"(?:instructions?|prompts?|directives?|commands?|utos|tagubilin|panuto|bilin"
    r"|instruksi?yon)"
)

# Matched against _normalise(text): lowercase ASCII words separated by single spaces.
_PATTERNS = [
    rf"\b{_IGNORE} (?:{_FILLER} ){{0,6}}{_ORDERS}\b",
    r"\b(?:ignore|disregard|forget) (?:everything |all )?(?:what )?you (?:were|ve been) told\b",
    r"\b(?:system|developer|hidden) prompt\b",
    r"\b(?:repeat|reveal|print|output|leak|dump) (?:\w+ ){0,2}(?:instructions|prompt)\b",
    r"(?<!\bif )(?<!\bwhen )\b(?:you|u) are now\b",
    r"\bfrom now on you\b",
    r"\bpretend (?:you are|you re|that you)\b",
    r"\bact as (?:an? )?(?:unrestricted|unfiltered|jailbroken|evil|uncensored)\b",
    r"\b(?:developer|god|dan|jailbreak|unrestricted) mode\b",
    r"\bjailbr(?:eak|oken)\b",
    r"\bdo anything now\b",
    r"\bmula ngayon ikaw\b",
    r"\bikaw na ngayon\b",
    r"\bmagpanggap ka(?:ng)? na\b",
    r"\bkunwari (?:ikaw|ka)\b",
]
# Matched against the lowercased text as written: chat-template and role markers.
_RAW_PATTERNS = [
    r"<\|[a-z_]+\|>",
    r"\[/?inst\]",
    r"<</?sys>>",
    r"(?:^|\n)\s*(?:system|assistant|developer)\s*:",
]
_INJECTION = re.compile("|".join(_PATTERNS))
_RAW_INJECTION = re.compile("|".join(_RAW_PATTERNS))


def looks_like_injection(text: str) -> bool:
    """Cheap pattern checks for well-known injection and jailbreak phrasings.

    Prompt Guard catches English attacks well but scores Filipino ones low, so the patterns cover
    English, Filipino and Taglish. They are tuned against the Golden Set to avoid flagging
    ordinary questions.
    """
    folded = unicodedata.normalize("NFKC", _strip_format_chars(text)).lower()
    return bool(_RAW_INJECTION.search(folded) or _INJECTION.search(_normalise(folded)))


def _strip_format_chars(text: str) -> str:
    """Drop zero-width and other invisible format characters used to split trigger words."""
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


# Cyrillic and Greek letters that look like Latin ones, which NFKC leaves alone.
_LOOKALIKES = str.maketrans("асеорхуіјѕԁɡαεικνορτυχ", "aceopxyijsdgaeikvoptux")


def _normalise(text: str) -> str:
    """Fold lookalike letters and accents, and reduce all but letters and digits to spaces."""
    text = text.translate(_LOOKALIKES)
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


class PromptGuard:
    """Llama Prompt Guard on Groq. Its reply is the probability that the text is an attack."""

    def __init__(
        self,
        api_key: str,
        model: str = PROMPT_GUARD_MODEL,
        client: Any = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        if client is not None:
            self._client = client

    @cached_property
    def _client(self) -> Any:
        # Fail fast: a slow or retired classifier must not hold up the request (ADR 0003).
        from groq import Groq

        return Groq(api_key=self.api_key, timeout=5.0, max_retries=0)

    def is_injection(self, text: str) -> bool:
        completion = self._client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": text}]
        )
        reply = completion.choices[0].message.content
        score = float(reply or "")  # ValueError on anything that isn't a number
        if math.isnan(score):
            raise ValueError(f"Prompt Guard returned {reply!r}")
        return score >= PROMPT_GUARD_THRESHOLD


class RateLimiter:
    """At most `limit` requests per client in any `window_seconds`. In memory, one process only."""

    def __init__(
        self, limit: int, window_seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        """Record a request from `client` and say whether it is within the limit.

        Only allowed requests are recorded, so a client that keeps retrying is not locked out
        for longer.
        """
        with self._lock:
            now = self._clock()
            if len(self._hits) > MAX_TRACKED_CLIENTS:
                self._forget_idle_clients(now)
            hits = self._hits[client]
            while hits and hits[0] <= now - self.window_seconds:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True

    def _forget_idle_clients(self, now: float) -> None:
        for client, hits in list(self._hits.items()):
            if not hits or hits[-1] <= now - self.window_seconds:
                del self._hits[client]
