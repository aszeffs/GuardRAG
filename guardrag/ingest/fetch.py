import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

USER_AGENT = "GuardRAG/0.1 (portfolio research project; +https://github.com/aszeffs/GuardRAG)"


@dataclass(frozen=True)
class Fetched:
    url: str
    path: Path
    is_pdf: bool
    fetched_at: date
    sha256: str


def fetch(url: str, raw_dir: Path, *, refresh: bool = False) -> Fetched:
    """Download url into raw_dir once; later runs reuse the cached copy unless refresh is set."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha1(url.encode()).hexdigest()[:16]  # noqa: S324 - cache key, not security
    cached = next(iter(sorted(raw_dir.glob(f"{stem}.*"))), None)

    if cached is None or refresh:
        resp = httpx.get(
            url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=120
        )
        resp.raise_for_status()
        is_pdf = "pdf" in resp.headers.get("content-type", "") or resp.content[:5] == b"%PDF-"
        if cached is not None:
            cached.unlink()
        cached = raw_dir / f"{stem}.{'pdf' if is_pdf else 'html'}"
        cached.write_bytes(resp.content)

    data = cached.read_bytes()
    return Fetched(
        url=url,
        path=cached,
        is_pdf=cached.suffix == ".pdf",
        fetched_at=date.fromtimestamp(cached.stat().st_mtime),
        sha256=hashlib.sha256(data).hexdigest(),
    )
