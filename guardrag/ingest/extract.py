"""Turn fetched HTML and PDF files into titled lists of Sections."""

import re
from pathlib import Path

import pdfplumber
import trafilatura

from guardrag.domain import Section

_MD_HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
# Statute provisions on lawphil.net: "Section 1.", "SEC. 2.", "Sec. 14-A.". lawphil wraps an
# act in a single table cell, so provisions can start mid-line. A cross-reference
# ("Section 22 of ...") has no period after the number and is not matched.
_STATUTE_HEADING = re.compile(
    r"(?:^|(?<=\s))\**(SECTION|SEC\.|Section|Sec\.)\s+\d+(-[A-Z])?\.", re.MULTILINE
)
# Citizen's Charter services are introduced by a numbered title a few lines above
# the "Office or Division:" row of the service's info table.
_CHARTER_ANCHOR = re.compile(r"^\s*Office\s*(or|/)\s*Division", re.IGNORECASE)
_NUMBERED_TITLE = re.compile(r"^\s*(\d{1,3}|[A-Z])\.\s+[A-Z]")
_WS = re.compile(r"[ \t]+")


def extract_html(path: Path) -> list[Section]:
    html = path.read_bytes()  # let trafilatura detect the encoding
    markdown = trafilatura.extract(
        html, output_format="markdown", include_tables=True, include_links=False
    )
    return _split_markdown(markdown) if markdown else []


def _split_markdown(markdown: str) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        text = _clean("\n".join(buf))
        if text:
            sections.append(Section(heading=heading, text=text))
        buf.clear()

    for line in markdown.splitlines():
        m = _MD_HEADING.match(line)
        if m:
            flush()
            heading = m.group(1).strip("* ")
        else:
            buf.append(line)
    flush()
    return _split_statute_sections(sections)


def _split_statute_sections(sections: list[Section]) -> list[Section]:
    """Split sections further at statute provisions, using each provision's label as its heading."""
    out: list[Section] = []
    for sec in sections:
        matches = list(_STATUTE_HEADING.finditer(sec.text))
        if not matches:
            out.append(sec)
            continue
        if matches[0].start() > 0:
            out.append(Section(heading=sec.heading, text=sec.text[: matches[0].start()].strip()))
        ends = [m.start() for m in matches[1:]] + [len(sec.text)]
        for m, end in zip(matches, ends, strict=True):
            heading = m.group(0).strip("* ").rstrip(".")
            out.append(Section(heading=heading, text=sec.text[m.start() : end].strip()))
    return [s for s in out if s.text]


def extract_pdf(path: Path) -> list[Section]:
    """One Section per (service, page). Headings come from Citizen's Charter service titles."""
    sections: list[Section] = []
    heading: str | None = None
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            lines = (page.extract_text() or "").splitlines()
            buf: list[str] = []
            titles = _charter_titles(lines)
            i = 0
            while i < len(lines):
                if i in titles:
                    text = _clean("\n".join(buf))
                    if text:
                        sections.append(Section(heading=heading, text=text, page=page_no))
                    buf = []
                    heading, consumed = titles[i]
                    i += consumed
                    continue
                buf.append(lines[i])
                i += 1
            text = _clean("\n".join(buf))
            if text:
                sections.append(Section(heading=heading, text=text, page=page_no))
    return sections


def _charter_titles(lines: list[str]) -> dict[int, tuple[str, int]]:
    """Map line index -> (service title, number of lines the title spans)."""
    titles: dict[int, tuple[str, int]] = {}
    for anchor, line in enumerate(lines):
        if not _CHARTER_ANCHOR.match(line):
            continue
        for start in range(anchor - 1, max(anchor - 8, -1), -1):
            if _NUMBERED_TITLE.match(lines[start]):
                title = lines[start].strip()
                consumed = 1
                nxt = lines[start + 1].strip() if start + 1 < anchor else ""
                if nxt and (title.count("(") > title.count(")") or len(nxt) < 40):
                    title = f"{title} {nxt}"
                    consumed = 2
                titles[start] = (title, consumed)
                break
    return titles


def _clean(text: str) -> str:
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()
