"""Extraction fixtures: small stand-ins for an agency page, a lawphil statute and a charter PDF."""

from pathlib import Path

from guardrag.ingest.extract import extract_html, extract_pdf

SERVICE_PAGE = """<!doctype html>
<html><head><title>Salary Loan | SSS</title></head>
<body>
<nav><a href="/">Home</a> <a href="/loans">Loans</a></nav>
<article>
<h1>Salary Loan</h1>
<p>The SSS Salary Loan is a cash loan granted to an employed, currently paying
self-employed or voluntary member who meets the qualifying conditions below.</p>
<h2>Qualifying Conditions</h2>
<p>The member must have at least thirty-six (36) posted monthly contributions, six (6)
of which should be posted within the last twelve (12) months prior to the month of
filing of the application.</p>
<h2>How to Apply</h2>
<p>Log in to your My.SSS account, select the Loans tab, choose Salary Loan, and follow
the instructions. Proceeds are credited to your enrolled disbursement account.</p>
</article>
<footer>Copyright Social Security System</footer>
</body></html>
"""

STATUTE_PAGE = """<!doctype html>
<html><head><title>Republic Act No. 99999</title></head>
<body><article><table><tr><td>
<p><b>REPUBLIC ACT NO. 99999</b></p>
<p>AN ACT GRANTING EXAMPLE BENEFITS TO CITIZENS AND FOR OTHER PURPOSES</p>
<p>Be it enacted by the Senate and House of Representatives of the Philippines in
Congress assembled: Section 1. Short Title. - This Act shall be known as the "Example
Benefits Act". Section 2. Declaration of Policy. - It is the policy of the State to
provide example benefits to every citizen, as provided in Section 22 of Republic Act
No. 11032. Sec. 3. Coverage. - This Act covers all citizens residing in the Philippines
who have registered with the proper Agency.</p>
<p>Approved: January 1, 2020</p>
</td></tr></table></article></body></html>
"""

CHARTER_LINES = [
    "BUREAU OF INTERNAL REVENUE CITIZEN'S CHARTER",
    "1. Issuance of Taxpayer Identification Number (TIN)",
    "Office or Division: Revenue District Office",
    "Classification: Simple",
    "Who may avail: Individuals earning purely compensation income",
    "CHECKLIST OF REQUIREMENTS WHERE TO SECURE",
    "BIR Form 1902 Revenue District Office",
    "2. Request for Certificate of Registration",
    "Office or Division: Revenue District Office",
    "Classification: Complex",
    "Who may avail: Registered taxpayers",
]


def test_html_page_splits_at_headings(tmp_path: Path) -> None:
    page = tmp_path / "salary-loan.html"
    page.write_text(SERVICE_PAGE, encoding="utf-8")

    sections = extract_html(page)

    by_heading = {s.heading: s.text for s in sections}
    assert "Qualifying Conditions" in by_heading
    assert "How to Apply" in by_heading
    assert "thirty-six (36) posted monthly contributions" in by_heading["Qualifying Conditions"]
    assert "My.SSS account" in by_heading["How to Apply"]
    assert "My.SSS" not in by_heading["Qualifying Conditions"]


def test_statute_page_splits_at_provisions(tmp_path: Path) -> None:
    page = tmp_path / "ra99999.html"
    page.write_text(STATUTE_PAGE, encoding="utf-8")

    sections = extract_html(page)

    headings = [s.heading for s in sections]
    assert headings.count("Section 1") == 1
    assert headings.count("Section 2") == 1
    assert headings.count("Sec. 3") == 1
    by_heading = {s.heading: s.text for s in sections}
    assert by_heading["Section 1"].startswith("Section 1. Short Title.")
    # A cross-reference ("Section 22 of ...") stays inside the provision that mentions it.
    assert "Section 22 of Republic Act" in by_heading["Section 2"]
    assert "Section 22" not in headings


def test_charter_pdf_picks_up_service_titles(tmp_path: Path) -> None:
    pdf = tmp_path / "charter.pdf"
    pdf.write_bytes(_make_pdf(CHARTER_LINES))

    sections = extract_pdf(pdf)

    by_heading = {s.heading: s for s in sections}
    tin = by_heading["1. Issuance of Taxpayer Identification Number (TIN)"]
    registration = by_heading["2. Request for Certificate of Registration"]
    assert "BIR Form 1902" in tin.text
    assert "Classification: Complex" in registration.text
    assert "BIR Form 1902" not in registration.text
    assert tin.page == registration.page == 1


def _make_pdf(lines: list[str]) -> bytes:
    """A one-page PDF with each line of text set in Helvetica, top to bottom."""
    text = "\n".join(
        f"1 0 0 1 50 {780 - 16 * i} Tm ({_pdf_escape(line)}) Tj" for i, line in enumerate(lines)
    )
    stream = f"BT /F1 10 Tf\n{text}\nET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (n, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
