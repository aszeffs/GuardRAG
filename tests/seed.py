"""A small fixed Corpus for tests: two Agencies, six Passages with fixed ids.

The same Passages back the seeded test database (for `Retriever.search`) and the static retriever
the `/ask` tests use, so a scripted draft can cite them by id either way.
"""

from dataclasses import dataclass
from datetime import date

import psycopg

from guardrag.domain import DocumentKind
from guardrag.embedder import Embedder
from guardrag.retrieval import RetrievedPassage


@dataclass(frozen=True)
class SeedDocument:
    id: int
    url: str
    title: str
    agency: str
    kind: DocumentKind
    fetched_at: date
    effective_date: date | None = None

    @property
    def as_of(self) -> date:
        return self.effective_date or self.fetched_at


@dataclass(frozen=True)
class SeedPassage:
    id: int
    document: SeedDocument
    ordinal: int
    section: str
    page: int | None
    text: str


BIR_CHARTER = SeedDocument(
    id=1,
    url="https://www.bir.gov.ph/test/rdo-citizens-charter.pdf",
    title="BIR Revenue District Office Citizen's Charter",
    agency="BIR",
    kind="service",
    fetched_at=date(2026, 10, 1),
    effective_date=date(2025, 3, 1),
)
SSS_SALARY_LOAN = SeedDocument(
    id=2,
    url="https://www.sss.gov.ph/test/salary-loan/",
    title="SSS Salary Loan",
    agency="SSS",
    kind="service",
    fetched_at=date(2026, 10, 1),
)

TIN_FOR_EMPLOYEES = SeedPassage(
    101,
    BIR_CHARTER,
    0,
    "Issuance of TIN to Local Employees",
    12,
    "Requirements: accomplished BIR Form 1902 and one valid government-issued ID showing the "
    "employee's name, address and birthdate. Submit to the RDO that has jurisdiction over the "
    "employer. Processing time: one day. No fee is charged.",
)
ONE_TIME_TAXPAYER = SeedPassage(
    102,
    BIR_CHARTER,
    1,
    "Registration of One-Time Taxpayers",
    15,
    "A person without a TIN who sells real property or pays estate or donor's tax files BIR Form "
    "1904 together with a copy of the deed of sale or the death certificate. Processing time: one "
    "day.",
)
CERTIFICATE_REPLACEMENT = SeedPassage(
    103,
    BIR_CHARTER,
    2,
    "Replacement of Lost Certificate of Registration",
    18,
    "To replace a lost Certificate of Registration, submit an affidavit of loss and pay the "
    "certification fee of ₱100 at an authorized agent bank. The replacement is released within "
    "three working days.",
)
LOAN_ELIGIBILITY = SeedPassage(
    201,
    SSS_SALARY_LOAN,
    0,
    "Salary Loan: Who may apply",
    None,
    "An employed, self-employed or voluntary member may apply for a salary loan after 36 posted "
    "monthly contributions, six of which were posted in the 12 months before the application.",
)
LOAN_APPLICATION = SeedPassage(
    202,
    SSS_SALARY_LOAN,
    1,
    "Salary Loan: How to apply",
    None,
    "Apply online through the My.SSS member portal. The loan proceeds are credited to the "
    "member's enrolled disbursement account, such as a bank account or e-wallet.",
)
LOAN_INTEREST = SeedPassage(
    203,
    SSS_SALARY_LOAN,
    2,
    "Salary Loan: Interest and penalty",
    None,
    "The salary loan earns interest of 10% per year until fully paid. A penalty of 1% per month "
    "is charged on any amortization not paid on time.",
)

DOCUMENTS = [BIR_CHARTER, SSS_SALARY_LOAN]
PASSAGES = [
    TIN_FOR_EMPLOYEES,
    ONE_TIME_TAXPAYER,
    CERTIFICATE_REPLACEMENT,
    LOAN_ELIGIBILITY,
    LOAN_APPLICATION,
    LOAN_INTEREST,
]


def retrieved(passage: SeedPassage, score: float) -> RetrievedPassage:
    """`passage` as a Retriever would return it."""
    doc = passage.document
    return RetrievedPassage(
        passage_id=passage.id,
        text=passage.text,
        section=passage.section,
        page=passage.page,
        document_title=doc.title,
        agency=doc.agency,
        url=doc.url,
        as_of=doc.as_of,
        score=score,
    )


def seed(conn: psycopg.Connection, embedder: Embedder) -> None:
    """Insert the seed Corpus, with real embeddings, into an empty schema."""
    embeddings = embedder.embed([p.text for p in PASSAGES])
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO documents
                 (id, url, title, agency, kind, effective_date, fetched_at, content_sha256)
               VALUES (%s, %s, %s, %s, %s, %s, %s, 'seed')""",
            [
                (d.id, d.url, d.title, d.agency, d.kind, d.effective_date, d.fetched_at)
                for d in DOCUMENTS
            ],
        )
        cur.executemany(
            """INSERT INTO passages (id, document_id, ordinal, section, page, text, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [
                (p.id, p.document.id, p.ordinal, p.section, p.page, p.text, emb)
                for p, emb in zip(PASSAGES, embeddings, strict=True)
            ],
        )
