"""Personal data (#11): redact PH identifiers and contact details from answers and logs, and spot
requests for someone else's personal data."""

import logging
import re
from collections.abc import Iterable, Mapping

_LABELS = {
    "email": "[redacted email]",
    "phone": "[redacted phone number]",
    "id": "[redacted ID number]",
}

# Not inside a longer word or number, and not the whole part of a decimal ("100000000.50").
_START = r"(?<![\w+.,/-])"
_END = r"(?![\w/]|[.,-]\d)"
_EMAIL = r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
_PHONE = "|".join(
    [
        r"(?:\+63|63|0)[ -]?9\d{2}[ -]?\d{3}[ -]?\d{4}",  # mobile: 0917 123 4567, +639171234567
        r"(?:\(02\)|02|\+63[ -]?2)[ -]?[2-9]\d{3}[ -]?\d{4}",  # Metro Manila: (02) 8123-4567
        r"(?:\(0\d{2,3}\)[ -]?|0\d{2}-)\d{3}[ -]?\d{4}",  # provincial: (032) 123-4567
    ]
)
# Grouped as each agency prints them. Bare digit runs need 9+ digits, past any fee written without
# separators; dates, form numbers and circulars ("RR No. 8-2018") never get there.
_ID = "|".join(
    [
        # TIN, with an optional branch code
        r"\d{3}(?P<sep>[ -])\d{3}(?P=sep)\d{3}(?:(?P=sep)\d{3,5})?",
        r"\d{2}[ -]\d{7}[ -]\d",  # SSS number
        r"\d{2}[ -]\d{9}[ -]\d",  # PhilHealth Identification Number
        r"\d{4}-\d{4}-\d{4}(?:-\d{4})?",  # Pag-IBIG MID, PhilSys Number
        r"\d{9,16}",
    ]
)
_PII = re.compile(
    rf"(?P<email>{_EMAIL})|{_START}(?:(?P<phone>{_PHONE})|(?P<id>{_ID})){_END}",
    re.IGNORECASE,
)


def redact(text: str, keep: Iterable[str] = ()) -> str:
    """`text` with identifiers, phone numbers and emails replaced by a label.

    Values that also appear in `keep` stay, however they are spaced or cased: an answer may repeat
    an Agency's hotline or email from the Passages it was given, but never the asker's own data.
    """
    kept = {_key(m) for k in keep for m in _PII.finditer(k)}
    return _PII.sub(lambda m: m[0] if _key(m) in kept else _label(m), text)


def _label(match: re.Match[str]) -> str:
    return next(label for kind, label in _LABELS.items() if match[kind])


def _key(match: re.Match[str]) -> str:
    if match["email"]:
        return match["email"].lower()
    digits = re.sub(r"\D", "", match[0])
    return "0" + digits[2:] if match["phone"] and digits.startswith("63") else digits


def install_log_redaction() -> None:
    """Redact every log record as it is created, whichever logger or handler it goes to."""
    make_record = logging.getLogRecordFactory()
    if getattr(make_record, "redacts_pii", False):
        return

    def redacting_record(*args, **kwargs) -> logging.LogRecord:
        record = make_record(*args, **kwargs)
        # The arguments are redacted one by one rather than merged into the message, because
        # some formatters unpack them (uvicorn's access log does).
        record.msg = _redacted_arg(record.msg)
        if isinstance(record.args, Mapping):
            record.args = {k: _redacted_arg(v) for k, v in record.args.items()}
        elif record.args:
            record.args = tuple(_redacted_arg(a) for a in record.args)
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return record

    redacting_record.redacts_pii = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(redacting_record)


def _redacted_arg(value: object) -> object:
    """Strings redacted, numbers as they are (for %d), anything else redacted when printed."""
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _Redacted(value)


class _Redacted:
    def __init__(self, value: object) -> None:
        self.value = value

    def __str__(self) -> str:
        return redact(str(self.value))

    def __repr__(self) -> str:
        return redact(repr(self.value))


# What counts as someone's personal data in a question. Case-insensitive; names are not.
_DATA = (
    r"(?i:tin|tax identification number|sss(?: number| no\.?| id)?"
    r"|philhealth(?: number| no\.?| id| pin)?|pag-?ibig(?: mid| number| no\.?)?"
    r"|mid(?: number| no\.?)"
    r"|philsys(?: number| id| card)?|psn|national id(?: number)?"
    r"|(?:mobile|cell(?:phone)?|cp|phone|contact|telephone|landline) (?:number|no\.?|details)"
    r"|number|e-?mail(?: address)?|(?:home )?address|birthday|birthdate|date of birth)"
)
_HONORIFIC = (
    r"(?i:mr|mrs|ms|miss|dr|atty|engr|sir|ma'?am|mayor|kap|kapitan|konsehal|gov|sen|cong|kuya"
    r"|ate|tito|tita|lolo|lola)\.?\s+"
)
_NAME_WORD = r"[A-Z][a-z]+(?![\w-])"
_NAME = (
    rf"(?:{_HONORIFIC})?{_NAME_WORD}(?:\s+(?:(?i:de|del|dela|de la|delos|de los)\s+)?{_NAME_WORD})*"
)
# People who are neither the asker nor someone the asker legitimately acts for (family, employees).
_OTHER = (
    r"(?i:neighbou?rs?|ex(?:-?(?:wife|husband|partner|girlfriend|boyfriend))?|boss|co-?workers?"
    r"|colleagues?|classmates?|friends?|tenants?|landlord|landlady|stranger|someone(?: else)?"
    r"|somebody(?: else)?|another person|other (?:person|people)|crush|debtors?)"
)
_OTHER_FIL = (
    r"(?i:kapitbahay|kaibigan|amo|katrabaho|kaklase|nangungupahan|ibang tao|isang tao|taong ito"
    r"|taong iyan|ex)"
)
_POSSESSIVE = r"(?:'s|’s|'|’)"
_PERSONAL_DATA_REQUEST = re.compile(
    "|".join(
        [
            rf"\b{_DATA} of (?P<name>{_NAME})",
            rf"\b(?P<owner>{_NAME}){_POSSESSIVE}\s+{_DATA}\b",
            rf"\b{_DATA} (?i:ni|nina|kay|kina)\s+\w+",
            rf"\b{_DATA} of (?i:(?:my|our|his|her|their|this|that|a|an|the|some) )?{_OTHER}\b",
            rf"\b{_OTHER}{_POSSESSIVE}\s+{_DATA}\b",
            rf"\b{_DATA} (?i:ng) (?i:(?:aking|aming|isang) )?{_OTHER_FIL}\b",
            rf"(?i:\b(?:who owns|whose|who is the owner of|kanino(?: ba)?(?: ang| yung)?))\s+"
            rf"(?i:(?:the|this|that|ang|yung) )?{_DATA}\b",
        ]
    )
)
# A word in a would-be name that shows it is an office, not a person.
_INSTITUTION = re.compile(
    r"\b(?:City|Hall|Office|Center|Centre|Bureau|Department|Municipal|Municipality|Barangay"
    r"|Hospital|Fund|System|Commission|Authority|Agency|Province|Provincial|Regional|District"
    r"|Revenue|Insurance|Corporation|Bank|School|University|Health|Security|Registry|Embassy)\b"
)
_HOW_TO = re.compile(r"(?i)\b(?:how|paano|saan|where)\b")


def asks_for_personal_data(question: str) -> bool:
    """True if `question` asks for, or about the owner of, someone else's identifier or contact
    details: an Out-of-Scope Request.

    Asking about one's own data, a family member's, an employee's, or an Agency's contacts is
    fine, and so is asking how to get such data, which the answer model then handles.
    """
    if _HOW_TO.search(question):
        return False
    return any(
        not _INSTITUTION.search(m["name"] or m["owner"] or "")
        for m in _PERSONAL_DATA_REQUEST.finditer(question)
    )
