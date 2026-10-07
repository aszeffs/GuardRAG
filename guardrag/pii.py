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
# Grouped as each Agency prints them. Bare digit runs need 9+ digits, past any fee written without
# separators; dates, form numbers and circulars ("RR No. 8-2018") never get there.
_ID = "|".join(
    [
        # TIN, with an optional branch code
        r"\d{3}(?P<sep>[ -])\d{3}(?P=sep)\d{3}(?:(?P=sep)\d{3,5})?",
        r"\d{2}[ -]\d{7}[ -]\d",  # SSS number
        r"\d{2}[ -]\d{9}[ -]\d",  # PhilHealth Identification Number
        r"\d{4}-\d{4}-\d{4}(?:-\d{4})?",  # Pag-IBIG MID, PhilSys Number
        r"\d{4} \d{4} \d{4} \d{4}",  # PhilSys Number as the PhilID card prints it
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

    def replace(m: re.Match[str]) -> str:
        return m[0] if _key(m) in kept or _is_peso_amount(m) else _label(m)

    return _PII.sub(replace, text)


def _is_peso_amount(match: re.Match[str]) -> bool:
    """A bare digit run written as a fee ("Php 123456789") is not an identifier."""
    before = match.string[max(0, match.start() - 4) : match.start()]
    return bool(match["id"]) and re.search(r"(?i)(?:php|₱|\bp) ?$", before) is not None


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
        if any(_is_long_number(a) for a in _values(record.args)):
            # "%d" can't print a label, so the message is formatted here instead.
            record.msg, record.args = redact(record.getMessage()), None
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


def _values(args: object) -> Iterable[object]:
    if isinstance(args, Mapping):
        return args.values()
    return args if isinstance(args, tuple) else ()


def _is_long_number(value: object) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and redact(str(value)) != str(value)
    )


def _redacted_arg(value: object) -> object:
    """Strings redacted, short numbers as they are (for %d), anything else redacted when printed."""
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


# Someone's identifiers and contact details, as a question names them. Case-insensitive; names
# are not.
_ID_FIELD = (
    r"(?i:tin|tax identification number|sss(?: number| no\.?| id)?"
    r"|philhealth(?: number| no\.?| id| pin)?|pag-?ibig(?: mid| number| no\.?)?"
    r"|mid(?: number| no\.?)"
    r"|philsys(?: number| id| card)?|psn|national id(?: number)?"
    r"|(?:mobile|cell(?:phone)?|cp|phone|contact|telephone|landline) (?:number|no\.?|details)"
    r"|e-?mail(?: address)?)"
)
# Also a place's or a famous person's ("the address of Makati", "the birthday of Jose Rizal"), so
# these only count for someone the question marks as a private person.
_PERSONAL_FIELD = rf"(?:{_ID_FIELD}|(?i:number|(?:home )?address|birthday|birthdate|date of birth))"
_HONORIFIC = (
    r"(?i:mr|mrs|ms|miss|dr|atty|engr|sir|ma'?am|mayor|kap|kapitan|konsehal|gov|sen|cong|kuya"
    r"|ate)\.?\s+"
)
_NAME_WORD = r"[A-Z][a-z]+(?![\w-])"
_MORE_NAME = rf"\s+(?:(?i:de|del|dela|de la|delos|de los)\s+)?{_NAME_WORD}"
_FULL_NAME = rf"{_NAME_WORD}(?:{_MORE_NAME})+"  # one capitalised word could be a town or a brand
_TITLED_NAME = rf"{_HONORIFIC}{_NAME_WORD}(?:{_MORE_NAME})*"
# People who are neither the asker nor someone the asker legitimately acts for (family, employees).
# Not a bare "someone": "the TIN of someone who died" is an estate tax question.
_THIRD_PARTY = (
    r"(?i:neighbou?rs?|ex(?:-?(?:wife|husband|partner|girlfriend|boyfriend))?|boss|co-?workers?"
    r"|colleagues?|classmates?|friends?|tenants?|landlord|landlady|stranger|someone else"
    r"|somebody else|another person|other (?:person|people)|crush|debtors?)"
)
_THIRD_PARTY_FIL = (
    r"(?i:kapitbahay|kaibigan|amo|katrabaho|kaklase|nangungupahan|ibang tao|isang tao|taong ito"
    r"|taong iyan|ex)"
)
_FAMILY_FIL = r"(?i:lolo|lola|nanay|tatay|inay|itay|mama|papa|anak|asawa|kapatid|tito|tita)\b"
_POSSESSIVE = r"(?:'s|’s|'|’)"
_PERSONAL_DATA_REQUEST = re.compile(
    "|".join(
        [
            rf"\b{_ID_FIELD} of (?P<name>{_FULL_NAME})",
            rf"\b{_PERSONAL_FIELD} of (?P<titled>{_TITLED_NAME})",
            rf"\b(?P<owner>{_FULL_NAME}){_POSSESSIVE}\s+{_ID_FIELD}\b",
            rf"\b(?P<titled_owner>{_TITLED_NAME}){_POSSESSIVE}\s+{_PERSONAL_FIELD}\b",
            rf"\b{_PERSONAL_FIELD} (?i:ni|nina|kay|kina)\s+(?!{_FAMILY_FIL})\w+",
            rf"\b{_PERSONAL_FIELD} of (?i:(?:my|our|his|her|their|this|that|a|an|the) )?"
            rf"{_THIRD_PARTY}\b",
            rf"\b{_THIRD_PARTY}{_POSSESSIVE}\s+{_PERSONAL_FIELD}\b",
            rf"\b{_PERSONAL_FIELD} (?i:ng) (?i:(?:aking|aming|isang) )?{_THIRD_PARTY_FIL}\b",
            rf"(?i:\b(?:who owns|whose|who is the owner of|kanino(?: ba)?(?: ang| yung)?))\s+"
            rf"(?i:(?:the|this|that|ang|yung) )?{_ID_FIELD}\b",
        ]
    )
)
# A word in a would-be name that shows it is an office, not a person.
_INSTITUTION = re.compile(
    r"\b(?:City|Hall|Office|Center|Centre|Bureau|Department|Municipal|Municipality|Barangay"
    r"|Hospital|Fund|System|Commission|Authority|Agency|Province|Provincial|Regional|District"
    r"|Revenue|Insurance|Corporation|Bank|School|University|Health|Security|Registry|Embassy)\b"
)


def asks_for_personal_data(question: str) -> bool:
    """True if `question` asks for, or about the owner of, someone else's identifier or contact
    details: an Out-of-Scope Request.

    Asking about one's own data, a family member's, an employee's, or an Agency's contacts is
    fine. So is anything the patterns miss, which the answer model is told to decline.
    """
    return any(
        not _INSTITUTION.search(_named(m)) for m in _PERSONAL_DATA_REQUEST.finditer(question)
    )


def _named(match: re.Match[str]) -> str:
    """The would-be person a request names, if it names one."""
    groups = ("name", "titled", "owner", "titled_owner")
    return next((name for g in groups if (name := match[g])), "")
