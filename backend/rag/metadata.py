"""
Canonical metadata schema for every document in the legal corpus.

Every doc stored in ChromaDB/BM25 carries exactly these keys (CLAUDE.md issue #3).
Chroma metadata values must be scalars, so "unknown" is "" — never None, and
never a guessed value: leave a field empty rather than invent a court, date or URL.
"""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from backend.config import LEGAL_DOMAINS

DocType = Literal[
    "ipc_section",      # penal provision (IPC with BNS equivalent in content)
    "act",              # summary of a statute
    "legal_guidance",   # procedural / practical guidance
    "helplines",        # contact numbers
    "case_law",         # judgment — requires court, date and source_url
]

_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")          # YYYY, YYYY-MM or YYYY-MM-DD
_JURISDICTION_RE = re.compile(r"^IN(-[A-Z]{2})?$")           # ISO 3166: IN, IN-UP, IN-DL ...


class DocMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    doc_type: DocType
    domain: str
    act: str = ""               # official short title, e.g. "The Consumer Protection Act, 2019"
    section: str = ""           # section/article within `act`, e.g. "498A"
    court: str = ""             # case_law only
    date: str = ""              # judgment / enactment date if known (ISO)
    source_url: str = ""        # authoritative source; "" if none
    source_name: str = ""       # who authored this text, e.g. curated summary vs official
    jurisdiction: str = "IN"    # IN (central) or IN-<state>
    topic: str = ""             # free-text tag for guidance docs

    @field_validator("domain")
    @classmethod
    def _domain(cls, v: str) -> str:
        if v not in LEGAL_DOMAINS:
            raise ValueError(f"domain must be one of {LEGAL_DOMAINS}, got {v!r}")
        return v

    @field_validator("date")
    @classmethod
    def _date(cls, v: str) -> str:
        if v and not _DATE_RE.match(v):
            raise ValueError(f"date must be ISO (YYYY[-MM[-DD]]), got {v!r}")
        return v

    @field_validator("source_url")
    @classmethod
    def _url(cls, v: str) -> str:
        if v and not v.startswith(("https://", "http://")):
            raise ValueError(f"source_url must be http(s), got {v!r}")
        return v

    @field_validator("jurisdiction")
    @classmethod
    def _jurisdiction(cls, v: str) -> str:
        if not _JURISDICTION_RE.match(v):
            raise ValueError(f"jurisdiction must be 'IN' or 'IN-XX', got {v!r}")
        return v

    @model_validator(mode="after")
    def _type_rules(self):
        if self.doc_type == "case_law" and not (self.court and self.date and self.source_url):
            raise ValueError("case_law requires court, date and source_url (never invent citations)")
        if self.doc_type in ("ipc_section", "act") and not self.act:
            raise ValueError(f"{self.doc_type} requires act")
        if self.doc_type == "ipc_section" and not self.section:
            raise ValueError("ipc_section requires section")
        return self

    def to_chroma(self) -> dict:
        return self.model_dump()
