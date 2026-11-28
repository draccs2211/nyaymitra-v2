"""
Document templates: single source of truth in data/templates/document_templates.json.

Read by the templates MCP server (what the drafting agent calls) and by the drafting
agent directly as a fallback when the MCP call fails. Edit the JSON (after lawyer
review), never a copy in code. Required fields are the template's {placeholders}.
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

TEMPLATES_FILE = Path(__file__).resolve().parents[1] / "data" / "templates" / "document_templates.json"
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


@lru_cache(maxsize=1)
def load_templates() -> Dict[str, dict]:
    with open(TEMPLATES_FILE, encoding="utf-8") as f:
        data = json.load(f)
    for key, t in data.items():
        missing = {"name", "description", "english", "hindi"} - set(t)
        if missing:
            raise ValueError(f"{TEMPLATES_FILE.name}: template {key!r} lacks {sorted(missing)}")
    return data


def get_template(doc_type: str) -> Optional[dict]:
    return load_templates().get(doc_type)


def placeholders(template: dict) -> List[str]:
    """Field names used in a template's English and Hindi text, in first-use order."""
    seen: List[str] = []
    for text in (template["english"], template["hindi"]):
        for name in _PLACEHOLDER.findall(text):
            if name not in seen:
                seen.append(name)
    return seen
