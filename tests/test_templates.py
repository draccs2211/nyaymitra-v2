"""
Document templates (CLAUDE.md issue 3): one source of truth
(data/templates/document_templates.json), served by the templates MCP server, used by
the drafting agent with a direct-file fallback. No paid APIs.
Run: pytest tests/test_templates.py -v
"""
import pytest

from backend.agents import drafting
from backend.mcp_client import MCPCallError, call_tool, run_sync
from backend.templates import load_templates, placeholders

DOC_TYPES = ["legal_notice", "rti_application", "consumer_complaint", "police_complaint", "vakalatnama"]


class TestTemplateFile:
    def test_all_document_types_present(self):
        assert sorted(load_templates()) == sorted(DOC_TYPES)

    @pytest.mark.parametrize("doc_type", DOC_TYPES)
    def test_every_placeholder_is_fillable(self, doc_type):
        """The deterministic fallback must be able to fill (or mark) every {placeholder}."""
        unknown = set(placeholders(load_templates()[doc_type])) - set(drafting.PLACEHOLDERS) - {"date"}
        assert not unknown

    def test_drafting_type_list_matches(self):
        assert set(DOC_TYPES) | {"other"} == set(drafting.REQUIRED_FIELDS)


class TestTemplatesServer:
    def test_get_template_serves_the_file(self):
        data = run_sync(call_tool("templates", "get_template", {"doc_type": "rti_application"}))
        tmpl = load_templates()["rti_application"]
        assert data["template_english"] == tmpl["english"] and data["template_hindi"] == tmpl["hindi"]
        assert data["required_fields"] == placeholders(tmpl)

    def test_unknown_type(self):
        data = run_sync(call_tool("templates", "get_template", {"doc_type": "will"}))
        assert data["status"] == "error" and "legal_notice" in data["available_templates"]

    def test_list_templates(self):
        data = run_sync(call_tool("templates", "list_templates", {}))
        assert data["total"] == len(DOC_TYPES)


class TestDraftingUsesServer:
    def test_fetch_via_mcp(self, monkeypatch):
        calls = []
        real = drafting.call_tool

        async def spy(server, tool, args=None, **kw):
            calls.append((server, tool))
            return await real(server, tool, args, **kw)
        monkeypatch.setattr(drafting, "call_tool", spy)
        tmpl = drafting.fetch_template("legal_notice")
        assert calls == [("templates", "get_template")]
        assert tmpl["english"] == load_templates()["legal_notice"]["english"]

    def test_falls_back_to_file_when_server_fails(self, monkeypatch):
        async def down(*a, **kw):
            raise MCPCallError("templates down")
        monkeypatch.setattr(drafting, "call_tool", down)
        assert drafting.fetch_template("vakalatnama")["hindi"] == load_templates()["vakalatnama"]["hindi"]

    def test_no_template_for_other(self):
        assert drafting.fetch_template("other") is None
