"""
MCP wiring tests (CLAUDE.md issue #8): client timeout/retry/fallback, real
in-process servers, and research-agent integration. No network, no paid APIs.
Run: pytest tests/test_mcp.py -v
"""
import asyncio

import pytest
from fastmcp import FastMCP

from backend import mcp_client
from backend.mcp_client import MCPCallError, call_tool, run_sync
from tests.test_pipeline import make_state


# ─── Client: timeout / retry / fallback ──────────────────────────────────────

@pytest.fixture
def flaky_server(monkeypatch):
    srv = FastMCP("flaky")
    calls = {"slow": 0, "boom": 0, "ok": 0}

    @srv.tool()
    async def ok(x: str) -> dict:
        calls["ok"] += 1
        return {"status": "success", "x": x}

    @srv.tool()
    async def slow_then_ok() -> dict:
        calls["slow"] += 1
        if calls["slow"] == 1:
            await asyncio.sleep(2)          # first attempt times out
        return {"status": "success", "attempt": calls["slow"]}

    @srv.tool()
    async def always_slow() -> dict:
        await asyncio.sleep(2)
        return {}

    @srv.tool()
    async def boom() -> dict:
        calls["boom"] += 1
        raise RuntimeError("tool bug")

    real = mcp_client._transport
    monkeypatch.setattr(mcp_client, "_transport", lambda s: srv if s == "flaky" else real(s))
    return calls


class TestMcpClient:
    @pytest.mark.asyncio
    async def test_success(self, flaky_server):
        assert await call_tool("flaky", "ok", {"x": "1"}) == {"status": "success", "x": "1"}

    @pytest.mark.asyncio
    async def test_timeout_then_retry_succeeds(self, flaky_server):
        out = await call_tool("flaky", "slow_then_ok", timeout=0.5, retries=1, backoff=0)
        assert out["attempt"] == 2

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises(self, flaky_server):
        with pytest.raises(MCPCallError, match="2 attempt"):
            await call_tool("flaky", "always_slow", timeout=0.3, retries=1, backoff=0)

    @pytest.mark.asyncio
    async def test_tool_error_not_retried(self, flaky_server):
        with pytest.raises(MCPCallError, match="tool bug"):
            await call_tool("flaky", "boom", retries=3, backoff=0)
        assert flaky_server["boom"] == 1

    @pytest.mark.asyncio
    async def test_unknown_server(self):
        with pytest.raises(MCPCallError, match="Unknown MCP server"):
            await call_tool("nope", "x")

    def test_run_sync_without_loop(self, flaky_server):
        assert run_sync(call_tool("flaky", "ok", {"x": "a"}))["x"] == "a"

    @pytest.mark.asyncio
    async def test_run_sync_inside_running_loop(self, flaky_server):
        # agents are sync but may be called from async code; must not deadlock
        assert run_sync(call_tool("flaky", "ok", {"x": "b"}))["x"] == "b"

    def test_remote_url_override(self, monkeypatch):
        monkeypatch.setenv("MCP_LEGAL_REFERENCE_URL", "http://example.invalid/mcp")
        assert mcp_client._transport("legal_reference") == "http://example.invalid/mcp"


# ─── Real servers in-process (Redis down must not break them) ─────────────────

class TestServersInProcess:
    @pytest.mark.asyncio
    async def test_ipc_lookup(self):
        out = await call_tool("legal_reference", "get_ipc_section", {"section_number": " 498a "})
        assert out["status"] == "found"
        assert "85 BNS" in out["bns_equivalent"]

    @pytest.mark.asyncio
    async def test_ipc_unknown_section(self):
        out = await call_tool("legal_reference", "get_ipc_section", {"section_number": "999"})
        assert out["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_kanoon_not_configured(self):
        out = await call_tool("indiankanoon", "search_case_laws", {"query": "deposit"})
        assert out["status"] == "error" and out["code"] == "not_configured"
        assert out["results"] == []


# ─── Research integration ────────────────────────────────────────────────────

RAG = [
    {"id": "ipc_section:a", "content": "Section 406 IPC / Section 316 BNS - Criminal breach of trust",
     "metadata": {"domain": "criminal", "doc_type": "ipc_section",
                  "act": "The Indian Penal Code, 1860", "section": "406"}},
    {"id": "legal_guidance:b", "content": "Security deposit return guidance",
     "metadata": {"domain": "property", "doc_type": "legal_guidance", "act": "", "section": ""}},
]


@pytest.fixture
def research(monkeypatch):
    from backend.agents import research as mod
    monkeypatch.setattr(mod.hybrid_rag, "search", lambda q, d=None: RAG)
    return mod


class TestSectionExtraction:
    @pytest.mark.parametrize("query,expected", [
        ("Is Section 420 IPC applicable?", ["420", "406"]),
        ("s. 498a of the IPC and 406 IPC", ["498A", "406"]),
        ("IPC 302 vs BNS 101", ["302", "406"]),
        ("धारा 376 IPC", ["376", "406"]),
        ("Section 154 CrPC and Rs 50000 deposit", ["406"]),    # non-IPC / amounts ignored
    ])
    def test_sections(self, research, query, expected):
        assert research.ipc_sections_to_lookup(query, RAG) == expected

    def test_capped(self, research):
        q = " ".join(f"Section {n} IPC" for n in range(300, 310))
        assert len(research.ipc_sections_to_lookup(q, [])) == research.MAX_SECTION_LOOKUPS


class TestResearchAgent:
    def test_real_mcp_servers(self, research):
        out = research.research_agent(make_state(user_query="landlord kept deposit, Section 420 IPC?"))
        assert out["source_status"] == {"indian_kanoon": "not_configured", "legal_reference": "ok"}
        assert out["kanoon_cases"] == []
        ids = [s["id"] for s in out["ipc_sections"]]
        assert ids == ["ipc_section:a", "mcp:legal_reference:ipc:420", "mcp:legal_reference:ipc:406"]
        mapped = out["ipc_sections"][1]
        assert "318 BNS" in mapped["content"] and mapped["source"] == "mcp"

    def test_kanoon_success_mapped(self, research, monkeypatch):
        real = research.call_tool

        async def fake(server, tool, args=None, **kw):
            if server == "indiankanoon":
                return {"status": "success", "results": [{
                    "title": "A v B", "citation": "", "court": "Allahabad High Court",
                    "date": "2021-03-01", "snippet": "deposit", "tid": 123,
                    "url": "https://indiankanoon.org/doc/123", "author": "X J"}]}
            return await real(server, tool, args, **kw)
        monkeypatch.setattr(research, "call_tool", fake)
        out = research.research_agent(make_state(user_query="deposit"))
        assert out["source_status"]["indian_kanoon"] == "ok"
        case = out["kanoon_cases"][0]
        assert case["tid"] == "123" and "author" not in case
        assert "verify" in case["source"].lower()

    def test_all_mcp_down_falls_back_to_rag(self, research, monkeypatch):
        async def down(*a, **kw):
            raise MCPCallError("connection refused")
        monkeypatch.setattr(research, "call_tool", down)
        out = research.research_agent(make_state(user_query="Section 420 IPC"))
        assert out["source_status"] == {"indian_kanoon": "unavailable", "legal_reference": "unavailable"}
        assert [r["id"] for r in out["research_results"]] == ["ipc_section:a", "legal_guidance:b"]
        assert [s["id"] for s in out["ipc_sections"]] == ["ipc_section:a"]   # corpus only

    def test_no_sections_means_legal_reference_skipped(self, research, monkeypatch):
        monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None: RAG[1:])
        out = research.research_agent(make_state(user_query="deposit"))
        assert out["source_status"]["legal_reference"] == "skipped"


class TestVerificationNotesMissingCaseLaw:
    def test_not_configured_noted(self):
        from backend.agents.verification import verification_agent
        out = verification_agent(make_state(final_response="x" * 300,
                                            source_status={"indian_kanoon": "not_configured"}))
        assert any("no case law was searched" in s for s in out["verified_sources"])

    def test_ok_not_noted(self):
        from backend.agents.verification import verification_agent
        out = verification_agent(make_state(final_response="x" * 300,
                                            source_status={"indian_kanoon": "ok"}))
        assert not any("no case law" in s for s in out["verified_sources"])
