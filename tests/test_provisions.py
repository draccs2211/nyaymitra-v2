"""
Exact lookup of user-cited provisions (CLAUDE.md issue 6): orchestrator extraction,
Act-name resolution, ordered section chunks, unresolved reporting, and wiring into
research / analysis / citation guard. No paid APIs, no real index.
Run: pytest tests/test_provisions.py -v
"""
import json

import pytest

from backend.rag import provisions as pv
from backend.utils.citations import research_context
from tests.test_agent_outputs import fake_openai
from tests.test_pipeline import make_state

BNS = "The Bharatiya Nyaya Sanhita, 2023"
RTI = "The Right to Information Act, 2005"
CPA = "The Consumer Protection Act, 2019"


def chunk(act, section, part=None, total=None, text="text"):
    label = f"{act} — Section {section}: T" + (f" (part {part}/{total})" if part else "")
    return {"id": f"s:{act[:5]}:{section}:{part}", "content": f"{label}\n{text}",
            "metadata": {"act": act, "section": section, "doc_type": "statute_section"}}


DOCS = [chunk(BNS, "318", 2, 2, "second"), chunk(BNS, "318", 1, 2, "first"),
        chunk(BNS, "319"), chunk(RTI, "6"), chunk(CPA, "35")]
ACTS = [BNS, RTI, CPA]


class TestResolveAct:
    @pytest.mark.parametrize("name,expected", [
        (BNS, BNS),
        ("Bharatiya Nyaya Sanhita", BNS),               # no "The", no year
        ("bharatiya nyaya sanhita 2023", BNS),
        ("Right to Information Act, 2005", RTI),
        ("The Consumer Protection Act, 1986", None),    # year differs: never the 2019 Act
        ("Factories Act, 1948", None),                  # not in corpus
        ("", None),
    ])
    def test_resolve(self, name, expected):
        assert pv.resolve_act(name, ACTS) == expected

    @pytest.mark.parametrize("raw,expected", [("318", "318"), ("s. 6", "6"), ("Section 498a", "498A"),
                                              ("धारा 85", "85"), (" 2 (1) ", "2(1)")])
    def test_normalize_section(self, raw, expected):
        assert pv.normalize_section(raw) == expected


class TestLookup:
    def test_found_in_part_order_and_unresolved(self):
        found, unresolved = pv.lookup_cited([
            {"act": "Bharatiya Nyaya Sanhita", "section": "318"},
            {"act": BNS, "section": "999"},
            {"act": "Factories Act, 1948", "section": "2"},
            {"act": "", "section": "6"}], DOCS)
        assert [d["content"].split("\n")[1] for d in found] == ["first", "second"]
        assert all(d["source"] == "exact" for d in found)
        assert unresolved == [f"{BNS} s.999", "Factories Act, 1948 s.2", "unknown Act s.6"]

    def test_dedup_and_cap(self):
        found, _ = pv.lookup_cited([{"act": BNS, "section": "319"}] * 3, DOCS)
        assert len(found) == 1
        many = [{"act": BNS, "section": str(n)} for n in range(1, 20)]
        _, unresolved = pv.lookup_cited(many, DOCS)
        assert len(unresolved) == pv.MAX_CITED_PROVISIONS


class TestOrchestratorExtraction:
    def test_cited_provisions_passed_through(self, monkeypatch):
        from backend.agents import orchestrator
        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(json.dumps({
            "legal_domain": "criminal",
            "cited_provisions": [{"act": BNS, "section": "318"}, "junk", {"act": BNS}]})))
        out = orchestrator.orchestrator_agent(make_state(user_query="kya BNS 318 lagegi?"))
        assert out["cited_provisions"] == [{"act": BNS, "section": "318"}]

    def test_none_cited(self, monkeypatch):
        from backend.agents import orchestrator
        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(json.dumps({"legal_domain": "civil"})))
        assert orchestrator.orchestrator_agent(make_state(user_query="q"))["cited_provisions"] == []


class TestWiring:
    @pytest.fixture
    def research(self, monkeypatch):
        from backend.agents import research
        monkeypatch.setattr(research.hybrid_rag, "search", lambda *a, **kw: [])
        monkeypatch.setattr(research.hybrid_rag, "corpus_docs", lambda: DOCS)
        return research

    def test_research_returns_exact_text(self, research):
        out = research.research_agent(make_state(
            user_query="BNS 318 and BNS 999", cited_provisions=[
                {"act": BNS, "section": "318"}, {"act": BNS, "section": "999"}]))
        assert [d["metadata"]["section"] for d in out["cited_sections"]] == ["318", "318"]
        assert out["unresolved_provisions"] == [f"{BNS} s.999"]

    def test_research_without_citations(self, research):
        out = research.research_agent(make_state(user_query="deposit"))
        assert out["cited_sections"] == [] and out["unresolved_provisions"] == []

    def test_analysis_prompt_and_citation_guard(self):
        from backend.agents.analysis import _build_prompt
        cited = [chunk(BNS, "318", i, 8, f"part{i}") for i in range(1, 9)] + [chunk(RTI, "6")]
        state = make_state(user_query="q", cited_sections=cited,
                           unresolved_provisions=["Factories Act, 1948 s.2"])
        prompt = _build_prompt(state)
        assert "Provisions cited by the user" in prompt
        assert "part1" in prompt and "part2" in prompt and "part3" not in prompt   # 2 chunks per provision
        assert f"{RTI} — Section 6" in prompt                                     # 2nd provision not crowded out
        assert "NOT in the corpus" in prompt and "Factories Act, 1948 s.2" in prompt
        assert "Section 318" in research_context(state)

    def test_state_law_cited_is_labelled(self):
        from backend.agents.analysis import _build_prompt
        up = chunk("The Uttar Pradesh Urban Buildings (Regulation of Letting, Rent and Eviction) Act, 1972", "21")
        up["metadata"]["jurisdiction"] = "IN-UP"
        prompt = _build_prompt(make_state(user_query="q", jurisdiction="IN-DL", cited_sections=[up]))
        assert "[applies only in IN-UP]" in prompt
