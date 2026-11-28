"""
Live check of the orchestrator's cited_provisions extraction against real OpenAI
(gpt-4o-mini), 1 call per case (15). Expected pairs are resolved with the same
resolver research uses, against Act names that exist in the corpus.
Run only deliberately: pytest tests/test_orchestrator_live.py -m paid -v
"""
import pytest

from backend.rag.provisions import normalize_section, resolve_act
from tests.test_pipeline import make_state

pytestmark = pytest.mark.paid

BNS = "The Bharatiya Nyaya Sanhita, 2023"
BNSS = "The Bharatiya Nagarik Suraksha Sanhita, 2023"
IPC = "The Indian Penal Code, 1860"
RTI = "The Right to Information Act, 2005"
CRPC = "The Code of Criminal Procedure, 1973"
CPA = "The Consumer Protection Act, 2019"
CORPUS_ACTS = [BNS, BNSS, IPC, RTI, CRPC, CPA]


@pytest.mark.parametrize("query,expected", [
    ("Is BNS 318 attracted if a buyer paid by a cheque that bounced?", {(BNS, "318")}),
    ("kya mere case mein BNS ki dhara 318 lagegi?", {(BNS, "318")}),
    ("पति ने दहेज के लिए मारा, क्या धारा 498A IPC लगेगी?", {(IPC, "498A")}),
    ("RTI ki section 6 ke under application kaise dein?", {(RTI, "6")}),
    ("Police refused FIR. What does Section 173 BNSS say, and is it the same as 154 CrPC?",
     {(BNSS, "173"), (CRPC, "154")}),
    ("Compare IPC 420 and BNS 318 for cheating", {(IPC, "420"), (BNS, "318")}),
    # the 1986 Act is not in the corpus: must stay unresolved, never become the 2019 Act
    ("Complaint under section 12 of the Consumer Protection Act, 1986 for a defective car", {(None, "12")}),
    ("anticipatory bail under section 482 of BNSS, what are the conditions?", {(BNSS, "482")}),
    ("Section 19 RTI appeal deadline?", {(RTI, "19")}),
    ("My landlord kept my deposit of Rs 50000 for 3 months", set()),
    ("boss ne do mahine se salary nahi di", set()),
    ("How do I file for divorce under Hindu law?", set()),
    ("मेरे खिलाफ झूठी FIR हुई है, क्या करूँ?", set()),
    ("I was fired without notice after 5 years of service", set()),
    ("Wife wants maintenance, what are her rights?", set()),
])
def test_cited_provisions(query, expected):
    from backend.agents.orchestrator import orchestrator_agent
    got = orchestrator_agent(make_state(user_query=query))["cited_provisions"]
    pairs = {(resolve_act(p["act"], CORPUS_ACTS), normalize_section(p["section"])) for p in got}
    assert pairs == expected, f"raw extraction: {got}"
