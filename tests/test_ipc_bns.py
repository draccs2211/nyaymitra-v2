"""IPC -> BNS exact lookup, checked against the BNS text in the corpus."""
import pytest
from backend.utils.ipc_bns import lookup_ipc, _bns_sections


@pytest.mark.parametrize("ipc,bns", [
    ("302", "103"), ("304", "105"), ("307", "109"), ("375", "63"), ("376", "64"),
    ("379", "303"), ("392", "309"), ("406", "316"), ("420", "318"), ("503", "351"), ("506", "351"),
])
def test_known_mappings(ipc, bns):
    assert bns in [b["section"] for b in lookup_ipc(ipc)["bns"]]


def test_input_normalised():
    assert [b["section"] for b in lookup_ipc(" 498a ")["bns"]] == ["85", "86"]


def test_murder_is_103_not_101():
    sec = lookup_ipc("302")["bns"][0]
    assert sec["section"] == "103" and "murder" in sec["title"].lower()


def test_rape_punishment_is_64_with_ten_year_minimum():
    sec = lookup_ipc("376")["bns"][0]
    assert sec["section"] == "64" and "ten years" in sec["punishment_text"]


def test_cbt_is_five_years_under_bns():
    assert "five years" in lookup_ipc("406")["bns"][0]["punishment_text"]


def test_dropped_and_unknown():
    assert lookup_ipc("124A")["carried_into_bns"] is False
    assert lookup_ipc("999") is None


def test_every_mapped_bns_section_exists_in_corpus():
    from backend.utils.ipc_bns import _mapping
    secs = _bns_sections()
    assert len(secs) == 358
    missing = {b for v in _mapping()["ipc_to_bns"].values() for b in v} - set(secs)
    assert not missing, missing
