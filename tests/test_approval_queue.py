"""
Durable approvals tests (production plan, Phase 1 item 3): approvals table,
reviewer queue, drafts saved as Documents, expiry when run state is lost.
SQLite + fake Redis + mocked LLMs. Run: pytest tests/test_approval_queue.py -v
"""
import asyncio

import pytest

from backend.database.models import Approval, Document
from tests.test_approval import mocked_agents  # noqa: F401  (fixture)
from tests.test_approval_api import client  # noqa: F401  (fixture)


def _pending_run(client, who, case_id=None):
    body = {"query": "Draft legal notice to landlord"}
    if case_id:
        body["case_id"] = case_id
    r = client.post("/api/query", json=body, headers=client.headers_for(who))
    assert r.status_code == 200 and r.json()["approval_status"] == "pending", r.text
    return r.json()["run_id"]


def _row(client, run_id):
    db = client.Session()
    try:
        return db.get(Approval, run_id)
    finally:
        db.close()


def _queue(client, who, status="pending"):
    r = client.get(f"/api/approvals?status={status}", headers=client.headers_for(who))
    assert r.status_code == 200, r.text
    return [a["run_id"] for a in r.json()]


class TestApprovalRecord:
    def test_pending_run_recorded_without_text(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = _pending_run(client, "associate")
        row = _row(client, run_id)
        assert row.status == "pending" and row.requester_id == client.ids["associate"]
        assert row.document_type == "legal_notice"
        assert not any("landlord" in str(v).lower() for v in vars(row).values())

    def test_non_review_run_not_recorded(self, client, mocked_agents):
        mocked_agents(["research"])
        r = client.post("/api/query", json={"query": "what is 406"}, headers=client.headers_for("associate"))
        assert r.json()["approval_status"] == "not_required"
        assert _row(client, r.json()["run_id"]) is None


class TestQueue:
    def test_visibility(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        a_run = _pending_run(client, "associate")
        s_run = _pending_run(client, "solo")
        assert set(_queue(client, "senior")) == {a_run}            # firm queue
        assert _queue(client, "associate") == [a_run]                # own requests
        assert _queue(client, "associate2") == []                    # firm associate: not a reviewer
        assert _queue(client, "other_senior") == []                  # other firm
        assert _queue(client, "solo") == [s_run]

    def test_status_filter(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = _pending_run(client, "associate")
        client.post(f"/api/approvals/{run_id}", json={"decision": "approved"}, headers=client.headers_for("senior"))
        assert _queue(client, "senior", "pending") == []
        assert _queue(client, "senior", "approved") == [run_id]
        assert _queue(client, "senior", "all") == [run_id]

    def test_invalid_status_422(self, client):
        assert client.get("/api/approvals?status=bogus", headers=client.headers_for("senior")).status_code == 422


class TestDocumentsSaved:
    def test_approved_draft_saved_final(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        h = client.headers_for("associate")
        case_id = client.post("/api/cases", json={"title": "Deposit"}, headers=h).json()["id"]
        run_id = _pending_run(client, "associate", case_id)
        r = client.post(f"/api/approvals/{run_id}", json={"decision": "approved"}, headers=client.headers_for("senior"))
        assert r.status_code == 200

        row = _row(client, run_id)
        assert row.status == "approved" and row.reviewer_id == client.ids["senior"] and row.decided_at
        docs = client.get(f"/api/documents/{case_id}", headers=h).json()
        assert len(docs) == 1 and docs[0]["status"] == "final"
        mine = client.get("/api/documents", headers=h).json()
        assert mine[0]["run_id"] == run_id and mine[0]["doc_type"] == "legal_notice"
        db = client.Session()
        try:
            doc = db.query(Document).filter_by(run_id=run_id).one()
            assert doc.user_id == client.ids["associate"] and doc.content_english
        finally:
            db.close()

    def test_rejected_draft_saved_as_rejected(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = _pending_run(client, "associate")
        client.post(f"/api/approvals/{run_id}", json={"decision": "rejected", "notes": "wrong forum"},
                    headers=client.headers_for("senior"))
        docs = client.get("/api/documents", headers=client.headers_for("associate")).json()
        assert [d["status"] for d in docs] == ["rejected"]
        assert _row(client, run_id).status == "rejected"


class TestExpiry:
    @pytest.fixture
    def lost_runs(self, monkeypatch):
        """Simulate a restart with the in-memory checkpointer: run state gone."""
        import backend.main as main

        async def gone(run_id):
            return None
        monkeypatch.setattr(main, "get_run", gone)

    def test_view_lost_run_410_and_expired(self, client, mocked_agents, lost_runs):
        mocked_agents(["research", "drafting"])
        run_id = _pending_run(client, "associate")
        r = client.get(f"/api/approvals/{run_id}", headers=client.headers_for("senior"))
        assert r.status_code == 410 and "expired" in r.json()["detail"]
        assert _row(client, run_id).status == "expired"
        assert _queue(client, "senior", "expired") == [run_id]

    def test_decide_lost_run_410(self, client, mocked_agents, lost_runs):
        mocked_agents(["research", "drafting"])
        run_id = _pending_run(client, "associate")
        r = client.post(f"/api/approvals/{run_id}", json={"decision": "approved"}, headers=client.headers_for("senior"))
        assert r.status_code == 410
        assert client.post(f"/api/approvals/{run_id}", json={"decision": "approved"},
                           headers=client.headers_for("senior")).status_code == 410

    def test_startup_sweep(self, client, mocked_agents, monkeypatch):
        import backend.main as main
        mocked_agents(["research", "drafting"])
        live = _pending_run(client, "associate")
        db = client.Session()
        db.add(Approval(run_id="lost-run", requester_id=client.ids["associate"], status="pending"))
        db.commit()
        try:
            assert asyncio.run(main.expire_orphaned_approvals(db)) == 1
        finally:
            db.close()
        assert _row(client, "lost-run").status == "expired"
        assert _row(client, live).status == "pending"

    def test_still_404_for_strangers(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = _pending_run(client, "associate")
        assert client.get(f"/api/approvals/{run_id}", headers=client.headers_for("other_senior")).status_code == 404
