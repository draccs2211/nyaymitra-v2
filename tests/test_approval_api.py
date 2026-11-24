"""
Approval API + reviewer authorization tests.
SQLite in-memory DB, fake Redis, mocked LLMs — no Postgres/Redis/paid APIs.
Run: pytest tests/test_approval_api.py -v
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import main
from backend.database.connection import get_db
from backend.database.models import Base, Firm, User
from backend.utils import cache
from backend.utils.auth import create_access_token
from tests.test_approval import mocked_agents  # noqa: F401  (fixture)
from tests.test_cache import FakeRedis


@pytest.fixture
def client(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def _get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(cache, "redis_client", FakeRedis())
    main.app.dependency_overrides[get_db] = _get_db
    db = Session()
    firm_a, firm_b = Firm(name="A", email="a@firm"), Firm(name="B", email="b@firm")
    db.add_all([firm_a, firm_b]); db.flush()
    users = {
        "associate": User(name="As", email="as@a", hashed_password="x", firm_id=firm_a.id, role="associate"),
        "associate2": User(name="As2", email="as2@a", hashed_password="x", firm_id=firm_a.id, role="associate"),
        "senior": User(name="Se", email="se@a", hashed_password="x", firm_id=firm_a.id, role="senior"),
        "other_senior": User(name="Os", email="os@b", hashed_password="x", firm_id=firm_b.id, role="senior"),
        "solo": User(name="So", email="so@x", hashed_password="x", firm_id=None, role="associate"),
    }
    db.add_all(users.values()); db.commit()
    ids = {k: u.id for k, u in users.items()}
    db.close()

    tc = TestClient(main.app)
    tc.headers_for = lambda who: {"Authorization": f"Bearer {create_access_token({'sub': ids[who]})}"}
    tc.ids = ids
    yield tc
    main.app.dependency_overrides.clear()


def ask(client, who, query="Draft legal notice to landlord"):
    r = client.post("/api/query", json={"query": query}, headers=client.headers_for(who))
    assert r.status_code == 200, r.text
    return r.json()


def decide(client, who, run_id, **body):
    body.setdefault("decision", "approved")
    return client.post(f"/api/approvals/{run_id}", json=body, headers=client.headers_for(who))


class TestApprovalApi:
    def test_query_returns_pending_run(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        body = ask(client, "associate")
        assert body["approval_status"] == "pending"
        assert body["run_id"]

    def test_non_draft_query_not_pending(self, client, mocked_agents):
        mocked_agents(["research"])
        body = ask(client, "associate", "What is section 406?")
        assert body["approval_status"] == "not_required"

    def test_associate_cannot_self_approve(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = ask(client, "associate")["run_id"]
        assert decide(client, "associate", run_id).status_code == 404
        assert decide(client, "associate2", run_id).status_code == 404

    def test_other_firm_cannot_see_or_approve(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = ask(client, "associate")["run_id"]
        h = client.headers_for("other_senior")
        assert client.get(f"/api/approvals/{run_id}", headers=h).status_code == 404
        assert decide(client, "other_senior", run_id).status_code == 404

    def test_senior_same_firm_approves(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = ask(client, "associate")["run_id"]
        r = decide(client, "senior", run_id, notes="ok", edited_response="Final text")
        assert r.status_code == 200, r.text
        assert r.json()["approval_status"] == "approved"
        assert r.json()["response"] == "Final text"
        # requester sees the outcome
        g = client.get(f"/api/approvals/{run_id}", headers=client.headers_for("associate"))
        assert g.status_code == 200 and g.json()["approval_status"] == "approved"
        # second decision rejected
        assert decide(client, "senior", run_id).status_code == 409

    def test_reviewer_id_comes_from_token(self, client, mocked_agents):
        import asyncio
        from backend.pipeline import get_run
        mocked_agents(["research", "drafting"])
        run_id = ask(client, "associate")["run_id"]
        r = decide(client, "senior", run_id, reviewer_id=client.ids["other_senior"])
        assert r.status_code == 200
        run = asyncio.run(get_run(run_id))
        assert run["approval"]["reviewer_id"] == client.ids["senior"]

    def test_reject_requires_notes(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = ask(client, "associate")["run_id"]
        assert decide(client, "senior", run_id, decision="rejected").status_code == 422
        r = decide(client, "senior", run_id, decision="rejected", notes="cite BNS")
        assert r.status_code == 200 and r.json()["approval_status"] == "rejected"

    def test_solo_user_self_approves(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = ask(client, "solo")["run_id"]
        assert decide(client, "solo", run_id).json()["approval_status"] == "approved"

    def test_unknown_run_404(self, client):
        assert decide(client, "senior", "nope").status_code == 404

    def test_pending_response_not_cached(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        a = ask(client, "associate")
        b = ask(client, "associate")
        assert a["run_id"] != b["run_id"]
        assert b["approval_status"] == "pending"
