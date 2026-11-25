"""
B2B auth tests (production plan, Phase 1 item 7): token claims, logout /
logout-all, invite-only onboarding, admin user management, create-firm CLI.
SQLite + fake Redis. Run: pytest tests/test_auth_b2b.py -v
"""
from datetime import datetime, timedelta

import pytest
from jose import jwt

from backend.config import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM
from backend.database.models import Invite, RevokedToken, User
from backend.utils.auth import SECRET_KEY
from backend.utils.invites import hash_token
from tests.test_approval_api import client  # noqa: F401  (fixture)

PW = "longenough1"


def register(client, email, *, firm_name=None, invite_token=None, name="N"):
    body = {"name": name, "email": email, "password": PW}
    if firm_name:
        body["firm_name"] = firm_name
    if invite_token:
        body["invite_token"] = invite_token
    return client.post("/api/auth/register", json=body)


def login(client, email, password=PW):
    return client.post("/api/auth/login", data={"username": email, "password": password})


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def me(client, token):
    return client.get("/api/auth/me", headers=bearer(token))


@pytest.fixture
def admin(client):
    r = register(client, "admin@firm.co", firm_name="Firm X")
    assert r.status_code == 200 and r.json()["user_role"] == "admin"
    return r.json()["access_token"]


def invite(client, admin_token, email, role="associate"):
    return client.post("/api/firms/invites", json={"email": email, "role": role}, headers=bearer(admin_token))


class TestTokens:
    def test_claims(self, client, admin):
        claims = jwt.decode(admin, SECRET_KEY, algorithms=[ALGORITHM])
        assert claims["jti"] and claims["tv"] == 0 and claims["iat"]
        assert claims["exp"] - claims["iat"] == ACCESS_TOKEN_EXPIRE_MINUTES * 60 == 8 * 3600

    def test_logout_revokes_only_that_token(self, client, admin):
        other = login(client, "admin@firm.co").json()["access_token"]
        assert client.post("/api/auth/logout", headers=bearer(admin)).status_code == 200
        assert me(client, admin).status_code == 401
        assert me(client, other).status_code == 200

    def test_logout_all(self, client, admin):
        other = login(client, "admin@firm.co").json()["access_token"]
        assert client.post("/api/auth/logout-all", headers=bearer(admin)).status_code == 200
        assert me(client, admin).status_code == 401 and me(client, other).status_code == 401
        fresh = login(client, "admin@firm.co").json()["access_token"]
        assert me(client, fresh).status_code == 200

    def test_tampered_version_rejected(self, client, admin):
        claims = jwt.decode(admin, SECRET_KEY, algorithms=[ALGORITHM])
        claims["tv"] = 99
        assert me(client, jwt.encode(claims, SECRET_KEY, algorithm=ALGORITHM)).status_code == 401

    def test_wrong_key_rejected(self, client):
        forged = jwt.encode({"sub": client.ids["senior"]}, "not-the-key" * 4, algorithm=ALGORITHM)
        assert me(client, forged).status_code == 401


class TestInvites:
    def test_invite_and_join(self, client, admin):
        r = invite(client, admin, "new.senior@firm.co", "senior")
        assert r.status_code == 200
        token = r.json()["invite_token"]
        joined = register(client, "New.Senior@firm.co", invite_token=token)
        assert joined.status_code == 200 and joined.json()["user_role"] == "senior"
        firm_id = me(client, admin).json()["firm_id"]
        assert me(client, joined.json()["access_token"]).json()["firm_id"] == firm_id
        listed = client.get("/api/firms/invites", headers=bearer(admin)).json()
        assert listed[0]["status"] == "used" and "invite_token" not in listed[0]

    def test_token_stored_hashed_only(self, client, admin):
        token = invite(client, admin, "x@firm.co").json()["invite_token"]
        db = client.Session()
        try:
            row = db.query(Invite).one()
            assert row.token_hash == hash_token(token) and token not in str(vars(row))
        finally:
            db.close()

    @pytest.mark.parametrize("case", ["reuse", "wrong_email", "expired", "garbage"])
    def test_invalid_invites(self, client, admin, case):
        token = invite(client, admin, "a@firm.co").json()["invite_token"]
        email = "a@firm.co"
        if case == "reuse":
            assert register(client, email, invite_token=token).status_code == 200
            email = "b@firm.co"
        elif case == "wrong_email":
            email = "someone.else@firm.co"
        elif case == "expired":
            db = client.Session()
            db.query(Invite).update({Invite.expires_at: datetime.utcnow() - timedelta(seconds=1)})
            db.commit(); db.close()
        elif case == "garbage":
            token = "not-a-token"
        r = register(client, email, invite_token=token)
        assert r.status_code == 400 and r.json()["detail"] == "Invalid or expired invite"

    def test_only_admins_invite(self, client):
        r = client.post("/api/firms/invites", json={"email": "z@x.co"}, headers=client.headers_for("senior"))
        assert r.status_code == 403

    def test_cannot_invite_existing_user(self, client, admin):
        assert invite(client, admin, "admin@firm.co").status_code == 409

    def test_production_is_invite_only(self, client, admin, monkeypatch):
        import backend.main as main
        token = invite(client, admin, "inv@firm.co").json()["invite_token"]
        monkeypatch.setattr(main, "ALLOW_OPEN_SIGNUP", False)
        r = register(client, "walkin@x.co", firm_name="Fake Firm")
        assert r.status_code == 403 and "invitation" in r.json()["detail"]
        assert register(client, "inv@firm.co", invite_token=token).status_code == 200


class TestUserManagement:
    @pytest.fixture
    def colleague(self, client, admin):
        token = invite(client, admin, "col@firm.co", "senior").json()["invite_token"]
        r = register(client, "col@firm.co", invite_token=token)
        return me(client, r.json()["access_token"]).json()["id"], r.json()["access_token"]

    def test_role_change_revokes_tokens(self, client, admin, colleague):
        uid, token = colleague
        r = client.post(f"/api/firms/users/{uid}", json={"role": "associate"}, headers=bearer(admin))
        assert r.status_code == 200 and r.json()["role"] == "associate"
        assert me(client, token).status_code == 401
        assert me(client, login(client, "col@firm.co").json()["access_token"]).json()["role"] == "associate"

    def test_deactivate_blocks_login(self, client, admin, colleague):
        uid, token = colleague
        client.post(f"/api/firms/users/{uid}", json={"is_active": False}, headers=bearer(admin))
        assert me(client, token).status_code == 401
        assert login(client, "col@firm.co").status_code == 401

    def test_guards(self, client, admin, colleague):
        admin_id = me(client, admin).json()["id"]
        assert client.post(f"/api/firms/users/{admin_id}", json={"role": "senior"},
                           headers=bearer(admin)).status_code == 400
        assert client.post(f"/api/firms/users/{client.ids['associate']}", json={"role": "senior"},
                           headers=bearer(admin)).status_code == 404          # other firm
        uid, _ = colleague
        assert client.post(f"/api/firms/users/{uid}", json={}, headers=bearer(admin)).status_code == 422
        assert client.post(f"/api/firms/users/{uid}", json={"role": "owner"},
                           headers=bearer(admin)).status_code == 422

    def test_list_colleagues(self, client, admin, colleague):
        emails = {u["email"] for u in client.get("/api/firms/users", headers=bearer(admin)).json()}
        assert emails == {"admin@firm.co", "col@firm.co"}


class TestCli:
    def test_create_firm_then_admin_joins(self, client):
        from backend.maintenance import create_firm
        token = create_firm(client.Session, name="Verma & Associates", admin_email="Head@Verma.co")
        r = register(client, "head@verma.co", invite_token=token)
        assert r.status_code == 200 and r.json()["user_role"] == "admin"

    def test_purge_expired_revoked_tokens(self, client):
        from backend.maintenance import purge
        db = client.Session()
        db.add_all([RevokedToken(jti="old", expires_at=datetime.utcnow() - timedelta(hours=1)),
                    RevokedToken(jti="live", expires_at=datetime.utcnow() + timedelta(hours=1))])
        db.commit(); db.close()
        assert purge(client.Session, query_log_days=90, audit_days=365)["revoked_tokens"] == 1
        db = client.Session()
        try:
            assert [t.jti for t in db.query(RevokedToken).all()] == ["live"]
        finally:
            db.close()
