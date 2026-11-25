"""
Firm invitations (B2B onboarding). The plaintext token is shown once to the
admin who creates it; only its SHA-256 is stored. An invite is bound to one
email address, a firm and a role, and can be used once before it expires.
"""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Tuple

from sqlalchemy.orm import Session

from backend.config import INVITE_TTL_DAYS
from backend.database.models import Invite

ROLES = ("associate", "senior", "admin")


class InviteError(Exception):
    """Invalid, used, expired or mismatched invite (deliberately one generic error)."""


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_invite(db: Session, *, firm_id: str, email: str, role: str,
                  created_by: str = None, ttl_days: int = INVITE_TTL_DAYS) -> Tuple[Invite, str]:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    token = secrets.token_urlsafe(32)
    invite = Invite(
        token_hash=hash_token(token), firm_id=firm_id, email=email.strip().lower(), role=role,
        created_by=created_by, expires_at=datetime.utcnow() + timedelta(days=ttl_days),
    )
    db.add(invite)
    db.flush()
    return invite, token


def find_usable_invite(db: Session, token: str, email: str) -> Invite:
    """Return the invite if it can be used by `email` now; raise InviteError otherwise."""
    invite = db.query(Invite).filter(Invite.token_hash == hash_token(token or "")).first()
    if (invite is None or invite.used_at is not None or invite.expires_at < datetime.utcnow()
            or invite.email != email.strip().lower()):
        raise InviteError("invalid or expired invite")
    return invite
