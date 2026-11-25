"""
Firm administration (admin only): invitations and colleague management.

Role changes and deactivation bump the user's token_version, which revokes
every token they hold immediately.
"""
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.database.connection import get_db, get_db_factory
from backend.database.models import Invite, User
from backend.utils.audit import audit, client_ip
from backend.utils.auth import get_current_user
from backend.utils.invites import create_invite

router = APIRouter(prefix="/api/firms", tags=["firms"])

Role = Literal["associate", "senior", "admin"]


def require_firm_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.firm_id or current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Firm admin only")
    return current_user


class InviteRequest(BaseModel):
    email: str = Field(max_length=254)
    role: Role = "associate"

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or v.startswith("@") or v.endswith("@"):
            raise ValueError("invalid email address")
        return v


class UserUpdate(BaseModel):
    role: Optional[Role] = None
    is_active: Optional[bool] = None


@router.post("/invites")
def invite_colleague(
    body: InviteRequest,
    http: Request,
    admin: User = Depends(require_firm_admin),
    db: Session = Depends(get_db),
    db_factory=Depends(get_db_factory),
):
    """Create a one-time invite. The token is returned ONCE; share it with the invitee."""
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="A user with this email already exists")
    invite, token = create_invite(db, firm_id=admin.firm_id, email=body.email, role=body.role,
                                  created_by=admin.id)
    db.commit()
    audit(db_factory, "firm.invite", user_id=admin.id, resource=f"invite:{invite.id}", ip=client_ip(http),
          details={"role": body.role})
    return {"invite_token": token, "email": invite.email, "role": invite.role,
            "expires_at": invite.expires_at.isoformat()}


@router.get("/invites")
def list_invites(admin: User = Depends(require_firm_admin), db: Session = Depends(get_db)):
    now = datetime.utcnow()
    rows = db.query(Invite).filter(Invite.firm_id == admin.firm_id) \
        .order_by(Invite.created_at.desc()).limit(100).all()
    return [
        {"id": i.id, "email": i.email, "role": i.role,
         "status": "used" if i.used_at else ("expired" if i.expires_at < now else "open"),
         "expires_at": i.expires_at.isoformat()}
        for i in rows
    ]


@router.get("/users")
def list_colleagues(admin: User = Depends(require_firm_admin), db: Session = Depends(get_db)):
    users = db.query(User).filter(User.firm_id == admin.firm_id).order_by(User.created_at).all()
    return [{"id": u.id, "name": u.name, "email": u.email, "role": u.role, "is_active": u.is_active}
            for u in users]


@router.post("/users/{user_id}")
def update_colleague(
    user_id: str,
    body: UserUpdate,
    http: Request,
    admin: User = Depends(require_firm_admin),
    db: Session = Depends(get_db),
    db_factory=Depends(get_db_factory),
):
    """Change a colleague's role or (de)activate them. Revokes all their tokens."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Admins cannot change their own role or status")
    target = db.query(User).filter(User.id == user_id, User.firm_id == admin.firm_id).first()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    if body.role is None and body.is_active is None:
        raise HTTPException(status_code=422, detail="Nothing to update")
    if body.role is not None:
        target.role = body.role
    if body.is_active is not None:
        target.is_active = body.is_active
    target.token_version = (target.token_version or 0) + 1
    db.commit()
    audit(db_factory, "firm.user_update", user_id=admin.id, resource=f"user:{target.id}", ip=client_ip(http),
          details={"role": body.role, "is_active": body.is_active})
    return {"id": target.id, "role": target.role, "is_active": target.is_active}
