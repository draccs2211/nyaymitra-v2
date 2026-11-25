"""
Audit trail (AuditLog table). Records WHO did WHAT to WHICH resource — never
client text. Writes use their own session so an audit failure can't break or
roll back the request, and never raise.
"""
import logging
from typing import Any, Callable, Dict, Optional

from backend.database.models import AuditLog

logger = logging.getLogger("nyaymitra.audit")


def audit(
    db_factory: Callable,
    action: str,
    *,
    user_id: Optional[str] = None,
    resource: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip: Optional[str] = None,
) -> None:
    db = db_factory()
    try:
        db.add(AuditLog(user_id=user_id, action=action, resource=resource,
                        details=details or {}, ip_address=ip))
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Audit write failed (action=%s)", action)
    finally:
        db.close()


def client_ip(request) -> Optional[str]:
    return request.client.host if request is not None and request.client else None
