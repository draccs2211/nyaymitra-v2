"""
Data retention maintenance.

    python -m backend.maintenance purge                 # use configured retention
    python -m backend.maintenance purge --dry-run
    python -m backend.maintenance purge --query-log-days 30 --audit-days 365

Deletes query logs and audit entries older than the retention period.
Run daily (cron / scheduled task). Documents (lawyer work product) are kept.
Redis data expires on its own TTLs (sessions 1 h, query cache 24 h).
"""
import argparse
from datetime import datetime, timedelta
from typing import Callable, Dict

from backend.config import RETENTION_AUDIT_DAYS, RETENTION_QUERY_LOG_DAYS
from backend.database.models import AuditLog, Firm, QueryLog, RevokedToken


def purge(db_factory: Callable, *, query_log_days: int, audit_days: int,
          dry_run: bool = False, now: datetime = None) -> Dict[str, int]:
    now = now or datetime.utcnow()
    targets = {
        "query_logs": (QueryLog, now - timedelta(days=query_log_days)),
        "audit_logs": (AuditLog, now - timedelta(days=audit_days)),
    }
    db = db_factory()
    counts: Dict[str, int] = {}
    try:
        for name, (model, cutoff) in targets.items():
            q = db.query(model).filter(model.created_at < cutoff)
            counts[name] = q.count() if dry_run else q.delete(synchronize_session=False)
        expired_tokens = db.query(RevokedToken).filter(RevokedToken.expires_at < now)
        counts["revoked_tokens"] = expired_tokens.count() if dry_run else expired_tokens.delete(synchronize_session=False)
        if dry_run:
            db.rollback()
        else:
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return counts


def create_firm(db_factory: Callable, *, name: str, admin_email: str) -> str:
    """Create a firm and a one-time admin invite; returns the invite token."""
    from backend.utils.invites import create_invite
    db = db_factory()
    try:
        firm = Firm(name=name, email=admin_email.strip().lower())
        db.add(firm)
        db.flush()
        _, token = create_invite(db, firm_id=firm.id, email=admin_email, role="admin")
        db.commit()
        return token
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("purge", help="delete data past its retention period")
    p.add_argument("--query-log-days", type=int, default=RETENTION_QUERY_LOG_DAYS)
    p.add_argument("--audit-days", type=int, default=RETENTION_AUDIT_DAYS)
    p.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("create-firm", help="create a firm + one-time admin invite (production onboarding)")
    c.add_argument("--name", required=True)
    c.add_argument("--admin-email", required=True)
    args = parser.parse_args()

    from backend.database.connection import get_db_factory
    if args.command == "create-firm":
        token = create_firm(get_db_factory(), name=args.name, admin_email=args.admin_email)
        print(f"Firm created. One-time admin invite for {args.admin_email} (shown once):\n{token}")
        return
    counts = purge(get_db_factory(), query_log_days=args.query_log_days,
                   audit_days=args.audit_days, dry_run=args.dry_run)
    verb = "Would delete" if args.dry_run else "Deleted"
    for name, n in counts.items():
        print(f"{verb} {n} {name}")


if __name__ == "__main__":
    main()
