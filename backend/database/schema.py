"""
Schema version check: compares the database's Alembic revision with the
migration head in backend/migrations. Production must run
`alembic upgrade head` before serving (see /ready).
"""
from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def migration_head() -> str:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(ALEMBIC_INI.parent / "backend" / "migrations"))
    return ScriptDirectory.from_config(cfg).get_current_head()


def schema_status(connection) -> dict:
    """{'current': db revision or None, 'head': latest migration, 'ok': current == head}"""
    current = MigrationContext.configure(connection).get_current_revision()
    head = migration_head()
    return {"current": current, "head": head, "ok": current == head}
