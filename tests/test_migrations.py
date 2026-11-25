"""
Migration tests (production plan, Phase 1 item 4): Alembic upgrade/downgrade on
SQLite, models and migrations agree (no autogenerate diff), indexes exist,
schema check used by /ready.
Run: pytest tests/test_migrations.py -v
"""
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from backend.database.models import Base
from backend.database.schema import ALEMBIC_INI, migration_head, schema_status


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'migr.db').as_posix()}"


def _alembic(db_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(ALEMBIC_INI.parent / "backend" / "migrations"))
    cfg.cmd_opts = type("Opts", (), {"x": [f"db_url={db_url}"]})()
    return cfg


class TestMigrations:
    def test_upgrade_creates_all_tables_and_indexes(self, db_url):
        command.upgrade(_alembic(db_url), "head")
        insp = inspect(create_engine(db_url))
        assert set(Base.metadata.tables) <= set(insp.get_table_names())
        ql_indexes = {i["name"] for i in insp.get_indexes("query_logs")}
        assert {"ix_query_logs_user_id", "ix_query_logs_created_at", "ix_query_logs_query_hmac"} <= ql_indexes
        assert "case_strength" not in {c["name"] for c in insp.get_columns("cases")}

    def test_models_match_migrations(self, db_url):
        """Fails when a model changes without a new migration (run alembic revision --autogenerate)."""
        command.upgrade(_alembic(db_url), "head")
        with create_engine(db_url).connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
        assert diff == [], f"models and migrations differ: {diff}"

    def test_downgrade_and_reupgrade(self, db_url):
        cfg = _alembic(db_url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")
        assert set(inspect(create_engine(db_url)).get_table_names()) <= {"alembic_version"}
        command.upgrade(cfg, "head")
        assert "users" in inspect(create_engine(db_url)).get_table_names()

    def test_schema_status(self, db_url):
        engine = create_engine(db_url)
        with engine.connect() as conn:
            assert schema_status(conn) == {"current": None, "head": migration_head(), "ok": False}
        command.upgrade(_alembic(db_url), "head")
        with engine.connect() as conn:
            assert schema_status(conn)["ok"] is True


class TestReadySchemaGate:
    def test_production_blocks_on_unmigrated_db(self, client, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(main, "IS_PRODUCTION", True)
        r = client.get("/ready")        # test DB built by create_all -> no alembic revision
        assert r.status_code == 503
        assert r.json()["checks"]["schema"]["ok"] is False

    def test_development_reports_schema_without_blocking(self, client, monkeypatch):
        import backend.config as config
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test-dummy")   # independent of .env
        r = client.get("/ready")
        assert r.status_code == 200 and "schema" in r.json()["checks"]


from tests.test_approval_api import client  # noqa: E402,F401  (fixture)
