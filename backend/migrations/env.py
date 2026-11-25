"""Alembic environment: models from backend.database.models, URL from config or -x db_url."""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.database.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _url() -> str:
    override = context.get_x_argument(as_dictionary=True).get("db_url")
    if override:
        return override
    from backend.config import DATABASE_URL
    return DATABASE_URL


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True,
                      render_as_batch=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection")
    if connectable is None:
        engine = engine_from_config({"sqlalchemy.url": _url()}, prefix="sqlalchemy.", poolclass=pool.NullPool)
        with engine.connect() as connection:
            _run(connection)
    else:
        _run(connectable)


def _run(connection) -> None:
    # render_as_batch: SQLite can't ALTER most things; batch mode recreates tables
    context.configure(connection=connection, target_metadata=target_metadata,
                      render_as_batch=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
