#!/bin/sh
# Container entrypoint: optional migrations / corpus load, then the CMD.
set -e

if [ "${RUN_MIGRATIONS:-true}" = "true" ]; then
  echo "Running database migrations..."
  alembic upgrade head
fi

# Off by default: the loader prunes docs not in its corpus list.
if [ "${LOAD_CORPUS_ON_START:-false}" = "true" ]; then
  echo "Loading legal corpus..."
  python -m backend.data_loader
fi

exec "$@"
