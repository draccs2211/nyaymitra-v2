# NyayMitra 2.0

Multi-agent AI legal research assistant for Indian law firms (B2B "Mode 2" first, citizen "Mode 1" later). Hindi / English / Hinglish.

## Environment
- Windows 11, PowerShell, Python 3.10, venv `.nyaymitra` (activate: `.\.nyaymitra\Scripts\Activate.ps1`)
- Always run from repo root with `python -m ...` (never `python backend/x.py`)
- Secrets only in `.env`. NEVER read, print, or commit `.env`.
- `.env` must contain `SECRET_KEY` (>= 32 random chars) or the API refuses to start.
- `APP_ENV` defaults to `production` (traces hide client text, strictest settings); set `APP_ENV=development` locally.
- Data minimisation: `STORE_QUERY_TEXT` (default false), `RETENTION_QUERY_LOG_DAYS` (90), `RETENTION_AUDIT_DAYS` (365).
- Auth: production is invite-only (`ALLOW_OPEN_SIGNUP` default false in prod, true in dev); tokens last 8 h (`ACCESS_TOKEN_EXPIRE_MINUTES`).

## Commands
- Install: `pip install -r requirements.txt` (unpinned versions on purpose)
- Load legal data: `python -m backend.data_loader [--dry-run]` — corpus files in `data/legal_corpus/` (format: its README.md); idempotent, prunes docs no longer in the files
- Run API: `uvicorn backend.main:app --reload --port 8000`
- Test one agent: `python -m backend.agents.<name>`
- Free tests: `$env:HF_HUB_OFFLINE='1'; pytest -m "not paid"` (OpenAI/Cohere/Kanoon blocked by tests/conftest.py)
- Paid tests (real OpenAI, ask first): `pytest -m paid`
- Retrieval eval (local, free): `python -m backend.evaluation.retrieval_eval`
- Retention purge (run daily): `python -m backend.maintenance purge [--dry-run]`
- Onboard a firm (prod): `python -m backend.maintenance create-firm --name "..." --admin-email ...` (prints one-time admin invite)
- DB migrations: `alembic upgrade head` (required in production; `/ready` is 503 until applied). After changing `backend/database/models.py`: `alembic revision --autogenerate -m "..."` (tests fail if models and migrations differ)

## Stack
FastAPI, LangGraph (interrupt + checkpointer), OpenAI gpt-4o-mini (NOT Groq), Pydantic, FastMCP (in-process client), ChromaDB + BM25 + RRF + Cohere rerank, sentence-transformers multilingual MiniLM, Redis (optional cache, circuit breaker), PostgreSQL/SQLAlchemy, JWT + bcrypt, LangSmith `@traceable`.

## Pipeline
guardian (urgency, never stops) ‖ orchestrator (`requested_actions`, `search_query_en`) run in parallel -> dispatch -> research | referral-only -> analysis -> drafting? -> referral? -> verification -> human_approval (interrupt when pending) -> END

## Key modules
- `backend/state.py` — `NyayMitraState` + `initial_state()` (single source of state keys)
- `backend/agents/schemas.py` — Pydantic output per agent; agents return `to_state(model)`
- `backend/mcp_client.py` — MCP `call_tool` with timeout/retry, `MCPCallError` for fallback
- `backend/utils/citations.py` — citation guard (provisions not in research context)
- `backend/rag/metadata.py` — corpus metadata schema; `backend/versioning.py` — cache fingerprints

## Rules
- No keyword/hardcoded classification; use LLM or model-based methods.
- Never invent citations. Indian Kanoon = discovery source only.
- No "strong/weak" case verdicts; use missing facts, uncertainties, questions for lawyer.
- Work on ONE agent/issue at a time; run its test after each change; small commits.
- Ask before adding dependencies, running paid-API loops, or touching `.env`.
- New tests must not call paid APIs unless marked `@pytest.mark.paid`.

## Known issues (priority order)
1. Legal content needs lawyer review: 9 Act files in `data/legal_corpus/` (BNS sections 20 and 34 were empty after conversion and were filled from the well-known one-line text — verify against the PDF); 8 Act files use the India Code home page as `source_url` (need the exact Act page); IPC->BNS mapping `data/reference/ipc_bns_mapping.json`; UP Rent Act "30-day deposit return" claim in `curated_summaries.jsonl`
2. Approval checkpointer is in-memory (`get_checkpointer()` in `pipeline.py`): pending reviews are marked `expired` (410) after a restart. Local Postgres `DATABASE_URL` doesn't connect; fix it, then use a Postgres saver (needs `langgraph-checkpoint-postgres`)
3. DLSA referral covers UP only (`data/referral/up_dlsa_directory.json`, 74 of 75 districts — Amethi missing, 16 without phone); templates MCP server not wired
4. Rate limiter + checkpointer are per-process (move to Redis for multi-worker); Redis not running locally
5. Cases are visible only to their creator, not firm colleagues (product decision)
6. Keyword-routed MCP tools (`get_legal_provision`, `search_up_laws`) not used — need LLM/exact lookup first
7. Indian Kanoon API key not purchased (research reports `not_configured`)

## Later
Frontend (React/Vite), Sarvam voice, RAGAS + larger legal eval sets, deployment.
