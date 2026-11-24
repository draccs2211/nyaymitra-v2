# NyayMitra 2.0

Multi-agent AI legal research assistant for Indian law firms (B2B "Mode 2" first, citizen "Mode 1" later). Hindi / English / Hinglish.

## Environment
- Windows 11, PowerShell, Python 3.10, venv `.nyaymitra` (activate: `.\.nyaymitra\Scripts\Activate.ps1`)
- Always run from repo root with `python -m ...` (never `python backend/x.py`)
- Secrets only in `.env`. NEVER read, print, or commit `.env`.
- `.env` must contain `SECRET_KEY` (>= 32 random chars) or the API refuses to start.

## Commands
- Install: `pip install -r requirements.txt` (unpinned versions on purpose)
- Load legal data: `python -m backend.data_loader` (idempotent; safe to re-run)
- Run API: `uvicorn backend.main:app --reload --port 8000`
- Test one agent: `python -m backend.agents.<name>`
- Free tests: `$env:HF_HUB_OFFLINE='1'; pytest -m "not paid"` (OpenAI/Cohere/Kanoon blocked by tests/conftest.py)
- Paid tests (real OpenAI, ask first): `pytest -m paid`
- Retrieval eval (local, free): `python -m backend.evaluation.retrieval_eval`

## Stack
FastAPI, LangGraph (interrupt + checkpointer), OpenAI gpt-4o-mini (NOT Groq), Pydantic, FastMCP (in-process client), ChromaDB + BM25 + RRF + Cohere rerank, sentence-transformers multilingual MiniLM, Redis (optional cache, circuit breaker), PostgreSQL/SQLAlchemy, JWT + bcrypt, LangSmith `@traceable`.

## Pipeline
guardian (flags urgency, never stops) -> orchestrator (`requested_actions`, `search_query_en`) -> research | referral-only -> analysis -> drafting? -> referral? -> verification -> human_approval (interrupt when pending) -> END

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
1. Legal content needs lawyer review: IPC 376 -> "BNS 63" (punishment is likely BNS 64) in `legal_reference_mcp.py`, `guardian.py`, `data_loader.py`; UP Rent Act "30-day deposit return" claim; whole IPC->BNS table
2. Approval checkpointer is in-memory (`get_checkpointer()` in `pipeline.py`): paused runs lost on restart. Local Postgres `DATABASE_URL` doesn't connect; fix it, then use a Postgres saver (needs `langgraph-checkpoint-postgres`)
3. No pending-approvals list endpoint for reviewers
4. Referral always returns Lucknow DLSA; DLSA + templates MCP servers not wired
5. `jurisdiction` is in request/state/cache key but retrieval doesn't filter by it
6. No DB migrations (Alembic); `Case.case_strength` column is unused
7. Rate limiter + checkpointer are per-process (move to Redis for multi-worker); Redis not running locally
8. Cases are visible only to their creator, not firm colleagues (product decision)
9. Keyword-routed MCP tools (`get_legal_provision`, `search_up_laws`) not used — need LLM/exact lookup first
10. Indian Kanoon API key not purchased (research reports `not_configured`)

## Later
Frontend (React/Vite), Sarvam voice, RAGAS + larger legal eval sets, deployment.
