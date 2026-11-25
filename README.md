# NyayMitra 2.0

Multi-agent AI legal research assistant for Indian law firms. Lawyers ask in English,
Hindi or Hinglish; NyayMitra researches the law, gives a structured analysis
(missing facts, uncertainties, questions for the lawyer — never "strong/weak case"
verdicts), drafts documents, and routes anything risky to a senior for approval.

> Status: backend pilot-ready for 1–5 firms. Legal content still needs lawyer review
> (see **Known limitations**). No frontend yet.

## How it works

```text
            ┌─ guardian ──────┐   urgency screening (LLM), never blocks
request ──► │                 ├─► dispatch ─► research ─► analysis ─► drafting? ─► referral? ─► verification ─► human approval?
            └─ orchestrator ──┘   domain, requested actions, English search query
```

- **Research**: hybrid retrieval (ChromaDB dense + BM25, RRF fusion, Cohere rerank) over a
  curated corpus, plus MCP tools for case-law discovery (Indian Kanoon) and IPC→BNS lookups.
- **Analysis / drafting**: gpt-4o-mini with structured (Pydantic-validated) output. A citation
  guard flags any provision not present in the retrieved sources.
- **Verification**: confidence, citation checks, disclaimers. Low confidence, unverified
  citations or any drafted document pause the run for **human approval** (LangGraph interrupt).

## Quick start (local development)

Requirements: Python 3.10, optionally Docker.

```powershell
python -m venv .nyaymitra
.\.nyaymitra\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # then fill in SECRET_KEY, OPENAI_API_KEY; set APP_ENV=development
python -m backend.data_loader   # load the legal corpus (idempotent)
uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000/docs>. In development, open signup is enabled
(`POST /api/auth/register`).

> Tip (Windows): keep the virtualenv **outside OneDrive** — importing torch from a synced
> folder can take minutes.

## Run with Docker (pilot deployment)

```bash
cp .env.example .env            # SECRET_KEY, OPENAI_API_KEY, POSTGRES_PASSWORD; APP_ENV=production
docker compose up -d --build    # api + postgres + redis; migrations run on start
docker compose exec api python -m backend.data_loader
docker compose exec api python -m backend.maintenance create-firm --name "Firm Name" --admin-email admin@firm.in
```

`create-firm` prints a one-time invite token; the admin registers with it
(`POST /api/auth/register` with `invite_token`) and invites colleagues from
`POST /api/firms/invites`. Production registration is invite-only.

Put a TLS-terminating reverse proxy in front of the API and set
`FORWARDED_ALLOW_IPS` to the proxy's address and `CORS_ORIGINS` to the frontend origin.

## Configuration

All settings are environment variables; `.env.example` documents each one. The important ones:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | JWT signing key, ≥ 32 random chars — the API refuses to start without it |
| `OPENAI_API_KEY` | LLM calls (required) |
| `APP_ENV` | `production` (default: JSON logs, hidden traces, invite-only, migrations required) / `development` / `test` |
| `DATABASE_URL`, `REDIS_URL` | PostgreSQL (required), Redis (optional cache; the API degrades gracefully) |
| `COHERE_API_KEY`, `INDIAN_KANOON_API_KEY` | Optional rerank and case-law discovery |
| `DAILY_TOKEN_BUDGET` | LLM tokens per firm per day (0 = unlimited) |
| `STORE_QUERY_TEXT`, `RETENTION_*` | Data minimisation and retention |

## Operations

| Task | Command |
|---|---|
| Apply DB migrations | `alembic upgrade head` (the container does this on start) |
| New migration after model changes | `alembic revision --autogenerate -m "..."` |
| Retention purge (daily) | `python -m backend.maintenance purge [--dry-run]` |
| Onboard a firm | `python -m backend.maintenance create-firm --name ... --admin-email ...` |
| Liveness / readiness | `GET /health`, `GET /ready` (503 until DB, corpus, OpenAI key and schema are ready) |

Logs are JSON in production, carry an `X-Request-ID`, and mask emails, phone, Aadhaar and PAN
numbers. Client text is never written to logs, the audit trail or (in production) LangSmith traces.

## API overview

| Endpoint | Description |
|---|---|
| `POST /api/auth/register`, `/login`, `/logout`, `/logout-all` | Auth (JWT, 8 h, revocable) |
| `POST /api/query` | Ask a question; returns analysis, draft, sources, `approval_status`, `run_id` |
| `GET /api/runs/{run_id}` | Progress while a query runs (stage: researching, drafting, ...) |
| `GET /api/approvals?status=pending` | Review queue (firm admins/seniors) |
| `GET/POST /api/approvals/{run_id}` | View / approve or reject (approved drafts are saved as documents) |
| `/api/cases`, `/api/documents` | Cases and saved documents |
| `/api/firms/invites`, `/api/firms/users` | Firm admin: invites, roles, deactivation |

## Testing

```powershell
$env:HF_HUB_OFFLINE='1'; pytest -m "not paid"     # ~320 tests, no paid APIs (blocked by tests/conftest.py)
pytest -m paid                                    # real OpenAI calls — costs money, run deliberately
python -m backend.evaluation.retrieval_eval       # retrieval quality per language (local, free)
```

CI (GitHub Actions) runs the free tests and builds the Docker image on every push.

## Known limitations

- **Legal content is unreviewed**: the corpus is 26 hand-written summaries and an IPC→BNS quick
  table; some mappings need a lawyer's check before real use (see `CLAUDE.md`).
- Pending approvals live in an in-memory checkpointer: after a restart they are marked
  `expired` and must be re-run (Postgres checkpointer planned).
- Single API worker: rate limits and run state are per process.
- Indian Kanoon case-law search needs an API key (reported as `not_configured` until then).

Developer guide, rules and the current issue list: [`CLAUDE.md`](CLAUDE.md).
