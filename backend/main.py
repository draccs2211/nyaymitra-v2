from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from sqlalchemy import func, text
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import datetime, timedelta
import logging
import re
import uuid
import time

from backend.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES, CORS_ORIGINS, DAILY_TOKEN_BUDGET, IS_PRODUCTION, STORE_QUERY_TEXT,
    RATE_LIMIT_APPROVAL_PER_MIN, RATE_LIMIT_AUTH_PER_MIN, RATE_LIMIT_QUERY_PER_MIN,
)
from backend.database.connection import get_db, get_db_factory, init_db
from backend.utils.audit import audit, client_ip
from backend.utils.rate_limit import limit_by_ip, limit_by_user
from backend.utils.redact import keyed_hash
from backend.logging_config import request_id_var, setup_logging
from backend.versioning import cache_version
from backend.database.models import User, Firm, Case, Document, QueryLog
from backend.utils.auth import (
    SECRET_KEY, verify_password, get_password_hash,
    create_access_token, get_current_user
)
from backend.utils.cache import (
    get_cached_query, set_cached_query,
    store_session, get_session
)
from backend.utils.language_detector import detect_language
from backend.pipeline import run_pipeline, resume_pipeline, get_run
from pydantic import ValidationError
from dotenv import load_dotenv

load_dotenv()

setup_logging()          # JSON in production, request IDs, PII redaction on every line
logger = logging.getLogger("nyaymitra.api")
access_logger = logging.getLogger("nyaymitra.access")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Production: schema is managed only by Alembic (`alembic upgrade head`);
    /ready reports 503 until the database is at the migration head.
    Development/test: create missing tables for convenience.
    """
    if not IS_PRODUCTION:
        try:
            init_db()
            logger.info("Database tables ensured (non-production create_all)")
        except Exception as e:
            logger.warning("Database init skipped: %s", e)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="NyayMitra 2.0 API",
    description="AI-powered legal intelligence platform for Indian lawyers",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,          # explicit allow-list, never "*"
    allow_credentials=False,             # bearer tokens in headers, no cookies
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Request ID (accepts a sane X-Request-ID, else generates one) + access log."""
    incoming = request.headers.get("x-request-id", "")
    rid = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
    token = request_id_var.set(rid)
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        access_logger.info(
            "%s %s -> %s", request.method, request.url.path, status_code,
            extra={"method": request.method, "path": request.url.path, "status": status_code,
                   "duration_ms": round((time.perf_counter() - start) * 1000, 1)},
        )
        request_id_var.reset(token)


# ─── Pydantic Models ────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str = Field(max_length=254)
    password: str
    firm_name: Optional[str] = Field(default=None, max_length=200)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("invalid email address")
        return v

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        # bcrypt only uses the first 72 bytes and errors beyond that
        if len(v) < 8 or len(v.encode("utf-8")) > 72:
            raise ValueError("password must be 8-72 bytes long")
        return v


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    # ISO 3166 code: IN (central) or IN-<state>, e.g. IN-UP. Part of the cache key;
    # retrieval does not filter by it yet.
    jurisdiction: str = Field(default="IN", pattern=r"^IN(-[A-Z]{2})?$")
    session_id: Optional[str] = Field(default=None, max_length=100)
    case_id: Optional[str] = Field(default=None, max_length=100)

class QueryResponse(BaseModel):
    session_id: str
    response: str
    confidence_score: float
    legal_domain: str
    missing_facts: List[str] = []           # facts that would change the analysis
    uncertainties: List[str] = []
    questions_for_lawyer: List[str] = []
    sources: List[str]
    next_steps: List[str]
    has_document: bool
    document_hindi: Optional[str] = None
    document_english: Optional[str] = None
    is_emergency: bool
    urgency_level: str = ""                 # CRITICAL / HIGH / MEDIUM when is_emergency
    human_approval_needed: bool
    source_status: dict = {}                # per external source: ok / not_configured / unavailable / error
    missing_fields: List[str] = []          # facts the lawyer must fill into the draft
    unverified_citations: List[str] = []    # provisions in draft not found in research
    tokens_used: int = 0
    run_id: Optional[str] = None
    approval_status: str = "not_required"   # not_required / pending / approved / rejected
    response_time_ms: int

class ApprovalRequest(BaseModel):
    decision: str                       # approved / rejected
    notes: str = ""
    edited_response: Optional[str] = None


class CreateCaseRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    legal_domain: Optional[str] = "general"
    notes: Optional[str] = ""

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_name: str
    user_role: str


def _to_response(result: dict, session_id: str, response_time_ms: int) -> QueryResponse:
    return QueryResponse(
        session_id=session_id,
        response=result.get("final_response", "No response generated"),
        confidence_score=result.get("confidence_score", 0.5),
        legal_domain=result.get("legal_domain", "general"),
        missing_facts=(result.get("case_analysis") or {}).get("missing_facts", []),
        uncertainties=(result.get("case_analysis") or {}).get("uncertainties", []),
        questions_for_lawyer=(result.get("case_analysis") or {}).get("questions_for_lawyer", []),
        sources=result.get("verified_sources", []),
        next_steps=result.get("next_steps", []),
        has_document=bool(result.get("draft_document")),
        document_hindi=result.get("document_hindi"),
        document_english=result.get("document_english"),
        is_emergency=result.get("is_emergency", False),
        urgency_level=result.get("urgency_level", ""),
        human_approval_needed=result.get("human_approval_needed", False),
        source_status=result.get("source_status", {}),
        missing_fields=result.get("draft_missing_fields", []),
        unverified_citations=sorted(set(result.get("draft_unverified_citations", []))
                                    | set(result.get("analysis_unverified_citations", []))),
        tokens_used=int(result.get("tokens_used") or 0),
        run_id=result.get("run_id"),
        approval_status=result.get("approval_status", "not_required"),
        response_time_ms=response_time_ms,
    )


def can_review(reviewer: User, requester: User) -> bool:
    """
    Who may approve/reject a run:
    - solo user (no firm): only themselves
    - firm member's run: an admin or senior of the SAME firm (associates can't
      approve their own output)
    """
    if requester.firm_id is None:
        return reviewer.id == requester.id
    return reviewer.firm_id == requester.firm_id and reviewer.role in ("admin", "senior")


# ─── Startup ─────────────────────────────────────────────────────────────────

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "NyayMitra 2.0 API",
        "status": "running",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Liveness: the process is up. No dependency checks (cheap, for restarts)."""
    return {"status": "ok", "version": "2.0.0"}


def _check_database(db: Session) -> dict:
    try:
        db.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}


def _check_schema(db: Session) -> dict:
    try:
        from backend.database.schema import schema_status
        return schema_status(db.connection())
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}


def _check_corpus() -> dict:
    try:
        from backend.rag.hybrid_search import hybrid_rag
        count = hybrid_rag.collection.count()
        return {"ok": count > 0, "documents": count}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}


@app.get("/ready")
async def ready(db: Session = Depends(get_db)):
    """
    Readiness: can this instance serve queries? 503 if a required dependency
    (database, loaded corpus, OpenAI key) is missing. Optional ones (Redis,
    Cohere, Indian Kanoon) only mark the response "degraded".
    """
    from backend.config import COHERE_API_KEY, INDIAN_KANOON_API_KEY, OPENAI_API_KEY
    from backend.utils.cache import redis_available
    db_check = _check_database(db)
    schema = _check_schema(db) if db_check["ok"] else {"ok": False, "error": "database unavailable"}
    required = {
        "database": db_check,
        "corpus": _check_corpus(),
        "openai_key": {"ok": bool(OPENAI_API_KEY)},
    }
    optional = {
        "redis": {"ok": redis_available()},
        "cohere_rerank": {"ok": bool(COHERE_API_KEY)},
        "indian_kanoon": {"ok": bool(INDIAN_KANOON_API_KEY)},
    }
    # Schema drift blocks traffic in production. In dev/test tables come from
    # create_all (no Alembic revision), so it's only informational there.
    if IS_PRODUCTION:
        required["schema"] = schema
    else:
        optional["schema"] = schema
    is_ready = all(c["ok"] for c in required.values())
    status_text = "ready" if is_ready and all(c["ok"] for c in optional.values()) else (
        "degraded" if is_ready else "not_ready")
    body = {"status": status_text, "checks": {**required, **optional}}
    return JSONResponse(status_code=200 if is_ready else 503, content=body)


# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=TokenResponse,
          dependencies=[Depends(limit_by_ip("register", RATE_LIMIT_AUTH_PER_MIN))])
async def register(request: RegisterRequest, http: Request,
                   db: Session = Depends(get_db), db_factory=Depends(get_db_factory)):
    # Check if email exists
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create firm if provided
    firm = None
    if request.firm_name:
        firm = Firm(name=request.firm_name, email=request.email)
        db.add(firm)
        db.flush()

    # Create user
    user = User(
        name=request.name,
        email=request.email,
        hashed_password=get_password_hash(request.password),
        firm_id=firm.id if firm else None,
        role="admin" if firm else "associate"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    audit(db_factory, "auth.register", user_id=user.id, ip=client_ip(http),
          details={"role": user.role, "created_firm": firm is not None})

    token = create_access_token(
        data={"sub": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_name=user.name,
        user_role=user.role
    )


@app.post("/api/auth/login", response_model=TokenResponse,
          dependencies=[Depends(limit_by_ip("login", RATE_LIMIT_AUTH_PER_MIN))])
async def login(
    http: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    db_factory=Depends(get_db_factory),
):
    email = form_data.username.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        audit(db_factory, "auth.login_failed", user_id=user.id if user else None, ip=client_ip(http),
              details={"email_hmac": keyed_hash(email, SECRET_KEY)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    audit(db_factory, "auth.login", user_id=user.id, ip=client_ip(http))

    token = create_access_token(
        data={"sub": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_name=user.name,
        user_role=user.role
    )


@app.get("/api/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "role": current_user.role,
        "firm_id": current_user.firm_id
    }


# ─── Query Routes ─────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    http: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(limit_by_user("query", RATE_LIMIT_QUERY_PER_MIN)),
    db: Session = Depends(get_db),
    db_factory=Depends(get_db_factory),
):
    start_time = time.time()
    if request.case_id:
        get_owned_case(db, request.case_id, current_user)   # 404 if not the user's case
    session_id = request.session_id or str(uuid.uuid4())

    # Get conversation history from session
    session_data = get_session(session_id, user_id=current_user.id) or {}
    conversation_history = session_data.get("history", [])

    # Detect language
    language = detect_language(request.query)

    # Check query cache
    version = cache_version()
    cached = get_cached_query(request.query, scope=current_user.id, language=language,
                              jurisdiction=request.jurisdiction, version=version)
    if cached and not request.session_id:
        return _to_response(cached, session_id, int((time.time() - start_time) * 1000))

    # Daily LLM budget per firm (or per user without a firm)
    if DAILY_TOKEN_BUDGET and tokens_used_today(db, current_user) >= DAILY_TOKEN_BUDGET:
        raise HTTPException(
            status_code=429,
            detail="Daily AI usage limit reached for your firm. It resets at 00:00 UTC.",
        )

    # Run pipeline
    try:
        result = await run_pipeline(
            query=request.query,
            session_id=session_id,
            user_id=current_user.id,
            conversation_history=conversation_history,
            jurisdiction=request.jurisdiction,
        )
    except Exception:
        logger.exception("Pipeline failed (user=%s session=%s)", current_user.id, session_id)
        raise HTTPException(status_code=500, detail="Internal error while processing the query")

    response_time = int((time.time() - start_time) * 1000)
    logger.info("Query completed", extra={
        "run_id": result.get("run_id"), "user_id": current_user.id, "latency_ms": response_time,
        "tokens_used": int(result.get("tokens_used") or 0), "approval_status": result.get("approval_status"),
        "legal_domain": result.get("legal_domain"), "is_emergency": bool(result.get("is_emergency")),
    })

    # Update conversation history
    conversation_history.append({"role": "user", "content": request.query})
    conversation_history.append({"role": "assistant", "content": result.get("final_response", "")})
    store_session(session_id, {"history": conversation_history[-20:]}, user_id=current_user.id)  # Keep last 20 turns

    # Cache only final, non-emergency responses — never ones awaiting review
    if (not result.get("is_emergency") and not request.session_id
            and result.get("approval_status") == "not_required"):
        set_cached_query(request.query, result, scope=current_user.id, language=language,
                         jurisdiction=request.jurisdiction, version=version)

    # Log query + audit in background (own sessions; no client text in the audit)
    background_tasks.add_task(
        log_query, db_factory, current_user.id, request.case_id,
        request.query, language, result, response_time
    )
    background_tasks.add_task(
        audit, db_factory, "query.run", user_id=current_user.id,
        resource=f"run:{result.get('run_id', '')}", ip=client_ip(http),
        details={"case_id": request.case_id, "approval_status": result.get("approval_status"),
                 "is_emergency": bool(result.get("is_emergency"))},
    )

    return _to_response(result, session_id, response_time)


# ─── Approval Routes ──────────────────────────────────────────────────────────

async def _load_run_for(run_id: str, current_user: User, db: Session, *, review: bool):
    run = await get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    requester = db.query(User).filter(User.id == run.get("user_id")).first()
    if requester is None:
        raise HTTPException(status_code=404, detail="Run not found")
    allowed = can_review(current_user, requester) or (
        not review and current_user.id == requester.id
    )
    if not allowed:
        # 404, not 403: don't reveal that the run exists
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/approvals/{run_id}", response_model=QueryResponse)
async def get_approval(
    run_id: str,
    http: Request,
    db_factory=Depends(get_db_factory),
    current_user: User = Depends(limit_by_user("approval", RATE_LIMIT_APPROVAL_PER_MIN)),
    db: Session = Depends(get_db)
):
    """View a run's response and approval status (requester or eligible reviewer)."""
    run = await _load_run_for(run_id, current_user, db, review=False)
    audit(db_factory, "approval.view", user_id=current_user.id, resource=f"run:{run_id}", ip=client_ip(http))
    return _to_response(run, run.get("session_id", ""), 0)


@app.post("/api/approvals/{run_id}", response_model=QueryResponse)
async def decide_approval(
    run_id: str,
    request: ApprovalRequest,
    http: Request,
    db_factory=Depends(get_db_factory),
    current_user: User = Depends(limit_by_user("approval", RATE_LIMIT_APPROVAL_PER_MIN)),
    db: Session = Depends(get_db)
):
    """Approve or reject a pending run (eligible reviewer only)."""
    run = await _load_run_for(run_id, current_user, db, review=True)
    if run.get("approval_status") != "pending":
        raise HTTPException(status_code=409, detail=f"Run is {run.get('approval_status')}, not pending")
    try:
        result = await resume_pipeline(run_id, {
            "decision": request.decision,
            "reviewer_id": current_user.id,      # from token, never from body
            "notes": request.notes,
            "edited_response": request.edited_response,
        })
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors(include_url=False, include_context=False))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    audit(db_factory, "approval.decide", user_id=current_user.id, resource=f"run:{run_id}",
          ip=client_ip(http), details={"decision": request.decision,
                                       "edited": request.edited_response is not None,
                                       "requester_id": run.get("user_id")})
    return _to_response(result, result.get("session_id", ""), 0)


def log_query(db_factory, user_id, case_id, query, language, result, response_time):
    """Background task: opens its own session (the request's is closed by now)."""
    db = db_factory()
    try:
        log = QueryLog(
            user_id=user_id,
            case_id=case_id,
            query=query if STORE_QUERY_TEXT else None,
            query_hmac=keyed_hash(query, SECRET_KEY),
            query_chars=len(query),
            language=language,
            legal_domain=result.get("legal_domain", "general"),
            confidence_score=result.get("confidence_score", 0.5),
            response_time_ms=response_time,
            is_emergency=result.get("is_emergency", False),
            tokens_used=int(result.get("tokens_used") or 0),
        )
        db.add(log)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to write query log (user=%s)", user_id)
    finally:
        db.close()


def tokens_used_today(db: Session, user: User) -> int:
    """LLM tokens used since 00:00 UTC by the user's firm (or the user if solo)."""
    since = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    q = db.query(func.coalesce(func.sum(QueryLog.tokens_used), 0)).filter(QueryLog.created_at >= since)
    if user.firm_id:
        q = q.join(User, QueryLog.user_id == User.id).filter(User.firm_id == user.firm_id)
    else:
        q = q.filter(QueryLog.user_id == user.id)
    return int(q.scalar() or 0)


def get_owned_case(db: Session, case_id: str, user: User) -> Case:
    """The user's own case, or 404 (never 403: don't reveal other users' cases)."""
    case = db.query(Case).filter(Case.id == case_id, Case.user_id == user.id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


# ─── Case Routes ──────────────────────────────────────────────────────────────

@app.post("/api/cases")
async def create_case(
    request: CreateCaseRequest,
    http: Request,
    db_factory=Depends(get_db_factory),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    case = Case(
        user_id=current_user.id,
        firm_id=current_user.firm_id,
        title=request.title,
        legal_domain=request.legal_domain,
        notes=request.notes
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    audit(db_factory, "case.create", user_id=current_user.id, resource=f"case:{case.id}", ip=client_ip(http))
    return {"id": case.id, "title": case.title, "status": "created"}


@app.get("/api/cases")
async def get_cases(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    cases = db.query(Case).filter(
        Case.user_id == current_user.id
    ).order_by(Case.created_at.desc()).limit(50).all()

    return [
        {
            "id": c.id,
            "title": c.title,
            "legal_domain": c.legal_domain,
            "status": c.status,
            "created_at": c.created_at.isoformat()
        }
        for c in cases
    ]


@app.get("/api/cases/{case_id}")
async def get_case(
    case_id: str,
    http: Request,
    db_factory=Depends(get_db_factory),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    case = get_owned_case(db, case_id, current_user)
    audit(db_factory, "case.view", user_id=current_user.id, resource=f"case:{case_id}", ip=client_ip(http))

    return {
        "id": case.id,
        "title": case.title,
        "legal_domain": case.legal_domain,
        "status": case.status,
        "notes": case.notes,
        "created_at": case.created_at.isoformat()
    }


# ─── Document Routes ───────────────────────────────────────────────────────────

@app.get("/api/documents/{case_id}")
async def get_documents(
    case_id: str,
    http: Request,
    db_factory=Depends(get_db_factory),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    get_owned_case(db, case_id, current_user)
    audit(db_factory, "documents.list", user_id=current_user.id, resource=f"case:{case_id}", ip=client_ip(http))
    docs = db.query(Document).filter(
        Document.case_id == case_id,
        Document.user_id == current_user.id
    ).all()

    return [
        {
            "id": d.id,
            "doc_type": d.doc_type,
            "status": d.status,
            "created_at": d.created_at.isoformat()
        }
        for d in docs
    ]


# ─── Analytics Routes ─────────────────────────────────────────────────────────

@app.get("/api/analytics/summary")
async def get_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    total_queries = db.query(QueryLog).filter(
        QueryLog.user_id == current_user.id
    ).count()

    total_cases = db.query(Case).filter(
        Case.user_id == current_user.id
    ).count()

    return {
        "total_queries": total_queries,
        "total_cases": total_cases,
        "user_name": current_user.name,
        "user_role": current_user.role
    }
