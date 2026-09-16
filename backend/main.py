from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import timedelta
import uuid
import time
import os

from backend.config import ACCESS_TOKEN_EXPIRE_MINUTES
from backend.database.connection import get_db, init_db
from backend.database.models import User, Firm, Case, Document, QueryLog
from backend.utils.auth import (
    verify_password, get_password_hash,
    create_access_token, get_current_user
)
from backend.utils.cache import (
    get_cached_query, set_cached_query,
    store_session, get_session
)
from backend.utils.language_detector import detect_language
from backend.pipeline import run_pipeline
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="NyayMitra 2.0 API",
    description="AI-powered legal intelligence platform for Indian lawyers",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    firm_name: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    case_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    response: str
    confidence_score: float
    legal_domain: str
    case_strength: str
    sources: List[str]
    next_steps: List[str]
    has_document: bool
    document_hindi: Optional[str] = None
    document_english: Optional[str] = None
    is_emergency: bool
    human_approval_needed: bool
    response_time_ms: int

class CreateCaseRequest(BaseModel):
    title: str
    legal_domain: Optional[str] = "general"
    notes: Optional[str] = ""

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_name: str
    user_role: str


# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    try:
        init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️ Database init skipped: {e}")


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
    return {"status": "healthy", "version": "2.0.0"}


# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
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


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

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
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # Get conversation history from session
    session_data = get_session(session_id) or {}
    conversation_history = session_data.get("history", [])

    # Detect language
    language = detect_language(request.query)

    # Check query cache
    cached = get_cached_query(request.query, "")
    if cached and not request.session_id:
        response_time = int((time.time() - start_time) * 1000)
        return QueryResponse(
            session_id=session_id,
            response=cached.get("final_response", ""),
            confidence_score=cached.get("confidence_score", 0.7),
            legal_domain=cached.get("legal_domain", "general"),
            case_strength=cached.get("case_strength", "medium"),
            sources=cached.get("verified_sources", []),
            next_steps=cached.get("next_steps", []),
            has_document=bool(cached.get("draft_document")),
            document_hindi=cached.get("document_hindi"),
            document_english=cached.get("document_english"),
            is_emergency=cached.get("is_emergency", False),
            human_approval_needed=cached.get("human_approval_needed", False),
            response_time_ms=response_time
        )

    # Run pipeline
    try:
        result = await run_pipeline(
            query=request.query,
            session_id=session_id,
            user_id=current_user.id,
            conversation_history=conversation_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    response_time = int((time.time() - start_time) * 1000)

    # Update conversation history
    conversation_history.append({"role": "user", "content": request.query})
    conversation_history.append({"role": "assistant", "content": result.get("final_response", "")})
    store_session(session_id, {"history": conversation_history[-20:]})  # Keep last 20 turns

    # Cache non-emergency responses
    if not result.get("is_emergency") and not request.session_id:
        set_cached_query(request.query, "", result)

    # Log query in background
    background_tasks.add_task(
        log_query, db, current_user.id, request.case_id,
        request.query, language, result, response_time
    )

    return QueryResponse(
        session_id=session_id,
        response=result.get("final_response", "No response generated"),
        confidence_score=result.get("confidence_score", 0.5),
        legal_domain=result.get("legal_domain", "general"),
        case_strength=result.get("case_strength", "medium"),
        sources=result.get("verified_sources", []),
        next_steps=result.get("next_steps", []),
        has_document=bool(result.get("draft_document")),
        document_hindi=result.get("document_hindi"),
        document_english=result.get("document_english"),
        is_emergency=result.get("is_emergency", False),
        human_approval_needed=result.get("human_approval_needed", False),
        response_time_ms=response_time
    )


def log_query(db, user_id, case_id, query, language, result, response_time):
    try:
        log = QueryLog(
            user_id=user_id,
            case_id=case_id,
            query=query,
            language=language,
            legal_domain=result.get("legal_domain", "general"),
            confidence_score=result.get("confidence_score", 0.5),
            response_time_ms=response_time,
            is_emergency=result.get("is_emergency", False)
        )
        db.add(log)
        db.commit()
    except Exception:
        pass


# ─── Case Routes ──────────────────────────────────────────────────────────────

@app.post("/api/cases")
async def create_case(
    request: CreateCaseRequest,
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
            "case_strength": c.case_strength,
            "status": c.status,
            "created_at": c.created_at.isoformat()
        }
        for c in cases
    ]


@app.get("/api/cases/{case_id}")
async def get_case(
    case_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    case = db.query(Case).filter(
        Case.id == case_id,
        Case.user_id == current_user.id
    ).first()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return {
        "id": case.id,
        "title": case.title,
        "legal_domain": case.legal_domain,
        "case_strength": case.case_strength,
        "status": case.status,
        "notes": case.notes,
        "created_at": case.created_at.isoformat()
    }


# ─── Document Routes ───────────────────────────────────────────────────────────

@app.get("/api/documents/{case_id}")
async def get_documents(
    case_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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
