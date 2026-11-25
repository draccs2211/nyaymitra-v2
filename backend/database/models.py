from sqlalchemy import Column, String, DateTime, Float, Boolean, Text, Integer, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import uuid

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class Firm(Base):
    __tablename__ = "firms"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    plan = Column(String, default="free")       # free, pro, firm
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="firm")
    cases = relationship("Case", back_populates="firm")


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    firm_id = Column(String, ForeignKey("firms.id"), nullable=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="associate")   # admin, senior, associate
    is_active = Column(Boolean, default=True)
    # Bumped on logout-all / role change / deactivation: invalidates every older token
    token_version = Column(Integer, nullable=False, default=0, server_default="0")
    created_at = Column(DateTime, default=datetime.utcnow)

    firm = relationship("Firm", back_populates="users")
    cases = relationship("Case", back_populates="user")
    queries = relationship("QueryLog", back_populates="user")


class Invite(Base):
    """One-time firm invitation. Only the SHA-256 of the token is stored."""
    __tablename__ = "invites"

    id = Column(String, primary_key=True, default=generate_uuid)
    token_hash = Column(String(64), unique=True, nullable=False)
    firm_id = Column(String, ForeignKey("firms.id"), nullable=False, index=True)
    email = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    created_by = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    used_by = Column(String, ForeignKey("users.id"), nullable=True)


class RevokedToken(Base):
    """Individually revoked JWTs (logout). Rows can be purged after expires_at."""
    __tablename__ = "revoked_tokens"

    jti = Column(String, primary_key=True)
    expires_at = Column(DateTime, nullable=False, index=True)


class Case(Base):
    __tablename__ = "cases"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    firm_id = Column(String, ForeignKey("firms.id"), nullable=True, index=True)
    title = Column(String, nullable=False)
    legal_domain = Column(String)
    status = Column(String, default="active")   # active, closed, archived
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="cases")
    firm = relationship("Firm", back_populates="cases")
    documents = relationship("Document", back_populates="case")
    queries = relationship("QueryLog", back_populates="case")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    case_id = Column(String, ForeignKey("cases.id"), nullable=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    run_id = Column(String, nullable=True, index=True)   # pipeline run that produced it
    doc_type = Column(String)                    # legal_notice, rti, complaint
    content_hindi = Column(Text)
    content_english = Column(Text)
    status = Column(String, default="draft")     # draft, final (approved), rejected, sent
    created_at = Column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="documents")


class Approval(Base):
    """
    Durable record of a run awaiting / given human review. Holds no client
    text (that lives in the run state). Authorisation and the reviewer queue
    use this table, so they keep working even if the run state is lost.
    """
    __tablename__ = "approvals"

    run_id = Column(String, primary_key=True)
    requester_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    firm_id = Column(String, ForeignKey("firms.id"), nullable=True, index=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=True)
    document_type = Column(String)
    status = Column(String, nullable=False, default="pending", index=True)  # pending/approved/rejected/expired
    reviewer_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    decided_at = Column(DateTime, nullable=True)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=True, index=True)
    query = Column(Text, nullable=True)            # raw text only if STORE_QUERY_TEXT=true
    query_hmac = Column(String(64), index=True)    # HMAC-SHA256 of the query (correlation without text)
    query_chars = Column(Integer)
    language = Column(String)
    legal_domain = Column(String)
    confidence_score = Column(Float)
    response_time_ms = Column(Integer)
    is_emergency = Column(Boolean, default=False)
    tokens_used = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="queries")
    case = relationship("Case", back_populates="queries")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    action = Column(String, nullable=False, index=True)
    resource = Column(String)
    details = Column(JSON)
    ip_address = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
