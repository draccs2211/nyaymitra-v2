from sqlalchemy import Column, String, DateTime, Float, Boolean, Text, Integer, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
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
    firm_id = Column(String, ForeignKey("firms.id"), nullable=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="associate")   # admin, senior, associate
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    firm = relationship("Firm", back_populates="users")
    cases = relationship("Case", back_populates="user")
    queries = relationship("QueryLog", back_populates="user")


class Case(Base):
    __tablename__ = "cases"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    firm_id = Column(String, ForeignKey("firms.id"), nullable=True)
    title = Column(String, nullable=False)
    legal_domain = Column(String)
    case_strength = Column(String)
    status = Column(String, default="active")   # active, closed, archived
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="cases")
    firm = relationship("Firm", back_populates="cases")
    documents = relationship("Document", back_populates="case")
    queries = relationship("QueryLog", back_populates="case")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    case_id = Column(String, ForeignKey("cases.id"), nullable=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    doc_type = Column(String)                    # legal_notice, rti, complaint
    content_hindi = Column(Text)
    content_english = Column(Text)
    status = Column(String, default="draft")     # draft, final, sent
    created_at = Column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="documents")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    case_id = Column(String, ForeignKey("cases.id"), nullable=True)
    query = Column(Text, nullable=True)            # raw text only if STORE_QUERY_TEXT=true
    query_hmac = Column(String(64), index=True)    # HMAC-SHA256 of the query (correlation without text)
    query_chars = Column(Integer)
    language = Column(String)
    legal_domain = Column(String)
    confidence_score = Column(Float)
    response_time_ms = Column(Integer)
    is_emergency = Column(Boolean, default=False)
    tokens_used = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="queries")
    case = relationship("Case", back_populates="queries")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    resource = Column(String)
    details = Column(JSON)
    ip_address = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
