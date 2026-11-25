from dotenv import load_dotenv
import os

load_dotenv()

# LLM
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# LLM call limits
LLM_MODEL = "gpt-4o-mini"
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# Daily token budget per firm (per user if no firm); 0 = unlimited
DAILY_TOKEN_BUDGET = int(os.getenv("DAILY_TOKEN_BUDGET", "2000000"))

# External APIs
INDIAN_KANOON_API_KEY = os.getenv("INDIAN_KANOON_API_KEY", "")
INDIAN_KANOON_BASE_URL = "https://api.indiankanoon.org"

# Environment: "production" (default, strictest), "development" or "test".
APP_ENV = os.getenv("APP_ENV", "production").strip().lower()
IS_PRODUCTION = APP_ENV == "production"


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def configure_tracing(env=os.environ) -> dict:
    """
    LangSmith reads its settings from the environment. Tracing is only on if
    LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 is explicitly true. In production,
    trace inputs/outputs (client facts, drafts) are hidden unless
    LANGSMITH_ALLOW_RAW_TRACES=true; timings, errors and run structure remain.
    """
    enabled = _truthy(env.get("LANGSMITH_TRACING")) or _truthy(env.get("LANGCHAIN_TRACING_V2"))
    production = env.get("APP_ENV", "production").strip().lower() == "production"
    hide = enabled and production and not _truthy(env.get("LANGSMITH_ALLOW_RAW_TRACES"))
    if hide:
        env["LANGSMITH_HIDE_INPUTS"] = "true"
        env["LANGSMITH_HIDE_OUTPUTS"] = "true"
    return {"enabled": enabled, "hide_payloads": hide}


TRACING = configure_tracing()

# Data retention / minimisation (DPDP Act 2023)
STORE_QUERY_TEXT = _truthy(os.getenv("STORE_QUERY_TEXT", "false"))   # raw client text in query_logs
RETENTION_QUERY_LOG_DAYS = int(os.getenv("RETENTION_QUERY_LOG_DAYS", "90"))
RETENTION_AUDIT_DAYS = int(os.getenv("RETENTION_AUDIT_DAYS", "365"))

# Database
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/nyaymitra")

# Auth
# No fallback: a missing/weak key must stop the app (see require_secret_key()).
SECRET_KEY = os.getenv("SECRET_KEY", "")
_PLACEHOLDER_SECRETS = {"your-secret-key-change-in-production", "changeme", "secret"}
MIN_SECRET_KEY_LENGTH = 32


def require_secret_key(key: str = None) -> str:
    """Return the JWT signing key or raise if it is missing, short or a placeholder."""
    key = SECRET_KEY if key is None else key
    if not key or key in _PLACEHOLDER_SECRETS or len(key) < MIN_SECRET_KEY_LENGTH:
        raise RuntimeError(
            "SECRET_KEY must be set in .env to a random value of at least "
            f"{MIN_SECRET_KEY_LENGTH} characters. Generate one with: "
            'python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
    return key


# CORS: comma-separated allowed origins (no wildcard)
CORS_ORIGINS = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",") if o.strip()
]

# Rate limits (requests per minute)
RATE_LIMIT_AUTH_PER_MIN = int(os.getenv("RATE_LIMIT_AUTH_PER_MIN", "10"))
RATE_LIMIT_QUERY_PER_MIN = int(os.getenv("RATE_LIMIT_QUERY_PER_MIN", "20"))
RATE_LIMIT_APPROVAL_PER_MIN = int(os.getenv("RATE_LIMIT_APPROVAL_PER_MIN", "60"))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# RAG
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PERSIST_DIR = "./data/chroma_db"
TOP_K_RESULTS = 10
RERANK_TOP_K = 5

# Confidence
CONFIDENCE_THRESHOLD = 0.70

# Cache TTL (seconds)
QUERY_CACHE_TTL = 86400       # 24 hours
EMBEDDING_CACHE_TTL = 604800  # 7 days
LLM_CACHE_TTL = 43200         # 12 hours
KANOON_CACHE_TTL = 86400      # 24 hours

# Legal domains
LEGAL_DOMAINS = [
    "criminal", "civil", "family", "property",
    "consumer", "rti", "labor", "constitutional", "general"
]

# MCP clients (timeout per attempt, retries after the first attempt)
MCP_TIMEOUT_SECONDS = float(os.getenv("MCP_TIMEOUT_SECONDS", "8"))
MCP_RETRIES = int(os.getenv("MCP_RETRIES", "1"))
