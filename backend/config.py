from dotenv import load_dotenv
import os

load_dotenv()

# LLM
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# External APIs
INDIAN_KANOON_API_KEY = os.getenv("INDIAN_KANOON_API_KEY", "")
INDIAN_KANOON_BASE_URL = "https://api.indiankanoon.org"

# LangSmith
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "nyaymitra-v2")

# Database
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/nyaymitra")

# Auth
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# RAG
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PERSIST_DIR = "./data/chroma_db"
TOP_K_RESULTS = 10
RERANK_TOP_K = 5

# Confidence
CONFIDENCE_THRESHOLD = 0.70
EMERGENCY_THRESHOLD = 0.85

# Cache TTL (seconds)
QUERY_CACHE_TTL = 86400       # 24 hours
EMBEDDING_CACHE_TTL = 604800  # 7 days
LLM_CACHE_TTL = 43200         # 12 hours
KANOON_CACHE_TTL = 86400      # 24 hours

# Emergency keywords
EMERGENCY_KEYWORDS_HI = [
    "maar raha", "maar rahi", "jaan ka khatra", "jaan lene",
    "domestic violence", "gharelu hinsa", "rape", "balatkaar",
    "kidnapping", "apaharan", "immediate threat", "turant khatra",
    "suicide", "aatmahatya", "help me", "bachao", "mujhe bachao"
]

EMERGENCY_KEYWORDS_EN = [
    "killing me", "threat to life", "domestic violence",
    "rape", "kidnapping", "immediate danger", "help me",
    "suicide", "assault", "physical abuse"
]

# Legal domains
LEGAL_DOMAINS = [
    "criminal", "civil", "family", "property",
    "consumer", "rti", "labor", "constitutional", "general"
]
