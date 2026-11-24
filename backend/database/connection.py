from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.config import DATABASE_URL
from backend.database.models import Base
import redis
from backend.config import REDIS_URL

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_factory():
    """Session factory for work that outlives the request (background tasks)."""
    return SessionLocal


def init_db():
    Base.metadata.create_all(bind=engine)


# Short timeouts: a down Redis must fail fast (cache is optional)
redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=1.0,
)


def get_redis():
    return redis_client
