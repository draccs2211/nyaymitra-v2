from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
import uuid
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from backend.config import ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, require_secret_key
from backend.database.connection import get_db
from backend.database.models import RevokedToken, User

SECRET_KEY = require_secret_key()   # raises at startup if missing/weak

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# bcrypt directly (passlib 1.7.4 is unmaintained and breaks with bcrypt>=4.1).
# Hashes are standard $2b$ strings, compatible with ones passlib produced.
_BCRYPT_MAX_BYTES = 72


def verify_password(plain_password: str, hashed_password: str) -> bool:
    pw = plain_password.encode("utf-8")
    if len(pw) > _BCRYPT_MAX_BYTES or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(pw, hashed_password.encode("utf-8"))
    except ValueError:      # malformed stored hash
        return False


def get_password_hash(password: str) -> str:
    pw = password.encode("utf-8")
    if len(pw) > _BCRYPT_MAX_BYTES:
        raise ValueError("password longer than 72 bytes")
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Signed JWT with exp, iat and a unique jti (for individual revocation)."""
    now = datetime.utcnow()
    to_encode = data.copy()
    to_encode.update({
        "exp": now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)),
        "iat": now,
        "jti": uuid.uuid4().hex,
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def issue_token(user: User) -> str:
    """Access token bound to the user's current token_version."""
    return create_access_token({"sub": user.id, "tv": user.token_version or 0})


def _credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_token_claims(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise _credentials_exception()
    if not payload.get("sub"):
        raise _credentials_exception()
    return payload


def get_current_user(
    claims: dict = Depends(get_token_claims),
    db: Session = Depends(get_db)
) -> User:
    """
    Valid only if: signature/expiry OK, jti not revoked (logout), user active,
    and the token's version matches the user's (logout-all / role change).
    Tokens without "tv" count as version 0.
    """
    jti = claims.get("jti")
    if jti and db.get(RevokedToken, jti) is not None:
        raise _credentials_exception()
    user = db.query(User).filter(User.id == claims["sub"]).first()
    if user is None or not user.is_active:
        raise _credentials_exception()
    if int(claims.get("tv", 0)) != (user.token_version or 0):
        raise _credentials_exception()
    return user
