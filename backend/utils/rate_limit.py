"""
In-memory sliding-window rate limiter (no extra dependency).

Per-process: with several uvicorn workers each worker counts separately, so
the effective limit is limit x workers. Move to Redis when running multi-worker.
"""
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from fastapi import Depends, HTTPException, Request

from backend.database.models import User
from backend.utils.auth import get_current_user


class SlidingWindowLimiter:
    def __init__(self):
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window: float) -> float:
        """Record a hit. Returns 0 if allowed, else seconds until a slot frees up."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= window:
                q.popleft()
            if len(q) >= limit:
                return max(0.0, window - (now - q[0]))
            q.append(now)
            return 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def _client_ip(request: Request) -> str:
    # Behind a reverse proxy, configure uvicorn --proxy-headers so this is the real client.
    return request.client.host if request.client else "unknown"


def _reject(retry_after: float):
    raise HTTPException(
        status_code=429,
        detail="Too many requests. Please slow down.",
        headers={"Retry-After": str(int(retry_after) + 1)},
    )


def limit_by_ip(scope: str, limit: int, window: float = 60.0) -> Callable:
    """Dependency for unauthenticated endpoints (login/register)."""
    def dependency(request: Request) -> None:
        wait = limiter.hit(f"{scope}:ip:{_client_ip(request)}", limit, window)
        if wait:
            _reject(wait)
    return dependency


def limit_by_user(scope: str, limit: int, window: float = 60.0) -> Callable:
    """Dependency for authenticated endpoints; returns the current user."""
    def dependency(current_user: User = Depends(get_current_user)) -> User:
        wait = limiter.hit(f"{scope}:user:{current_user.id}", limit, window)
        if wait:
            _reject(wait)
        return current_user
    return dependency
