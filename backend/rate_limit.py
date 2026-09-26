"""Per-IP rate limiting (sliding window, in-memory).

Generation is autoregressive and far more expensive per request than a
single ESM2 embedding batch, so /generate/* gets a stricter limit than
/predict/*. Limits come from backend.config and are per IP per minute.
"""

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from . import config


class RateLimiter:
    def __init__(self, limit_per_minute: int, scope: str):
        self.limit = limit_per_minute
        self.scope = scope
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - 60.0
        hits = self._hits[ip]
        while hits and hits[0] <= window_start:
            hits.popleft()
        if len(hits) >= self.limit:
            retry_after = max(1, int(hits[0] + 60.0 - now))
            raise HTTPException(
                status_code=429,
                detail={
                    "message": f"Rate limit exceeded for {self.scope} endpoints "
                    f"({self.limit} requests/minute per IP). Retry after "
                    f"{retry_after}s.",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)


predict_limiter = RateLimiter(config.PREDICT_PER_MINUTE_PER_IP, "predict")
generate_limiter = RateLimiter(config.GENERATE_PER_MINUTE_PER_IP, "generate")


async def check_predict_rate(request: Request) -> None:
    predict_limiter.check(request)


async def check_generate_rate(request: Request) -> None:
    generate_limiter.check(request)
