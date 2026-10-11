import time
from typing import Dict, Tuple
from app.config import settings

class TokenBucketRateLimiter:
    """
    High-performance in-memory Token Bucket rate limiter with sliding replenishment.
    Keyed per tenant_id with automatic stale client eviction to prevent memory leaks.
    """
    def __init__(self, rate_rps: int = settings.DEFAULT_RATE_LIMIT_RPS, burst_capacity: int = settings.DEFAULT_BURST_CAPACITY):
        self.rate_rps = rate_rps
        self.capacity = burst_capacity
        # Storage format: client_id -> (current_tokens: float, last_updated_timestamp: float)
        self.buckets: Dict[str, Tuple[float, float]] = {}
        self.last_cleanup_time = time.monotonic()

    def is_allowed(self, client_id: str) -> Tuple[bool, float]:
        """
        Check if request is allowed for this client_id / tenant_id.
        Returns: (allowed: bool, retry_after_seconds: float)
        """
        now = time.monotonic()

        # Opportunistic lazy cleanup every 60 seconds to purge inactive clients
        if now - self.last_cleanup_time > 60.0:
            self.cleanup_stale_buckets(ttl_seconds=1800.0)
            self.last_cleanup_time = now

        if client_id not in self.buckets:
            # First request from this client
            self.buckets[client_id] = (self.capacity - 1.0, now)
            return True, 0.0

        current_tokens, last_time = self.buckets[client_id]

        # Calculate tokens replenished since last request
        elapsed = now - last_time
        replenished = elapsed * self.rate_rps
        new_tokens = min(self.capacity, current_tokens + replenished)

        if new_tokens >= 1.0:
            # Consume 1 token
            self.buckets[client_id] = (new_tokens - 1.0, now)
            return True, 0.0
        else:
            # Rate limit exceeded. Calculate wait time for at least 1 token.
            tokens_needed = 1.0 - new_tokens
            retry_after = tokens_needed / self.rate_rps
            self.buckets[client_id] = (new_tokens, now)
            return False, max(0.01, retry_after)

    def cleanup_stale_buckets(self, ttl_seconds: float = 1800.0):
        """Clean up inactive tenant/client buckets to prevent memory leaks."""
        now = time.monotonic()
        stale_keys = [
            k for k, (_, last_time) in self.buckets.items()
            if now - last_time > ttl_seconds
        ]
        for k in stale_keys:
            del self.buckets[k]

rate_limiter = TokenBucketRateLimiter()
