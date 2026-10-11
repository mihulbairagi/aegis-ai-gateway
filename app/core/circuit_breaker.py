import time
from enum import Enum
from typing import Tuple
from app.config import settings

class CircuitState(str, Enum):
    CLOSED = "CLOSED"       # Normal operation, primary provider healthy
    OPEN = "OPEN"           # Primary down, all traffic routed directly to fallback
    HALF_OPEN = "HALF_OPEN" # Testing primary with probe requests

class CircuitBreaker:
    """
    State machine for upstream provider resiliency and automatic failover.
    Protects downstream latency when OpenAI/Anthropic/Gemini fail or return 500/503.
    """
    def __init__(
        self,
        failure_threshold: int = settings.CIRCUIT_FAILURE_THRESHOLD,
        recovery_timeout: float = settings.CIRCUIT_RECOVERY_TIMEOUT
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_state_change = time.monotonic()
        self.last_failure_time = 0.0

    def get_route(self) -> Tuple[str, CircuitState]:
        """
        Determines which upstream provider to route to.
        Returns: ('primary' | 'fallback', CircuitState)
        """
        now = time.monotonic()

        if self.state == CircuitState.OPEN:
            if now - self.last_state_change > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                self.consecutive_successes = 0
                return "primary", self.state
            return "fallback", self.state

        elif self.state == CircuitState.HALF_OPEN:
            # Send single probe to primary
            return "primary", self.state

        # CLOSED state
        return "primary", self.state

    def record_success(self):
        """Called when an upstream request succeeds."""
        if self.state == CircuitState.HALF_OPEN:
            self.consecutive_successes += 1
            if self.consecutive_successes >= 2:
                # Primary proven healthy again, close circuit
                self.state = CircuitState.CLOSED
                self.consecutive_failures = 0
                self.last_state_change = time.monotonic()
        elif self.state == CircuitState.CLOSED:
            self.consecutive_failures = 0

    def record_failure(self):
        """Called when an upstream request times out or returns 5xx."""
        now = time.monotonic()
        self.last_failure_time = now
        self.consecutive_failures += 1

        if self.state == CircuitState.CLOSED and self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_state_change = now

        elif self.state == CircuitState.HALF_OPEN:
            # Probe failed, trip back to OPEN immediately
            self.state = CircuitState.OPEN
            self.last_state_change = now

circuit_breaker = CircuitBreaker()
