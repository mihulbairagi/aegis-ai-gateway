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
    Limits concurrent probes in HALF_OPEN to avoid overwhelming struggling upstreams.
    """
    def __init__(
        self,
        failure_threshold: int = settings.CIRCUIT_FAILURE_THRESHOLD,
        recovery_timeout: float = settings.CIRCUIT_RECOVERY_TIMEOUT,
        half_open_max_probes: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_probes = half_open_max_probes
        
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_state_change = time.monotonic()
        self.last_failure_time = 0.0
        self.inflight_probes = 0
        self.last_probe_time = 0.0

    def peek_state(self) -> CircuitState:
        """Read-only view of the state (no transitions, no probe slot consumed). Use for /metrics."""
        return self.state

    def release_probe(self):
        """Free a HALF_OPEN probe slot when a request ends without a success/failure verdict
        (4xx, client disconnect, cancelled stream)."""
        self.inflight_probes = max(0, self.inflight_probes - 1)

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
                self.inflight_probes = 1
                self.last_probe_time = now
                return "primary", self.state
            return "fallback", self.state

        elif self.state == CircuitState.HALF_OPEN:
            # Only permit up to half_open_max_probes to primary; excess traffic routes safely to fallback
            # Self-heal: a probe that never reported back must not wedge the breaker forever.
            if self.inflight_probes > 0 and (time.monotonic() - self.last_probe_time) > self.recovery_timeout:
                self.inflight_probes = 0
            if self.inflight_probes < self.half_open_max_probes:
                self.inflight_probes += 1
                self.last_probe_time = time.monotonic()
                return "primary", self.state
            return "fallback", self.state

        # CLOSED state
        return "primary", self.state

    def record_success(self):
        """Called when an upstream request succeeds."""
        if self.state == CircuitState.HALF_OPEN:
            self.inflight_probes = max(0, self.inflight_probes - 1)
            self.consecutive_successes += 1
            if self.consecutive_successes >= 2:
                # Primary proven healthy again, close circuit
                self.state = CircuitState.CLOSED
                self.consecutive_failures = 0
                self.inflight_probes = 0
                self.last_state_change = time.monotonic()
        elif self.state == CircuitState.CLOSED:
            self.consecutive_failures = 0

    def record_failure(self):
        """Called when an upstream request times out or returns 5xx."""
        now = time.monotonic()
        self.last_failure_time = now
        self.inflight_probes = 0

        if self.state == CircuitState.CLOSED:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.last_state_change = now

        elif self.state == CircuitState.HALF_OPEN:
            # Probe failed, trip back to OPEN immediately
            self.state = CircuitState.OPEN
            self.last_state_change = now

    def reset(self):
        """Reset circuit breaker to pristine CLOSED state."""
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.inflight_probes = 0
        self.last_probe_time = 0.0
        self.last_state_change = time.monotonic()
        self.last_failure_time = 0.0

circuit_breaker = CircuitBreaker()
