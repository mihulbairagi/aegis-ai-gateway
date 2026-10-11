import sys
import os
import time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.core.rate_limiter import rate_limiter
from app.core.circuit_breaker import circuit_breaker, CircuitState
from app.core.semantic_cache import semantic_cache

client = TestClient(app)

def reset_state():
    """Clear caches and state before each test."""
    semantic_cache.l1_cache.clear()
    semantic_cache.l2_vectors.clear()
    semantic_cache.l2_entries.clear()
    semantic_cache.l2_prompts.clear()
    rate_limiter.buckets.clear()
    circuit_breaker.state = CircuitState.CLOSED
    circuit_breaker.consecutive_failures = 0

def test_cache_miss_then_l1_hit():
    reset_state()
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is Python asyncio?"}]
    }
    
    # 1. First request -> MISS
    res1 = client.post("/v1/chat/completions", json=payload)
    assert res1.status_code == 200, f"Expected 200, got {res1.status_code}"
    assert res1.headers.get("x-gateway-cache-status") == "MISS"
    assert res1.json()["cached"] is False

    # 2. Second request with exact prompt -> L1 HIT
    res2 = client.post("/v1/chat/completions", json=payload)
    assert res2.status_code == 200
    assert res2.headers.get("x-gateway-cache-status") == "HIT-L1_EXACT"
    assert res2.json()["cached"] is True
    assert res2.json()["cache_type"] == "L1_EXACT"
    print("[PASS] test_cache_miss_then_l1_hit")

def test_l2_semantic_cache_hit():
    reset_state()
    # 1. First request
    p1 = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "How do I cancel my subscription plan?"}]
    }
    res1 = client.post("/v1/chat/completions", json=p1)
    assert res1.status_code == 200
    assert res1.headers.get("x-gateway-cache-status") == "MISS"

    # 2. Semantically similar query (different wording)
    p2 = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "What is the procedure to cancel subscription plan?"}]
    }
    res2 = client.post("/v1/chat/completions", json=p2)
    assert res2.status_code == 200
    assert "HIT-L2_SEMANTIC" in res2.headers.get("x-gateway-cache-status")
    assert res2.json()["cached"] is True
    print("[PASS] test_l2_semantic_cache_hit")

def test_rate_limiter_exceeded():
    reset_state()
    headers = {"Authorization": "Bearer rate-limited-client"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Quick request"}]
    }

    # Exhaust tokens
    rate_limiter.buckets["rate-limited-client"] = (0.0, time.monotonic())

    res = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert res.status_code == 429
    assert "Retry-After" in res.headers
    print("[PASS] test_rate_limiter_exceeded")

def test_circuit_breaker_manual_trip_and_reset():
    reset_state()
    # Trip circuit
    res_trip = client.post("/v1/gateway/circuit-breaker/trip")
    assert res_trip.status_code == 200
    assert circuit_breaker.state == CircuitState.OPEN

    # Check metrics reports OPEN
    metrics = client.get("/metrics").json()
    assert metrics["circuit_breaker_state"] == "OPEN"

    # Reset circuit
    res_reset = client.post("/v1/gateway/circuit-breaker/reset")
    assert res_reset.status_code == 200
    assert circuit_breaker.state == CircuitState.CLOSED
    print("[PASS] test_circuit_breaker_manual_trip_and_reset")

def test_l2_semantic_cache_hit_paraphrase_intent():
    """Verify distinct wording with identical intent yields an L2 semantic cache hit (>=0.85 cosine sim)."""
    reset_state()
    p1 = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "cancel my order"}]
    }
    res1 = client.post("/v1/chat/completions", json=p1)
    assert res1.status_code == 200
    assert res1.headers.get("x-gateway-cache-status") == "MISS"

    # Distinct wording but identical intent ("cancel my order" vs "how to cancel subscription")
    p2 = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "how to cancel subscription"}]
    }
    res2 = client.post("/v1/chat/completions", json=p2)
    assert res2.status_code == 200
    assert "HIT-L2_SEMANTIC" in res2.headers.get("x-gateway-cache-status")
    assert res2.json()["cached"] is True
    print("[PASS] test_l2_semantic_cache_hit_paraphrase_intent")

def test_circuit_breaker_auto_failover():
    """Verify 3 consecutive failures trips circuit breaker to OPEN and routes to fallback."""
    reset_state()
    assert circuit_breaker.state == CircuitState.CLOSED

    # Simulate 3 consecutive upstream failures
    for _ in range(3):
        circuit_breaker.record_failure()

    assert circuit_breaker.state == CircuitState.OPEN
    route, state = circuit_breaker.get_route()
    assert route == "fallback"
    assert state == CircuitState.OPEN
    print("[PASS] test_circuit_breaker_auto_failover")

if __name__ == "__main__":
    test_cache_miss_then_l1_hit()
    test_l2_semantic_cache_hit()
    test_l2_semantic_cache_hit_paraphrase_intent()
    test_rate_limiter_exceeded()
    test_circuit_breaker_manual_trip_and_reset()
    test_circuit_breaker_auto_failover()
    print("\nALL 6 CORE GATEWAY TESTS PASSED SUCCESSFULLY!")
