import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Response, status, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings, authenticate_tenant, verify_admin_key
from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse, GatewayMetricsResponse
from app.core.rate_limiter import rate_limiter
from app.core.semantic_cache import semantic_cache
from app.core.circuit_breaker import circuit_breaker, CircuitState
from app.core.proxy import proxy_client
from app.core.metrics import metrics_tracker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan task for periodic rate limiter bucket cleanup."""
    async def periodic_bucket_cleaner():
        while True:
            try:
                await asyncio.sleep(60.0)
                rate_limiter.cleanup_stale_buckets(ttl_seconds=1800.0)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    cleaner_task = asyncio.create_task(periodic_bucket_cleaner())
    yield
    cleaner_task.cancel()


app = FastAPI(
    title="Aegis AI Gateway & Semantic Cache",
    description="High-Throughput Enterprise AI Reverse Proxy with L1/L2 Semantic Caching, Token-Bucket Rate Limiting, Multi-Tenant Isolation, and Circuit Breakers.",
    version="1.2.0",
    lifespan=lifespan
)

# Enable CORS for dashboards and frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_api_token(request: Request) -> str:
    """Extracts bearer token or API key from request headers."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return request.headers.get("X-API-Key", "").strip()


def _require_admin(request: Request):
    """Enforces admin API key authentication for protected operations."""
    admin_key = request.headers.get("X-Admin-Key", "")
    if not admin_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            admin_key = auth_header[7:].strip()

    if not verify_admin_key(admin_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid or missing admin API key (Pass via 'X-Admin-Key' or 'Authorization: Bearer <ADMIN_API_KEY>')"
        )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest, request: Request, response: Response):
    start_time = time.monotonic()

    # 1. Reject streaming requests with explicit 400 Bad Request
    if req.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Streaming SSE not supported in this gateway version. Set stream: false or omit the stream parameter."
        )

    # 2. Client Authentication & Multi-Tenant Identification
    token = _get_api_token(request)
    tenant_id = authenticate_tenant(token)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing client API key. Pre-configured dev keys include: 'sk-aegis-tenant-alpha', 'sk-aegis-tenant-beta', 'sk-aegis-test', 'sk-aegis-benchmark'."
        )

    # 3. Rate Limiting (Token Bucket per tenant)
    allowed, retry_after = rate_limiter.is_allowed(tenant_id)
    if not allowed:
        retry_val = str(round(retry_after, 2))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for tenant '{tenant_id}'. Retry after {retry_val} seconds.",
            headers={"Retry-After": retry_val}
        )

    # 4. Cache-Control & TTL extraction
    bypass_cache = (
        request.headers.get("X-Bypass-Cache", "").lower() in ("true", "1") or
        request.headers.get("Cache-Control", "").lower() == "no-cache"
    )
    raw_ttl = request.headers.get("X-Cache-TTL")
    custom_ttl = float(raw_ttl) if raw_ttl and raw_ttl.isdigit() else None

    # 5. Check Tiered Multi-Tenant Cache (L1 Exact & L2 Semantic Cosine Match)
    cached_resp, cache_type = await semantic_cache.get(
        req,
        tenant_id=tenant_id,
        bypass_cache=bypass_cache
    )
    if cached_resp:
        latency_ms = (time.monotonic() - start_time) * 1000.0
        cached_resp.latency_ms = round(latency_ms, 2)
        
        # Track metrics
        saved_tokens = cached_resp.usage.total_tokens
        metrics_tracker.record_request(latency_ms, cache_type=cache_type, tokens_saved=saved_tokens)
        
        # Set observability headers
        response.headers["X-Gateway-Tenant-Id"] = tenant_id
        response.headers["X-Gateway-Cache-Status"] = f"HIT-{cache_type}"
        response.headers["X-Gateway-Latency-Ms"] = str(round(latency_ms, 2))
        return cached_resp

    # 6. Cache Miss -> Forward to Upstream Provider via Circuit Breaker
    upstream_resp, provider = await proxy_client.forward(req)
    latency_ms = (time.monotonic() - start_time) * 1000.0

    upstream_resp.latency_ms = round(latency_ms, 2)
    upstream_resp.provider = provider
    upstream_resp.cached = False

    # 7. Store in Cache (partitioned strictly by tenant)
    await semantic_cache.put(
        req,
        upstream_resp,
        tenant_id=tenant_id,
        bypass_cache=bypass_cache,
        ttl_seconds=custom_ttl
    )

    # 8. Record telemetry
    metrics_tracker.record_request(latency_ms, cache_type=None)

    response.headers["X-Gateway-Tenant-Id"] = tenant_id
    response.headers["X-Gateway-Cache-Status"] = "MISS"
    response.headers["X-Gateway-Latency-Ms"] = str(round(latency_ms, 2))
    response.headers["X-Gateway-Provider"] = provider
    return upstream_resp


@app.get("/metrics", response_model=GatewayMetricsResponse)
async def get_metrics():
    summary = metrics_tracker.get_summary()
    route, state = circuit_breaker.get_route()
    summary["circuit_breaker_state"] = state.value
    summary["active_provider"] = route
    return summary


# --- Protected Admin Endpoints ---

@app.post("/v1/gateway/circuit-breaker/trip")
async def force_trip_circuit(request: Request):
    """Admin endpoint: Simulate upstream outage for chaos testing."""
    _require_admin(request)
    circuit_breaker.state = CircuitState.OPEN
    circuit_breaker.last_state_change = time.monotonic()
    return {"message": "Circuit breaker forced to OPEN state (Failover active)"}


@app.post("/v1/gateway/circuit-breaker/reset")
async def reset_circuit(request: Request):
    """Admin endpoint: Reset circuit breaker to healthy state."""
    _require_admin(request)
    circuit_breaker.state = CircuitState.CLOSED
    circuit_breaker.consecutive_failures = 0
    return {"message": "Circuit breaker reset to CLOSED state"}


@app.delete("/v1/gateway/cache")
async def flush_cache(request: Request):
    """Admin endpoint: Purge all L1 and L2 tenant caches."""
    _require_admin(request)
    semantic_cache.clear()
    return {"message": "All L1 & L2 tenant caches purged successfully"}


@app.get("/dashboard", response_class=HTMLResponse)
async def live_dashboard():
    """High-visibility web dashboard displaying real-time metrics."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Aegis AI Gateway & Semantic Cache</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <meta http-equiv="refresh" content="3">
    </head>
    <body class="bg-slate-950 text-slate-100 font-sans p-6 min-h-screen">
        <div class="max-w-6xl mx-auto space-y-6">
            <header class="flex justify-between items-center border-b border-slate-800 pb-4">
                <div>
                    <h1 class="text-3xl font-extrabold text-indigo-400 tracking-tight">AEGIS AI GATEWAY</h1>
                    <p class="text-slate-400 text-sm mt-1">Multi-Tenant Semantic Reverse Proxy • ONNX Dense Cache • Rate Limiting • Circuit Breakers</p>
                </div>
                <div class="flex items-center space-x-3">
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                        ● LIVE CLUSTER
                    </span>
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                        PORT 8000
                    </span>
                </div>
            </header>

            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                    <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">L1 Exact Hits</div>
                    <div id="l1-hits" class="text-2xl font-bold text-white mt-1">--</div>
                    <div class="text-xs text-slate-400 mt-1">&lt;0.1ms LRU in-memory</div>
                </div>
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                    <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">L2 Semantic Hits</div>
                    <div id="l2-hits" class="text-2xl font-bold text-indigo-400 mt-1">--</div>
                    <div class="text-xs text-slate-400 mt-1">ONNX Dense Vector Cosine</div>
                </div>
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                    <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Cache Hit Ratio</div>
                    <div id="hit-ratio" class="text-2xl font-bold text-emerald-400 mt-1">--%</div>
                    <div class="text-xs text-slate-400 mt-1">Token avoidance</div>
                </div>
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
                    <div class="text-xs font-medium text-slate-400 uppercase tracking-wider">Estimated Savings</div>
                    <div id="savings" class="text-2xl font-bold text-amber-400 mt-1">$0.00</div>
                    <div class="text-xs text-slate-400 mt-1">Direct token cost avoided</div>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5">
                    <h2 class="text-lg font-bold text-emerald-400 mb-3">⚡ In-Memory Cache Latency</h2>
                    <div class="space-y-3">
                        <div class="flex justify-between items-center text-sm border-b border-slate-800 pb-2">
                            <span class="text-slate-400">p50 Median</span>
                            <span id="hit-p50" class="font-mono text-emerald-400 font-bold">-- ms</span>
                        </div>
                        <div class="flex justify-between items-center text-sm border-b border-slate-800 pb-2">
                            <span class="text-slate-400">p95 Percentile</span>
                            <span id="hit-p95" class="font-mono text-emerald-400 font-bold">-- ms</span>
                        </div>
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">p99 Tail</span>
                            <span id="hit-p99" class="font-mono text-emerald-400 font-bold">-- ms</span>
                        </div>
                    </div>
                </div>

                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5">
                    <h2 class="text-lg font-bold text-amber-400 mb-3">🌐 Upstream Round-Trip Latency</h2>
                    <div class="space-y-3">
                        <div class="flex justify-between items-center text-sm border-b border-slate-800 pb-2">
                            <span class="text-slate-400">p50 Median</span>
                            <span id="miss-p50" class="font-mono text-amber-400 font-bold">-- ms</span>
                        </div>
                        <div class="flex justify-between items-center text-sm border-b border-slate-800 pb-2">
                            <span class="text-slate-400">p95 Percentile</span>
                            <span id="miss-p95" class="font-mono text-amber-400 font-bold">-- ms</span>
                        </div>
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">p99 Tail</span>
                            <span id="miss-p99" class="font-mono text-amber-400 font-bold">-- ms</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 flex justify-between items-center">
                <div>
                    <span class="text-xs uppercase text-slate-400 font-semibold">Circuit Breaker Status</span>
                    <div id="cb-state" class="text-lg font-bold text-emerald-400 mt-1">CLOSED</div>
                </div>
                <div>
                    <span class="text-xs uppercase text-slate-400 font-semibold">Active Provider</span>
                    <div id="active-provider" class="text-lg font-bold text-white mt-1">--</div>
                </div>
                <div>
                    <span class="text-xs uppercase text-slate-400 font-semibold">Total Requests</span>
                    <div id="total-reqs" class="text-lg font-bold text-white mt-1">--</div>
                </div>
            </div>
        </div>

        <script>
            async function updateDashboard() {
                try {
                    const res = await fetch('/metrics');
                    const data = await res.json();
                    document.getElementById('l1-hits').innerText = data.l1_cache_hits;
                    document.getElementById('l2-hits').innerText = data.l2_cache_hits;
                    document.getElementById('hit-ratio').innerText = data.cache_hit_ratio_percent + '%';
                    document.getElementById('savings').innerText = '$' + data.estimated_cost_saved_usd.toFixed(4);
                    document.getElementById('total-reqs').innerText = data.total_requests;
                    document.getElementById('cb-state').innerText = data.circuit_breaker_state;
                    document.getElementById('active-provider').innerText = data.active_provider;
                    document.getElementById('hit-p50').innerText = data.cache_hit_p50_latency_ms + ' ms';
                    document.getElementById('hit-p95').innerText = data.cache_hit_p95_latency_ms + ' ms';
                    document.getElementById('hit-p99').innerText = data.cache_hit_p99_latency_ms + ' ms';
                    document.getElementById('miss-p50').innerText = data.cache_miss_p50_latency_ms + ' ms';
                    document.getElementById('miss-p95').innerText = data.cache_miss_p95_latency_ms + ' ms';
                    document.getElementById('miss-p99').innerText = data.cache_miss_p99_latency_ms + ' ms';
                } catch (e) {
                    console.error("Dashboard metrics fetch failed", e);
                }
            }
            updateDashboard();
        </script>
    </body>
    </html>
    """
