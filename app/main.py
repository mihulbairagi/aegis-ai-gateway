import time
from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse, GatewayMetricsResponse
from app.core.rate_limiter import rate_limiter
from app.core.semantic_cache import semantic_cache
from app.core.circuit_breaker import circuit_breaker, CircuitState
from app.core.proxy import proxy_client
from app.core.metrics import metrics_tracker

app = FastAPI(
    title="Aegis AI Gateway & Semantic Cache",
    description="High-Throughput Enterprise AI Reverse Proxy with L1/L2 Semantic Caching, Token-Bucket Rate Limiting, and Circuit Breakers.",
    version="1.0.0"
)

# Enable CORS for all frontends/dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest, request: Request, response: Response):
    start_time = time.monotonic()
    client_ip = request.client.host if request.client else "127.0.0.1"
    auth_header = request.headers.get("Authorization", "")
    client_id = auth_header.replace("Bearer ", "").strip() or client_ip

    # 1. Rate Limiting (Token Bucket)
    allowed, retry_after = rate_limiter.is_allowed(client_id)
    if not allowed:
        retry_val = str(round(retry_after, 2))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_val} seconds.",
            headers={"Retry-After": retry_val}
        )

    # 2. Check Tiered Cache (L1 Exact & L2 Semantic Cosine Match)
    cached_resp, cache_type = semantic_cache.get(req)
    if cached_resp:
        latency_ms = (time.monotonic() - start_time) * 1000.0
        cached_resp.latency_ms = round(latency_ms, 2)
        
        # Track metrics
        saved_tokens = cached_resp.usage.total_tokens
        metrics_tracker.record_request(latency_ms, cache_type=cache_type, tokens_saved=saved_tokens)
        
        # Set observability headers
        response.headers["X-Gateway-Cache-Status"] = f"HIT-{cache_type}"
        response.headers["X-Gateway-Latency-Ms"] = str(round(latency_ms, 2))
        return cached_resp

    # 3. Cache Miss -> Forward to Upstream Provider via Circuit Breaker
    upstream_resp, provider = await proxy_client.forward(req)
    latency_ms = (time.monotonic() - start_time) * 1000.0

    upstream_resp.latency_ms = round(latency_ms, 2)
    upstream_resp.provider = provider
    upstream_resp.cached = False

    # 4. Store in Cache for future hits
    semantic_cache.put(req, upstream_resp)

    # 5. Record telemetry
    metrics_tracker.record_request(latency_ms, cache_type=None)

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

@app.post("/v1/gateway/circuit-breaker/trip")
async def force_trip_circuit():
    """Simulate upstream outage for live interview demo."""
    circuit_breaker.state = CircuitState.OPEN
    circuit_breaker.last_state_change = time.monotonic()
    return {"message": "Circuit breaker forced to OPEN state (Failover active)"}

@app.post("/v1/gateway/circuit-breaker/reset")
async def reset_circuit():
    circuit_breaker.state = CircuitState.CLOSED
    circuit_breaker.consecutive_failures = 0
    return {"message": "Circuit breaker reset to CLOSED state"}

@app.delete("/v1/gateway/cache")
async def flush_cache():
    semantic_cache.l1_cache.clear()
    semantic_cache.l2_vectors.clear()
    semantic_cache.l2_entries.clear()
    semantic_cache.l2_prompts.clear()
    return {"message": "L1 & L2 caches purged successfully"}

@app.get("/dashboard", response_class=HTMLResponse)
async def live_dashboard():
    """Clean, high-visibility dashboard for live demonstrations."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Aegis AI Gateway & Semantic Cache</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <meta http-equiv="refresh" content="2">
    </head>
    <body class="bg-slate-950 text-slate-100 font-sans p-8">
        <div class="max-w-6xl mx-auto">
            <header class="flex justify-between items-center border-b border-slate-800 pb-6 mb-8">
                <div>
                    <h1 class="text-3xl font-extrabold text-indigo-400 tracking-tight">AEGIS AI GATEWAY</h1>
                    <p class="text-slate-400 text-sm">High-Throughput Semantic Reverse Proxy & Resilience Control Plane</p>
                </div>
                <div class="flex gap-3">
                    <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                        ● Engine Online
                    </span>
                </div>
            </header>

            <div id="stats" class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                    <div class="text-slate-400 text-xs font-bold uppercase tracking-wider mb-1">Total Requests</div>
                    <div id="total-req" class="text-3xl font-black text-white">0</div>
                </div>
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                    <div class="text-slate-400 text-xs font-bold uppercase tracking-wider mb-1">Cache Hit Ratio</div>
                    <div id="hit-ratio" class="text-3xl font-black text-emerald-400">0.0%</div>
                </div>
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                    <div class="text-slate-400 text-xs font-bold uppercase tracking-wider mb-1">Cache-Hit p95 (Memory)</div>
                    <div id="hit-p95-lat" class="text-3xl font-black text-cyan-400">0.0 ms</div>
                </div>
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                    <div class="text-slate-400 text-xs font-bold uppercase tracking-wider mb-1">Cache-Miss p95 (Upstream)</div>
                    <div id="miss-p95-lat" class="text-3xl font-black text-amber-400">0.0 ms</div>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
                    <h3 class="text-lg font-bold text-slate-200 mb-4">Tiered In-Memory Caching (L1/L2)</h3>
                    <div class="space-y-3">
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">L1 Exact Hits (O(1) LRU):</span>
                            <span id="l1-hits" class="font-bold text-indigo-400">0</span>
                        </div>
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">L2 Semantic Hits (FastEmbed ONNX):</span>
                            <span id="l2-hits" class="font-bold text-emerald-400">0</span>
                        </div>
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">Cache-Hit Median (p50):</span>
                            <span id="hit-p50-lat" class="font-bold text-cyan-400">0.0 ms</span>
                        </div>
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">Estimated Cost Saved:</span>
                            <span id="cost-saved" class="font-bold text-emerald-400">$0.0000</span>
                        </div>
                    </div>
                </div>

                <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
                    <h3 class="text-lg font-bold text-slate-200 mb-4">Upstream Provider & Resilience</h3>
                    <div class="space-y-3">
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">Circuit Breaker State:</span>
                            <span id="cb-state" class="font-bold text-emerald-400 uppercase">CLOSED</span>
                        </div>
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">Active Upstream Route:</span>
                            <span id="provider" class="font-bold text-indigo-400">primary_routerplus</span>
                        </div>
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">Upstream Miss Median (p50):</span>
                            <span id="miss-p50-lat" class="font-bold text-amber-400">0.0 ms</span>
                        </div>
                        <div class="flex justify-between items-center text-sm">
                            <span class="text-slate-400">Total Upstream Misses:</span>
                            <span id="misses" class="font-bold text-rose-400">0</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <h3 class="text-lg font-bold text-slate-200 mb-2">Live Integration Code (OpenAI SDK Drop-in)</h3>
                <p class="text-slate-400 text-xs mb-4">Point baseURL to Aegis Gateway to automatically route through RouterPlus / Claude with 90%+ cache offloading:</p>
                <pre class="bg-slate-950 p-4 rounded-lg text-emerald-400 font-mono text-xs overflow-x-auto">
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Aegis Gateway URL
    api_key="your-client-key"
)

response = client.chat.completions.create(
    model="claude-sonnet-5-5",
    messages=[{"role": "user", "content": "How do I optimize SQL queries?"}]
)
print(response.choices[0].message.content)
                </pre>
            </div>
        </div>

        <script>
            async function updateStats() {
                try {
                    const res = await fetch('/metrics');
                    const d = await res.json();
                    document.getElementById('total-req').innerText = d.total_requests;
                    document.getElementById('hit-ratio').innerText = d.cache_hit_ratio_percent + '%';
                    document.getElementById('hit-p95-lat').innerText = d.cache_hit_p95_latency_ms + ' ms';
                    document.getElementById('miss-p95-lat').innerText = d.cache_miss_p95_latency_ms + ' ms';
                    document.getElementById('cost-saved').innerText = '$' + d.estimated_cost_saved_usd;
                    document.getElementById('l1-hits').innerText = d.l1_cache_hits;
                    document.getElementById('l2-hits').innerText = d.l2_cache_hits;
                    document.getElementById('hit-p50-lat').innerText = d.cache_hit_p50_latency_ms + ' ms';
                    document.getElementById('miss-p50-lat').innerText = d.cache_miss_p50_latency_ms + ' ms';
                    document.getElementById('misses').innerText = d.cache_misses;
                    document.getElementById('cb-state').innerText = d.circuit_breaker_state;
                    document.getElementById('provider').innerText = d.active_provider;
                } catch(e) {}
            }
            updateStats();
        </script>
    </body>
    </html>
    """
