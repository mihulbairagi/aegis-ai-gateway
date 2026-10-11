import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import FastAPI, Request, HTTPException, Response, status, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings, authenticate_tenant, verify_admin_key, get_active_provider_info
from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, UsageInfo, GatewayMetricsResponse
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
                await semantic_cache.purge_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    cleaner_task = asyncio.create_task(periodic_bucket_cleaner())
    yield
    cleaner_task.cancel()


app = FastAPI(
    title="Aegis AI Gateway & Semantic Cache",
    description="High-Throughput Enterprise AI Reverse Proxy with L1/L2 Semantic Caching, SQLite WAL Persistence, Token-Bucket Rate Limiting, Multi-Tenant Isolation, and Multi-Provider Routing.",
    version="1.3.1",
    lifespan=lifespan
)

# Enable CORS for all frontends/dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # auth uses headers, not cookies; wildcard origin + credentials is invalid/unsafe
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


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request, response: Response):
    start_time = time.monotonic()

    # 1. Client Authentication & Multi-Tenant Identification
    token = _get_api_token(request)
    tenant_id = authenticate_tenant(token)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing client API key."
        )

    # 2. Rate Limiting (Token Bucket per tenant)
    allowed, retry_after = rate_limiter.is_allowed(tenant_id)
    if not allowed:
        retry_val = str(round(retry_after, 2))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for tenant '{tenant_id}'. Retry after {retry_val} seconds.",
            headers={"Retry-After": retry_val}
        )

    # 3. Cache-Control, Dynamic Threshold, & TTL extraction
    bypass_cache = (
        request.headers.get("X-Bypass-Cache", "").lower() in ("true", "1") or
        request.headers.get("Cache-Control", "").lower() == "no-cache"
    )
    raw_ttl = request.headers.get("X-Cache-TTL")
    custom_ttl = (
        min(float(raw_ttl), settings.CACHE_MAX_TTL_SECONDS)
        if raw_ttl and raw_ttl.isdigit() and float(raw_ttl) > 0
        else None
    )

    # Dynamic threshold override from request header or body
    raw_thresh = request.headers.get("X-Similarity-Threshold")
    custom_thresh = float(raw_thresh) if raw_thresh else req.similarity_threshold

    # 4. Check Tiered Multi-Tenant Cache (L1 Exact & L2 Semantic Cosine Match)
    cached_resp, cache_type, similarity_score = await semantic_cache.get(
        req,
        tenant_id=tenant_id,
        bypass_cache=bypass_cache,
        custom_threshold=custom_thresh
    )

    # --- STREAMING RESPONSE HANDLING (req.stream == True) ---
    if req.stream:
        if cached_resp:
            # Stream from in-memory cached response with simulated typewriter emission
            async def stream_cached_chunks():
                full_text = cached_resp.choices[0].message.content
                words = full_text.split(" ")
                chunk_id = f"chatcmpl-cached-{uuid.uuid4().hex[:8]}"
                
                for i, word in enumerate(words):
                    chunk_payload = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant" if i == 0 else None,
                                "content": word + (" " if i < len(words) - 1 else "")
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_payload)}\n\n"
                    await asyncio.sleep(0.005)  # 5ms ultra-fast emission

                stop_payload = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(stop_payload)}\n\n"
                yield "data: [DONE]\n\n"

            latency_ms = (time.monotonic() - start_time) * 1000.0
            metrics_tracker.record_request(latency_ms, cache_type=cache_type, tokens_saved=cached_resp.usage.total_tokens)
            stream_headers = {
                "X-Gateway-Tenant-Id": tenant_id,
                "X-Gateway-Cache-Status": f"HIT-{cache_type}",
                "X-Gateway-Latency-Ms": str(round(latency_ms, 2)),
                "X-Gateway-Similarity-Score": str(round(similarity_score, 4)),
                "Content-Type": "text/event-stream"
            }
            return StreamingResponse(stream_cached_chunks(), media_type="text/event-stream", headers=stream_headers)

        # Cache Miss on Streaming: Stream live from upstream and accumulate in background
        stream_gen, provider_name = await proxy_client.forward_stream(req)

        async def stream_and_cache_wrapper():
            accumulated_chunks = []
            try:
                async for line in stream_gen:
                    yield line
                    if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                        try:
                            data = json.loads(line[6:].strip())
                            delta_content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta_content:
                                accumulated_chunks.append(delta_content)
                        except Exception:
                            pass
            finally:
                if accumulated_chunks:
                    full_content = "".join(accumulated_chunks)
                    p_tokens = sum(len(m.content.split()) for m in req.messages) + 10
                    c_tokens = len(full_content.split()) + 5
                    reconstructed = ChatCompletionResponse(
                        model=req.model,
                        choices=[ChatChoice(message=ChatMessage(role="assistant", content=full_content))],
                        usage=UsageInfo(
                            prompt_tokens=p_tokens,
                            completion_tokens=c_tokens,
                            total_tokens=p_tokens + c_tokens
                        ),
                        provider=provider_name
                    )
                    await semantic_cache.put(req, reconstructed, tenant_id=tenant_id, bypass_cache=bypass_cache, ttl_seconds=custom_ttl)
                    latency_ms = (time.monotonic() - start_time) * 1000.0
                    metrics_tracker.record_request(latency_ms, cache_type=None)

        stream_headers = {
            "X-Gateway-Tenant-Id": tenant_id,
            "X-Gateway-Cache-Status": "MISS",
            "X-Gateway-Provider": provider_name,
            "X-Gateway-Similarity-Score": str(round(similarity_score, 4)),
            "Content-Type": "text/event-stream"
        }
        return StreamingResponse(stream_and_cache_wrapper(), media_type="text/event-stream", headers=stream_headers)

    # --- NON-STREAMING STANDARD RESPONSE ---
    if cached_resp:
        latency_ms = (time.monotonic() - start_time) * 1000.0
        cached_resp.latency_ms = round(latency_ms, 2)
        cached_resp.similarity_score = round(similarity_score, 4)
        
        # Track metrics
        saved_tokens = cached_resp.usage.total_tokens
        metrics_tracker.record_request(latency_ms, cache_type=cache_type, tokens_saved=saved_tokens)
        
        # Set observability headers
        response.headers["X-Gateway-Tenant-Id"] = tenant_id
        response.headers["X-Gateway-Cache-Status"] = f"HIT-{cache_type}"
        response.headers["X-Gateway-Latency-Ms"] = str(round(latency_ms, 2))
        response.headers["X-Gateway-Similarity-Score"] = str(round(similarity_score, 4))
        return cached_resp

    # Cache Miss -> Forward to Upstream Provider via Circuit Breaker
    upstream_resp, provider = await proxy_client.forward(req)
    latency_ms = (time.monotonic() - start_time) * 1000.0

    upstream_resp.latency_ms = round(latency_ms, 2)
    upstream_resp.provider = provider
    upstream_resp.cached = False
    upstream_resp.similarity_score = round(similarity_score, 4)

    # Store in Cache (partitioned by tenant and persisted to SQLite)
    await semantic_cache.put(
        req,
        upstream_resp,
        tenant_id=tenant_id,
        bypass_cache=bypass_cache,
        ttl_seconds=custom_ttl
    )

    # Record telemetry
    metrics_tracker.record_request(latency_ms, cache_type=None)

    response.headers["X-Gateway-Tenant-Id"] = tenant_id
    response.headers["X-Gateway-Cache-Status"] = "MISS"
    response.headers["X-Gateway-Latency-Ms"] = str(round(latency_ms, 2))
    response.headers["X-Gateway-Provider"] = provider
    response.headers["X-Gateway-Similarity-Score"] = str(round(similarity_score, 4))
    return upstream_resp


@app.get("/metrics", response_model=GatewayMetricsResponse)
async def get_metrics():
    summary = metrics_tracker.get_summary()
    route, state = circuit_breaker.get_route()
    provider_info = get_active_provider_info()
    summary["circuit_breaker_state"] = state.value
    summary["active_provider"] = f"{provider_info['name']} ({route})"
    return summary


# --- Protected Admin Endpoints ---

@app.post("/v1/gateway/circuit-breaker/trip")
async def force_trip_circuit(request: Request):
    """Admin endpoint: Chaos testing for circuit breaker failover."""
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
    """Admin endpoint: Purge all in-memory L1/L2 caches and SQLite persistent storage."""
    _require_admin(request)
    semantic_cache.clear()
    return {"message": "All L1 & L2 tenant caches and SQLite records purged successfully"}


# --- Web UI: Interactive Chat Playground ---

@app.get("/playground", response_class=HTMLResponse)
async def interactive_playground():
    """Interactive Chat Playground with real-time semantic caching badges and threshold slider."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Aegis AI Gateway Playground</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; }
            code, pre, .font-mono { font-family: 'JetBrains Mono', monospace; }
        </style>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">
        <!-- Top Navigation -->
        <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur px-6 py-3 flex justify-between items-center sticky top-0 z-50">
            <div class="flex items-center space-x-3">
                <span class="text-xl font-black bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">AEGIS PLAYGROUND</span>
                <span class="text-xs px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">v1.3.1</span>
            </div>
            <div class="flex items-center space-x-4">
                <a href="/dashboard" class="text-xs text-slate-400 hover:text-white transition">📊 Live Dashboard</a>
                <a href="/docs" target="_blank" class="text-xs text-slate-400 hover:text-white transition">📖 API Docs</a>
                <button onclick="purgeCache()" class="text-xs px-3 py-1 bg-rose-500/10 text-rose-400 hover:bg-rose-500/20 border border-rose-500/30 rounded transition">
                    🗑️ Purge Cache
                </button>
            </div>
        </header>

        <!-- Main Workspace -->
        <div class="flex-1 max-w-6xl w-full mx-auto p-6 grid grid-cols-1 md:grid-cols-4 gap-6">
            <!-- Left Config Sidebar -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4 h-fit">
                <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400">Gateway Controls</h3>
                
                <div>
                    <label class="text-xs text-slate-400 block mb-1">Tenant API Key</label>
                    <input id="api-key" type="password" placeholder="sk-your-tenant-key" autocomplete="off" class="w-full bg-slate-950 border border-slate-700 rounded px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono">
                </div>

                <div>
                    <label class="text-xs text-slate-400 block mb-1">Admin Key (only for Purge Cache)</label>
                    <input id="admin-key" type="password" placeholder="ADMIN_API_KEY" autocomplete="off" class="w-full bg-slate-950 border border-slate-700 rounded px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono">
                </div>

                <div>
                    <label class="text-xs text-slate-400 block mb-1">Model Target</label>
                    <input id="model" type="text" value="gpt-4o-mini" class="w-full bg-slate-950 border border-slate-700 rounded px-3 py-1.5 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono">
                </div>

                <div>
                    <label class="text-xs text-slate-400 block mb-1">Temperature: <span id="temp-val" class="text-indigo-400 font-mono">0.0</span></label>
                    <input id="temperature" type="range" min="0" max="1" step="0.1" value="0.0" oninput="document.getElementById('temp-val').innerText = this.value" class="w-full accent-indigo-500 cursor-pointer">
                </div>

                <div class="space-y-1 pt-2 border-t border-slate-800">
                    <div class="flex justify-between items-center text-xs text-slate-400">
                        <span>Match Threshold</span>
                        <span id="threshold-val" class="font-mono text-emerald-400 font-bold">0.80</span>
                    </div>
                    <input id="threshold-slider" type="range" min="0.70" max="0.95" step="0.01" value="0.80" oninput="document.getElementById('threshold-val').innerText = this.value" class="w-full accent-emerald-500 cursor-pointer">
                    <p class="text-[10px] text-slate-500">Sweet spot: 0.78 - 0.82. Lower catches more paraphrases; higher requires strict phrasing.</p>
                </div>

                <div class="pt-2 border-t border-slate-800 space-y-2">
                    <label class="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
                        <input id="bypass-cache" type="checkbox" class="accent-indigo-500">
                        <span>Bypass Cache (X-Bypass-Cache)</span>
                    </label>
                    <label class="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
                        <input id="use-stream" type="checkbox" checked class="accent-indigo-500">
                        <span>Enable SSE Streaming</span>
                    </label>
                </div>

                <div class="pt-3 border-t border-slate-800">
                    <div class="text-[11px] text-slate-400">
                        <div class="font-semibold text-slate-300 mb-1">💡 Real-World Test Pair:</div>
                        1. <span class="text-indigo-300 font-mono">"How do I cancel my food order on Swiggy"</span><br>
                        2. <span class="text-emerald-300 font-mono">"swiggy food canceling process ?"</span> (Hits L2 at threshold &le;0.80)!
                    </div>
                </div>
            </div>

            <!-- Right Chat Container -->
            <div class="md:col-span-3 bg-slate-900 border border-slate-800 rounded-xl flex flex-col h-[75vh]">
                <!-- Chat Message Feed -->
                <div id="chat-feed" class="flex-1 overflow-y-auto p-5 space-y-4">
                    <div class="text-center text-xs text-slate-400 py-4">
                        🚀 Welcome to Aegis Playground. Start chatting to test instant L1 exact hits & ONNX dense L2 semantic matches.
                    </div>
                </div>

                <!-- Chat Input Form -->
                <div class="p-4 border-t border-slate-800 bg-slate-900/80">
                    <form id="chat-form" onsubmit="sendMessage(event)" class="flex space-x-3">
                        <input id="user-input" type="text" placeholder="Type a prompt to test caching behavior..." autocomplete="off" class="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500">
                        <button id="send-btn" type="submit" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-5 py-2.5 rounded-lg text-sm transition flex items-center space-x-1">
                            <span>Send</span>
                            <span>➤</span>
                        </button>
                    </form>
                </div>
            </div>
        </div>

        <script>
            async function sendMessage(e) {
                e.preventDefault();
                const input = document.getElementById('user-input');
                const prompt = input.value.trim();
                if (!prompt) return;

                const apiKey = document.getElementById('api-key').value.trim();
                const model = document.getElementById('model').value.trim();
                const temperature = parseFloat(document.getElementById('temperature').value);
                const threshold = parseFloat(document.getElementById('threshold-slider').value);
                const bypass = document.getElementById('bypass-cache').checked;
                const stream = document.getElementById('use-stream').checked;

                input.value = '';
                appendUserMessage(prompt);

                const botMsgId = 'msg-' + Date.now();
                const botBox = appendBotPlaceholder(botMsgId);

                const headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + apiKey,
                    'X-Similarity-Threshold': threshold.toString()
                };
                if (bypass) headers['X-Bypass-Cache'] = 'true';

                const payload = {
                    model: model,
                    messages: [{ role: 'user', content: prompt }],
                    temperature: temperature,
                    stream: stream,
                    similarity_threshold: threshold
                };

                const t0 = performance.now();

                try {
                    const response = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: headers,
                        body: JSON.stringify(payload)
                    });

                    const latencyMs = Math.round(performance.now() - t0);
                    const cacheStatus = response.headers.get('X-Gateway-Cache-Status') || 'MISS';
                    const similarityScore = parseFloat(response.headers.get('X-Gateway-Similarity-Score') || '0.0');

                    if (!response.ok) {
                        const err = await response.json();
                        botBox.innerHTML = `<span class="text-rose-400">Error ${response.status}: ${err.detail || 'Request failed'}</span>`;
                        return;
                    }

                    if (stream) {
                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let fullText = '';
                        botBox.innerHTML = '';

                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            const chunk = decoder.decode(value);
                            const lines = chunk.split('\\n');
                            for (const line of lines) {
                                if (line.startsWith('data: ') && !line.includes('[DONE]')) {
                                    try {
                                        const parsed = JSON.parse(line.replace('data: ', ''));
                                        const content = parsed.choices[0]?.delta?.content || '';
                                        fullText += content;
                                        botBox.innerText = fullText;
                                    } catch(e) {}
                                }
                            }
                        }
                        updateBadge(botMsgId, cacheStatus, latencyMs, similarityScore, threshold);
                    } else {
                        const data = await response.json();
                        botBox.innerText = data.choices[0].message.content;
                        updateBadge(botMsgId, cacheStatus, latencyMs, similarityScore, threshold);
                    }
                } catch (err) {
                    botBox.innerHTML = `<span class="text-rose-400">Network Error: ${err.message}</span>`;
                }
            }

            function appendUserMessage(text) {
                const feed = document.getElementById('chat-feed');
                const div = document.createElement('div');
                div.className = 'flex justify-end';
                div.innerHTML = `<div class="bg-indigo-600/20 border border-indigo-500/30 text-indigo-100 rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-xl text-sm leading-relaxed">${escapeHtml(text)}</div>`;
                feed.appendChild(div);
                feed.scrollTop = feed.scrollHeight;
            }

            function appendBotPlaceholder(id) {
                const feed = document.getElementById('chat-feed');
                const div = document.createElement('div');
                div.className = 'flex flex-col space-y-1.5 items-start';
                div.innerHTML = `
                    <div id="badge-${id}" class="text-[11px] font-mono px-2.5 py-0.5 rounded-full inline-flex items-center space-x-1.5 bg-slate-800 text-slate-400 animate-pulse">
                        ⏳ Routing through Aegis Gateway...
                    </div>
                    <div id="${id}" class="bg-slate-950 border border-slate-800 text-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 max-w-2xl text-sm leading-relaxed whitespace-pre-wrap">Thinking...</div>
                `;
                feed.appendChild(div);
                feed.scrollTop = feed.scrollHeight;
                return document.getElementById(id);
            }

            function updateBadge(id, status, latency, score, threshold) {
                const badge = document.getElementById(`badge-${id}`);
                if (!badge) return;
                badge.classList.remove('animate-pulse');

                if (status.includes('L1_EXACT')) {
                    badge.className = 'text-[11px] font-mono px-2.5 py-0.5 rounded-full inline-flex items-center space-x-1.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 shadow-sm';
                    badge.innerHTML = `⚡ <b>L1 EXACT HIT</b> • ${latency}ms • Cost: $0.00 (Saved!)`;
                } else if (status.includes('L2_SEMANTIC')) {
                    badge.className = 'text-[11px] font-mono px-2.5 py-0.5 rounded-full inline-flex items-center space-x-1.5 bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 shadow-sm';
                    badge.innerHTML = `🧠 <b>L2 SEMANTIC HIT</b> (Cosine: ${score.toFixed(3)} ≥ ${threshold.toFixed(2)}) • ${latency}ms • Cost: $0.00 (Saved!)`;
                } else {
                    badge.className = 'text-[11px] font-mono px-2.5 py-0.5 rounded-full inline-flex items-center space-x-1.5 bg-amber-500/10 text-amber-400 border border-amber-500/30 shadow-sm';
                    if (score > 0) {
                        badge.innerHTML = `🌐 <b>UPSTREAM MISS</b> (Best match: ${score.toFixed(3)} &lt; ${threshold.toFixed(2)}) • ${latency}ms • Cost: $0.003`;
                    } else {
                        badge.innerHTML = `🌐 <b>UPSTREAM MISS</b> • ${latency}ms • Cost: $0.003`;
                    }
                }
            }

            async function purgeCache() {
                const adminKey = document.getElementById('admin-key').value.trim();
                if (!adminKey) { alert('Enter the admin key in the sidebar first.'); return; }
                if (!confirm("Are you sure you want to flush all L1/L2 caches and SQLite records?")) return;
                try {
                    const res = await fetch('/v1/gateway/cache', {
                        method: 'DELETE',
                        headers: { 'X-Admin-Key': adminKey }
                    });
                    const data = await res.json();
                    alert(data.message || data.detail || 'Cache flushed');
                } catch(e) {
                    alert('Purge failed: ' + e);
                }
            }

            function escapeHtml(str) {
                return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            }
        </script>
    </body>
    </html>
    """


# --- Web UI: Operational Metrics Dashboard ---

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
                    <p class="text-slate-400 text-sm mt-1">Multi-Tenant Semantic Reverse Proxy • SQLite WAL • ONNX Dense Cache • Rate Limiter</p>
                </div>
                <div class="flex items-center space-x-3">
                    <a href="/playground" class="text-xs px-3 py-1 rounded-full font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30 transition">
                        🎮 Open Playground
                    </a>
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
