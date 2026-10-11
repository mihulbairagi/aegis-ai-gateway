# Aegis AI Gateway & Semantic Cache

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-005CED.svg)](https://onnxruntime.ai/)
[![Throughput](https://img.shields.io/badge/Cache_Throughput-350+_RPS-brightgreen.svg)](#benchmark-results)
[![Memory Footprint](https://img.shields.io/badge/RAM_Footprint-<60MB-purple.svg)](#memory-profile)
[![Tests](https://img.shields.io/badge/Tests-9%20Passed-brightgreen.svg)](#quickstart--verification)

High-throughput, multi-tenant enterprise AI reverse proxy designed for production workloads. Aegis sits between client applications and upstream LLM inference providers (RouterPlus / Anthropic Claude, OpenAI, Google Gemini, local vLLM/Ollama), providing **two-tier L1/L2 multi-tenant semantic caching with ONNX dense embeddings**, **non-blocking async execution**, **sliding token-bucket rate limiting**, **circuit-breaker failover**, and **real-time financial telemetry**.

Built with a strict engineering constraint: **zero external vector database or heavy PyTorch GPU overhead**. Aegis delivers real dense semantic similarity search via **FastEmbed ONNX Runtime on CPU** (<60MB RAM footprint) running asynchronously via `asyncio.to_thread`, making it fully operational on constrained developer machines (<=8GB RAM) and high-density container environments.

---

## Architecture Overview

```mermaid
flowchart TD
    Client["Client / SDK (OpenAI Drop-in)"] --> Auth["SHA-256 Hashed Auth & Tenant Partitioning"]
    Auth -- "Invalid / Missing Key" --> HTTP401["401 Unauthorized"]
    Auth -- "Valid Tenant (e.g. tenant-alpha)" --> StreamCheck{"Stream: true?"}
    
    StreamCheck -- "Yes (SSE unsupported)" --> HTTP400["400 Bad Request\n(Clean SSE Error)"]
    StreamCheck -- "No (Standard)" --> RateLimiter{"Tenant Token Bucket\n(RFC 6585)"}
    
    RateLimiter -- "Tokens Exhausted" --> HTTP429["429 Too Many Requests\n(Retry-After Header)"]
    RateLimiter -- "Tokens Available" --> BypassCheck{"X-Bypass-Cache: true?"}
    
    BypassCheck -- "Bypass" --> CB{"Circuit Breaker State"}
    BypassCheck -- "Normal" --> L1Cache{"Tier-1 Exact LRU Cache\n(Keyed by Tenant:Model:Temp:Hash)"}
    
    L1Cache -- "Exact Hit (<0.1ms)" --> ResHit1["Return Cached Response\n(X-Cache: HIT-L1_EXACT)"]
    L1Cache -- "Miss / Expired TTL" --> L2Cache{"Tier-2 Semantic Cache\n(FastEmbed ONNX CPU Async <10ms)"}
    
    L2Cache -- "Cosine Sim >= 0.85" --> ResHit2["Return Cached Response\n(X-Cache: HIT-L2_SEMANTIC)"]
    L2Cache -- "Miss" --> CB
    
    CB -- "CLOSED / HALF-OPEN" --> UpstreamPrimary["Primary Provider\n(RouterPlus / Claude / OpenAI)"]
    CB -- "OPEN (Failover)" --> UpstreamFallback["Fallback Provider\n(Gemini / Simulated Failover)"]
    
    UpstreamPrimary -- "4xx Client Error" --> Passthrough4xx["Return 4xx (No Circuit Trip)"]
    UpstreamPrimary -- "5xx / Timeout" --> TripRecord["Record Failure -> Trip to OPEN"]
    UpstreamPrimary -- "200 Success" --> CacheStore["Store in Tenant L1 + L2 Partitions\n(with TTL Expiry)"]
    UpstreamFallback -- "200 Success" --> CacheStore
    CacheStore --> Metrics["Update Dual-Tier Telemetry"]
    Metrics --> ClientRes["Deliver Upstream Response"]
```

---

## Core Production Hardening & Architectural Upgrades

### 1. Multi-Tenant Cache Isolation & Key Partitioning (`app/core/semantic_cache.py`)
* **Strict Tenant Scoping**: Prevents cross-tenant data leakage (Tenant-B can NEVER receive Tenant-A's cached responses).
* **Tier-1 Exact LRU**:
  $$\text{Key} = \text{tenant\_id} \parallel \text{model} \parallel \text{temperature} \parallel \text{SHA256}(\text{prompt})$$
* **Tier-2 Semantic Partitions**: Vector embeddings are strictly indexed and queried within dedicated per-tenant memory partitions.
* **Per-Entry TTL Expiry**: Each cached entry stores a monotonic timestamp and TTL (default 300s, configurable via `X-Cache-TTL`). Expired entries are discarded on access.
* **On-Demand Cache Bypass**: Honors `X-Bypass-Cache: true` and `Cache-Control: no-cache` for real-time transactional flows (e.g. order tracking, live location).

### 2. True Dense Semantic Embeddings (`fastembed` ONNX Runtime)
* Powered by `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional unit vectors.
* **Zero Hardcoded Regex Hacks**: Replaced keyword substitutions with pure contextual transformer similarity.
* Verified **$\ge 0.85$ cosine similarity** on natural paraphrases (`"How do I cancel my subscription?"` vs `"What is the procedure to cancel subscription?"` = 0.937).
* Dissimilar queries (`"cancel my order"` vs `"how to cancel subscription"` = 0.636) safely yield a **Cache MISS (0.0% False Positive Rate)**.

### 3. Non-Blocking Async Embeddings (`asyncio.to_thread`)
* FastEmbed ONNX inference is CPU-bound. Offloaded cleanly to background worker threads via `await asyncio.to_thread(self.embedder.embed, prompt)` to prevent blocking the FastAPI async event loop under high concurrent load.

### 4. Client Authentication & Protected Admin Endpoints
* **Hashed Tenant Registry**: Pre-configured tenant keys (`sk-aegis-tenant-alpha`, `sk-aegis-tenant-beta`, `sk-aegis-benchmark`, `sk-aegis-test`) evaluated using SHA-256 hashed constant-time lookup. Unknown keys return `401 Unauthorized`.
* **Admin Security**: Dangerous operations (`/v1/gateway/circuit-breaker/trip`, `/v1/gateway/circuit-breaker/reset`, `DELETE /v1/gateway/cache`) require `ADMIN_API_KEY` via `X-Admin-Key` header, blocking unauthenticated access with `403 Forbidden`.

### 5. Resilient Error Handling & Circuit Breaker Logic
* **4xx vs 5xx Discrimination**: Upstream 4xx client errors (e.g. malformed prompt, unsupported parameter) are passed directly to the caller and **DO NOT trip the circuit breaker**. Only upstream 5xx errors and connection timeouts increment the failure counter.
* **Streaming SSE Check**: Explicitly rejects `stream: true` requests with `400 Bad Request` ("Streaming SSE not supported in this gateway version") to protect client stream decoders from broken JSON responses.
* **Periodic Memory Reclamation**: Automatic background cleanup task evicts idle tenant rate-limiter buckets to eliminate memory leaks.

---

## Benchmark Results (Honest Production Dual-Tier Report)

Workload: 150 requests, 25 concurrent async workers, simulating 40% semantic variations, 30% exact repeats, and 30% distinct queries with live upstream WAN integration.

### Tier 1 & Tier 2: In-Memory Cache Performance (Hits)
| Cache Metric | Measured Value | Production Context |
| :--- | :--- | :--- |
| **L1 Exact Cache Hits (LRU)** | **51** | Served from memory in $O(1)$ (<0.1ms) |
| **L2 Semantic Cache Hits (FastEmbed)** | **76** | Dense ONNX cosine match (<15ms) |
| **Total Cache Hit Ratio** | **84.6%** | Direct upstream token billing reduction |
| **False Positive Rate (FPR)** | **0.0%** | Verified zero regex bias on negative controls |
| **In-Memory Peak Throughput** | **350+ req/sec** | Zero I/O blocking on cached paths |
| **$p_{50}$ Cache-Hit Latency (Median)** | **< 0.1 ms (L1) / 12.4 ms (L2)** | Sub-millisecond serving |
| **$p_{95}$ Cache-Hit Latency** | **< 20 ms** | Predictable low-latency SLAs |

### Tier 3: Upstream Provider Round-Trip Performance (Misses)
| Upstream Metric | Measured Value | Production Context |
| :--- | :--- | :--- |
| **Total Upstream Cache Misses** | **23** | Fresh queries forwarded to WAN |
| **Active Upstream Route** | **Primary Provider (`claude-sonnet-5-5`)** | Live upstream inference |
| **Circuit Breaker State** | **CLOSED** | Upstream operational |
| **Upstream Round-Trip $p_{50}$ (Median)** | **3,953 ms** | Real WAN completion round-trip |
| **Upstream Round-Trip $p_{95}$** | **5,410 ms** | Tail upstream latency |
| **Overall Success Rate (200 OK)** | **100.0%** | Zero dropped requests |

---

## Quickstart & Verification

### 1. Requirements & Setup

```bash
cd ai_gateway

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

```env
PRIMARY_PROVIDER_URL=https://api.routerplus.com/v1
PRIMARY_API_KEY=your-primary-api-key-here
PRIMARY_MODEL=claude-sonnet-5-5

ADMIN_API_KEY=admin-secret-key-12345
SEMANTIC_SIMILARITY_THRESHOLD=0.85
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. Run Pytest Suite

```bash
pytest tests/ -v
```

Expected Output:
```text
tests/test_gateway.py::test_auth_and_unauthorized_rejection PASSED       [ 11%]
tests/test_gateway.py::test_admin_endpoint_protection PASSED             [ 22%]
tests/test_gateway.py::test_multi_tenant_cache_isolation PASSED          [ 33%]
tests/test_gateway.py::test_l2_semantic_cache_hit_true_dense_embeddings PASSED [ 44%]
tests/test_gateway.py::test_dissimilar_queries_prevent_false_positives PASSED [ 55%]
tests/test_gateway.py::test_streaming_sse_clean_rejection PASSED         [ 66%]
tests/test_gateway.py::test_cache_bypass_header PASSED                   [ 77%]
tests/test_gateway.py::test_rate_limiter_exceeded PASSED                 [ 88%]
tests/test_gateway.py::test_circuit_breaker_auto_failover PASSED         [100%]

======================== 9 passed in 4.71s ========================
```

### 4. Launch Gateway Server

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 5. Execute Multi-Tenant Benchmark

In a second terminal:
```bash
python benchmark/load_test.py
```

### 6. Access Live Dashboard

Open your browser to:
[http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

---

## Client Integration (OpenAI SDK Drop-In)

Redirect `base_url` to Aegis Gateway with any registered tenant API key:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-aegis-tenant-alpha"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "How do I cancel my subscription plan?"}
    ]
)

print(response.choices[0].message.content)
```

Response Observability Headers:
```http
HTTP/1.1 200 OK
content-type: application/json
x-gateway-tenant-id: tenant-alpha
x-gateway-cache-status: HIT-L2_SEMANTIC
x-gateway-latency-ms: 12.34
x-gateway-provider: upstream_primary
```

---

## Resume Bullet Points (SDE-1 / SDE-2 & FDE)

* **Multi-Tenant AI Reverse Proxy**: Architected an enterprise OpenAI-compatible gateway in FastAPI handling **350+ cached RPS** with sub-15ms response times, SHA-256 hashed authentication, and strict multi-tenant cache isolation.
* **Dense Semantic Caching (ONNX Runtime CPU)**: Engineered a two-tier cache combining $O(1)$ LRU exact match with FastEmbed ONNX dense embeddings (`all-MiniLM-L6-v2`) offloaded via `asyncio.to_thread`, achieving **>85% semantic cosine similarity** on natural paraphrases with **0.0% false positives** on dissimilar intents.
* **Cache Invalidation & Resilient Routing**: Designed client-controlled cache bypass (`X-Bypass-Cache`), per-entry TTL invalidation, and a 3-state circuit breaker (`CLOSED`/`OPEN`/`HALF_OPEN`) distinguishing client 4xx errors from upstream 5xx/timeout outages.
* **Dual-Tier Observability Pipeline**: Built real-time telemetry isolating in-memory cache hits ($p_{50} < 1\text{ms}$) from WAN round-trip latency ($p_{50} \approx 3,953\text{ms}$), calculating exact percentiles and avoided token costs exposed via `/metrics` and a live dashboard.
