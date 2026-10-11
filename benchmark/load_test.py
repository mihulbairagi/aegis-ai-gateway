import asyncio
import sys
import time
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

console = Console()

GATEWAY_URL = "http://127.0.0.1:8000"

# Realistic prompt pairs to test semantic similarity caching
SEMANTIC_PAIRS = [
    ("How do I cancel my subscription?", "What is the procedure to cancel subscription?"),
    ("What is your refund policy?", "How can I request a refund for my order?"),
    ("How to reset password?", "I forgot my password, how to change it?"),
    ("Where is my shipment?", "Can you track my package delivery status?"),
    ("Tell me about company history.", "When was this enterprise founded?")
]

async def run_benchmark(total_requests: int = 100, concurrency: int = 20):
    console.print(Panel.fit(
        "[bold cyan]AEGIS AI GATEWAY: HIGH-CONCURRENCY STRESS TEST[/bold cyan]\n"
        f"Simulating [bold yellow]{total_requests} requests[/bold yellow] with [bold yellow]{concurrency} concurrent workers[/bold yellow]...",
        border_style="cyan"
    ))

    client = httpx.AsyncClient(timeout=10.0)

    # 1. Warm-up request & seed initial cache
    console.print("[dim]Seeding base queries into Gateway...[/dim]")
    for p1, _ in SEMANTIC_PAIRS:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": p1}]
        }
        await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload)

    # 2. Run concurrent requests mixing exact matches, semantic matches, and fresh queries
    tasks = []
    latencies = []
    status_codes = []
    cache_statuses = []

    sem = asyncio.Semaphore(concurrency)

    async def send_req(prompt: str, client_key: str = "benchmark-client-1"):
        async with sem:
            t0 = time.monotonic()
            headers = {"Authorization": f"Bearer {client_key}"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            }
            try:
                res = await client.post(f"{GATEWAY_URL}/v1/chat/completions", json=payload, headers=headers)
                latency = (time.monotonic() - t0) * 1000.0
                latencies.append(latency)
                status_codes.append(res.status_code)
                cache_statuses.append(res.headers.get("X-Gateway-Cache-Status", "MISS"))
            except Exception as e:
                latencies.append(9999.0)
                status_codes.append(500)

    start_bench = time.monotonic()

    # Generate workload:
    # 40% Semantic matches, 30% Exact matches, 30% New queries
    req_list = []
    for i in range(total_requests):
        pair_idx = i % len(SEMANTIC_PAIRS)
        if i % 3 == 0:
            # Semantic variation (should trigger L2)
            req_list.append(SEMANTIC_PAIRS[pair_idx][1])
        elif i % 3 == 1:
            # Exact repeat (should trigger L1)
            req_list.append(SEMANTIC_PAIRS[pair_idx][0])
        else:
            # Unique prompt (Miss)
            req_list.append(f"Generate unique calculation for ID-{i}-{time.time()}")

    # Launch concurrent workers
    tasks = [send_req(p, client_key=f"bench-client-{i % 10}") for i, p in enumerate(req_list)]
    await asyncio.gather(*tasks)

    total_time = time.monotonic() - start_bench
    rps = total_requests / total_time if total_time > 0 else 0

    # Separate client-observed hit vs miss latencies
    hit_lats = [latencies[i] for i in range(len(latencies)) if "HIT" in cache_statuses[i]]
    miss_lats = [latencies[i] for i in range(len(latencies)) if "HIT" not in cache_statuses[i]]

    import numpy as np
    def get_pcts(arr):
        if not arr:
            return 0.0, 0.0, 0.0
        a = np.array(arr)
        return round(float(np.percentile(a, 50)), 2), round(float(np.percentile(a, 95)), 2), round(float(np.percentile(a, 99)), 2)

    hit_p50, hit_p95, hit_p99 = get_pcts(hit_lats)
    miss_p50, miss_p95, miss_p99 = get_pcts(miss_lats)

    # Fetch gateway metrics
    metrics_res = await client.get(f"{GATEWAY_URL}/metrics")
    m = metrics_res.json()
    await client.aclose()

    # 1. Cache-Hit In-Memory Performance Table
    hit_table = Table(title="[bold green]Tier 1 & Tier 2: In-Memory Cache Performance[/bold green]", border_style="green")
    hit_table.add_column("Cache Metric", style="cyan", no_wrap=True)
    hit_table.add_column("Measured Value", style="bold green")

    hit_table.add_row("Cache Hits (L1 Exact LRU)", str(m.get('l1_cache_hits', 0)))
    hit_table.add_row("Cache Hits (L2 Semantic FastEmbed)", str(m.get('l2_cache_hits', 0)))
    hit_table.add_row("Total Cache Hit Ratio", f"{m.get('cache_hit_ratio_percent', 0.0)}%")
    hit_table.add_row("In-Memory Throughput", f"{round(rps, 1)} Requests/Sec")
    hit_table.add_row("Cache-Hit p50 Latency (Median)", f"{hit_p50} ms")
    hit_table.add_row("Cache-Hit p95 Latency", f"{hit_p95} ms")
    hit_table.add_row("Cache-Hit p99 Latency (Tail)", f"{hit_p99} ms")
    console.print(hit_table)

    # 2. Upstream Cache-Miss Round-Trip Table
    miss_table = Table(title="[bold yellow]Tier 3: Upstream Provider Round-Trip Performance[/bold yellow]", border_style="yellow")
    miss_table.add_column("Upstream Metric", style="cyan", no_wrap=True)
    miss_table.add_column("Measured Value", style="bold yellow")

    miss_table.add_row("Total Cache Misses (Upstream Calls)", str(m.get('cache_misses', len(miss_lats))))
    miss_table.add_row("Active Upstream Route", str(m.get('active_provider', 'primary_routerplus')))
    miss_table.add_row("Circuit Breaker State", str(m.get('circuit_breaker_state', 'CLOSED')))
    miss_table.add_row("Upstream Round-Trip p50 (Median)", f"{miss_p50} ms")
    miss_table.add_row("Upstream Round-Trip p95", f"{miss_p95} ms")
    miss_table.add_row("Upstream Round-Trip p99 (Tail)", f"{miss_p99} ms")
    miss_table.add_row("Overall Success Rate (200 OK)", f"{round((status_codes.count(200)/total_requests)*100, 1)}%")
    console.print(miss_table)

    # 3. Financial Telemetry Panel
    saved_usd = m.get('estimated_cost_saved_usd', 0.0)
    console.print(Panel.fit(
        f"[bold cyan]FINANCIAL IMPACT & TOKEN AVOIDANCE[/bold cyan]\n"
        f"• Total Direct Savings (This Run): [bold green]${saved_usd:.4f}[/bold green]\n"
        f"• Estimated Monthly Savings (@10M reqs/mo, 90% hit ratio): [bold green]${(saved_usd * (10_000_000 / total_requests)):,.2f}[/bold green]",
        border_style="cyan"
    ))

if __name__ == "__main__":
    asyncio.run(run_benchmark(total_requests=150, concurrency=25))
