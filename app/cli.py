import sys
import os

# Ensure project root is always in sys.path regardless of where 'aegis' is invoked
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import asyncio
import uvicorn
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def start_server(host: str, port: int, reload: bool):
    """Starts the Aegis AI Gateway using uvicorn."""
    console.print(Panel.fit(
        f"[bold cyan]AEGIS AI GATEWAY[/bold cyan] v1.3.0\n"
        f"Server listening on [bold green]http://{host}:{port}[/bold green]\n"
        f"• Playground: [bold yellow]http://{host}:{port}/playground[/bold yellow]\n"
        f"• Dashboard:  [bold yellow]http://{host}:{port}/dashboard[/bold yellow]\n"
        f"• Metrics:    [bold yellow]http://{host}:{port}/metrics[/bold yellow]",
        border_style="cyan"
    ))
    uvicorn.run("app.main:app", host=host, port=port, reload=reload, app_dir=ROOT_DIR)

def flush_cache(url: str, admin_key: str):
    """Flushes all L1/L2 caches and SQLite records."""
    try:
        res = httpx.delete(f"{url.rstrip('/')}/v1/gateway/cache", headers={"X-Admin-Key": admin_key})
        if res.status_code == 200:
            console.print(f"[bold green]✔ Success:[/bold green] {res.json().get('message')}")
        else:
            console.print(f"[bold red]✘ Failed ({res.status_code}):[/bold red] {res.text}")
    except Exception as e:
        console.print(f"[bold red]✘ Error connecting to gateway:[/bold red] {e}")

def check_status(url: str):
    """Fetches and displays live gateway health and cache metrics."""
    try:
        res = httpx.get(f"{url.rstrip('/')}/metrics")
        if res.status_code == 200:
            m = res.json()
            table = Table(title="[bold cyan]Aegis Gateway Status[/bold cyan]", border_style="cyan")
            table.add_column("Property", style="white")
            table.add_column("Value", style="bold green")
            table.add_row("Circuit Breaker", m.get("circuit_breaker_state", "CLOSED"))
            table.add_row("Active Provider", m.get("active_provider", "--"))
            table.add_row("Total Requests", str(m.get("total_requests", 0)))
            table.add_row("L1 Exact Hits", str(m.get("l1_cache_hits", 0)))
            table.add_row("L2 Semantic Hits", str(m.get("l2_cache_hits", 0)))
            table.add_row("Hit Ratio", f"{m.get('cache_hit_ratio_percent', 0.0)}%")
            table.add_row("Estimated Savings", f"${m.get('estimated_cost_saved_usd', 0.0):.4f}")
            console.print(table)
        else:
            console.print(f"[bold red]✘ Failed ({res.status_code}):[/bold red] {res.text}")
    except Exception as e:
        console.print(f"[bold red]✘ Error connecting to gateway:[/bold red] {e}")

def run_benchmark_cli(requests: int, concurrency: int):
    """Runs the high-concurrency load benchmark."""
    from benchmark.load_test import run_benchmark
    asyncio.run(run_benchmark(total_requests=requests, concurrency=concurrency))

def main():
    parser = argparse.ArgumentParser(
        prog="aegis",
        description="Aegis AI Gateway: Enterprise Semantic Cache & Multi-Provider LLM Proxy"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Command: start
    start_parser = subparsers.add_parser("start", help="Start the gateway server")
    start_parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    start_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    start_parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")

    # Command: flush
    flush_parser = subparsers.add_parser("flush", help="Purge all in-memory and SQLite cache records")
    flush_parser.add_argument("--url", default="http://127.0.0.1:8000", help="Gateway URL")
    flush_parser.add_argument("--admin-key", default="admin-secret-key-12345", help="Admin API Key")

    # Command: status
    status_parser = subparsers.add_parser("status", help="Display live gateway operational telemetry")
    status_parser.add_argument("--url", default="http://127.0.0.1:8000", help="Gateway URL")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run high-concurrency load test")
    bench_parser.add_argument("--requests", type=int, default=150, help="Total requests (default: 150)")
    bench_parser.add_argument("--concurrency", type=int, default=25, help="Concurrent workers (default: 25)")

    args = parser.parse_args()

    if args.command == "start" or args.command is None:
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8000)
        reload = getattr(args, "reload", False)
        start_server(host=host, port=port, reload=reload)
    elif args.command == "flush":
        flush_cache(url=args.url, admin_key=args.admin_key)
    elif args.command == "status":
        check_status(url=args.url)
    elif args.command == "benchmark":
        run_benchmark_cli(requests=args.requests, concurrency=args.concurrency)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
