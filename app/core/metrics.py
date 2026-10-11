import time
from typing import List, Dict, Any
import numpy as np

class GatewayMetricsTracker:
    """
    Real-time dual-tier performance and financial telemetry tracker.
    Distinguishes between:
    1. Cache-Hit Latency (served from in-memory L1/L2 <0.5ms)
    2. Cache-Miss Latency (real upstream network round-trip time)
    Computes p50/p90/p95/p99 percentiles for both tiers and financial USD savings.
    """
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.total_requests = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.cache_misses = 0
        
        # Dual-tier latency reservoirs in milliseconds
        self.cache_hit_latencies_ms: List[float] = []
        self.cache_miss_latencies_ms: List[float] = []
        self.all_latencies_ms: List[float] = []
        
        # Token savings: Assuming average cost of $0.005 per 1,000 prompt+completion tokens
        self.saved_tokens = 0
        self.cost_per_token_usd = 0.000005 # $5 per million tokens

    def record_request(
        self,
        latency_ms: float,
        cache_type: str = None,
        tokens_saved: int = 0
    ):
        self.total_requests += 1

        if cache_type in ("L1_EXACT", "L2_SEMANTIC"):
            if cache_type == "L1_EXACT":
                self.l1_hits += 1
            else:
                self.l2_hits += 1
            self.saved_tokens += tokens_saved
            if len(self.cache_hit_latencies_ms) >= self.max_history:
                self.cache_hit_latencies_ms.pop(0)
            self.cache_hit_latencies_ms.append(latency_ms)
        else:
            self.cache_misses += 1
            if len(self.cache_miss_latencies_ms) >= self.max_history:
                self.cache_miss_latencies_ms.pop(0)
            self.cache_miss_latencies_ms.append(latency_ms)

        if len(self.all_latencies_ms) >= self.max_history:
            self.all_latencies_ms.pop(0)
        self.all_latencies_ms.append(latency_ms)

    def _calc_percentiles(self, samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(samples)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
        }

    def get_summary(self) -> Dict[str, Any]:
        total_hits = self.l1_hits + self.l2_hits
        hit_ratio = (total_hits / self.total_requests * 100.0) if self.total_requests > 0 else 0.0

        all_pct = self._calc_percentiles(self.all_latencies_ms)
        hit_pct = self._calc_percentiles(self.cache_hit_latencies_ms)
        miss_pct = self._calc_percentiles(self.cache_miss_latencies_ms)

        return {
            "total_requests": self.total_requests,
            "l1_cache_hits": self.l1_hits,
            "l2_cache_hits": self.l2_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio_percent": round(hit_ratio, 2),
            # Overall latency
            "p50_latency_ms": all_pct["p50"],
            "p90_latency_ms": all_pct["p90"],
            "p95_latency_ms": all_pct["p95"],
            "p99_latency_ms": all_pct["p99"],
            # Honest Dual-Metric Breakdown
            "cache_hit_p50_latency_ms": hit_pct["p50"],
            "cache_hit_p95_latency_ms": hit_pct["p95"],
            "cache_hit_p99_latency_ms": hit_pct["p99"],
            "cache_miss_p50_latency_ms": miss_pct["p50"],
            "cache_miss_p95_latency_ms": miss_pct["p95"],
            "cache_miss_p99_latency_ms": miss_pct["p99"],
            "estimated_cost_saved_usd": round(self.saved_tokens * self.cost_per_token_usd, 4),
        }

metrics_tracker = GatewayMetricsTracker()
