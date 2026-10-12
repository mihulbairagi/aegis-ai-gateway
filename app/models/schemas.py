from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
import time
import uuid

class ChatMessage(BaseModel):
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    similarity_threshold: Optional[float] = None

class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    
    # Custom gateway metadata for observability
    cached: bool = False
    cache_type: Optional[str] = None  # 'L1_EXACT', 'L2_SEMANTIC', or None
    latency_ms: Optional[float] = None
    similarity_score: Optional[float] = None
    provider: Optional[str] = None

class GatewayMetricsResponse(BaseModel):
    total_requests: int
    l1_cache_hits: int
    l2_cache_hits: int
    cache_misses: int
    cache_hit_ratio_percent: float
    circuit_breaker_state: str
    active_provider: str
    is_dense_active: bool = False
    embedding_engine: str = "fastembed"
    p50_latency_ms: float
    p90_latency_ms: float = 0.0
    p95_latency_ms: float
    p99_latency_ms: float
    # Honest Dual-Metric Breakdown
    cache_hit_p50_latency_ms: float = 0.0
    cache_hit_p95_latency_ms: float = 0.0
    cache_hit_p99_latency_ms: float = 0.0
    cache_miss_p50_latency_ms: float = 0.0
    cache_miss_p95_latency_ms: float = 0.0
    cache_miss_p99_latency_ms: float = 0.0
    estimated_cost_saved_usd: float
