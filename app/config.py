import os
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class GatewaySettings(BaseModel):
    # Gateway Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Upstream AI Providers
    PRIMARY_PROVIDER_URL: str = os.getenv("PRIMARY_PROVIDER_URL", "https://api.routerplus.com/v1")
    PRIMARY_API_KEY: str = os.getenv("PRIMARY_API_KEY", "")
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "claude-sonnet-5-5")
    
    FALLBACK_PROVIDER_URL: str = os.getenv("FALLBACK_PROVIDER_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    FALLBACK_API_KEY: str = os.getenv("FALLBACK_API_KEY", os.getenv("GEMINI_API_KEY", "mock-fallback-key"))
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "gemini-2.0-flash")

    # Caching Settings
    L1_CACHE_MAX_ITEMS: int = 1000           # In-memory exact match cache size
    L2_SEMANTIC_CACHE_MAX_ITEMS: int = 5000   # Vector similarity cache entries
    SEMANTIC_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.85")) # Cosine similarity threshold for L2 cache hit
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Rate Limiting Settings
    DEFAULT_RATE_LIMIT_RPS: int = 20         # 20 requests per second per client
    DEFAULT_BURST_CAPACITY: int = 50         # Burst capacity for token bucket

    # Circuit Breaker Settings
    CIRCUIT_FAILURE_THRESHOLD: int = 3       # Trip circuit after 3 consecutive failures
    CIRCUIT_RECOVERY_TIMEOUT: float = 15.0   # Wait 15s in OPEN state before trying HALF-OPEN
    CIRCUIT_TIMEOUT_SECONDS: float = 15.0    # Upstream request timeout threshold (allow live LLM completion)

settings = GatewaySettings()
