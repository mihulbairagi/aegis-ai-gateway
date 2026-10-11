import os
import hashlib
from typing import Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class GatewaySettings(BaseModel):
    # Gateway Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Upstream AI Providers
    PRIMARY_PROVIDER_URL: str = os.getenv("PRIMARY_PROVIDER_URL", "https://api.routerplus.com/v1")
    PRIMARY_API_KEY: str = os.getenv("PRIMARY_API_KEY", "")
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "claude-sonnet-5-5")
    
    FALLBACK_PROVIDER_URL: str = os.getenv("FALLBACK_PROVIDER_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    FALLBACK_API_KEY: str = os.getenv("FALLBACK_API_KEY", "")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "gemini-2.0-flash")

    # Admin & Authentication Security
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "admin-secret-key-12345")
    
    # Caching Settings
    L1_CACHE_MAX_ITEMS: int = int(os.getenv("L1_CACHE_MAX_ITEMS", "1000"))
    L2_SEMANTIC_CACHE_MAX_ITEMS: int = int(os.getenv("L2_SEMANTIC_CACHE_MAX_ITEMS", "5000"))
    SEMANTIC_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.85"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CACHE_DEFAULT_TTL_SECONDS: float = float(os.getenv("CACHE_DEFAULT_TTL_SECONDS", "300.0"))

    # Rate Limiting Settings
    DEFAULT_RATE_LIMIT_RPS: int = int(os.getenv("DEFAULT_RATE_LIMIT_RPS", "20"))
    DEFAULT_BURST_CAPACITY: int = int(os.getenv("DEFAULT_BURST_CAPACITY", "50"))

    # Circuit Breaker Settings
    CIRCUIT_FAILURE_THRESHOLD: int = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))
    CIRCUIT_RECOVERY_TIMEOUT: float = float(os.getenv("CIRCUIT_RECOVERY_TIMEOUT", "15.0"))
    CIRCUIT_TIMEOUT_SECONDS: float = float(os.getenv("CIRCUIT_TIMEOUT_SECONDS", "10.0"))

settings = GatewaySettings()

# Multi-tenant API key hash registry (Pre-configured tenants: alpha, beta, benchmark, test)
# Keyed by SHA-256 hash of API token for constant-time cryptographic lookup
RAW_TENANT_KEYS: Dict[str, str] = {
    "sk-aegis-tenant-alpha": "tenant-alpha",
    "sk-aegis-tenant-beta": "tenant-beta",
    "sk-aegis-benchmark": "tenant-benchmark",
    "sk-aegis-test": "tenant-test",
}

# Allow external additions via env var (format: "key1:tenant1,key2:tenant2")
_env_tenant_keys = os.getenv("TENANT_API_KEYS", "")
if _env_tenant_keys:
    for pair in _env_tenant_keys.split(","):
        if ":" in pair:
            k, t = pair.strip().split(":", 1)
            RAW_TENANT_KEYS[k.strip()] = t.strip()

TENANT_KEY_HASH_MAP: Dict[str, str] = {
    hashlib.sha256(key.encode()).hexdigest(): tenant_id
    for key, tenant_id in RAW_TENANT_KEYS.items()
}

def authenticate_tenant(api_key: str) -> Optional[str]:
    """
    Validates client API key against hashed tenant registry.
    Returns tenant_id if valid, None if unauthorized.
    """
    if not api_key:
        return None
    key_hash = hashlib.sha256(api_key.strip().encode()).hexdigest()
    return TENANT_KEY_HASH_MAP.get(key_hash)

def verify_admin_key(admin_key: str) -> bool:
    """Validates admin API key for protected gateway operations."""
    if not admin_key or not settings.ADMIN_API_KEY:
        return False
    return admin_key.strip() == settings.ADMIN_API_KEY.strip()
