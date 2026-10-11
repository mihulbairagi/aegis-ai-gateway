import os
import hashlib
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class GatewaySettings(BaseModel):
    # Gateway Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Multi-Provider Routing: "ollama" | "groq" | "openai" | "routerplus" | "custom"
    UPSTREAM_PROVIDER: str = os.getenv("UPSTREAM_PROVIDER", "routerplus").lower()

    # Individual Provider Credentials (zero hardcoded secrets)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    ROUTERPLUS_API_KEY: str = os.getenv("ROUTERPLUS_API_KEY", os.getenv("PRIMARY_API_KEY", ""))
    ROUTERPLUS_BASE_URL: str = os.getenv("ROUTERPLUS_BASE_URL", os.getenv("PRIMARY_PROVIDER_URL", "https://api.routerplus.com/v1"))
    ROUTERPLUS_MODEL: str = os.getenv("ROUTERPLUS_MODEL", os.getenv("PRIMARY_MODEL", "claude-sonnet-5-5"))

    # Fallback Provider Settings
    FALLBACK_PROVIDER_URL: str = os.getenv("FALLBACK_PROVIDER_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    FALLBACK_API_KEY: str = os.getenv("FALLBACK_API_KEY", "")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "gemini-2.0-flash")

    # Admin & Authentication Security
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "admin-secret-key-12345")
    
    # Caching Settings
    L1_CACHE_MAX_ITEMS: int = int(os.getenv("L1_CACHE_MAX_ITEMS", "1000"))
    L2_SEMANTIC_CACHE_MAX_ITEMS: int = int(os.getenv("L2_SEMANTIC_CACHE_MAX_ITEMS", "5000"))
    SEMANTIC_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.80"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CACHE_DEFAULT_TTL_SECONDS: float = float(os.getenv("CACHE_DEFAULT_TTL_SECONDS", "300.0"))

    # Rate Limiting Settings
    DEFAULT_RATE_LIMIT_RPS: int = int(os.getenv("DEFAULT_RATE_LIMIT_RPS", "20"))
    DEFAULT_BURST_CAPACITY: int = int(os.getenv("DEFAULT_BURST_CAPACITY", "50"))

    # Circuit Breaker Settings
    CIRCUIT_FAILURE_THRESHOLD: int = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))
    CIRCUIT_RECOVERY_TIMEOUT: float = float(os.getenv("CIRCUIT_RECOVERY_TIMEOUT", "15.0"))
    CIRCUIT_TIMEOUT_SECONDS: float = float(os.getenv("CIRCUIT_TIMEOUT_SECONDS", "15.0"))

settings = GatewaySettings()

def get_active_provider_info() -> Dict[str, str]:
    """Returns the URL, API key, model, and display name for the configured provider."""
    provider = settings.UPSTREAM_PROVIDER
    if provider == "ollama":
        return {
            "name": "ollama",
            "url": settings.OLLAMA_BASE_URL,
            "api_key": "ollama-local",
            "default_model": settings.OLLAMA_MODEL
        }
    elif provider == "groq":
        return {
            "name": "groq",
            "url": settings.GROQ_BASE_URL,
            "api_key": settings.GROQ_API_KEY,
            "default_model": settings.GROQ_MODEL
        }
    elif provider == "openai":
        return {
            "name": "openai",
            "url": settings.OPENAI_BASE_URL,
            "api_key": settings.OPENAI_API_KEY,
            "default_model": settings.OPENAI_MODEL
        }
    else:
        # Default: routerplus
        return {
            "name": "routerplus",
            "url": settings.ROUTERPLUS_BASE_URL,
            "api_key": settings.ROUTERPLUS_API_KEY,
            "default_model": settings.ROUTERPLUS_MODEL
        }

# Multi-tenant API key hash registry (Pre-configured tenants: alpha, beta, benchmark, test)
RAW_TENANT_KEYS: Dict[str, str] = {
    "sk-aegis-tenant-alpha": "tenant-alpha",
    "sk-aegis-tenant-beta": "tenant-beta",
    "sk-aegis-benchmark": "tenant-benchmark",
    "sk-aegis-test": "tenant-test",
}

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
    """Validates client API key against hashed tenant registry."""
    if not api_key:
        return None
    key_hash = hashlib.sha256(api_key.strip().encode()).hexdigest()
    return TENANT_KEY_HASH_MAP.get(key_hash)

def verify_admin_key(admin_key: str) -> bool:
    """Validates admin API key for protected gateway operations."""
    if not admin_key or not settings.ADMIN_API_KEY:
        return False
    return admin_key.strip() == settings.ADMIN_API_KEY.strip()
