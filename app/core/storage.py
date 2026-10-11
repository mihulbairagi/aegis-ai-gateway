import os
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
DB_PATH = os.path.join(DB_DIR, "cache.db")

class SQLiteCacheStorage:
    """
    High-throughput embedded disk storage for L1/L2 semantic cache using SQLite in WAL mode.
    Guarantees instant startup warmup and zero cache loss across process restarts.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    prompt TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    hits INTEGER DEFAULT 0,
                    ttl REAL DEFAULT 300.0
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tenant_model 
                ON semantic_cache(tenant_id, model, temperature);
            """)
            conn.commit()

    def load_all_records(self) -> List[Dict[str, Any]]:
        """Loads all persisted cache records for in-memory index warmup on startup."""
        records = []
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, tenant_id, model, temperature, prompt, 
                           response_json, embedding, created_at, hits, ttl 
                    FROM semantic_cache
                """)
                for row in cursor.fetchall():
                    emb_bytes = row["embedding"]
                    vector = np.frombuffer(emb_bytes, dtype=np.float32) if emb_bytes else None
                    records.append({
                        "id": row["id"],
                        "tenant_id": row["tenant_id"],
                        "model": row["model"],
                        "temperature": row["temperature"],
                        "prompt": row["prompt"],
                        "response_json": row["response_json"],
                        "vector": vector,
                        "created_at": row["created_at"],
                        "hits": row["hits"],
                        "ttl": row["ttl"]
                    })
        except Exception as e:
            logger.warning(f"Failed to load cache warmup from SQLite: {e}")
        return records

    def save_record(
        self,
        cache_id: str,
        tenant_id: str,
        model: str,
        temperature: float,
        prompt: str,
        response_json: str,
        embedding: np.ndarray,
        created_at: float,
        ttl: float
    ):
        """Persists a new cache entry and its ONNX dense vector."""
        emb_bytes = embedding.astype(np.float32).tobytes() if embedding is not None else b""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO semantic_cache 
                    (id, tenant_id, model, temperature, prompt, response_json, embedding, created_at, hits, ttl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (
                    cache_id,
                    tenant_id,
                    model,
                    temperature,
                    prompt,
                    response_json,
                    emb_bytes,
                    created_at,
                    ttl
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"SQLite save_record failed: {e}")

    def increment_hits(self, cache_id: str):
        """Increments hit count for analytics."""
        try:
            with self._get_connection() as conn:
                conn.execute("UPDATE semantic_cache SET hits = hits + 1 WHERE id = ?", (cache_id,))
                conn.commit()
        except Exception:
            pass

    def clear(self):
        """Purges all records from SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM semantic_cache")
                conn.commit()
        except Exception as e:
            logger.error(f"SQLite clear failed: {e}")

sqlite_storage = SQLiteCacheStorage()
