import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict, Set
import numpy as np

from app.config import settings
from app.models.schemas import ChatCompletionResponse, ChatCompletionRequest
from app.core.storage import sqlite_storage

logger = logging.getLogger(__name__)

# Zero-overhead lexical & typo normalization map for common keyboard slips,
# chat abbreviations, and high-frequency user typos in LLM queries.
LEXICAL_NORMALIZATION_MAP: Dict[str, str] = {
    # Forgot typos
    "fogot": "forgot", "forgt": "forgot", "forgott": "forgot", "frogot": "forgot", "fogott": "forgot",
    # Password typos & shortcuts
    "passward": "password", "passwrd": "password", "pasword": "password",
    "passwrod": "password", "paword": "password", "passord": "password",
    "psswd": "password", "pswd": "password",
    # Reset typos
    "resset": "reset", "rset": "reset", "resett": "reset",
    # Change typos
    "chnage": "change", "chagne": "change", "chng": "change", "chnge": "change",
    # Cancel typos
    "cancle": "cancel", "cancell": "cancel", "cancled": "canceled", "cancl": "cancel",
    "cancelling": "canceling",
    # Account typos
    "accout": "account", "acount": "account", "acnt": "account", "acct": "account",
    # Delete typos
    "dlete": "delete", "delte": "delete", "delt": "delete",
    # Update typos
    "updt": "update", "updte": "update",
    # Refund typos
    "refnd": "refund", "refun": "refund", "refud": "refund",
    # Order typos
    "ordr": "order", "oder": "order",
    # Payment typos
    "pymnt": "payment", "paymnt": "payment", "pyment": "payment",
    # Subscription typos
    "subcription": "subscription", "subscrip": "subscription", "subscripton": "subscription",
    # Address typos
    "addres": "address", "adress": "address", "adrs": "address",
    # Delivery typos
    "delivry": "delivery", "delvery": "delivery", "dlvry": "delivery",
    # Status typos
    "sttus": "status", "statuss": "status",
    # Common service brands
    "swigy": "swiggy", "swigi": "swiggy",
    "zomto": "zomato", "zomatto": "zomato",
    # Chat slang & quick abbreviations
    "hw": "how", "wat": "what", "wht": "what",
    "plz": "please", "pls": "please",
    "ur": "your",
    "bcz": "because", "cuz": "because", "coz": "because",
    "hlp": "help", "thx": "thanks", "thanx": "thanks", "tnx": "thanks",
}


def normalize_text_for_embedding(text: str) -> str:
    """
    Clean conversational role prefixes, collapse repeated characters (e.g. 'pleaaase' -> 'please'),
    and normalize common keyboard slip typos / chat slang to preserve high-confidence dense
    semantic cosine similarity on human input before vector embedding.
    """
    t = text.lower().strip()
    t = re.sub(r'^(user|system|assistant|tool):\s*', '', t)
    # Deduplicate 3+ repeated LETTERS only (e.g. 'pleaaase' -> 'please'). Never digits/underscores:
    # '1000' must not collapse to '10', or different amounts/IDs share one cached answer.
    t = re.sub(r'([^\W\d_])\1{2,}', r'\1', t)
    words = t.split()
    normalized_words = []
    for w in words:
        m = re.match(r"^([^\w]*)([\w'-]+)([^\w]*)$", w)
        if m:
            prefix, core, suffix = m.groups()
            core_replacement = LEXICAL_NORMALIZATION_MAP.get(core, core)
            normalized_words.append(f"{prefix}{core_replacement}{suffix}")
        else:
            normalized_words.append(LEXICAL_NORMALIZATION_MAP.get(w, w))
    t = " ".join(normalized_words)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


# Antonym / opposing polarity pairs for semantic cache negation guard.
# Prevents false positive cache hits between opposing boolean requests (e.g. enable 2FA vs disable 2FA).
POLARITY_PAIRS = [
    ({"enable", "enabled", "enabling", "turn on", "activate", "activating"}, {"disable", "disabled", "disabling", "turn off", "deactivate", "deactivating"}),
    ({"allow", "allowed", "allowing"}, {"deny", "denied", "denying", "block", "blocked", "blocking"}),
    ({"start", "started", "starting"}, {"stop", "stopped", "stopping"}),
    ({"increase", "increasing"}, {"decrease", "decreasing"}),
    ({"add", "adding"}, {"remove", "removing", "delete", "deleting"}),
    ({"connect", "connected"}, {"disconnect", "disconnected"}),
    ({"lock", "locked"}, {"unlock", "unlocked"}),
    ({"subscribe", "subscribing"}, {"unsubscribe", "unsubscribing"}),
    ({"credit", "credited"}, {"debit", "debited"}),
    ({"deposit", "depositing"}, {"withdraw", "withdrawing"}),
]


def _extract_guard_features(text: str) -> Tuple[Set[str], Set[Tuple[int, str]]]:
    """Extract numeric entities and polarity categories from query prompts."""
    # Extract numbers (normalize out thousands separators e.g. 1,000 -> 1000)
    raw_nums = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)
    numbers = {num.replace(',', '') for num in raw_nums}

    polarities: Set[Tuple[int, str]] = set()
    text_lower = text.lower()
    for i, (pos_set, neg_set) in enumerate(POLARITY_PAIRS):
        for term in pos_set:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                polarities.add((i, "pos"))
                break
        for term in neg_set:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                polarities.add((i, "neg"))
                break

    return numbers, polarities


def entity_and_polarity_guard(candidate_prompt: str, query_prompt: str) -> bool:
    """
    High-precision post-embedding guard for semantic cache verification.
    Rejects hits with mismatched numeric entities (e.g. refund 500 vs 5000, transfer 10 vs 1000)
    or contradictory antonym polarities (e.g. enable 2FA vs disable 2FA).
    """
    c_nums, c_pol = _extract_guard_features(candidate_prompt)
    q_nums, q_pol = _extract_guard_features(query_prompt)

    # 1. Number integrity: if numbers are present, sets must be identical
    if (c_nums or q_nums) and (c_nums != q_nums):
        return False

    # 2. Polarity integrity: opposing actions reject
    for p in c_pol:
        opp = (p[0], "neg" if p[1] == "pos" else "pos")
        if opp in q_pol:
            return False

    return True


class FastNumpyEmbedder:
    """
    Lightweight, fallback CPU semantic vector embedder using tokenization
    and character/word n-gram hashing projection normalized to unit length.
    Zero PyTorch/GPU dependency.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        normalized = normalize_text_for_embedding(text)
        raw_tokens = re.findall(r'[a-z0-9]+', normalized)
        vector = np.zeros(self.dim, dtype=np.float32)

        for word in raw_tokens:
            if word in ('user', 'system', 'assistant', 'tool'):
                continue

            # Word hash projection
            h_word = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vector[h_word] += 2.0

            # Character trigrams for morphological and typo tolerance
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    trigram = word[i:i+3]
                    h_tri = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % self.dim
                    vector[h_tri] += 0.5

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

class FastEmbedDenseEmbedder:
    """
    Production dense semantic vector embedder using fastembed (ONNX Runtime on CPU).
    Uses 'sentence-transformers/all-MiniLM-L6-v2' (384-dimensional unit vector)
    with sub-10ms CPU inference, <60MB memory footprint, and zero PyTorch/GPU overhead.
    No hardcoded regex hacks — true contextual embeddings with zero-latency lexical normalization.
    """
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self.dim = 384
        self._fallback = FastNumpyEmbedder(dim=self.dim)
        self._model = None
        self._init_model()

    @property
    def is_dense_active(self) -> bool:
        """Returns True if the true transformer ONNX model is loaded and active."""
        return self._model is not None

    def _init_model(self):
        try:
            from fastembed import TextEmbedding
            logger.info(f"Loading dense FastEmbed ONNX model '{self.model_name}'...")
            self._model = TextEmbedding(model_name=self.model_name)
            logger.info(f"FastEmbed dense model '{self.model_name}' loaded successfully (ONNX CPU).")
        except ImportError as e:
            self._model = None
            msg = (
                "CRITICAL: 'fastembed' package is not installed in the active Python environment! "
                "Running on low-accuracy FastNumpy MD5 fallback (paraphrases will yield low 0.30-0.49 cosine scores). "
                "Run 'pip install fastembed' to enable true dense semantic caching."
            )
            logger.error(msg)
            print(f"\n[AEGIS WARNING] {msg}\n", file=sys.stderr)
        except Exception as e:
            self._model = None
            msg = (
                f"CRITICAL: FastEmbed failed to load ({e})! "
                "Running on low-accuracy FastNumpy MD5 fallback. "
                "Run 'pip install fastembed' or verify ONNX runtime to enable true dense semantic caching."
            )
            logger.error(msg)
            print(f"\n[AEGIS WARNING] {msg}\n", file=sys.stderr)

    def _normalize_text(self, text: str) -> str:
        """
        Clean conversational role prefixes, repeated characters, and common keyboard typos.
        """
        return normalize_text_for_embedding(text)

    def embed(self, text: str) -> np.ndarray:
        normalized = self._normalize_text(text)
        if self._model is not None:
            try:
                embeddings = list(self._model.embed([normalized]))
                vec = np.array(embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception as e:
                logger.warning(f"FastEmbed inference error ({e}), using FastNumpy fallback for this query.")
        return self._fallback.embed(normalized)


class TenantL2Partition:
    """Isolated L2 Semantic Cache partition for a specific tenant."""
    def __init__(self, max_items: int = settings.L2_SEMANTIC_CACHE_MAX_ITEMS):
        self.max_items = max_items
        self.vectors: List[np.ndarray] = []
        self.entries: List[ChatCompletionResponse] = []
        self.prompts: List[str] = []
        self.models: List[str] = []
        self.temperatures: List[float] = []
        self.timestamps: List[float] = []
        self.ttls: List[float] = []

    def evict_oldest(self):
        if self.vectors:
            self.vectors.pop(0)
            self.entries.pop(0)
            self.prompts.pop(0)
            self.models.pop(0)
            self.temperatures.pop(0)
            self.timestamps.pop(0)
            self.ttls.pop(0)

    def clear(self):
        self.vectors.clear()
        self.entries.clear()
        self.prompts.clear()
        self.models.clear()
        self.temperatures.clear()
        self.timestamps.clear()
        self.ttls.clear()

    def purge_expired(self, now: float) -> int:
        """Drops entries whose TTL elapsed. Returns how many were removed."""
        keep = [i for i in range(len(self.vectors)) if now - self.timestamps[i] <= self.ttls[i]]
        removed = len(self.vectors) - len(keep)
        if removed:
            self.vectors = [self.vectors[i] for i in keep]
            self.entries = [self.entries[i] for i in keep]
            self.prompts = [self.prompts[i] for i in keep]
            self.models = [self.models[i] for i in keep]
            self.temperatures = [self.temperatures[i] for i in keep]
            self.timestamps = [self.timestamps[i] for i in keep]
            self.ttls = [self.ttls[i] for i in keep]
        return removed


class SemanticCache:
    """
    Production-Hardened Multi-Tenant Semantic Cache with SQLite WAL Persistence:
    - Tier 1 (L1): O(1) Exact-match LRU partitioned by (tenant_id, model, temperature) (<0.1ms)
    - Tier 2 (L2): Dense ONNX Semantic Vector Search partitioned strictly per tenant (<15ms)
    - Persistent SQLite disk backing: Automatic warmup on reboot, zero loss across restarts.
    - Non-blocking async embedding execution via asyncio.to_thread
    - Per-entry TTL invalidation and Client-controlled bypass (X-Bypass-Cache)
    """
    def __init__(
        self,
        l1_max_items: int = settings.L1_CACHE_MAX_ITEMS,
        l2_max_items: int = settings.L2_SEMANTIC_CACHE_MAX_ITEMS,
        similarity_threshold: float = settings.SEMANTIC_SIMILARITY_THRESHOLD,
        default_ttl: float = settings.CACHE_DEFAULT_TTL_SECONDS
    ):
        self.l1_max_items = l1_max_items
        self.l2_max_items = l2_max_items
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        
        # L1: exact_key -> (ChatCompletionResponse, created_at, ttl_seconds)
        self.l1_cache: OrderedDict[str, Tuple[ChatCompletionResponse, float, float]] = OrderedDict()
        
        # L2: tenant_id -> TenantL2Partition
        self.l2_partitions: Dict[str, TenantL2Partition] = {}
        
        self.embedder = FastEmbedDenseEmbedder(model_name=settings.EMBEDDING_MODEL)

        # Warm up existing entries from SQLite disk persistence
        self.warmup()

    @property
    def is_dense_active(self) -> bool:
        """Returns True if the ONNX transformer model is active (not fallback)."""
        return self.embedder.is_dense_active

    def warmup(self):
        """Warm up in-memory indices from persistent SQLite storage on process start."""
        # Timestamps are wall-clock epoch seconds so TTLs survive restarts/reboots
        # (time.monotonic() resets at boot and made old entries look "from the future").
        now = time.time()
        sqlite_storage.delete_expired(now)
        records = sqlite_storage.load_all_records()
        for r in records:
            try:
                if now - float(r["created_at"]) > float(r["ttl"]):
                    continue  # expired while the process was down
                resp_data = json.loads(r["response_json"])
                resp = ChatCompletionResponse.model_validate(resp_data)
                tenant_id = r["tenant_id"]
                model = r["model"]
                temp = float(r["temperature"])
                prompt = r["prompt"]
                ttl = float(r["ttl"])
                created_at = float(r["created_at"])
                vector = r["vector"]

                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                l1_key = f"{tenant_id}:{model}:{temp}:{prompt_hash}"
                self.l1_cache[l1_key] = (resp, created_at, ttl)

                if vector is not None and len(vector) == self.embedder.dim:
                    partition = self._get_partition(tenant_id)
                    partition.vectors.append(vector)
                    partition.entries.append(resp)
                    partition.prompts.append(prompt)
                    partition.models.append(model)
                    partition.temperatures.append(temp)
                    partition.timestamps.append(created_at)
                    partition.ttls.append(ttl)
            except Exception as e:
                logger.debug(f"Cache warmup item skipped: {e}")

    def _get_partition(self, tenant_id: str) -> TenantL2Partition:
        if tenant_id not in self.l2_partitions:
            self.l2_partitions[tenant_id] = TenantL2Partition(max_items=self.l2_max_items)
        return self.l2_partitions[tenant_id]

    def _extract_prompt_text(self, request: ChatCompletionRequest) -> str:
        """Extracts concatenated prompt messages for hashing and embedding."""
        return "\n".join([f"{m.role}:{m.content}" for m in request.messages])

    def _get_l1_key(self, request: ChatCompletionRequest, tenant_id: str) -> str:
        """Partitioned L1 cache key strictly bound to tenant_id, model, and temperature."""
        prompt_text = self._extract_prompt_text(request)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
        temp = request.temperature if request.temperature is not None else 0.0
        return f"{tenant_id}:{request.model}:{temp}:{prompt_hash}"

    async def get(
        self,
        request: ChatCompletionRequest,
        tenant_id: str,
        bypass_cache: bool = False,
        custom_threshold: Optional[float] = None
    ) -> Tuple[Optional[ChatCompletionResponse], Optional[str], float]:
        """
        Check cache for matching response within tenant partition.
        Returns: (Response | None, cache_type: 'L1_EXACT' | 'L2_SEMANTIC' | None, similarity_score: float)
        """
        if bypass_cache:
            return None, None, 0.0

        now = time.time()
        effective_threshold = (
            custom_threshold
            if custom_threshold is not None and 0.5 <= custom_threshold <= 1.0
            else self.similarity_threshold
        )

        # 1. Check Tier 1 (L1 Exact Match)
        l1_key = self._get_l1_key(request, tenant_id)
        if l1_key in self.l1_cache:
            resp, created_at, ttl = self.l1_cache[l1_key]
            # Check TTL expiration
            if now - created_at > ttl:
                del self.l1_cache[l1_key]
            else:
                self.l1_cache.move_to_end(l1_key)
                hit_resp = resp.model_copy(deep=True)
                hit_resp.cached = True
                hit_resp.cache_type = "L1_EXACT"
                hit_resp.similarity_score = 1.0
                sqlite_storage.increment_hits(l1_key)
                return hit_resp, "L1_EXACT", 1.0

        # 2. Check Tier 2 (L2 Semantic Cosine Similarity within tenant partition)
        partition = self._get_partition(tenant_id)
        if len(partition.vectors) > 0:
            req_temp = request.temperature if request.temperature is not None else 0.0
            
            # Find candidate indices matching model, temperature, and valid TTL
            candidate_indices = [
                i for i in range(len(partition.vectors))
                if partition.models[i] == request.model
                and abs(partition.temperatures[i] - req_temp) < 1e-4
                and (now - partition.timestamps[i] <= partition.ttls[i])
            ]

            if candidate_indices:
                prompt_text = self._extract_prompt_text(request)
                query_vec = await asyncio.to_thread(self.embedder.embed, prompt_text)

                candidate_vectors = [partition.vectors[i] for i in candidate_indices]
                matrix = np.vstack(candidate_vectors)  # shape: (K, 384)
                
                # Vectors are unit normalized, cosine similarity = dot product
                scores = np.dot(matrix, query_vec)
                sorted_sub_indices = np.argsort(scores)[::-1]
                best_overall_score = float(scores[sorted_sub_indices[0]])

                for sub_idx in sorted_sub_indices:
                    score = float(scores[sub_idx])
                    if score < effective_threshold:
                        break
                    matched_idx = candidate_indices[sub_idx]
                    candidate_prompt = partition.prompts[matched_idx]

                    # Verify high-precision entity & polarity guard (antonyms, number integrity)
                    if entity_and_polarity_guard(candidate_prompt, prompt_text):
                        hit_resp = partition.entries[matched_idx].model_copy(deep=True)
                        hit_resp.cached = True
                        hit_resp.cache_type = "L2_SEMANTIC"
                        hit_resp.similarity_score = round(score, 4)
                        return hit_resp, "L2_SEMANTIC", score
                    else:
                        logger.debug(
                            f"L2 candidate rejected by Entity/Polarity Guard (score: {score:.4f}): "
                            f"'{candidate_prompt}' vs '{prompt_text}'"
                        )

                return None, None, best_overall_score

        return None, None, 0.0

    async def put(
        self,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
        tenant_id: str,
        bypass_cache: bool = False,
        ttl_seconds: Optional[float] = None
    ):
        """Stores response in L1, L2, and SQLite disk storage asynchronously."""
        if bypass_cache:
            return

        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None and ttl_seconds > 0 else self.default_ttl
        ttl = min(ttl, settings.CACHE_MAX_TTL_SECONDS)
        prompt_text = self._extract_prompt_text(request)
        req_temp = request.temperature if request.temperature is not None else 0.0

        # Store in L1
        l1_key = self._get_l1_key(request, tenant_id)
        if len(self.l1_cache) >= self.l1_max_items:
            self.l1_cache.popitem(last=False)
        self.l1_cache[l1_key] = (response, now, ttl)

        # Store in L2 (isolated tenant partition)
        partition = self._get_partition(tenant_id)
        vec = await asyncio.to_thread(self.embedder.embed, prompt_text)
        
        if len(partition.vectors) >= partition.max_items:
            partition.evict_oldest()

        partition.vectors.append(vec)
        partition.entries.append(response)
        partition.prompts.append(prompt_text)
        partition.models.append(request.model)
        partition.temperatures.append(req_temp)
        partition.timestamps.append(now)
        partition.ttls.append(ttl)

        # Asynchronously persist to SQLite disk in worker thread (zero event loop delay)
        resp_json = response.model_dump_json()
        asyncio.create_task(asyncio.to_thread(
            sqlite_storage.save_record,
            l1_key,
            tenant_id,
            request.model,
            req_temp,
            prompt_text,
            resp_json,
            vec,
            now,
            ttl
        ))

    async def purge_expired(self) -> int:
        """Removes expired entries from L1, every L2 partition and SQLite (called periodically)."""
        now = time.time()
        removed = 0

        for key in [k for k, (_, created_at, ttl) in self.l1_cache.items() if now - created_at > ttl]:
            del self.l1_cache[key]
            removed += 1

        for partition in self.l2_partitions.values():
            removed += partition.purge_expired(now)

        removed += await asyncio.to_thread(sqlite_storage.delete_expired, now)
        return removed

    def clear(self):
        """Flushes all tenant L1 and L2 caches and clears SQLite database."""
        self.l1_cache.clear()
        for partition in self.l2_partitions.values():
            partition.clear()
        self.l2_partitions.clear()
        sqlite_storage.clear()

semantic_cache = SemanticCache()
