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
from app.models.schemas import ChatCompletionResponse, ChatCompletionRequest, content_to_text
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


# --- Post-embedding guard -----------------------------------------------------------------
# Dense embeddings pack "enable/disable", "500/5000", "can/can't" into near-identical vectors
# (cosine 0.88-0.92 measured). After a vector match we therefore require that the two prompts
# agree on: (1) every number-bearing token, (2) negation, (3) antonym polarity.

POLARITY_PAIRS = [
    ({"enable", "activate"}, {"disable", "deactivate"}),
    ({"allow", "permit", "grant"}, {"deny", "block", "revoke", "forbid"}),
    ({"start", "begin"}, {"stop", "halt"}),
    ({"pause"}, {"resume"}),
    ({"increase", "raise"}, {"decrease", "reduce", "lower"}),
    ({"add"}, {"remove", "delete"}),
    ({"connect"}, {"disconnect"}),
    ({"lock"}, {"unlock"}),
    ({"subscribe"}, {"unsubscribe"}),
    ({"credit"}, {"debit"}),
    ({"deposit"}, {"withdraw"}),
    ({"buy", "purchase"}, {"sell"}),
    ({"open"}, {"close"}),
    ({"upgrade"}, {"downgrade"}),
    ({"show"}, {"hide"}),
    ({"login", "signin"}, {"logout", "signout"}),
    ({"send"}, {"receive"}),
    ({"install"}, {"uninstall"}),
    ({"encrypt"}, {"decrypt"}),
    ({"import"}, {"export"}),
    ({"mute"}, {"unmute"}),
    ({"approve", "accept"}, {"reject", "decline"}),
    ({"cancel"}, {"renew"}),
    ({"on"}, {"off"}),
    ({"free"}, {"paid"}),
    ({"public"}, {"private"}),
    ({"monthly"}, {"yearly", "annual"}),
    ({"today"}, {"tomorrow", "yesterday"}),
    ({"before"}, {"after"}),
    ({"first"}, {"last", "final"}),
    ({"inbound"}, {"outbound"}),
]

_POLARITY_LOOKUP: Dict[str, Tuple[int, str]] = {}
for _i, (_pos, _neg) in enumerate(POLARITY_PAIRS):
    for _w in _pos:
        _POLARITY_LOOKUP[_w] = (_i, "pos")
    for _w in _neg:
        _POLARITY_LOOKUP[_w] = (_i, "neg")

_NEGATORS = {
    "not", "no", "never", "cannot", "without", "nothing", "neither", "nor",
    "cant", "dont", "doesnt", "didnt", "wont", "isnt", "arent", "wasnt", "werent",
    "shouldnt", "couldnt", "wouldnt", "havent", "hasnt", "hadnt",
}
_NUMBER_WORDS = {
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
    "ninety", "hundred", "thousand", "lakh", "lakhs", "crore", "crores", "million", "billion",
    "half", "double", "triple",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_PHRASAL_ON_OFF = re.compile(r"\b(?:turn|turning|switch|switching|toggle)\w*\s+(?:\w+\s+){0,2}?(on|off)\b")
_ON_OFF_PAIR_IDX = next(
    (idx for idx, (p, n) in enumerate(POLARITY_PAIRS) if "on" in p and "off" in n),
    -1
)


def _polarity_of(token: str):
    """Look up a token, trying simple inflection stripping (enables/enabled/enabling -> enable)."""
    if token in _POLARITY_LOOKUP:
        return _POLARITY_LOOKUP[token]
    cands = []
    for suf, repl in (("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("es", ""), ("s", ""), ("d", "")):
        if token.endswith(suf) and len(token) > len(suf) + 2:
            cands.append(token[: -len(suf)] + repl)
    for c in cands:
        if c in _POLARITY_LOOKUP:
            return _POLARITY_LOOKUP[c]
    return None


def _extract_guard_features(text: str):
    """Returns (number_like_tokens, has_negation, polarity_set)."""
    t = text.lower().replace("\u2019", "'").replace("'", "")
    tokens = [x.replace(",", "") for x in _TOKEN_RE.findall(t)]
    numbers = {x for x in tokens if any(ch.isdigit() for ch in x) or x in _NUMBER_WORDS}
    has_neg = any(x in _NEGATORS for x in tokens)
    polarities: Set[Tuple[int, str]] = set()
    for x in tokens:
        if x in ("on", "off"):
            continue  # bare on/off is too ambiguous; only handled via phrasal forms below
        pol = _polarity_of(x)
        if pol:
            polarities.add(pol)
    for m in _PHRASAL_ON_OFF.finditer(t):
        if _ON_OFF_PAIR_IDX >= 0:
            polarities.add((_ON_OFF_PAIR_IDX, "pos" if m.group(1) == "on" else "neg"))
    return numbers, has_neg, polarities


def entity_and_polarity_guard(candidate_prompt: str, query_prompt: str) -> bool:
    """True if the vector match may be served; False if the prompts differ in a meaning-critical way."""
    c_nums, c_neg, c_pol = _extract_guard_features(candidate_prompt)
    q_nums, q_neg, q_pol = _extract_guard_features(query_prompt)

    if c_nums != q_nums:          # 500 vs 5000, 10gb vs 100gb, python3 vs python2, A1234 vs A1243
        return False
    if c_neg != q_neg:            # "can log in" vs "can't log in"
        return False
    for idx, sign in c_pol:       # enable vs disable, buy vs sell, turn it on vs turn it off ...
        if (idx, "neg" if sign == "pos" else "pos") in q_pol:
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
        self.semantic_prompts: List[str] = []
        self.sys_hashes: List[str] = []
        self.user_scopes: List[str] = []
        self.params_hashes: List[str] = []
        self.models: List[str] = []
        self.temperatures: List[float] = []
        self.timestamps: List[float] = []
        self.ttls: List[float] = []

    def evict_oldest(self):
        if self.vectors:
            self.vectors.pop(0)
            self.entries.pop(0)
            self.prompts.pop(0)
            if self.semantic_prompts:
                self.semantic_prompts.pop(0)
            if self.sys_hashes:
                self.sys_hashes.pop(0)
            if self.user_scopes:
                self.user_scopes.pop(0)
            if self.params_hashes:
                self.params_hashes.pop(0)
            self.models.pop(0)
            self.temperatures.pop(0)
            self.timestamps.pop(0)
            self.ttls.pop(0)

    def clear(self):
        self.vectors.clear()
        self.entries.clear()
        self.prompts.clear()
        self.semantic_prompts.clear()
        self.sys_hashes.clear()
        self.user_scopes.clear()
        self.params_hashes.clear()
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
            self.semantic_prompts = [self.semantic_prompts[i] for i in keep]
            self.sys_hashes = [self.sys_hashes[i] for i in keep]
            self.user_scopes = [self.user_scopes[i] for i in keep]
            self.params_hashes = [self.params_hashes[i] for i in keep]
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

                user_scope = r.get("user_scope")
                params_hash = r.get("params_hash")
                # Legacy rows missing user_scope or params_hash must NOT enter L1 or L2 on warmup
                if not user_scope or not params_hash:
                    continue

                # On warmup use the stored row id as the L1 key instead of recomputing it
                l1_key = r["id"]
                self.l1_cache[l1_key] = (resp, created_at, ttl)

                sys_hash = r.get("sys_hash")
                semantic_text = r.get("semantic_text")
                if vector is not None and len(vector) == self.embedder.dim and sys_hash and semantic_text:
                    partition = self._get_partition(tenant_id)
                    partition.vectors.append(vector)
                    partition.entries.append(resp)
                    partition.prompts.append(prompt)
                    partition.semantic_prompts.append(semantic_text)
                    partition.sys_hashes.append(sys_hash)
                    partition.user_scopes.append(user_scope)
                    partition.params_hashes.append(params_hash)
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

    def _compute_user_scope(self, request: ChatCompletionRequest) -> str:
        """Computes user scope hash if request.user is a non-empty string, else literal 'shared'."""
        if isinstance(request.user, str) and request.user:
            return hashlib.sha256(request.user.encode()).hexdigest()[:16]
        return "shared"

    def _compute_params_hash(self, request: ChatCompletionRequest) -> str:
        """
        Computes a deterministic 16-hex sha256 hash of sampling parameters.
        Excludes keys: messages, model, temperature, stream, stream_options,
        user, similarity_threshold, tools, tool_choice, functions, function_call.
        """
        data = request.model_dump(exclude_none=True)
        excluded_keys = {
            "messages", "model", "temperature", "stream", "stream_options",
            "user", "similarity_threshold", "tools", "tool_choice",
            "functions", "function_call"
        }
        filtered = {k: v for k, v in data.items() if k not in excluded_keys}
        canonical_json = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]

    def is_cacheable(self, request: ChatCompletionRequest) -> Tuple[bool, str]:
        """
        Determines whether a request can be safely cached.
        Returns (False, reason) if the request involves tools, non-idempotent operations, or n > 1.
        """
        tools = getattr(request, "tools", None)
        if tools:
            return False, "tools"

        functions = getattr(request, "functions", None)
        if functions:
            return False, "functions"

        tool_choice = getattr(request, "tool_choice", None)
        if tool_choice:
            return False, "tool_choice"

        if request.messages:
            for m in request.messages:
                if m.role in {"tool", "function"}:
                    return False, f"message role '{m.role}'"
                if m.tool_calls:
                    return False, "tool_calls"
                if isinstance(m.content, list):
                    for part in m.content:
                        if not isinstance(part, dict) or part.get("type") != "text":
                            return False, "multimodal"

        if request.n is not None and request.n > 1:
            return False, "n > 1"

        return True, ""

    def _extract_prompt_text(self, request: ChatCompletionRequest) -> str:
        """Extracts concatenated prompt messages for hashing and embedding."""
        return "\n".join([f"{m.role}:{content_to_text(m.content)}" for m in request.messages])

    def _split_request_prompt(self, request: ChatCompletionRequest) -> Tuple[str, str, str]:
        """
        Splits request messages into:
        (full_prompt, sys_hash, semantic_text)

        - full_prompt: complete conversation used for L1 key and persistence
        - sys_hash: 16-char SHA256 of system instructions or 'none' (for strict persona isolation)
        - semantic_text: user/assistant conversation intent without system prompt dilution
        """
        full_prompt = "\n".join([f"{m.role}:{content_to_text(m.content)}" for m in request.messages])
        sys_parts = [content_to_text(m.content) for m in request.messages if m.role in ("system", "developer")]
        if sys_parts:
            sys_hash = hashlib.sha256("\n".join(sys_parts).encode()).hexdigest()[:16]
        else:
            sys_hash = "none"

        semantic_parts = [f"{m.role}:{content_to_text(m.content)}" for m in request.messages if m.role not in ("system", "developer")]
        if semantic_parts:
            semantic_text = "\n".join(semantic_parts).strip()
        else:
            semantic_text = full_prompt

        return full_prompt, sys_hash, semantic_text

    def _get_l1_key(self, request: ChatCompletionRequest, tenant_id: str) -> str:
        """Partitioned L1 cache key strictly bound to tenant_id, user_scope, params_hash, model, and temperature."""
        prompt_text = self._extract_prompt_text(request)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
        temp = request.temperature if request.temperature is not None else 0.0
        user_scope = self._compute_user_scope(request)
        params_hash = self._compute_params_hash(request)
        return f"{tenant_id}:{user_scope}:{params_hash}:{request.model}:{temp}:{prompt_hash}"

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

        full_prompt, sys_hash, semantic_text = self._split_request_prompt(request)
        req_temp = request.temperature if request.temperature is not None else 0.0
        user_scope = self._compute_user_scope(request)
        params_hash = self._compute_params_hash(request)

        # 1. Check Tier 1 (L1 Exact Match on full prompt)
        prompt_hash = hashlib.sha256(full_prompt.encode()).hexdigest()
        l1_key = f"{tenant_id}:{user_scope}:{params_hash}:{request.model}:{req_temp}:{prompt_hash}"
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
            # Find candidate indices matching model, temperature, system persona hash, user scope, params_hash, and valid TTL
            candidate_indices = [
                i for i in range(len(partition.vectors))
                if partition.models[i] == request.model
                and abs(partition.temperatures[i] - req_temp) < 1e-4
                and (partition.sys_hashes[i] == sys_hash if partition.sys_hashes else True)
                and (partition.user_scopes[i] == user_scope if partition.user_scopes else True)
                and (partition.params_hashes[i] == params_hash if partition.params_hashes else True)
                and (now - partition.timestamps[i] <= partition.ttls[i])
            ]

            if candidate_indices:
                query_vec = await asyncio.to_thread(self.embedder.embed, semantic_text)

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
                    candidate_semantic = (
                        partition.semantic_prompts[matched_idx]
                        if partition.semantic_prompts
                        else partition.prompts[matched_idx]
                    )

                    # Verify high-precision entity & polarity guard on user conversation intent
                    if entity_and_polarity_guard(candidate_semantic, semantic_text):
                        hit_resp = partition.entries[matched_idx].model_copy(deep=True)
                        hit_resp.cached = True
                        hit_resp.cache_type = "L2_SEMANTIC"
                        hit_resp.similarity_score = round(score, 4)
                        return hit_resp, "L2_SEMANTIC", score
                    else:
                        logger.debug(
                            f"L2 candidate rejected by Entity/Polarity Guard (score: {score:.4f}): "
                            f"'{candidate_semantic}' vs '{semantic_text}'"
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

        cacheable, _ = self.is_cacheable(request)
        if not cacheable:
            return

        # Only cache when every choice has finish_reason == "stop"
        if not response.choices or not all(c.finish_reason == "stop" for c in response.choices):
            return

        # Responses containing tool_calls must never be stored
        if any(getattr(c.message, "tool_calls", None) for c in response.choices):
            return

        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None and ttl_seconds > 0 else self.default_ttl
        ttl = min(ttl, settings.CACHE_MAX_TTL_SECONDS)
        
        full_prompt, sys_hash, semantic_text = self._split_request_prompt(request)
        req_temp = request.temperature if request.temperature is not None else 0.0
        user_scope = self._compute_user_scope(request)
        params_hash = self._compute_params_hash(request)

        # Store in L1
        prompt_hash = hashlib.sha256(full_prompt.encode()).hexdigest()
        l1_key = f"{tenant_id}:{user_scope}:{params_hash}:{request.model}:{req_temp}:{prompt_hash}"
        if len(self.l1_cache) >= self.l1_max_items:
            self.l1_cache.popitem(last=False)
        self.l1_cache[l1_key] = (response, now, ttl)

        # Store in L2 (isolated tenant partition, embedding pure semantic intent)
        partition = self._get_partition(tenant_id)
        vec = await asyncio.to_thread(self.embedder.embed, semantic_text)
        
        if len(partition.vectors) >= partition.max_items:
            partition.evict_oldest()

        partition.vectors.append(vec)
        partition.entries.append(response)
        partition.prompts.append(full_prompt)
        partition.semantic_prompts.append(semantic_text)
        partition.sys_hashes.append(sys_hash)
        partition.user_scopes.append(user_scope)
        partition.params_hashes.append(params_hash)
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
            full_prompt,
            resp_json,
            vec,
            now,
            ttl,
            sys_hash,
            semantic_text,
            user_scope,
            params_hash,
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
