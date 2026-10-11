import time
import httpx
from typing import Tuple, Dict, Any, Optional
from fastapi import HTTPException
from app.config import settings
from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, UsageInfo
from app.core.circuit_breaker import circuit_breaker

class UpstreamProxy:
    """
    High-concurrency async reverse proxy client with connection pooling,
    automatic circuit-breaker failover, and zero-token simulated mock fallback.
    Distinguishes client 4xx errors (no circuit trip) from 5xx/timeouts (circuit trip).
    """
    def __init__(self):
        # High-throughput connection pool: 100 keep-alive, 500 max connections
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=500),
            timeout=httpx.Timeout(settings.CIRCUIT_TIMEOUT_SECONDS)
        )

    def _is_live_key(self, key: str) -> bool:
        if not key:
            return False
        return key not in ("mock-primary-key", "mock-fallback-key", "your-primary-api-key-here", "")

    async def forward(self, request: ChatCompletionRequest) -> Tuple[ChatCompletionResponse, str]:
        """
        Routes request to healthy upstream provider.
        Returns: (ChatCompletionResponse, provider_name: str)
        """
        route, state = circuit_breaker.get_route()
        is_live_primary = self._is_live_key(settings.PRIMARY_API_KEY)

        if is_live_primary and route == "primary":
            try:
                payload = request.model_dump(exclude_unset=True)
                if not payload.get("model"):
                    payload["model"] = settings.PRIMARY_MODEL

                response = await self._call_upstream(
                    url=f"{settings.PRIMARY_PROVIDER_URL}/chat/completions",
                    api_key=settings.PRIMARY_API_KEY,
                    payload=payload
                )
                circuit_breaker.record_success()
                return response, "primary_live"
            except httpx.HTTPStatusError as e:
                # 4xx client errors MUST NOT trip the circuit breaker
                if 400 <= e.response.status_code < 500:
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=f"Upstream provider client error ({e.response.status_code}): {e.response.text[:200]}"
                    )
                # 5xx server error from upstream: record failure and try fallback
                circuit_breaker.record_failure()
                return await self._call_fallback(request)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError):
                circuit_breaker.record_failure()
                return await self._call_fallback(request)
            except Exception:
                circuit_breaker.record_failure()
                return await self._call_fallback(request)

        # Fallback / Simulated mode for fast local testing and benchmarks
        return await self._call_simulated_upstream(request, route)

    async def _call_fallback(self, request: ChatCompletionRequest) -> Tuple[ChatCompletionResponse, str]:
        """Calls secondary fallback provider."""
        is_live_fallback = self._is_live_key(settings.FALLBACK_API_KEY)
        if is_live_fallback:
            try:
                payload = request.model_dump(exclude_unset=True)
                if not payload.get("model"):
                    payload["model"] = settings.FALLBACK_MODEL

                response = await self._call_upstream(
                    url=f"{settings.FALLBACK_PROVIDER_URL}/chat/completions",
                    api_key=settings.FALLBACK_API_KEY,
                    payload=payload
                )
                return response, "fallback_live"
            except httpx.HTTPStatusError as e:
                if 400 <= e.response.status_code < 500:
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=f"Fallback provider client error ({e.response.status_code}): {e.response.text[:200]}"
                    )
            except Exception:
                pass
        return await self._call_simulated_upstream(request, "fallback_failover")

    async def _call_upstream(self, url: str, api_key: str, payload: Dict[str, Any]) -> ChatCompletionResponse:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        res = await self.client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        data = res.json()
        return ChatCompletionResponse.model_validate(data)

    async def _call_simulated_upstream(self, request: ChatCompletionRequest, route: str) -> Tuple[ChatCompletionResponse, str]:
        """
        Simulated upstream response (simulating real LLM completion)
        Enables stress-testing at scale without external API dependency.
        """
        last_msg = request.messages[-1].content if request.messages else "Hello"
        answer = f"Processed response for: '{last_msg[:60]}...' via {route} engine."

        prompt_tokens = sum(len(m.content.split()) for m in request.messages) + 10
        completion_tokens = len(answer.split()) + 5

        resp = ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=answer),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            ),
            provider=f"upstream_{route}"
        )
        return resp, f"upstream_{route}"

proxy_client = UpstreamProxy()
