import time
import json
import uuid
import asyncio
import httpx
from typing import Tuple, Dict, Any, Optional, AsyncGenerator
from fastapi import HTTPException
from app.config import settings, get_active_provider_info
from app.models.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, UsageInfo
from app.core.circuit_breaker import circuit_breaker

class UpstreamProxy:
    """
    High-concurrency async reverse proxy supporting multi-provider routing
    (Ollama, Groq, OpenAI, RouterPlus) with full SSE streaming passthrough
    and automatic circuit-breaker failover.
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
        return key not in ("mock-primary-key", "mock-fallback-key", "your-primary-api-key-here", "dummy", "")

    async def forward(self, request: ChatCompletionRequest) -> Tuple[ChatCompletionResponse, str]:
        """
        Routes non-streaming request to active healthy upstream provider.
        Returns: (ChatCompletionResponse, provider_name: str)
        """
        route, state = circuit_breaker.get_route()
        active = get_active_provider_info()
        is_live = self._is_live_key(active["api_key"]) or active["name"] == "ollama"

        if is_live and route == "primary":
            try:
                payload = request.model_dump(exclude_unset=True)
                if not payload.get("model"):
                    payload["model"] = active["default_model"]

                response = await self._call_upstream(
                    url=f"{active['url'].rstrip('/')}/chat/completions",
                    api_key=active["api_key"],
                    payload=payload
                )
                circuit_breaker.record_success()
                return response, f"{active['name']}_live"
            except httpx.HTTPStatusError as e:
                # 4xx client errors MUST NOT trip the circuit breaker
                if 400 <= e.response.status_code < 500:
                    circuit_breaker.release_probe()
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=f"Upstream provider client error ({e.response.status_code}): {e.response.text[:200]}"
                    )
                circuit_breaker.record_failure()
                return await self._call_fallback(request)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError):
                circuit_breaker.record_failure()
                return await self._call_fallback(request)
            except Exception:
                circuit_breaker.record_failure()
                return await self._call_fallback(request)

        # Fallback / Simulated mode for fast local testing and offline benchmarks
        return await self._call_simulated_upstream(request, route)

    async def forward_stream(
        self,
        request: ChatCompletionRequest
    ) -> Tuple[AsyncGenerator[str, None], str]:
        """
        Routes streaming request to upstream with SSE chunk passthrough.
        Returns: (AsyncGenerator yielding 'data: {...}\n\n', provider_name)
        """
        route, state = circuit_breaker.get_route()
        active = get_active_provider_info()
        is_live = self._is_live_key(active["api_key"]) or active["name"] == "ollama"

        if is_live and route == "primary":
            try:
                payload = request.model_dump(exclude_unset=True)
                payload["stream"] = True
                if not payload.get("model"):
                    payload["model"] = active["default_model"]

                url = f"{active['url'].rstrip('/')}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {active['api_key']}",
                    "Content-Type": "application/json"
                }

                async def live_stream_generator():
                    settled = False  # did we report a success/failure verdict to the breaker?
                    try:
                        async with self.client.stream("POST", url, headers=headers, json=payload) as resp:
                            if resp.status_code >= 400:
                                error_text = await resp.aread()
                                if resp.status_code >= 500:
                                    circuit_breaker.record_failure()
                                    settled = True
                                yield f"data: {json.dumps({'error': error_text.decode('utf-8', errors='ignore')})}\n\n"
                                return
                            circuit_breaker.record_success()
                            settled = True
                            async for line in resp.aiter_lines():
                                if line:
                                    yield f"{line}\n\n"
                    except Exception as exc:
                        if not settled:
                            circuit_breaker.record_failure()
                            settled = True
                        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        if not settled:  # 4xx, client disconnect, cancellation
                            circuit_breaker.release_probe()

                return live_stream_generator(), f"{active['name']}_stream"
            except Exception:
                circuit_breaker.record_failure()

        # Simulated stream for local development / testing
        return self._simulated_stream_generator(request, route), f"simulated_{route}_stream"

    async def _simulated_stream_generator(
        self,
        request: ChatCompletionRequest,
        route: str
    ) -> AsyncGenerator[str, None]:
        """Simulates token-by-token OpenAI SSE streaming for offline/testing use."""
        last_msg = request.messages[-1].content if request.messages else "Hello"
        answer = f"Processed response for: '{last_msg[:60]}...' via {route} engine."
        words = answer.split()
        chunk_id = f"chatcmpl-stream-{uuid.uuid4().hex[:8]}"

        for i, word in enumerate(words):
            delta = {"role": "assistant"} if i == 0 else {}
            delta["content"] = word + (" " if i < len(words) - 1 else "")
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.015)  # 15ms typewriter delay

        # Final stop chunk
        stop_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(stop_chunk)}\n\n"
        yield "data: [DONE]\n\n"

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
        """Simulated upstream response (for testing and offline benchmarking)."""
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
proxy = proxy_client
