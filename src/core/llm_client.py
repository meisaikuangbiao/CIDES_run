from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

# 默认让 httpx / openai 保持 INFO 级别（每个请求都打 1 行日志，便于观察评测进度
# 与 503/429 重试情况）。如果你想压制，设环境变量 LLM_HTTP_LOG_LEVEL=WARNING 即可。
_http_log_level = os.getenv("LLM_HTTP_LOG_LEVEL")
if _http_log_level:
    for _noisy in ("httpx", "httpcore", "openai._base_client"):
        logging.getLogger(_noisy).setLevel(_http_log_level.upper())


class EmptyLLMResponseError(RuntimeError):
    """Raised when the provider returns a successful response with empty content."""


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResult:
    content: str
    model: str
    latency_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False
    raw: Optional[dict[str, Any]] = field(default=None, repr=False)


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat endpoint with retry and disk cache."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: Optional[float] = None,
        cache_dir: Optional[str | Path] = None,
        max_retries: int = 3,
        default_thinking: bool = False,
        default_reasoning_effort: Optional[str] = None,
        max_in_flight: Optional[int] = None,
        in_flight_semaphore: Optional[threading.BoundedSemaphore] = None,
    ) -> None:
        # 兼容 DeepSeek 官方推荐的 DEEPSEEK_API_KEY 变量名；OPENAI_API_KEY 也可用
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
        self.default_model = default_model or os.getenv("SUT_MODEL", "deepseek-v4-flash")
        timeout_env = os.getenv("LLM_REQUEST_TIMEOUT")
        self.timeout = timeout if timeout is not None else (float(timeout_env) if timeout_env else 60.0)
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.api_key:
            logger.warning(
                "LLMClient initialised without API key; live calls will fail. "
                "Set DEEPSEEK_API_KEY or OPENAI_API_KEY in your .env file."
            )
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key or "dummy",
            "timeout": self.timeout,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)
        self._is_deepseek = bool(self.base_url and "deepseek" in self.base_url)
        self.default_thinking = default_thinking
        self.default_reasoning_effort = default_reasoning_effort
        # in-flight 信号量优先沿用外部传入的实例（多个 LLMClient 共享同一上限），
        # 否则按 max_in_flight 创建独立的；非正值或 None 表示不限制。
        self._in_flight_semaphore: Optional[threading.BoundedSemaphore]
        if in_flight_semaphore is not None:
            self._in_flight_semaphore = in_flight_semaphore
            self.max_in_flight = max_in_flight or 0
        elif max_in_flight and max_in_flight > 0:
            self._in_flight_semaphore = threading.BoundedSemaphore(max_in_flight)
            self.max_in_flight = max_in_flight
        else:
            self._in_flight_semaphore = None
            self.max_in_flight = 0

    def _cache_key(self, payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[ChatResult]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ChatResult(
                content=data["content"],
                model=data["model"],
                latency_ms=data.get("latency_ms", 0),
                tokens_in=data.get("tokens_in", 0),
                tokens_out=data.get("tokens_out", 0),
                cached=True,
            )
        except Exception:
            logger.exception("Failed to read cache entry %s", path)
            return None

    def _cache_put(self, key: str, result: ChatResult) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.json"
        payload = {
            "content": result.content,
            "model": result.model,
            "latency_ms": result.latency_ms,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
        }
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.exception("Failed to write cache entry %s", path)

    def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
        cache: bool = True,
        extra: Optional[dict[str, Any]] = None,
        thinking: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> ChatResult:
        """Send a chat completion request.

        DeepSeek-specific knobs:
        - ``thinking=True`` enables v4-pro/v4-flash's reasoning mode by passing
          ``thinking={"type": "enabled"}`` in the request body (via ``extra_body``).
        - ``reasoning_effort`` (``low``/``medium``/``high``) only applies when
          ``thinking`` is on. Non-DeepSeek backends should leave these defaults.
        """
        target_model = model or self.default_model
        normalized: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                normalized.append(m.to_dict())
            else:
                normalized.append({"role": m["role"], "content": m["content"]})
        payload: dict[str, Any] = {
            "model": target_model,
            "messages": normalized,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed
        effective_thinking = thinking or self.default_thinking
        effective_reasoning = reasoning_effort or self.default_reasoning_effort
        if effective_thinking and self._is_deepseek:
            extra_body = payload.setdefault("extra_body", {})
            extra_body["thinking"] = {"type": "enabled"}
            if effective_reasoning:
                payload["reasoning_effort"] = effective_reasoning
        if extra:
            for k, v in extra.items():
                if k == "extra_body" and isinstance(v, dict):
                    eb = payload.setdefault("extra_body", {})
                    eb.update(v)
                else:
                    payload[k] = v
        cache_key = self._cache_key(payload) if cache else None
        if cache_key is not None:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
        result = self._chat_with_retry(payload)
        if cache_key is not None:
            self._cache_put(cache_key, result)
        return result

    def _chat_with_retry(self, payload: dict[str, Any]) -> ChatResult:
        sem_ctx: contextlib.AbstractContextManager[Any] = (
            self._in_flight_semaphore
            if self._in_flight_semaphore is not None
            else contextlib.nullcontext()
        )

        @retry(
            stop=stop_after_attempt(self.max_retries),
            # 5xx / 429 风暴时把退避拉得更长，避免 thundering herd
            # 重击同一个临时过载的实例。
            wait=wait_exponential(multiplier=1.5, min=2.0, max=30.0),
            retry=retry_if_exception_type(
                (
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                    APIStatusError,
                    EmptyLLMResponseError,
                )
            ),
            reraise=True,
        )
        def _call() -> ChatResult:
            with sem_ctx:
                start = time.perf_counter()
                response = self._client.chat.completions.create(**payload)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                if not content:
                    raise EmptyLLMResponseError(
                        f"Empty content from model {payload['model']}"
                    )
                usage = getattr(response, "usage", None)
                return ChatResult(
                    content=content,
                    model=payload["model"],
                    latency_ms=elapsed_ms,
                    tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
                    raw=None,
                )

        return _call()


def build_default_client(
    cache_dir: Optional[str | Path] = "data/.llm_cache",
    *,
    thinking: bool = False,
    reasoning_effort: Optional[str] = None,
    max_in_flight: Optional[int] = None,
    in_flight_semaphore: Optional[threading.BoundedSemaphore] = None,
) -> LLMClient:
    return LLMClient(
        cache_dir=cache_dir,
        default_thinking=thinking,
        default_reasoning_effort=reasoning_effort,
        max_in_flight=max_in_flight,
        in_flight_semaphore=in_flight_semaphore,
    )
