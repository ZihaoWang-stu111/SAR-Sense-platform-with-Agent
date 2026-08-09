"""外部调用轻量治理：超时、错误分类、有限重试、退避+jitter、脱敏日志。

max_attempts=3（首次 + 最多重试 2 次）。
Chat 只靠 factory 对 chat_model.invoke/ainvoke 的 monkey-patch；
嵌套 call_with_retries 靠 _in_governed_call ContextVar 短路，避免 3×3。
"""

from __future__ import annotations

import contextvars
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ATTEMPTS = 3
BACKOFF_BASES = (0.5, 1.0)  # 第 1/2 次重试前的基数；实际等待 base*(1+jitter)
BACKOFF_JITTER = 0.5  # 额外 0~50%
RETRY_AFTER_CAP_S = 10.0

# 单次 HTTP 读超时默认值（秒）。重试由本模块负责，provider SDK / langchain 内置重试应关掉。
DEFAULT_OLLAMA_TIMEOUT_S = 120.0
DEFAULT_OPENAI_TIMEOUT_S = 120.0
DEFAULT_DASHSCOPE_TIMEOUT_S = 60.0
DEFAULT_TAVILY_TIMEOUT_S = 60.0

# 防止嵌套 call_with_retries 叠成 3×3
_in_governed_call: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "in_governed_call", default=False
)


def get_timeout_s(env_key: str, default: float) -> float:
    """从环境变量读超时秒数；非法/空值回落到 default。"""
    raw = os.getenv(env_key)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def resolve_chat_provider_model() -> tuple[str, str]:
    """统一解析 CHAT_PROVIDER / CHAT_MODEL_NAME（env 优先，否则 rag.yml）。"""
    from utils.config_handler import rag_conf

    provider = os.getenv(
        "CHAT_PROVIDER",
        rag_conf.get("chat_provider", "dashscope"),
    ).lower()
    model_name = os.getenv(
        "CHAT_MODEL_NAME",
        rag_conf.get("chat_model_name", ""),
    )
    return provider, model_name


@dataclass(frozen=True)
class ClassifiedError:
    error_type: str
    retryable: bool
    status_code: int | None = None
    retry_after_s: float | None = None
    message: str = ""


def _safe_message(exc: BaseException, limit: int = 200) -> str:
    text = str(exc) or type(exc).__name__
    for marker in ("sk-", "Bearer ", "api_key", "api-key", "Authorization"):
        if marker.lower() in text.lower():
            return f"{type(exc).__name__}(redacted)"
    return text[:limit]


def _get_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and 100 <= val <= 599:
            return val
    response = getattr(exc, "response", None)
    if response is not None:
        val = getattr(response, "status_code", None)
        if isinstance(val, int):
            return val
    body = getattr(exc, "body", None) or getattr(exc, "args", None)
    if isinstance(body, dict):
        code = body.get("code") or body.get("statusCode")
        if isinstance(code, int):
            return code
    return None


def _get_retry_after(exc: BaseException) -> float | None:
    headers = None
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if raw is None:
            return None
        return min(float(raw), RETRY_AFTER_CAP_S)
    except (TypeError, ValueError):
        return None


def classify_error(exc: BaseException) -> ClassifiedError:
    """把异常归类为可重试 / 不可重试。"""
    status = _get_status_code(exc)
    name = type(exc).__name__.lower()
    msg = (str(exc) or "").lower()
    retry_after = _get_retry_after(exc)

    if status in (401, 403):
        return ClassifiedError("auth_error", False, status, None, _safe_message(exc))
    if status in (400, 404, 422):
        return ClassifiedError("client_error", False, status, None, _safe_message(exc))
    if status == 429:
        return ClassifiedError("rate_limit", True, status, retry_after, _safe_message(exc))
    if status in (500, 502, 503, 504):
        return ClassifiedError("server_error", True, status, retry_after, _safe_message(exc))

    if isinstance(exc, TimeoutError) or "timeout" in name or "timed out" in msg or "timeout" in msg:
        return ClassifiedError("timeout", True, status, None, _safe_message(exc))

    # "connection" 已覆盖 connectionreset / aborted / refused
    conn_markers = (
        "connection",
        "connecterror",
        "failed to connect",
        "temporarily unavailable",
        "broken pipe",
        "nodename nor servname",
        "name or service not known",
        "max retries exceeded",
        "remoteprotocolerror",
    )
    if any(m in name or m in msg for m in conn_markers):
        return ClassifiedError("network_error", True, status, None, _safe_message(exc))

    if any(k in msg for k in ("unauthorized", "invalid api key", "authentication", "api key")):
        return ClassifiedError("auth_error", False, status, None, _safe_message(exc))

    return ClassifiedError("unknown_error", False, status, None, _safe_message(exc))


def _log_attempt(
    *,
    provider: str,
    model: str,
    classified: ClassifiedError,
    attempt: int,
    elapsed_ms: float,
    will_retry: bool,
) -> None:
    logger.warning(
        "external_call_failed provider=%s model=%s error_type=%s status_code=%s "
        "attempt=%s elapsed_ms=%.0f will_retry=%s message=%s",
        provider,
        model or "-",
        classified.error_type,
        classified.status_code if classified.status_code is not None else "-",
        attempt,
        elapsed_ms,
        will_retry,
        classified.message,
    )


def _sleep_before_retry(attempt_index: int, classified: ClassifiedError) -> None:
    if classified.retry_after_s is not None:
        time.sleep(max(0.0, classified.retry_after_s))
        return
    base = BACKOFF_BASES[min(attempt_index, len(BACKOFF_BASES) - 1)]
    time.sleep(base * (1.0 + random.random() * BACKOFF_JITTER))


def call_with_retries(
    fn: Callable[[], T],
    *,
    provider: str,
    model: str = "",
    max_attempts: int = MAX_ATTEMPTS,
) -> T:
    """同步调用：最多 max_attempts 次（默认 3 = 1 次首次 + 2 次重试）。"""
    if _in_governed_call.get():
        return fn()

    token = _in_governed_call.set(True)
    try:
        last_exc: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                classified = classify_error(exc)
                elapsed_ms = (time.perf_counter() - started) * 1000
                will_retry = classified.retryable and attempt < max_attempts
                _log_attempt(
                    provider=provider,
                    model=model,
                    classified=classified,
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                    will_retry=will_retry,
                )
                if not will_retry:
                    raise
                _sleep_before_retry(attempt - 1, classified)
        assert last_exc is not None
        raise last_exc
    finally:
        _in_governed_call.reset(token)


async def acall_with_retries(
    fn: Callable[[], T],
    *,
    provider: str,
    model: str = "",
    max_attempts: int = MAX_ATTEMPTS,
) -> T:
    """异步调用版本（fn 应返回 awaitable）。"""
    import asyncio

    if _in_governed_call.get():
        return await fn()

    token = _in_governed_call.set(True)
    try:
        last_exc: BaseException | None = None
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                return await fn()
            except Exception as exc:
                last_exc = exc
                classified = classify_error(exc)
                elapsed_ms = (time.perf_counter() - started) * 1000
                will_retry = classified.retryable and attempt < max_attempts
                _log_attempt(
                    provider=provider,
                    model=model,
                    classified=classified,
                    attempt=attempt,
                    elapsed_ms=elapsed_ms,
                    will_retry=will_retry,
                )
                if not will_retry:
                    raise
                if classified.retry_after_s is not None:
                    await asyncio.sleep(max(0.0, classified.retry_after_s))
                else:
                    base = BACKOFF_BASES[min(attempt - 1, len(BACKOFF_BASES) - 1)]
                    await asyncio.sleep(base * (1.0 + random.random() * BACKOFF_JITTER))
        assert last_exc is not None
        raise last_exc
    finally:
        _in_governed_call.reset(token)
