"""Muse Spark Open on Baseten (plan 3.4), as a Transport for the M1 client.

Baseten's Model APIs are OpenAI-compatible chat completions. This transport takes the client's request shape
(input messages with input_text / input_image parts, max_output_tokens) and returns the client's response
shape (output message items, usage), so the cache, ledger, budgets and the paid-call gate apply unchanged.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from longtail.llm.client import ModelClient, TransportError

BASE_URL = "https://inference.baseten.co/v1"
# ASSUMPTION A-218: the Spark Open model slug on Baseten is meta-models/muse-spark-open-1026 (plan 3); VERIFY
SPARK_OPEN = "meta-models/muse-spark-open-1026"


def _part(part: Any) -> Any:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    kind = part.get("type")
    if kind in ("input_text", "text"):
        return {"type": "text", "text": part["text"]}
    if kind in ("input_image", "image_url"):
        url = part.get("image_url")
        url = url["url"] if isinstance(url, dict) else url
        return {"type": "image_url", "image_url": {"url": url}}
    raise ValueError(f"unsupported content part {kind!r}")


def to_chat(request: dict[str, Any]) -> dict[str, Any]:
    """The client's request shape to a chat-completions payload."""
    items = request["input"]
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    messages = []
    for m in items:
        content = m["content"]
        messages.append(
            {
                "role": m["role"],
                "content": content if isinstance(content, str) else [_part(p) for p in content],
            }
        )
    payload: dict[str, Any] = {"model": request["model"], "messages": messages}
    if "max_output_tokens" in request:
        payload["max_tokens"] = request["max_output_tokens"]
    for k in ("temperature", "response_format"):
        if k in request:
            payload[k] = request[k]
    return payload


def from_chat(raw: dict[str, Any]) -> dict[str, Any]:
    """A chat-completions response to the client's response shape."""
    text = "".join((c.get("message") or {}).get("content") or "" for c in raw.get("choices") or [])
    usage = raw.get("usage") or {}
    return {
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
        ],
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        },
    }


class BasetenTransport:
    """Exercised only by a live call. VERIFY on the first one: the slug, image parts and usage fields."""

    def __init__(self, api_key: str, base_url: str = BASE_URL, chat: Any | None = None) -> None:
        if chat is None:
            from openai import APIStatusError, OpenAI

            chat = OpenAI(api_key=api_key, base_url=base_url, max_retries=0).chat.completions
            self._status_error: type[Exception] | None = APIStatusError
        else:
            self._status_error = None
        self._chat = chat

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._chat.create(**to_chat(request))
        except Exception as e:
            if self._status_error is not None and isinstance(e, self._status_error):
                raise TransportError(e.status_code, str(e)) from e
            raise
        return from_chat(resp if isinstance(resp, dict) else resp.model_dump())

    def upload(self, path: Path) -> str:
        raise NotImplementedError("Spark Open takes images inline; there is no file upload")

    def stream(self, request: dict[str, Any]) -> Iterator[str]:
        raise NotImplementedError("the Spark Open calls here are not streamed")


def spark_open_client() -> ModelClient:
    """A client for Spark Open: same cache, ledger and gates as the Meta client."""
    key = os.environ.get("BASETEN_API_KEY", "").strip()
    return ModelClient(BasetenTransport(key) if key else None)
