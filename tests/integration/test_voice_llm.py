from __future__ import annotations

import asyncio
import json

import httpx2
import pytest

from vllm_realtime_lab.voice.llm import stream_reply

from .helpers import recording_client, sse_event


async def collect(client: httpx2.AsyncClient) -> list[str]:
    messages = [{"role": "user", "content": "Hallo"}]
    async with client:
        return [piece async for piece in stream_reply(client, "tutor", messages, "t-1")]


def test_yields_only_the_spoken_text() -> None:
    body = b"".join(
        [
            sse_event(role="assistant"),
            sse_event(reasoning="Der Nutzer"),
            sse_event(content="Gut"),
            sse_event(content=". Weiter"),
            b"data: [DONE]\n\n",
        ]
    )
    assert asyncio.run(collect(recording_client(body, []))) == ["Gut", ". Weiter"]


def test_sends_a_streamed_chat_request_with_the_turn_id() -> None:
    seen: list[httpx2.Request] = []
    asyncio.run(collect(recording_client(b"data: [DONE]\n\n", seen)))

    [request] = seen
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["x-turn-id"] == "t-1"
    body = json.loads(request.content)
    assert body["model"] == "tutor"
    assert body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "Hallo"}]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] > 0


def test_raises_when_the_server_rejects_the_request() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401)

    client = httpx2.AsyncClient(base_url="http://test", transport=httpx2.MockTransport(handler))
    with pytest.raises(httpx2.HTTPStatusError):
        asyncio.run(collect(client))
