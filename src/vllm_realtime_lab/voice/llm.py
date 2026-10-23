from __future__ import annotations

from collections.abc import AsyncIterator

import httpx2

from vllm_realtime_lab.engines import vllm

_DONE = vllm.DONE.decode()
_READY_ATTEMPTS = 10
# Qwen's settings for replies without thinking. mlx-lm would otherwise decode greedily.
_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


async def stream_reply(
    client: httpx2.AsyncClient, model: str, messages: list[dict[str, str]], turn_id: str
) -> AsyncIterator[str]:
    """Stream the tutor's reply and yield each new piece of spoken text.

    Thinking text is dropped, because only the spoken reply goes to TTS. The turn ID header
    joins this turn to the server's turn log.

    Raises:
        httpx2.HTTPStatusError: the server rejected the request.
    """
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        **_SAMPLING,
        # vLLM's qwen3 reasoning parser assumes thinking unless the request turns it off.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"x-turn-id": turn_id}
    async with client.sse(vllm.CHAT_PATH, method="POST", json=body, headers=headers) as events:
        events.response.raise_for_status()
        async for event in events:
            if event.data == _DONE:
                return
            for delta in vllm.decode(event.data):
                if delta.content:
                    yield delta.content


async def wait_until_ready(client: httpx2.AsyncClient) -> None:
    """Return once the tutor model answers. A cold GPU container takes minutes to answer.

    Modal answers a request held longer than 150 s with a redirect, so each try stops at 140 s.

    Raises:
        httpx2.HTTPStatusError: the server rejected the request, for example without proxy auth.
        TimeoutError: the model did not answer in time.
    """
    for _ in range(_READY_ATTEMPTS):
        try:
            response = await client.get(vllm.MODELS_PATH, timeout=140)
        except httpx2.TimeoutException:
            continue
        if response.status_code == 200:
            return
        if 400 <= response.status_code < 500:
            response.raise_for_status()
    raise TimeoutError("the tutor model did not answer")
