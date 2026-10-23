from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import msgspec

TTS_PATH = "/tts/v1/voice:stream"
VOICES_PATH = "/tts/v1/voices"
SAMPLE_RATE = 24000


class TtsError(Exception):
    """Inworld TTS reported an error inside a stream that started with HTTP 200."""


class _Audio(msgspec.Struct, rename="camel"):
    audio_content: str = ""


class _Line(msgspec.Struct):
    result: _Audio | None = None
    error: dict[str, Any] | None = None


async def synthesize(
    client: httpx2.AsyncClient, text: str, *, voice: str, model: str
) -> AsyncIterator[bytes]:
    """Stream German speech for one sentence as 16-bit mono PCM chunks at `SAMPLE_RATE`.

    Raises:
        httpx2.HTTPStatusError: Inworld rejected the request.
        TtsError: Inworld reported an error after the stream started.
    """
    body = {
        "text": text,
        "voiceId": voice,
        "modelId": model,
        "language": "de-DE",
        "audioConfig": {"audioEncoding": "PCM", "sampleRateHertz": SAMPLE_RATE},
    }
    async with client.stream("POST", TTS_PATH, json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            chunk = msgspec.json.decode(line, type=_Line)
            if chunk.error is not None:
                raise TtsError(str(chunk.error.get("message", chunk.error)))
            if chunk.result is not None and chunk.result.audio_content:
                yield base64.b64decode(chunk.result.audio_content)


async def open_connection(client: httpx2.AsyncClient) -> None:
    """Open a connection to Inworld with a cheap request, so the next sentence reuses it."""
    await client.get(VOICES_PATH, params={"filter": "language=de"})
