from __future__ import annotations

import asyncio
import base64
import json

import httpx2
import pytest

from vllm_realtime_lab.voice.tts import SAMPLE_RATE, TtsError, synthesize

from .helpers import recording_client


def ndjson(*lines: object) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in lines).encode()


def audio(pcm: bytes) -> dict[str, object]:
    return {"result": {"audioContent": base64.b64encode(pcm).decode(), "usage": {}}}


async def collect(client: httpx2.AsyncClient) -> list[bytes]:
    async with client:
        speech = synthesize(client, "Gut.", voice="Carina", model="tts")
        return [chunk async for chunk in speech]


def test_yields_the_pcm_chunks_in_order() -> None:
    body = ndjson(audio(b"\x01\x02"), audio(b"\x03\x04"))
    assert asyncio.run(collect(recording_client(body, []))) == [b"\x01\x02", b"\x03\x04"]


def test_asks_for_raw_pcm_in_german() -> None:
    seen: list[httpx2.Request] = []
    asyncio.run(collect(recording_client(ndjson(audio(b"\x00\x00")), seen)))

    [request] = seen
    assert request.url.path == "/tts/v1/voice:stream"
    body = json.loads(request.content)
    assert body["text"] == "Gut."
    assert body["voiceId"] == "Carina"
    assert body["modelId"] == "tts"
    assert body["language"] == "de-DE"
    assert body["audioConfig"] == {"audioEncoding": "PCM", "sampleRateHertz": SAMPLE_RATE}


def test_raises_on_an_error_line_after_a_200() -> None:
    body = ndjson(audio(b"\x00\x00"), {"error": {"code": 8, "message": "quota"}})
    with pytest.raises(TtsError, match="quota"):
        asyncio.run(collect(recording_client(body, [])))
