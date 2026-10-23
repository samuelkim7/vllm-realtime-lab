from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from vllm_realtime_lab.voice.stt import SttError, transcribe


class FakeInworld:
    """Answers like Inworld STT with turn detection off. Tests set the events to send."""

    def __init__(self, before_end: list[dict[str, Any]], after_end: list[dict[str, Any]]) -> None:
        self.before_end = before_end
        self.after_end = after_end
        self.config: dict[str, Any] = {}
        self.audio = b""
        self.controls: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}

    async def handle(self, ws: ServerConnection) -> None:
        self.headers = dict(ws.request.headers) if ws.request else {}
        self.config = json.loads(await ws.recv())
        async for message in ws:
            if isinstance(message, bytes):
                if not self.audio:
                    for event in self.before_end:
                        await ws.send(json.dumps(event))
                self.audio += message
                continue
            control = json.loads(message)
            self.controls.append(control)
            if "endTurn" in control:
                for event in self.after_end:
                    await ws.send(json.dumps(event))


def final(text: str) -> dict[str, Any]:
    return {"result": {"transcription": {"transcript": text, "isFinal": True}}}


async def frames(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk
        # Real audio arrives in real time, so server events can land between frames.
        await asyncio.sleep(0.05)


async def run(fake: FakeInworld, *chunks: bytes) -> str:
    async with serve(fake.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}/stt"
        return await transcribe(url, {"Authorization": "Basic k"}, frames(*chunks))


def test_streams_the_turn_and_returns_the_final_transcript() -> None:
    fake = FakeInworld(before_end=[], after_end=[final("Ich wohne in Berlin.")])
    assert asyncio.run(run(fake, b"\x01\x00", b"\x02\x00")) == "Ich wohne in Berlin."

    assert fake.audio == b"\x01\x00\x02\x00"
    assert fake.controls[0] == {"endTurn": {}}
    assert fake.headers["authorization"] == "Basic k"


def test_turns_server_turn_detection_off() -> None:
    fake = FakeInworld(before_end=[], after_end=[final("Ja.")])
    asyncio.run(run(fake, b"\x00\x00"))

    config = fake.config["transcribeConfig"]
    assert fake.config["audioProtocol"] == "BINARY"
    assert config["language"] == "de"
    assert config["sampleRateHertz"] == 16000
    assert config["inworldSttV1Config"]["vadThreshold"] == 0


def test_joins_finals_that_arrive_before_the_end_of_the_turn() -> None:
    fake = FakeInworld(before_end=[final("Erstens.")], after_end=[final("Zweitens.")])
    assert asyncio.run(run(fake, b"\x00\x00")) == "Erstens. Zweitens."


def test_raises_on_an_error_event() -> None:
    fake = FakeInworld(before_end=[], after_end=[{"error": {"code": 3, "message": "bad audio"}}])
    with pytest.raises(SttError, match="bad audio"):
        asyncio.run(run(fake, b"\x00\x00"))
