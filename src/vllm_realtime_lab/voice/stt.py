from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable
from typing import Any

import msgspec
import websockets
from websockets.asyncio.client import ClientConnection

STT_PATH = "/stt/v1/transcribe:streamBidirectional"
SAMPLE_RATE = 16000

_CONFIG = json.dumps(
    {
        "transcribeConfig": {
            "modelId": "inworld/inworld-stt-1",
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": SAMPLE_RATE,
            "numberOfChannels": 1,
            "language": "de",
            "inworldSttV1Config": {"vadThreshold": 0},
        },
        "audioProtocol": "BINARY",
    }
)
_END_TURN = json.dumps({"endTurn": {}})
_CLOSE_STREAM = json.dumps({"closeStream": {}})
_closing: set[asyncio.Task[None]] = set()


class SttError(Exception):
    """Inworld STT reported an error or closed the stream before the final transcript."""


class _Transcription(msgspec.Struct, rename="camel"):
    transcript: str = ""
    is_final: bool = False


class _Result(msgspec.Struct):
    transcription: _Transcription | None = None


class _Event(msgspec.Struct):
    result: _Result | None = None
    error: dict[str, Any] | None = None


async def transcribe(url: str, headers: dict[str, str], frames: AsyncIterable[bytes]) -> str:
    """Stream one turn of 16 kHz PCM to Inworld STT and return the final transcript.

    Server turn detection is off. The end of `frames` ends the turn, so the speaker decides when
    the turn is over. Final transcripts that arrive before `frames` ends are joined with the last
    one.

    Raises:
        SttError: Inworld reported an error or closed the stream early.
    """
    ws = await websockets.connect(url, additional_headers=headers, max_size=None)
    sender: asyncio.Task[None] | None = None
    try:
        await ws.send(_CONFIG)
        ended = asyncio.Event()

        async def send_turn() -> None:
            async for frame in frames:
                await ws.send(frame)
            await ws.send(_END_TURN)
            ended.set()

        sender = asyncio.create_task(send_turn())
        finals: list[str] = []
        async for message in ws:
            event = msgspec.json.decode(message, type=_Event)
            if event.error is not None:
                raise SttError(str(event.error.get("message", event.error)))
            transcription = event.result.transcription if event.result else None
            if transcription is None or not transcription.is_final:
                continue
            finals.append(transcription.transcript)
            if ended.is_set():
                await ws.send(_CLOSE_STREAM)
                return " ".join(text for text in finals if text)
        if sender.done() and not sender.cancelled() and sender.exception() is not None:
            raise SttError("sending the audio failed") from sender.exception()
        raise SttError("the stream closed before the final transcript")
    finally:
        if sender is not None:
            sender.cancel()
        _close_later(ws)


def _close_later(ws: ClientConnection) -> None:
    # Closing waits a round trip for the server, and the transcript must not wait for it.
    task = asyncio.create_task(ws.close())
    _closing.add(task)
    task.add_done_callback(_closing.discard)
