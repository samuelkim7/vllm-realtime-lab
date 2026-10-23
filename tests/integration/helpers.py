from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx2
from starlette.testclient import TestClient

from vllm_realtime_lab.voice.orchestrator import Pipeline, create_app, tutor_prompt
from vllm_realtime_lab.voice.tts import TtsError


def sse_event(**delta: Any) -> bytes:
    """Return one streamed chat completion chunk as a server-sent event."""
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()


def read_turns(path: Path) -> list[dict[str, Any]]:
    """Return the records of a JSON lines log."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def recording_client(
    body: bytes, seen: list[httpx2.Request], content_type: str = "text/event-stream"
) -> httpx2.AsyncClient:
    """Return a client whose server answers every request with the body and records it."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, content=body, headers={"content-type": content_type})

    return httpx2.AsyncClient(base_url="http://test", transport=httpx2.MockTransport(handler))


class Fakes:
    """Stands in for Inworld and the tutor model. The audio of a sentence is its own text."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.histories: list[list[str]] = []
        self.system_prompts: list[str] = []
        self.broken_speech = False
        self.broken_warm_up = False

    async def transcribe(self, frames: AsyncIterable[bytes]) -> str:
        audio = b"".join([frame async for frame in frames])
        return f"Ich habe {len(audio)} Bytes gesagt."

    async def reply(self, messages: list[dict[str, str]], turn_id: str) -> AsyncIterator[str]:
        self.histories.append([message["role"] for message in messages])
        self.system_prompts.append(messages[0]["content"])
        for piece in ["Gut", ". Weiter", " so."]:
            yield piece

    async def speak(self, text: str) -> AsyncIterator[bytes]:
        if self.broken_speech:
            raise TtsError("quota")
        yield text.encode()

    async def warm_up(self) -> None:
        if self.broken_warm_up:
            raise TimeoutError

    def pipeline(self) -> Pipeline:
        return Pipeline(
            transcribe=self.transcribe,
            reply=self.reply,
            speak=self.speak,
            warm_up=self.warm_up,
            system_prompt=tutor_prompt("Carina"),
            log_path=self.log_path,
        )


@contextmanager
def talk_session(fakes: Fakes) -> Iterator[Any]:
    """Open the talk WebSocket on an orchestrator that runs on the fakes."""
    with (
        TestClient(create_app(fakes.pipeline())) as client,
        client.websocket_connect("/talk") as ws,
    ):
        yield ws
