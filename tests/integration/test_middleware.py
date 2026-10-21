from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from vllm_realtime_lab.server import middleware
from vllm_realtime_lab.server.middleware import ASGIApp, Message, Scope, TurnLogMiddleware

CHAT_PATH = "/v1/chat/completions"


class FakeClock:
    """Stands in for time.time. Tests set now by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Replace the middleware's clock with a FakeClock that starts at 1000.0."""
    fake = FakeClock()
    monkeypatch.setattr(middleware, "_now", fake)
    return fake


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the turn log at a file in tmp_path."""
    path = tmp_path / "turns.jsonl"
    monkeypatch.setenv(middleware.LOG_ENV, str(path))
    return path


def sse_event(**delta: Any) -> bytes:
    chunk = {"choices": [{"delta": delta}]}
    return f"data: {json.dumps(chunk)}\n\n".encode()


def build_app(
    chunks: list[tuple[float, bytes]],
    clock: FakeClock,
    content_type: bytes = b"text/event-stream",
) -> ASGIApp:
    async def app(scope: Scope, receive: Any, send: Any) -> None:
        start = {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", content_type)],
        }
        await send(start)
        for at, body in chunks:
            clock.now = at
            await send({"type": "http.response.body", "body": body, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    return app


def build_scope(path: str = CHAT_PATH, turn_id: str | None = None) -> Scope:
    headers = []
    if turn_id is not None:
        headers.append((b"x-turn-id", turn_id.encode()))
    return {"type": "http", "method": "POST", "path": path, "headers": headers}


def call_app(app: ASGIApp, scope: Scope) -> list[Message]:
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    async def serve_once() -> None:
        await app(scope, receive, send)

    asyncio.run(serve_once())
    return sent


def read_turns(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text().splitlines():
        yield json.loads(line)


def test_passes_lifespan_through(log_path: Path) -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    call_app(TurnLogMiddleware(app), {"type": "lifespan"})
    assert seen == ["lifespan"]


def test_records_the_turn_timestamps(log_path: Path, clock: FakeClock) -> None:
    app = build_app(
        [
            (1000.1, sse_event(role="assistant")),
            (1000.2, sse_event(reasoning="Der Nutzer")),
            (1000.5, sse_event(content="Gut")),
            (1000.6, sse_event(content=".")),
            (1000.7, sse_event(content=" Du hast")),
            (1000.9, sse_event(content=" recht.")),
            (1001.0, b"data: [DONE]\n\n"),
        ],
        clock,
    )
    sent = call_app(TurnLogMiddleware(app), build_scope(turn_id="t-1"))

    assert sent[0]["status"] == 200
    [turn] = read_turns(log_path)
    assert turn == {
        "turn_id": "t-1",
        "status": 200,
        "cold": True,
        "t_received": 1000.0,
        "t_first_reasoning": 1000.2,
        "t_first_content": 1000.5,
        "t_first_sentence": 1000.7,
        "t_last_token": 1000.9,
        "t_end": 1001.0,
    }


def test_parses_an_event_split_across_chunks(log_path: Path, clock: FakeClock) -> None:
    event = sse_event(content="Hallo.")
    app = build_app([(1000.3, event[:15]), (1000.4, event[15:])], clock)
    call_app(TurnLogMiddleware(app), build_scope())

    [turn] = read_turns(log_path)
    assert turn["t_first_content"] == 1000.4
    assert turn["t_first_sentence"] == 1000.4


def test_skips_an_unreadable_chunk_without_breaking_the_reply(
    log_path: Path, clock: FakeClock
) -> None:
    app = build_app([(1000.1, b"data: not json\n\n"), (1000.2, sse_event(content="Ja."))], clock)
    sent = call_app(TurnLogMiddleware(app), build_scope())

    assert sent[-1]["more_body"] is False
    [turn] = read_turns(log_path)
    assert turn["t_first_content"] == 1000.2


def test_only_the_first_turn_is_cold(log_path: Path, clock: FakeClock) -> None:
    tutor = TurnLogMiddleware(build_app([(1000.1, sse_event(content="Ja."))], clock))
    call_app(tutor, build_scope())
    call_app(tutor, build_scope())
    assert [t["cold"] for t in read_turns(log_path)] == [True, False]


def test_logs_an_aborted_turn_and_reraises(log_path: Path, clock: FakeClock) -> None:
    async def app(scope: Scope, receive: Any, send: Any) -> None:
        raise ConnectionResetError

    with pytest.raises(ConnectionResetError):
        call_app(TurnLogMiddleware(app), build_scope())
    [turn] = read_turns(log_path)
    assert turn["t_end"] is None


def test_does_not_log_other_paths(log_path: Path, clock: FakeClock) -> None:
    sent = call_app(TurnLogMiddleware(build_app([], clock)), build_scope("/v1/models"))
    assert sent[0]["status"] == 200
    assert not log_path.exists()


def test_reads_a_non_streamed_reply(log_path: Path, clock: FakeClock) -> None:
    body = json.dumps({"choices": [{"message": {"content": "Gut. Weiter", "reasoning": "x"}}]})
    app = build_app([(1002.0, body.encode())], clock, content_type=b"application/json")
    call_app(TurnLogMiddleware(app), build_scope())
    [turn] = read_turns(log_path)
    assert turn["t_first_content"] == turn["t_first_sentence"] == turn["t_end"] == 1002.0
