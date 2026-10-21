from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

import msgspec

from vllm_realtime_lab.engines import vllm
from vllm_realtime_lab.speech.sentences import first_sentence_end

log = logging.getLogger(__name__)

# ASGI types, spelled out because Starlette is not a dependency of the lab.
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

LOG_ENV = "VRL_TURN_LOG"

_now = time.time


class TurnRecord(msgspec.Struct, kw_only=True):
    """One chat turn as the server saw it. The middleware writes it as one JSON line.

    Timestamps are Unix seconds. A timestamp is None when the reply had no token of that kind.

    Attributes:
        turn_id: The x-turn-id header the client sent, or a random id.
        status: The HTTP status vLLM answered with. None when vLLM never answered.
        cold: True for the first turn after the server started.
        t_received: The request reached the server.
        t_first_reasoning: The first thinking token left the server.
        t_first_content: The first token of the spoken reply left the server. This is the TTFT.
        t_first_sentence: The end of the first sentence became known. The speech model can
            start at this point.
        t_last_token: The last token of any kind left the server.
        t_end: The response finished. None when it never finished, for example after a client
            disconnect.
    """

    turn_id: str
    status: int | None = None
    cold: bool
    t_received: float
    t_first_reasoning: float | None = None
    t_first_content: float | None = None
    t_first_sentence: float | None = None
    t_last_token: float | None = None
    t_end: float | None = None


class TurnTracker:
    """Reads one chat response as vLLM sends it and notes when each kind of token first appears.

    An unreadable chunk is logged and skipped. The measurement must never break the reply.
    """

    def __init__(self, record: TurnRecord) -> None:
        self.record = record
        self.content = ""
        self.streamed = True
        self.pending = b""

    def on_response_start(self, message: Message) -> None:
        """Note the status and whether the body is a stream of events."""
        self.record.status = message["status"]
        content_type = dict(message.get("headers", [])).get(b"content-type", b"")
        self.streamed = content_type.startswith(vllm.EVENT_STREAM)

    def on_body(self, body: bytes, at: float) -> None:
        """Read every complete event in the bytes received so far."""
        self.pending += body
        if not self.streamed:
            return
        *lines, self.pending = self.pending.split(b"\n")
        for line in lines:
            if not line.startswith(vllm.DATA_PREFIX):
                continue
            payload = line.removeprefix(vllm.DATA_PREFIX).strip()
            if payload != vllm.DONE:
                self._read(payload, at)

    def on_end(self, at: float) -> None:
        """Close the turn when vLLM sends the last body message."""
        if not self.streamed and self.pending:
            self._read(self.pending, at)
        if self.record.t_first_sentence is None:
            self._note_sentence(at, final=True)
        self.record.t_end = at

    def _read(self, payload: bytes, at: float) -> None:
        try:
            deltas = vllm.decode(payload)
        except msgspec.DecodeError as exc:
            log.warning("turn %s: unreadable response chunk: %s", self.record.turn_id, exc)
            return
        for delta in deltas:
            self._apply(delta, at)

    def _apply(self, delta: vllm.Delta, at: float) -> None:
        record = self.record
        if delta.reasoning and record.t_first_reasoning is None:
            record.t_first_reasoning = at
        if delta.content:
            if record.t_first_content is None:
                record.t_first_content = at
            if record.t_first_sentence is None:
                self.content += delta.content
                self._note_sentence(at, final=False)
        if delta.reasoning or delta.content:
            record.t_last_token = at

    def _note_sentence(self, at: float, *, final: bool) -> None:
        if first_sentence_end(self.content, final=final) is not None:
            self.record.t_first_sentence = at


class TurnLogMiddleware:
    """Logs one TurnRecord per chat completion and passes every request through unchanged.

    vLLM loads it with `--middleware` and constructs it with the app alone, so the log path comes
    from the environment. Requests are already authenticated by Modal's proxy before they reach
    the container.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.log_path = Path(os.environ[LOG_ENV])
        self.cold = True

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == vllm.CHAT_PATH:
            await self._track(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def _track(self, scope: Scope, receive: Receive, send: Send) -> None:
        turn_id = dict(scope["headers"]).get(b"x-turn-id", b"").decode("latin-1")
        turn = TurnTracker(
            TurnRecord(turn_id=turn_id or uuid.uuid4().hex, cold=self.cold, t_received=_now())
        )
        self.cold = False

        async def send_tracked(message: Message) -> None:
            at = _now()
            await send(message)
            if message["type"] == "http.response.start":
                turn.on_response_start(message)
            elif message["type"] == "http.response.body":
                turn.on_body(message.get("body", b""), at)
                if not message.get("more_body", False):
                    turn.on_end(at)

        try:
            await self.app(scope, receive, send_tracked)
        finally:
            self._append(turn.record)

    def _append(self, record: TurnRecord) -> None:
        with self.log_path.open("ab") as f:
            f.write(msgspec.json.encode(record) + b"\n")
