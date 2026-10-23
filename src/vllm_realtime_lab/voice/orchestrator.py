from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import httpx2
import msgspec
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.types import Lifespan
from starlette.websockets import WebSocket

from vllm_realtime_lab import jsonl
from vllm_realtime_lab.speech.sentences import split_sentences
from vllm_realtime_lab.voice import llm, stt, tts
from vllm_realtime_lab.voice.protocol import (
    TALK_PATH,
    ClientMessage,
    Failed,
    Ready,
    Sentence,
    Start,
    Stop,
    Transcript,
    TurnEnd,
    encode,
)
from vllm_realtime_lab.voice.queues import drain
from vllm_realtime_lab.voice.settings import VoiceSettings

log = logging.getLogger(__name__)

# Without this, a connection idle for 5 s closes, and the next turn pays a new TLS handshake.
_KEEP_ALIVE = httpx2.Limits(keepalive_expiry=120)


def tutor_prompt(name: str) -> str:
    """Return the system prompt for a tutor that introduces itself by the name of its voice."""
    return (
        f"Du bist {name} aus Berlin und unterhältst dich ganz locker mit jemandem, der Deutsch "
        "lernt. Rede wie mit einer Freundin oder einem Freund: Geh auf das ein, was die Person "
        "erzählt, erzähl auch kurz etwas von dir und stell dann eine offene Frage, damit das "
        "Gespräch weiterläuft. Benutze einfache Alltagssprache auf B1-Niveau. Antworte mit "
        "höchstens zwei oder drei kurzen Sätzen und ohne Listen, Emojis oder Sonderzeichen, weil "
        "deine Antwort vorgelesen wird. Korrigiere nie ausdrücklich. Wenn die Person einen Fehler "
        "macht, benutze die richtige Form einfach selbst in deiner Antwort."
    )


@dataclass(frozen=True)
class Pipeline:
    """The services that a voice session calls, its system prompt, and the file it logs turns to.

    Attributes:
        system_prompt: The first message of every conversation.
        transcribe: Turns one turn of 16 kHz PCM into text.
        reply: Streams the tutor's reply to a conversation, piece by piece.
        speak: Streams the PCM audio of one sentence at `tts.SAMPLE_RATE`.
        warm_up: Opens the connections a turn needs, and returns once the tutor model answers.
        log_path: The JSON lines file that gets one VoiceTurn per turn.
    """

    transcribe: Callable[[AsyncIterable[bytes]], Coroutine[Any, Any, str]]
    reply: Callable[[list[dict[str, str]], str], AsyncIterator[str]]
    speak: Callable[[str], AsyncIterator[bytes]]
    warm_up: Callable[[], Coroutine[Any, Any, None]]
    system_prompt: str
    log_path: Path


class VoiceTurn(msgspec.Struct, kw_only=True):
    """One voice turn as the orchestrator saw it. The orchestrator writes it as one JSON line.

    Timestamps are Unix seconds on the orchestrator's clock. A timestamp is None when the turn
    failed before that point.

    Attributes:
        turn_id: The turn ID the client chose. vLLM's turn log has the same ID.
        t_speech_start: The client started the turn.
        t_speech_end: The client ended the turn. The person stopped talking here.
        t_stt_final: The final transcript arrived.
        t_llm_request: The request to the tutor model left.
        t_first_token: The first piece of the spoken reply arrived.
        t_first_sentence: The first whole sentence was ready for TTS.
        t_tts_request: The TTS request for the first sentence left.
        t_first_audio: The first audio of the reply was sent to the client.
        t_last_audio: The last audio of the reply was sent to the client.
        error: The stage and the error type that ended the turn early, or None.
    """

    turn_id: str
    t_speech_start: float
    t_speech_end: float | None = None
    t_stt_final: float | None = None
    t_llm_request: float | None = None
    t_first_token: float | None = None
    t_first_sentence: float | None = None
    t_tts_request: float | None = None
    t_first_audio: float | None = None
    t_last_audio: float | None = None
    error: str | None = None


@dataclass
class _OpenTurn:
    record: VoiceTurn
    frames: asyncio.Queue[bytes]
    transcript: asyncio.Task[str]


class Session:
    """One person's conversation over one WebSocket. Turns run one after another.

    The client sends `Start`, then binary 16 kHz PCM frames, then `Stop`. The session answers
    with the transcript, then the audio of each sentence followed by its text, then `TurnEnd`.
    """

    def __init__(self, ws: WebSocket, pipeline: Pipeline) -> None:
        self.ws = ws
        self.pipeline = pipeline
        self.history = [{"role": "system", "content": pipeline.system_prompt}]
        self.warming: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        await self.ws.accept()
        try:
            await self.pipeline.warm_up()
        except Exception as exc:
            log.warning("warm-up failed", exc_info=True)
            await self.ws.send_text(encode(Failed(stage="warm_up", kind=type(exc).__name__)))
            await self.ws.close(code=1013)
            return
        await self.ws.send_text(encode(Ready(sample_rate=tts.SAMPLE_RATE)))
        turn: _OpenTurn | None = None
        try:
            while (message := await self.ws.receive())["type"] != "websocket.disconnect":
                if message.get("bytes") is not None:
                    if turn is not None:
                        turn.frames.put_nowait(message["bytes"])
                    continue
                try:
                    control = msgspec.json.decode(message["text"], type=ClientMessage)
                except msgspec.ValidationError as exc:
                    await self.ws.send_text(encode(Failed(stage="protocol", kind=str(exc))))
                    continue
                match control:
                    case Start(turn_id=turn_id):
                        turn = self._start(turn_id)
                    case Stop() if turn is not None:
                        turn.frames.shutdown()
                        turn.record.t_speech_end = time.time()
                        await self._answer(turn)
                        turn = None
        finally:
            if turn is not None:
                turn.transcript.cancel()

    def _start(self, turn_id: str) -> _OpenTurn:
        # Reopening idle connections while the person talks keeps the handshakes off the turn.
        warming = asyncio.create_task(self._warm_up_quietly())
        self.warming.add(warming)
        warming.add_done_callback(self.warming.discard)
        frames: asyncio.Queue[bytes] = asyncio.Queue()
        return _OpenTurn(
            record=VoiceTurn(turn_id=turn_id, t_speech_start=time.time()),
            frames=frames,
            transcript=asyncio.create_task(self.pipeline.transcribe(drain(frames))),
        )

    async def _warm_up_quietly(self) -> None:
        try:
            await self.pipeline.warm_up()
        except Exception:
            log.warning("warm-up before a turn failed", exc_info=True)

    async def _answer(self, turn: _OpenTurn) -> None:
        record = turn.record
        sentences: asyncio.Queue[str] = asyncio.Queue()
        thinking: asyncio.Task[None] | None = None
        stage = "stt"
        try:
            text = await turn.transcript
            record.t_stt_final = time.time()
            self.history.append({"role": "user", "content": text})
            thinking = asyncio.create_task(self._think(record, sentences))
            await self.ws.send_text(encode(Transcript(text=text)))
            stage = "tts"
            spoken = await self._speak(record, sentences)
            stage = "llm"
            await thinking
            self.history.append({"role": "assistant", "content": " ".join(spoken)})
        except Exception as exc:
            record.error = f"{stage}: {type(exc).__name__}"
            log.warning("turn %s failed at %s", record.turn_id, stage, exc_info=True)
            await self.ws.send_text(encode(Failed(stage=stage, kind=type(exc).__name__)))
        finally:
            if thinking is not None:
                thinking.cancel()
        await self.ws.send_text(encode(TurnEnd(turn_id=record.turn_id)))
        jsonl.append(self.pipeline.log_path, record)

    async def _think(self, record: VoiceTurn, sentences: asyncio.Queue[str]) -> None:
        async def timed() -> AsyncIterator[str]:
            async for piece in self.pipeline.reply(list(self.history), record.turn_id):
                if record.t_first_token is None:
                    record.t_first_token = time.time()
                yield piece

        record.t_llm_request = time.time()
        try:
            async for sentence in split_sentences(timed()):
                if record.t_first_sentence is None:
                    record.t_first_sentence = time.time()
                sentences.put_nowait(sentence)
        finally:
            sentences.shutdown()

    async def _speak(self, record: VoiceTurn, sentences: asyncio.Queue[str]) -> list[str]:
        spoken = []
        async for sentence in drain(sentences):
            if record.t_tts_request is None:
                record.t_tts_request = time.time()
            async for audio in self.pipeline.speak(sentence):
                await self.ws.send_bytes(audio)
                record.t_last_audio = time.time()
                if record.t_first_audio is None:
                    record.t_first_audio = record.t_last_audio
            await self.ws.send_text(encode(Sentence(text=sentence)))
            spoken.append(sentence)
        return spoken


def create_app(pipeline: Pipeline, lifespan: Lifespan[Starlette] | None = None) -> Starlette:
    """Return the ASGI app that serves the talk WebSocket."""

    async def talk(ws: WebSocket) -> None:
        await Session(ws, pipeline).run()

    return Starlette(routes=[WebSocketRoute(TALK_PATH, talk)], lifespan=lifespan)


def create_app_from_settings(
    settings: VoiceSettings, llm_headers: dict[str, str] | None = None
) -> Starlette:
    """Return the app wired to the tutor model and Inworld that the settings name.

    Every request to the tutor model carries `llm_headers`. Pass the Modal proxy auth headers only
    with a tutor URL that Modal itself returned.
    """
    tutor = httpx2.AsyncClient(
        base_url=settings.llm_base_url,
        headers=llm_headers,
        timeout=httpx2.Timeout(30, read=60),
        limits=_KEEP_ALIVE,
    )
    inworld = httpx2.AsyncClient(
        base_url=settings.inworld_base_url,
        headers=settings.inworld_headers(),
        timeout=30,
        limits=_KEEP_ALIVE,
    )
    stt_url = httpx2.URL(settings.inworld_base_url).copy_with(scheme="wss", path=stt.STT_PATH)

    async def warm_up() -> None:
        await asyncio.gather(llm.wait_until_ready(tutor), tts.open_connection(inworld))

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        await tutor.aclose()
        await inworld.aclose()

    pipeline = Pipeline(
        transcribe=partial(stt.transcribe, str(stt_url), settings.inworld_headers()),
        reply=partial(llm.stream_reply, tutor, settings.llm_model),
        speak=partial(tts.synthesize, inworld, voice=settings.tts_voice, model=settings.tts_model),
        warm_up=warm_up,
        system_prompt=tutor_prompt(settings.tts_voice),
        log_path=settings.voice_log,
    )
    return create_app(pipeline, lifespan=lifespan)


def create_app_from_env() -> Starlette:
    """Return the app for a local run, configured from the environment and `.env`."""
    return create_app_from_settings(VoiceSettings())
