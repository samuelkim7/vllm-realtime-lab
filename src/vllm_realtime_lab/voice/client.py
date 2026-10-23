"""Talk to the German tutor from the Mac, or replay recorded turns through it.

Live mode records the microphone between two presses of Enter and plays the reply. Replay mode
sends 16 kHz mono WAV files instead, one file per turn, so every run gets the same input.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import threading
import time
import uuid
import wave
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import msgspec
import websockets
from websockets.asyncio.client import ClientConnection

from vllm_realtime_lab import jsonl
from vllm_realtime_lab.deployment import VOICE_APP, VOICE_FUNCTION
from vllm_realtime_lab.voice.protocol import (
    TALK_PATH,
    Failed,
    Ready,
    Sentence,
    ServerMessage,
    Start,
    Stop,
    Transcript,
    TurnEnd,
    encode,
)
from vllm_realtime_lab.voice.queues import drain
from vllm_realtime_lab.voice.settings import ProxyAuth
from vllm_realtime_lab.voice.stt import SAMPLE_RATE

FRAME_SECONDS = 0.1
FRAME_BYTES = int(SAMPLE_RATE * 2 * FRAME_SECONDS)


class ClientTurn(msgspec.Struct, kw_only=True):
    """One turn as the client saw it, written as one JSON line. Timestamps are Unix seconds.

    Attributes:
        turn_id: The turn ID. The orchestrator log and vLLM's turn log have the same ID.
        session: The replay session that sent the turn. Live mode is session 0.
        t_speech_end: The person stopped talking, or the replayed file ended.
        t_first_audio: The first audio of the reply arrived.
        t_first_played: The first audio of the reply reached the speaker. Live mode only.
        t_end: The orchestrator ended the turn.
        error: The stage and the error type the orchestrator reported, or None.
    """

    turn_id: str
    session: int
    t_speech_end: float | None = None
    t_first_audio: float | None = None
    t_first_played: float | None = None
    t_end: float | None = None
    error: str | None = None


class _Player:
    """Plays reply audio as it arrives. The audio callback reads from a buffer that feed fills."""

    def __init__(self, sd: Any, sample_rate: int) -> None:
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.first_played: float | None = None
        self.stream = sd.RawOutputStream(
            samplerate=sample_rate, channels=1, dtype="int16", callback=self._fill
        )

    def reset(self) -> None:
        """Drop audio left from the last reply, so the next reply's first audio is its own."""
        with self.lock:
            self.buffer.clear()
            self.first_played = None

    def feed(self, audio: bytes) -> None:
        with self.lock:
            self.buffer += audio

    def _fill(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        with self.lock:
            chunk = bytes(self.buffer[: len(outdata)])
            del self.buffer[: len(outdata)]
            if chunk and self.first_played is None:
                to_speaker = time_info.outputBufferDacTime - time_info.currentTime
                self.first_played = time.time() + to_speaker
        outdata[: len(chunk)] = chunk
        outdata[len(chunk) :] = b"\x00" * (len(outdata) - len(chunk))


async def run_turn(
    ws: ClientConnection,
    frames: AsyncIterable[bytes],
    session: int,
    player: _Player | None = None,
    show: bool = False,
) -> tuple[ClientTurn, str]:
    """Send one turn of 16 kHz PCM and receive the reply. Return the turn and its transcript."""
    turn = ClientTurn(turn_id=uuid.uuid4().hex, session=session)
    if player is not None:
        player.reset()
    await ws.send(encode(Start(turn_id=turn.turn_id)))
    async for frame in frames:
        await ws.send(frame)
    turn.t_speech_end = time.time()
    await ws.send(encode(Stop()))
    transcript = ""
    async for message in ws:
        if isinstance(message, bytes):
            if turn.t_first_audio is None:
                turn.t_first_audio = time.time()
            if player is not None:
                player.feed(message)
            continue
        match msgspec.json.decode(message, type=ServerMessage):
            case Transcript(text=text):
                transcript = text
                if show:
                    print(f"Du:    {text}")
            case Sentence(text=text) if show:
                print(f"Tutor: {text}")
            case Failed(stage=stage, kind=kind):
                turn.error = f"{stage}: {kind}"
                print(f"Error: {turn.error}")
            case TurnEnd():
                turn.t_end = time.time()
                break
    if player is not None:
        turn.t_first_played = player.first_played
    return turn, transcript


async def wait_ready(ws: ClientConnection) -> Ready:
    """Wait until the orchestrator says the tutor model answers.

    Raises:
        RuntimeError: the orchestrator failed to reach the tutor model.
    """
    match msgspec.json.decode(await ws.recv(), type=ServerMessage):
        case Ready() as ready:
            return ready
        case Failed(stage=stage, kind=kind):
            raise RuntimeError(f"the orchestrator failed at {stage}: {kind}")
        case other:
            raise RuntimeError(f"expected Ready, got {other}")


def read_turn(path: Path) -> bytes:
    """Return the PCM frames of a WAV file.

    Raises:
        ValueError: the file is not 16 kHz mono 16-bit PCM.
    """
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: replay needs 16 kHz mono 16-bit PCM")
        return w.readframes(w.getnframes())


async def replay_sessions(
    url: str,
    turns: list[bytes],
    sessions: int,
    log_path: Path,
    *,
    pace: float = FRAME_SECONDS,
    headers: dict[str, str] | None = None,
    show: bool = False,
) -> list[list[str]]:
    """Replay the turns in several sessions at once and return each session's transcripts.

    Session i starts at turn i and wraps around. Audio mixed up between sessions shows as
    transcripts in the wrong order.
    """
    rotations = [turns[i % len(turns) :] + turns[: i % len(turns)] for i in range(sessions)]
    return await asyncio.gather(
        *(
            _replay(url, headers or {}, rotation, session, log_path, pace, show)
            for session, rotation in enumerate(rotations)
        )
    )


async def _replay(
    url: str,
    headers: dict[str, str],
    turns: list[bytes],
    session: int,
    log_path: Path,
    pace: float,
    show: bool,
) -> list[str]:
    transcripts = []
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        await wait_ready(ws)
        for pcm in turns:
            turn, transcript = await run_turn(ws, _paced(pcm, pace), session, show=show)
            jsonl.append(log_path, turn)
            transcripts.append(transcript)
    return transcripts


async def _paced(pcm: bytes, pace: float) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    deadline = loop.time()
    for i in range(0, len(pcm), FRAME_BYTES):
        yield pcm[i : i + FRAME_BYTES]
        deadline += pace
        await asyncio.sleep(max(0.0, deadline - loop.time()))


async def live(url: str, headers: dict[str, str], log_path: Path) -> None:
    """Hold a spoken conversation. Enter starts a turn, and Enter again ends it."""
    # PortAudio is loaded on import, and only live mode needs it.
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    presses: asyncio.Queue[None] = asyncio.Queue()
    mic: asyncio.Queue[bytes] | None = None

    def read_enter() -> None:
        while True:
            input()
            loop.call_soon_threadsafe(presses.put_nowait, None)

    def on_mic(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if (queue := mic) is not None:
            loop.call_soon_threadsafe(_put, queue, bytes(indata))

    threading.Thread(target=read_enter, daemon=True).start()
    print("Waiting for the tutor model. A cold GPU takes a few minutes.")
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        ready = await wait_ready(ws)
        player = _Player(sd, ready.sample_rate)
        microphone = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_BYTES // 2,
            callback=on_mic,
        )
        with microphone, player.stream:
            while True:
                print("Press Enter, then speak.")
                await presses.get()
                queue: asyncio.Queue[bytes] = asyncio.Queue()
                mic = queue
                turn = asyncio.create_task(run_turn(ws, drain(queue), 0, player, True))
                print("Listening. Press Enter when you are done.")
                await presses.get()
                mic = None
                # Runs after the frames the microphone already handed to the loop.
                loop.call_soon(queue.shutdown)
                record, _ = await turn
                jsonl.append(log_path, record)


def _put(queue: asyncio.Queue[bytes], frame: bytes) -> None:
    with contextlib.suppress(asyncio.QueueShutDown):
        queue.put_nowait(frame)


def _deployed_url() -> str:
    import modal

    web_url = modal.Function.from_name(VOICE_APP, VOICE_FUNCTION).get_web_url()
    if web_url is None:
        raise SystemExit(f"{VOICE_APP} is not deployed")
    return str(httpx2.URL(web_url).copy_with(scheme="wss", path=TALK_PATH))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--url", default=f"ws://127.0.0.1:8001{TALK_PATH}")
    target.add_argument("--modal", action="store_true", help="use the deployed orchestrator")
    parser.add_argument("--replay", nargs="+", type=Path, help="WAV files, one turn each")
    parser.add_argument("--sessions", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--pace", type=float, default=FRAME_SECONDS, help="seconds per frame")
    parser.add_argument("--log", type=Path, default=Path(".dev/voice/client.jsonl"))
    args = parser.parse_args()

    # The proxy token opens every endpoint in the workspace, so it goes only to Modal's own URL.
    url, headers = (_deployed_url(), ProxyAuth().headers()) if args.modal else (args.url, {})
    args.log.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not args.replay:
            asyncio.run(live(url, headers, args.log))
            return
        turns = [read_turn(path) for path in args.replay]
        results = asyncio.run(
            replay_sessions(
                url,
                turns,
                args.sessions,
                args.log,
                pace=args.pace,
                headers=headers,
                show=args.sessions == 1,
            )
        )
    except KeyboardInterrupt:
        print("\nBis bald!")
        return
    for session, transcripts in enumerate(results):
        print(f"session {session}: {transcripts}")


if __name__ == "__main__":
    main()
