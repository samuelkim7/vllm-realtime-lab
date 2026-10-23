from __future__ import annotations

import asyncio
from pathlib import Path

import uvicorn

from vllm_realtime_lab.voice.client import replay_sessions
from vllm_realtime_lab.voice.orchestrator import create_app

from .helpers import Fakes, read_turns


async def replay_against_fakes(
    tmp_path: Path, turns: list[bytes], sessions: int
) -> list[list[str]]:
    app = create_app(Fakes(tmp_path / "voice.jsonl").pipeline())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        url = f"ws://127.0.0.1:{port}/talk"
        return await replay_sessions(url, turns, sessions, tmp_path / "client.jsonl", pace=0)
    finally:
        server.should_exit = True
        await serving


def test_each_session_hears_back_its_own_turns(tmp_path: Path) -> None:
    turns = [b"\x00" * 3200, b"\x00" * 6400, b"\x00" * 9600]
    transcripts = asyncio.run(replay_against_fakes(tmp_path, turns, sessions=3))

    said = [f"Ich habe {len(turn)} Bytes gesagt." for turn in turns]
    assert transcripts == [said, said[1:] + said[:1], said[2:] + said[:2]]


def test_logs_each_turn_on_the_client_clock(tmp_path: Path) -> None:
    asyncio.run(replay_against_fakes(tmp_path, [b"\x00" * 3200], sessions=2))

    turns = read_turns(tmp_path / "client.jsonl")
    assert sorted(turn["session"] for turn in turns) == [0, 1]
    for turn in turns:
        assert turn["t_speech_end"] <= turn["t_first_audio"] <= turn["t_end"]
