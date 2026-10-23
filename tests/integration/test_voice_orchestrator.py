from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vllm_realtime_lab.voice.tts import SAMPLE_RATE

from .helpers import Fakes, read_turns, talk_session


def talk(ws: Any, turn_id: str, audio: bytes) -> None:
    ws.send_json({"type": "start", "turn_id": turn_id})
    ws.send_bytes(audio)
    ws.send_json({"type": "stop"})


def until_turn_end(ws: Any) -> list[dict[str, Any]]:
    """Return the JSON messages before `turn_end`, skipping audio."""
    events: list[dict[str, Any]] = []
    while True:
        message = ws.receive()
        if message.get("text") is None:
            continue
        event = json.loads(message["text"])
        if event["type"] == "turn_end":
            return events
        events.append(event)


def test_answers_a_turn_sentence_by_sentence(tmp_path: Path) -> None:
    fakes = Fakes(tmp_path / "voice.jsonl")
    with talk_session(fakes) as ws:
        assert ws.receive_json() == {"type": "ready", "sample_rate": SAMPLE_RATE}
        talk(ws, "t-1", b"\x00" * 20)

        assert ws.receive_json() == {"type": "transcript", "text": "Ich habe 20 Bytes gesagt."}
        assert ws.receive_bytes() == b"Gut."
        assert ws.receive_json() == {"type": "sentence", "text": "Gut."}
        assert ws.receive_bytes() == b"Weiter so."
        assert ws.receive_json() == {"type": "sentence", "text": "Weiter so."}
        assert ws.receive_json() == {"type": "turn_end", "turn_id": "t-1"}

    [turn] = read_turns(fakes.log_path)
    assert turn["turn_id"] == "t-1"
    assert turn["error"] is None
    times = [turn[key] for key in turn if key.startswith("t_")]
    assert None not in times
    assert times == sorted(times)


def test_keeps_the_conversation_across_turns(tmp_path: Path) -> None:
    fakes = Fakes(tmp_path / "voice.jsonl")
    with talk_session(fakes) as ws:
        ws.receive_json()
        for turn_id in ["t-1", "t-2"]:
            talk(ws, turn_id, b"\x00\x00")
            until_turn_end(ws)

    assert fakes.histories == [["system", "user"], ["system", "user", "assistant", "user"]]
    assert all(prompt.startswith("Du bist Carina aus Berlin") for prompt in fakes.system_prompts)


def test_reports_a_failed_stage_and_keeps_the_session(tmp_path: Path) -> None:
    fakes = Fakes(tmp_path / "voice.jsonl")
    fakes.broken_speech = True
    with talk_session(fakes) as ws:
        ws.receive_json()
        talk(ws, "t-1", b"\x00\x00")
        assert {"type": "error", "stage": "tts", "kind": "TtsError"} in until_turn_end(ws)

        fakes.broken_speech = False
        talk(ws, "t-2", b"\x00\x00")
        until_turn_end(ws)

    failed, recovered = read_turns(fakes.log_path)
    assert failed["error"] == "tts: TtsError"
    assert recovered["error"] is None


def test_answers_a_malformed_message_with_an_error(tmp_path: Path) -> None:
    fakes = Fakes(tmp_path / "voice.jsonl")
    with talk_session(fakes) as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        assert ws.receive_json()["stage"] == "protocol"

        talk(ws, "t-1", b"\x00" * 4)
        assert ws.receive_json()["text"] == "Ich habe 4 Bytes gesagt."


def test_reports_a_failed_warm_up_and_closes(tmp_path: Path) -> None:
    fakes = Fakes(tmp_path / "voice.jsonl")
    fakes.broken_warm_up = True
    with talk_session(fakes) as ws:
        assert ws.receive_json() == {"type": "error", "stage": "warm_up", "kind": "TimeoutError"}
        assert ws.receive()["type"] == "websocket.close"


def test_ignores_audio_outside_a_turn(tmp_path: Path) -> None:
    fakes = Fakes(tmp_path / "voice.jsonl")
    with talk_session(fakes) as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00" * 99)
        talk(ws, "t-1", b"\x00" * 4)
        assert ws.receive_json()["text"] == "Ich habe 4 Bytes gesagt."
