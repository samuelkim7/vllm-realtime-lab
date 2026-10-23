"""The messages the Mac client and the orchestrator exchange over the `/talk` WebSocket.

Audio travels in binary frames: 16 kHz PCM from the client during a turn, and the reply's PCM at
the `Ready` sample rate from the orchestrator. Every other message is one of the structs below.
"""

from __future__ import annotations

import msgspec

TALK_PATH = "/talk"


class Start(msgspec.Struct, tag="start"):
    """The person starts talking. Audio frames follow."""

    turn_id: str


class Stop(msgspec.Struct, tag="stop"):
    """The person stopped talking."""


class Ready(msgspec.Struct, tag="ready"):
    """The tutor model answers, so turns can start."""

    sample_rate: int


class Transcript(msgspec.Struct, tag="transcript"):
    """What the person said, as the speech-to-text model heard it."""

    text: str


class Sentence(msgspec.Struct, tag="sentence"):
    """The text of the sentence whose audio the orchestrator just sent."""

    text: str


class Failed(msgspec.Struct, tag="error"):
    """A stage failed. The orchestrator log has the details."""

    stage: str
    kind: str


class TurnEnd(msgspec.Struct, tag="turn_end"):
    """The orchestrator sent the whole reply to the turn."""

    turn_id: str


ClientMessage = Start | Stop
ServerMessage = Ready | Transcript | Sentence | Failed | TurnEnd


def encode(message: msgspec.Struct) -> str:
    """Return the message as the JSON text of one WebSocket frame."""
    return msgspec.json.encode(message).decode()
