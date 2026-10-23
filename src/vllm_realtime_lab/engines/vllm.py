from __future__ import annotations

import msgspec

CHAT_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
EVENT_STREAM = b"text/event-stream"
DATA_PREFIX = b"data:"
DONE = b"[DONE]"


class Delta(msgspec.Struct):
    """New text in one chunk of a reply, or the whole reply in a non-streamed response.

    Attributes:
        content: Text the user sees or hears.
        reasoning: Thinking text. vLLM 0.28.0 sends it here, not in reasoning_content.
    """

    content: str | None = None
    reasoning: str | None = None


class Choice(msgspec.Struct):
    """A streamed chunk carries a delta. A non-streamed response carries a message."""

    delta: Delta | None = None
    message: Delta | None = None


class ChatBody(msgspec.Struct):
    """One server-sent event of a streamed chat completion, or a whole non-streamed one."""

    choices: list[Choice] = []


def decode(payload: bytes | str) -> list[Delta]:
    """Return the new text in one server-sent event payload or one non-streamed body.

    Raises:
        msgspec.DecodeError: the payload is not a chat completion.
    """
    body = msgspec.json.decode(payload, type=ChatBody)
    return [delta for choice in body.choices if (delta := choice.delta or choice.message)]
