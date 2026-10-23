from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

_ENDS = ".?!"
_CLOSERS = "\"'“”‘’»«)]"
_OPENERS = "\"'„“‘«»(["
_ABBREVIATIONS = frozenset(
    {"usw", "bzw", "ca", "dr", "fr", "hr", "nr", "vgl", "evtl", "ggf", "inkl", "etc", "bspw"}
)


def first_sentence_end(text: str, *, final: bool) -> int | None:
    """Return the index just past the first complete sentence in text, or None.

    TTS starts only on a whole sentence, so this index is the earliest point speech can start.
    A sentence ends at `.`, `?` or `!` and any closing quotes after it. Whitespace must follow.
    At the end of text, the sentence ends only when final is true, because the next chunk can
    still continue the word. A `.` after a number or an abbreviation does not end a sentence.

    Args:
        text: The reply so far.
        final: True when the stream has ended.
    """
    for i, char in enumerate(text):
        if char not in _ENDS:
            continue
        end = i + 1
        while end < len(text) and text[end] in _CLOSERS:
            end += 1
        at_end = end == len(text)
        if at_end and not final:
            return None
        if not at_end and not text[end].isspace():
            continue
        if char == "." and _is_abbreviation(text[:i]):
            continue
        return end
    return None


async def split_sentences(deltas: AsyncIterable[str]) -> AsyncIterator[str]:
    """Yield each sentence of a streamed reply as soon as it ends, then the rest at the end.

    Each sentence ends where `first_sentence_end` puts it. TTS therefore starts at the moment
    the middleware logs as the end of the first sentence.
    """
    text = ""
    async for delta in deltas:
        text += delta
        while (end := first_sentence_end(text, final=False)) is not None:
            yield text[:end].strip()
            text = text[end:]
    if rest := text.strip():
        yield rest


def _is_abbreviation(before: str) -> bool:
    """Return True when the word before a `.` is a number or an abbreviation like `z.` or `usw`."""
    words = before.rsplit(maxsplit=1)
    if not words:
        return False
    word = words[-1].lstrip(_OPENERS)
    if word.isdigit() or "." in word:
        return True
    return (len(word) == 1 and word.isalpha()) or word.lower() in _ABBREVIATIONS
