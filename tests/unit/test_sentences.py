from __future__ import annotations

import pytest

from vllm_realtime_lab.speech.sentences import first_sentence_end


@pytest.mark.parametrize(
    ("text", "final", "expected"),
    [
        ("Gut. Du hast", False, 4),
        ("Was? Nichts", False, 4),
        ("Wirklich?! Ja", False, 10),
        ("Gut.", False, None),
        ("Gut.", True, 4),
        ("Hallo", True, None),
        ("", True, None),
        ("Ich meine z. B. das Auto. Ja", False, 25),
        ("Ich meine z.B. das Auto. Ja", False, 24),
        ("Das heißt d. h. nichts. Ja", False, 23),
        ("Äpfel, Birnen usw. Dann", False, None),
        ("Am 3. Mai fahre ich. Und", False, 20),
        ("Das kostet 3.50 Euro. Ja", False, 21),
        ("Man sagt „Ich bin gegangen.“ Gut", False, 28),
        ('Er sagte "Nein!" Dann', False, 16),
        ("Hmm... ja. Gut", False, 10),
    ],
)
def test_first_sentence_end(text: str, final: bool, expected: int | None) -> None:
    assert first_sentence_end(text, final=final) == expected


def test_boundary_waits_for_the_next_chunk() -> None:
    text = "Gut."
    assert first_sentence_end(text, final=False) is None
    assert first_sentence_end(text + " Du", final=False) == 4
