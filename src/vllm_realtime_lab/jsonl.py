from __future__ import annotations

from pathlib import Path

import msgspec


def append(path: Path, record: msgspec.Struct) -> None:
    """Append the record to the file as one JSON line."""
    with path.open("ab") as f:
        f.write(msgspec.json.encode(record) + b"\n")
