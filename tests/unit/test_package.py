from __future__ import annotations

import vllm_realtime_lab


def test_version_is_set() -> None:
    assert vllm_realtime_lab.__version__ == "0.1.0"
