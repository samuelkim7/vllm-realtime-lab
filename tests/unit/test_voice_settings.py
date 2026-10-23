from __future__ import annotations

import pytest

from vllm_realtime_lab.voice.settings import ProxyAuth, VoiceSettings


@pytest.fixture
def voice_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VRL_LLM_MODEL", "tutor")
    monkeypatch.setenv("VRL_VOICE_LOG", "voice.jsonl")
    monkeypatch.setenv("INWORLD_API_KEY", "inworld-v1")


@pytest.mark.usefixtures("voice_env")
def test_reads_the_inworld_key_under_its_env_name() -> None:
    settings = VoiceSettings(_env_file=None)

    assert settings.llm_model == "tutor"
    assert settings.inworld_headers() == {"Authorization": "Basic inworld-v1"}
    assert "inworld-v1" not in repr(settings)


@pytest.mark.usefixtures("voice_env")
def test_arguments_win_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VRL_LLM_BASE_URL", "http://stale")

    settings = VoiceSettings(_env_file=None, llm_base_url="http://from-modal")

    assert settings.llm_base_url == "http://from-modal"


def test_reads_the_proxy_token_under_its_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_PROXY_KEY", "key-v2")
    monkeypatch.setenv("MODAL_PROXY_SECRET", "secret-v3")

    auth = ProxyAuth(_env_file=None)

    assert auth.headers() == {"Modal-Key": "key-v2", "Modal-Secret": "secret-v3"}
    assert not any(value in repr(auth) for value in ("key-v2", "secret-v3"))
