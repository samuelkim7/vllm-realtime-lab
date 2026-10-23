from __future__ import annotations

import time

import modal
from starlette.applications import Starlette

from vllm_realtime_lab.deployment import (
    TURNS_DIR,
    TURNS_VOLUME,
    TUTOR_APP,
    TUTOR_FUNCTION,
    TUTOR_MODEL,
    VOICE_APP,
)
from vllm_realtime_lab.voice.orchestrator import create_app_from_settings
from vllm_realtime_lab.voice.settings import ProxyAuth, VoiceSettings

app = modal.App(VOICE_APP)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(groups=["voice"], extra_options="--no-default-groups")
    .add_local_python_source("vllm_realtime_lab")
)
turns = modal.Volume.from_name(TURNS_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    # Holds INWORLD_API_KEY, MODAL_PROXY_KEY and MODAL_PROXY_SECRET.
    secrets=[modal.Secret.from_name("vrl-voice")],
    volumes={str(TURNS_DIR): turns},
    max_containers=1,
    scaledown_window=5 * 60,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(requires_proxy_auth=True)
def web() -> Starlette:
    """Serve the orchestrator, pointed at the tutor model of the other app."""
    tutor_url = modal.Function.from_name(TUTOR_APP, TUTOR_FUNCTION).get_web_url()
    if tutor_url is None:
        raise RuntimeError(f"{TUTOR_APP} is not deployed")
    settings = VoiceSettings(
        llm_base_url=tutor_url,
        llm_model=TUTOR_MODEL,
        voice_log=TURNS_DIR / f"voice-{time.strftime('%Y%m%d-%H%M%S')}.jsonl",
    )
    return create_app_from_settings(settings, llm_headers=ProxyAuth().headers())
