from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

import modal

from vllm_realtime_lab.server.middleware import LOG_ENV, TurnLogMiddleware

VLLM_VERSION = "0.28.0"
# devel ships the nvcc FlashInfer needs at startup. Its CUDA must match vLLM's torch.
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
MODEL_REVISION = "95a723d08a9490559dae23d0cff1d9466213d989"
GPU = "L40S"
PORT = 8000
TURNS = Path("/turns")

app = modal.App("german-tutor")
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.13")
    .pip_install(f"vllm=={VLLM_VERSION}")
    # The FlashInfer sampler JIT-compiles on first use; the default sampler needs no compile.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("vllm_realtime_lab")
)
hf_cache = modal.Volume.from_name("vrl-hf-cache", create_if_missing=True)
turns = modal.Volume.from_name("vrl-turns", create_if_missing=True)
compile_cache = modal.Volume.from_name("vrl-compile-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": compile_cache,
        str(TURNS): turns,
    },
    # One container caps what a leaked proxy token can spend.
    max_containers=1,
    scaledown_window=5 * 60,
    timeout=24 * 60 * 60,
)
@modal.concurrent(max_inputs=128)
# Modal rejects requests without Modal-Key and Modal-Secret before a container starts.
@modal.web_server(port=PORT, startup_timeout=20 * 60, requires_proxy_auth=True)
def serve() -> None:
    """Start vLLM with the turn log middleware and write the server's metadata next to the log."""
    started = time.strftime("%Y%m%d-%H%M%S")
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--revision",
        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--middleware",
        f"{TurnLogMiddleware.__module__}.{TurnLogMiddleware.__name__}",
        "--max-model-len",
        "32768",
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
    ]
    meta = {"vllm_version": version("vllm"), "gpu": read_gpu_name(), "cmd": cmd}
    (TURNS / f"server-{started}.json").write_text(json.dumps(meta, indent=2))

    log_path = TURNS / f"turns-{started}.jsonl"
    env = os.environ | {LOG_ENV: str(log_path), "PYTHONPATH": "/root"}
    server = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    threading.Thread(
        target=copy_log, args=(server, TURNS / f"vllm-{started}.log"), daemon=True
    ).start()
    threading.Thread(target=exit_with_vllm, args=(server,), daemon=True).start()


def read_gpu_name() -> str:
    """Return the GPU name and memory as nvidia-smi reports them."""
    query = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    return subprocess.run(query, capture_output=True, text=True, check=True).stdout.strip()


def copy_log(server: subprocess.Popen[bytes], path: Path) -> None:
    """Print vLLM's output and append it to a file on the Volume, which keeps the whole log."""
    assert server.stdout is not None
    with path.open("ab") as f:
        for line in server.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            # A failed Volume write must not stop draining the pipe, or vLLM blocks on its log.
            with contextlib.suppress(OSError):
                f.write(line)
                f.flush()


def exit_with_vllm(server: subprocess.Popen[bytes]) -> None:
    """Stop the container as soon as vLLM exits, so a failed start does not hold a GPU."""
    code = server.wait()
    os._exit(code or 1)
