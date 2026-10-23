# serving

`tutor.py` runs the tutor model on Modal. Modal's proxy rejects any request without the workspace
proxy auth token, before a container starts.

| | |
|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B-FP8`, revision `95a723d08a9490559dae23d0cff1d9466213d989` |
| vLLM | 0.28.0 |
| GPU | one NVIDIA L40S, 48 GB |

The container starts vLLM with this command:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --revision 95a723d08a9490559dae23d0cff1d9466213d989 \
  --host 0.0.0.0 --port 8000 \
  --middleware vllm_realtime_lab.server.middleware.TurnLogMiddleware \
  --max-model-len 32768 --language-model-only --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Each start also writes the command, the installed vLLM version and the GPU that `nvidia-smi`
reports to `server-<start>.json` on the `vrl-turns` Volume.

```bash
uv run modal deploy serving/tutor.py
curl -H "Modal-Key: $MODAL_PROXY_KEY" -H "Modal-Secret: $MODAL_PROXY_SECRET" https://<app-url>/health
uv run modal volume get vrl-turns / .dev/turns/
```

The container sleeps after five idle minutes. The `/health` request wakes it, and the first turn
after that waits for vLLM to load. The last command copies the turn logs.

## Voice

`voice.py` runs the voice orchestrator as its own app, `german-tutor-voice`, in a CPU container.
Redeploying it leaves the tutor model loaded. The orchestrator runs each spoken turn: Inworld
speech-to-text, the tutor model, the sentence splitter, and Inworld text-to-speech. It writes one
line per turn to `voice-<start>.jsonl` on the `vrl-turns` Volume, next to the tutor's turn log.

The orchestrator needs a Modal Secret with `INWORLD_API_KEY`, `MODAL_PROXY_KEY` and
`MODAL_PROXY_SECRET`. Put only these three keys in a file, because the command uploads the whole
file.

```bash
uv run modal secret create vrl-voice --from-dotenv <file with the three keys>
uv run modal deploy serving/voice.py
uv run python -m vllm_realtime_lab.voice.client --modal
uv run python -m vllm_realtime_lab.voice.client --modal --replay turn1.wav turn2.wav
```

In live mode, Enter starts a turn and Enter again ends it. Replay mode sends 16 kHz mono WAV
files, one turn each, so every run gets the same input. `--sessions 4` replays four sessions at
once. The client sends the proxy token only to the URL that Modal returns for the deployed app.

### On the Mac

A small model on mlx-lm stands in for the tutor, so the pipeline can be built and debugged
without a GPU. Inworld still runs in the cloud.

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen3.5-2B-MLX-4bit --port 8000 \
  --chat-template-args '{"enable_thinking": false}'
VRL_LLM_MODEL=mlx-community/Qwen3.5-2B-MLX-4bit VRL_VOICE_LOG=.dev/voice/turns.jsonl \
  uv run uvicorn --factory vllm_realtime_lab.voice.orchestrator:create_app_from_env --port 8001
uv run python -m vllm_realtime_lab.voice.client --replay turn1.wav turn2.wav
```
