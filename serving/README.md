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
