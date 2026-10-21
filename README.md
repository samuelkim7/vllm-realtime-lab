# vllm-realtime-lab

**vLLM latency tuning for real-time workloads, hunting down milliseconds with caching, quantization and speculative decoding.**

Each experiment changes one setting of a vLLM deployment and measures what it bought. The settings are prefix caching, FP8 weights, an FP8 KV cache, and speculative decoding with the model's own MTP head or with n-grams. Every setting runs at concurrency 1, 8, 32 and 64, so the numbers show where batching starts to cost latency.

The workload is my own German speaking tutor. Each turn sends the whole conversation so far and gets a short reply, so the prompt is many times longer than the answer. A spoken reply also waits for its first complete sentence, because speech starts on whole sentences. The lab records the time to first sentence next to TTFT.

The model is Qwen3.6-35B-A3B, a mixture of experts with 3B active parameters. 30 of its 40 layers use linear attention. Those layers keep a recurrent state instead of KV blocks, so prefix caching has to store that state as well.

These are personal experiments, not affiliated with the vLLM project.

## Layout

- `serving/`: the tutor model on Modal, behind Modal proxy auth. `serving/README.md` has the commands.
- `src/vllm_realtime_lab/engines/`: every field name the lab reads from a vLLM response.
- `src/vllm_realtime_lab/server/`: the middleware that runs inside vLLM and logs one line of timestamps per turn.
- `src/vllm_realtime_lab/speech/`: the rule for where a sentence ends.
- `tests/`: unit tests, and integration tests that write files.
- `docs/plan.md`: the experiments in order, and what each one measures.

## Setup

```bash
uv sync
cp .env.example .env
modal token new
```

`.env` holds the API keys and stays out of git. Modal keeps its own token in `~/.modal.toml`. Requests to the tutor need a Modal proxy auth token. Create one in the Modal dashboard and put it in `.env` as `MODAL_PROXY_KEY` and `MODAL_PROXY_SECRET`.

```bash
uv run pytest
```
