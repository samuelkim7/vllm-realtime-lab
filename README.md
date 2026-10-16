# vllm-realtime-lab

**vLLM latency tuning for real-time workloads, hunting down milliseconds with caching, quantization and speculative decoding.**

Each experiment changes one thing in a vLLM deployment and measures what it bought: thinking off, prefix caching, FP8 and INT4 weights, an FP8 KV cache, speculative decoding at batch size one, engine settings, and a concurrency sweep under a latency SLO. Every result in `results/` comes with the command that produced it.

The first workload is a voice agent: my own German B1 speaking tutor, one concurrent user. A voice turn cares about the time to the first complete sentence, not only TTFT, because the speech model starts on whole sentences.

These are personal experiments, not affiliated with the vLLM project.

## Layout

- `serving/`: Modal apps that start vLLM with a given configuration.
- `workloads/voice_agent/`: the tutor prompt, the replay conversations and the quality set.
- `src/vllm_realtime_lab/`: the replay client and the per-turn metrics.
- `results/`: one folder per experiment, with the raw numbers and the command.
- `docs/plan.md`: the experiments in order, and what each one measures.

## Setup

```bash
uv sync
cp .env.example .env
modal token new
```

`.env` holds the API keys and stays out of git. Modal keeps its own token in `~/.modal.toml`.
