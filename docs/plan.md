# Plan

The experiments run in order. Each one ends with a measurement, and the results stay usable wherever the work stops.

## Baseline (P0)

vLLM serves Qwen3.6-35B-A3B in BF16 on one H200, with prefix caching switched off by `--no-enable-prefix-caching`. vLLM 0.28.0 turns prefix caching on by default for this model. Thinking is off.

vLLM's multi-turn benchmark (`benchmarks/multi_turn`) replays 24 synthetic conversations of 12 to 18 turns, with replies of 80 to 120 tokens. The benchmark and the server run in one Modal container, so no network sits between them. The middleware logs the server-side timestamps of every turn, including the moment the first sentence ended.

Each run reports TTFT p50 and p90, time to first sentence, TPOT and throughput at concurrency 1, 8, 32 and 64. It also records the KV cache size that vLLM reports at startup, and the highest concurrency that keeps p90 TTFT under one second.

## The latency ladder (P1)

Each step keeps the settings before it and adds one more.

| Step | Setting | Expected change |
|---|---|---|
| 1 | Prefix caching back on, vLLM's default (`mamba_cache_mode=align`) | Lower TTFT on later turns, because the history is already cached |
| 2 | FP8 weights | Lower TPOT, because each decode step reads half the bytes |
| 3 | FP8 KV cache | More KV capacity, so TTFT holds up at higher concurrency |
| 4a | MTP speculative decoding | Lower TPOT at low concurrency |
| 4b | n-gram speculative decoding | The same, with a draft that costs nothing to run |

Speculative decoding should help at concurrency 1 and fade as batches grow. The concurrency where it stops helping is one of the results. Negative results get the same space as gains.

## The voice loop

A spoken German tutor. The Mac only records and plays audio over one WebSocket. A CPU function on Modal runs each turn: Inworld STT, the tutor model on vLLM, a sentence splitter, and Inworld TTS, which speaks each sentence as soon as it ends. The tutor runs the best setting from the ladder. The orchestrator times every part of a turn, and the Mac times the whole turn from the end of speech to the first audio.

## kvstat on a real workload (P2)

[kvstat](https://github.com/samuelkim7/kvstat) records the KV events of the prefix-caching run and shows the prefix hits grow turn by turn.

## Later

- P3. Serve an open TTS model that generates audio tokens, and check which LLM techniques carry over to it.
- P4. Take one hot operation from a profile that is not a matrix multiply, and rewrite it in Triton.
- P5. Distil the tutor into a small model. Use it as the tutor, and as a speculative decoding draft.

## Scope

One GPU and one model. A practice session has one user. The results make no claim about serving at scale, and every write-up says so in its first paragraph.
