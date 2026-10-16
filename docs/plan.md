# Plan

The experiments run in order. Each phase ends with one measurement, and the results stay usable wherever the work stops.

## P0: measurement harness and baseline

- Connect a self-served vLLM endpoint as the LLM of a voice agent. The agent handles speech-to-text, turn detection and text-to-speech.
- Record four timestamps per turn on the server: request received, first token, first sentence boundary, last token. Record the full turn time on the client.
- Replay 32 tutor conversations, written by a large model, as text at concurrency 1, 8 and 32.
- Build a quality set of 60 learner sentences with errors and their correct fixes.

Output: a baseline table (TTFT p50/p90, first-sentence time, tokens per turn, correction accuracy) and one profiler trace of a single turn.

## P1: the LLM latency ladder

| # | Lever | Measure |
|---|---|---|
| 1 | Thinking off, `max_tokens`, a sentence-length instruction | First-sentence time |
| 2 | Prefix caching on the system prompt and the growing conversation | TTFT by turn number, on and off |
| 3 | Weights BF16 / FP8 / AWQ INT4 × KV cache BF16 / FP8 | TTFT, ITL, KV capacity, correction accuracy |
| 4 | Speculative decoding at batch size one: n-gram, a small draft model, EAGLE | Acceptance rate, ITL, first-sentence time |
| 5 | CUDA graphs, compile settings, chunked prefill, `max_num_seqs` | Launch overhead at small batch sizes |
| 6 | Concurrency sweep 1 → 32 | Goodput under the TTFT SLO |
| 7 | MoE behavior (Qwen3 30B-A3B class, FP8, one L40S) | ITL by batch size, expert routing in the profile |

Each lever runs on top of the best setting so far, so the table reads as one ladder from the baseline to the tuned server. The tutor repeats the corrected sentence, so n-gram speculation should accept many tokens. Negative results get the same space as positive ones.

## P2: kvstat on a real workload

Use [kvstat](https://github.com/samuelkim7/kvstat) to show prefix hits growing turn by turn and the KV capacity an FP8 cache adds.

## P3: serving a speech model

Serve an open TTS model that generates audio tokens autoregressively, and decode its output with a codec. Measure time to first audio (TTFA), the real-time factor (RTF), and how many streams one GPU holds at RTF < 1. Check which LLM techniques carry over to the speech model and what the codec decoder needs instead.

## P4: one custom kernel

Pick one hot operation from the P3 profile that is not a matrix multiply. Rewrite it in Triton, check that it matches the PyTorch original, and measure how close it gets to the memory bandwidth limit.

## P5: distillation (optional)

Generate tutor conversations with a large model and train a small model on them with LoRA. Use the small model as the tutor itself and as the speculative decoding draft.

## Scope

One concurrent user. No scale claims: every post says so in its first paragraph.
