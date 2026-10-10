# exl3-server

Native **EXL3** (exllamav3) inference for Apple Silicon: a small C++ engine on top of
Apple's [MLX](https://github.com/ml-explore/mlx), with a CLI and an OpenAI-compatible
HTTP API. It loads EXL3-quantized Llama-family checkpoints directly from their
safetensors files — for example `MiniCPM5-2B-EXL3` — and reconstructs the weights to
fp16 at load time.

Scope is deliberately narrow: **CLI + HTTP server**. No conversion, no training, no
CUDA/ROCm backend, no plugin system.

## Features

- EXL3 `.trellis` / `.suh` / `.svh` (and `.mul1` / `.mcg` codebook markers) loading,
  reconstructed to fp16 (`mul1`, `mcg` and `3inst` procedural codebooks, K = 1..8).
- Llama-family forward pass on MLX: RMSNorm, GQA attention, RoPE, SwiGLU, fp16 KV cache.
- BPE tokenizer built from `tokenizer.json` (HF byte-level BPE, the model's two-stage
  pre-tokenizer, added tokens) plus the model's ChatML chat template, including the
  optional `thinking` block.
- Streaming, prefix-aware generation: the KV cache keeps the previous conversation and
  reuses the longest common token prefix on the next request.
- OpenAI-compatible API: `/v1/models`, `/v1/chat/completions`, `/v1/completions`
  (SSE streaming and non-streaming), `/health`.
- Request-level batching for GLM-5-Next (`--serve --batch N`): up to N concurrent
  requests advance through one fused decode step (rows = sequences), each with its
  own KDA/MLA/indexer state; a batch of one is bit-identical to serial decoding.
- Samplers: greedy, temperature, top-k, top-p, min-p, repetition / presence /
  frequency penalties, seeded RNG, stop strings.

## Requirements

- macOS on Apple Silicon (MLX uses Metal; unified memory means weights are read in place).
- `brew install mlx cmake` — built and tested against MLX 0.32.
- A C++20 toolchain (Xcode command line tools).

## Build

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The binary lands in `build/bin/exl3-server`. MLX is located via `/opt/homebrew/opt/mlx`
or `brew --prefix mlx`; if it lives elsewhere, pass `-DMLX_ROOT=/path/to/mlx`.

## Model directory

A standard HF layout: `config.json`, `tokenizer.json`, `chat_template.jinja` (optional)
and one or more safetensors shards. Quantized linears are stored as

```
<key>.trellis  int16  [in/16, out/16, 16*K]   packed 16x16 tiles, K bits per weight
<key>.suh      float16 [in]                   input-side scale vector
<key>.svh      float16 [out]                  output-side scale vector
<key>.mul1     int32  scalar marker           selects the mul1 procedural codebook
<key>.mcg      int32  scalar marker           selects the mcg procedural codebook
```

Unquantized tensors (`model.embed_tokens.weight`, norms, or a fully unquantized
checkpoint such as `openbmb/MiniCPM5-2B`) are also supported; `nn.Linear` weights are
transposed to the engine's `(in, out)` layout on load.

## GLM-5.3-Flash (GLM-5-Next)

Native support, verified against the real 121 GB checkpoint
(`MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3`, 13 shards):

| piece | what the engine implements |
|---|---|
| loader | 147039 of 151554 tensors, zero-copy mmap, ~10 s load (vision 1007 and the MTP layer 45 are skipped by design) |
| layers | 45 = 34 KDA (gated delta rule with a short conv) + 11 DeepSeek-style sparse attention (MLA + lightning indexer) |
| MoE | 288 experts, top-8, sigmoid routing with `e_score_correction_bias` used only for selection, shared expert, `swiglu_limit` 10, `routed_scaling_factor` 2.5; layers 0-2 dense |
| hyper-connections | mhc with 4 residual streams, per-sublayer mixing, 20-iteration Sinkhorn normalisation |
| tokenizer | vocabulary 154880 with the pre-tokenizer chain interpreted from `tokenizer.json` and the checkpoint's chat template reproduced byte-for-byte |

`--model DIR` auto-detects the architecture; the CLI and the HTTP server both
work with it. Notes on the layout discovered from the real weights: q/k/v are
fused into one `self_attn.qkv_proj`, the KDA tensors live under `self_attn.*`,
`conv1d` has no bias, plain weights are stored (in, out) after loading, the
forget gate uses the config's `gate_lower_bound = -5`, and bitrates vary per
tensor (e.g. `kv_a_proj_with_mqa` at K=5 while most are K=4).

### Request-level batching

`--serve --batch N` (default 8 for GLM-5-Next, `0` = one at a time) runs up to N
concurrent sequences through one fused decode step: the rows of the step are the
sequences, so a step's weight reads serve the whole batch while each slot keeps
its own state (KDA conv window + delta-rule matrix, the compressed MLA latent
cache, the packed DSA indexer cache). A joining request is prefilled through the
ordinary single-sequence path and its state copied in, so a batch of one is
bit-identical to serial decoding, and four concurrent greedy runs produce exactly
the tokens four serial runs do (`tests/glm5_batch_test.cpp`). Every request is
built and evaluated on the scheduler's own thread - MLX streams are thread-local,
so serving from an HTTP worker thread cannot work - and `--batch 0` is that
thread's serial mode.

The routed experts are the part batching cannot amortize by itself: 8 rows over
top-8 of 288 experts select ~58 distinct experts, so those ~2.4 GB/token are read
either way, while the ~3.3 GB of dense weight reads per token are shared. The
step's B*K (row, expert) pairs are therefore executed in *expert order* (sorted by
expert id, in chunks of 14/29 pairs per launch) so pairs sharing an expert run
back to back and its tile is read from cache: bit-identical per pair (only the
order changes) and worth 0-28% depending on the cell, but the mixed-prompt rate
still sits on a line set by the number of distinct experts - 19.3 tok/s aggregate
at B=8 with 4 distinct prompts against 29.7 tok/s with one. Measured gradients,
tables and the honest reading: `docs/perf.md`, "expert-order pair execution" and
"request-level batching".

## CLI

```sh
# chat-formatted single prompt
exl3-server --model /models/MiniCPM5-2B-EXL3-Quantized -p "Explain photosynthesis in two sentences."

# raw text completion
exl3-server --model /models/MiniCPM5-2B-EXL3-Quantized --raw -p "The capital of France is"

# interactive chat
exl3-server --model /models/MiniCPM5-2B-EXL3-Quantized -i

# OpenAI-compatible server
exl3-server --model /models/MiniCPM5-2B-EXL3-Quantized --serve --host 127.0.0.1 --port 8080
```

| Flag | Meaning |
| --- | --- |
| `--model DIR` | model directory (required) |
| `-p, --prompt TEXT` | generate from a prompt (repeatable) |
| `-i, --interactive` | chat REPL (`/exit`, `/reset`, `/think`, `/nothink`, `/system TEXT`, `/raw TEXT`) |
| `--serve`, `--host`, `--port` | HTTP server (default `127.0.0.1:8080`) |
| `--batch N` | serve up to N concurrent requests through one fused decode step (GLM-5-Next only; default 8, `0` = one request at a time through the serial path on the same scheduler thread) |
| `--raw` | treat `--prompt` as a raw completion (no chat template) |
| `--system TEXT` | system prompt for chat modes |
| `--think` / `--no-think` | start the reply inside / outside a `<think>` block (default: empty think block) |
| `--show-reasoning` | print reasoning deltas dimmed |
| `--max-tokens N` | generation limit (default 512) |
| `--temp F`, `--top-p F`, `--top-k N`, `--min-p F` | sampling (`--temp 0` = greedy, the default) |
| `--repetition-penalty F`, `--presence-penalty F`, `--frequency-penalty F` | penalties |
| `--seed N`, `--stop TEXT`, `--ignore-eos` | RNG seed, stop strings, EOS handling |
| `--threads N` | threads used for weight reconstruction (default: all cores) |
| `--model-name NAME`, `--quiet` | reported model id, silence per-request stats |

## API

```sh
# non-streaming chat
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Name the largest planet in the solar system."}],
  "max_tokens": 32, "temperature": 0
}'

# streaming chat
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Count from 1 to 5."}],
  "max_tokens": 64, "stream": true
}'

# legacy completions
curl http://127.0.0.1:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "2 + 2 =", "max_tokens": 8}'
```

Endpoints: `GET /v1/models`, `GET /v1/models/{id}`, `POST /v1/chat/completions`,
`POST /v1/completions`, `GET /health`.

Supported request fields: `messages`, `prompt` (string or array), `model`, `stream`,
`max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `stop`
(string or array), `n` (1..8, non-streaming), `ignore_eos`, and
`chat_template_kwargs.enable_thinking` (or top-level `enable_thinking`).
Unknown/unsupported OpenAI fields (`logprobs`, `logit_bias`, `tools`, …) are ignored;
`usage` is reported in every response, and in the final SSE chunk when `stream` is set.

### Thinking mode

The model's template opens the assistant turn with `<|im_start|>assistant\n`. Thinking
is **off by default** (the template's empty `<think>\n\n</think>\n\n` is rendered), which
gives direct answers. With `enable_thinking: true` the block is left open; text produced
inside `<think>…</think>` is streamed in the `reasoning_content` field and everything
after it in `content`.

## How it works

1. **Trellis decode.** Each 16x16 tile is a tail-biting ring of `256*K` bits; states are
   stored as 16-bit words with the two halves of each 32-bit word swapped, and the state
   for stream position `p` is the 16-bit window ending at ring bit `(p+1)*K`. Positions
   map to tile elements through exllamav3's tensor-core permutation, and the state is
   turned into an fp16 weight by the procedural codebook
   (`mul1`: `(1024 + bytesum(state * 0x83DCD12D)) * fp16(0x1eee) + fp16(0xc931)`, one
   rounding).
2. **Rotation.** The decoded tile is in the quantizer's rotated basis:
   `W = diag(svh) · H128 · Ŵ · H128 / 128 · diag(suh)`, with the Sylvester Hadamard matrix
   applied independently to every 128-row and 128-column block. The engine keeps a
   128-row block in fp32 across both rotations and rounds to fp16 once.
3. **Forward pass.** MLX `matmul` / `fast::rms_norm` / `fast::rope` (HF-style half-split
   convention) / `fast::scaled_dot_product_attention` with an additive causal mask; the
   KV cache is fp16 and lives per layer, so multi-turn chats only prefill the new suffix.
4. **Sampling** happens on the host: penalties, temperature, top-k/min-p/top-p, then
   inverse-CDF sampling over a seeded xorshift RNG (greedy = argmax).
5. **Detokenization** is byte-level and incremental: only complete UTF-8 sequences are
   emitted, and stop strings are matched with a holdback window so they never leak.

## Performance

Measured on an M4 Max (128 GB), MiniCPM5-2B-EXL3 (4 bpw body, 6 bpw head), greedy,
single stream, 96-token generations:

| Path | Decode | Notes |
| --- | ---: | --- |
| fp16 weights materialized (`--fp16-weights`) | ~50 tok/s | 4.5 GB of weights read per token |
| **trellis-resident, in-kernel decode (default)** | **~105 tok/s** | **1.19 GB resident, ~1.2 GB read per token** |

The default path never materializes fp16 weights: the packed trellis stays in memory
and a custom Metal GEMV decodes values inside the kernel, with both 128-point
Hadamard rotations (+ `suh`/`svh` scaling) fused into single kernels rather than
expressed as MLX operators. That keeps the per-token graph at ~4 operators per
linear instead of ~11, which matters because decode is as much CPU-graph-bound as
bandwidth-bound at these sizes.

Absolute limits on this machine, for reference: the model reads ~1.2 GB of weights
per token and the measured streaming bandwidth is ~425 GB/s, so a perfect kernel
could not exceed ~350 tok/s without speculative decoding.

Tuning knobs (environment, mainly for experiments): `EXL3_QMV_NT` (output tiles per
threadgroup), `EXL3_QMV_SG` (simdgroups per threadgroup), `EXL3_QMV_MB` (rows per
threadgroup block), `EXL3_QMV_SPLITS` (split-K factor).

## Verification

The implementation was checked against independent references, not just by "it runs":

- fp16 rounding matches round-to-nearest-even hardware/numpy behaviour on 85k cases
  (normals, subnormals, ties, overflow).
- All three procedural codebooks are bit-identical to a numpy reference for all 65536
  states.
- Trellis decoding is **bit-exact** against a numpy reference for K=4 and K=6 tensors,
  and the decoding rules were cross-checked against upstream exllamav3 kernels and the
  independent Rust implementation in `0xZKnw/mlxl3`.
- Reconstructed weights vs the original `openbmb/MiniCPM5-2B` BF16 checkpoint:
  cosine 0.997, relative Frobenius error 7.5% for 4-bit tensors and 2.2% for the 6-bit
  `lm_head` — i.e. the checkpoint's own quantization error, consistent across layers.
- One-shot prefill vs token-by-token decoding: cosine 0.999997 with identical argmax.
- Measured on an M4 Max (128 GB) with MiniCPM5-2B-EXL3: ~1 s to load 1.19 GB of weights,
  ~105 tokens/s decode (see Performance), ~2k tokens/s prefill.

## Limitations

- **GLM-5.3-Flash is exact only against our own reference-derived oracles and
  the prefill/decode invariant, not against an independent implementation**: the
  source bf16 weights and a reference runtime are not available locally, so a
  wrong norm placement or layer-plan detail would still produce fluent, factual
  text. Rotate on that when the checkpoint or a reference runtime becomes
  available.
- Speculation for GLM-5.3-Flash uses the checkpoint's MTP block (layer 45,
  `--draft-mtp N`, loaded from the two public implementations of this
  architecture since the HF reference ignores it): 78-100% proposal acceptance
  and ~30-50% faster decode for 32-token generations, but the gain shrinks with
  context length (fresh process, 128 tokens: 3.9 vs 3.8 tok/s) because the
  per-row work grows. Greedy output is identical to non-speculative greedy for
  32-token generations; over longer runs the trajectories can diverge after
  ~40 tokens because partially accepted rounds rewind f16 recurrent state, so
  the trajectories are numerically distinct even when every component is
  correct.
- GLM-5.3-Flash is trellis-resident only (no f16 fallback: materialising 100 GB
  of f16 beside the checkpoint is not feasible) and its speculative decoding has
  not been exercised; `model.visual.*` and the MTP layer 45 are skipped.
- GLM-5.3-Flash decode runs at **8.6 tok/s (116 ms/step) serial and 11.4 tok/s
  effective with `--draft-mtp 4` (88 ms/step)** on an M4 Max, measured
  back-to-back on the real checkpoint (best of 3 repeats of a 16-token greedy
  run; the box drifts ~30% between sessions and a warm page cache is worth
  another 20%, so only same-session numbers are comparable - the previous
  pass's 6.5/8.6 tok/s are the same binary on a colder cache, and the
  interleaved before/after of the last pass is inside this spread).
  The step's time is spread over the components, not concentrated in the MoE:
  measured uninstrumented (one component per eval cycle, minus the 0.12 ms a
  sync costs - `tests/moe_bench.cpp`) the 42 MoE layers are ~36 ms/step, the
  34 KDA layers ~25 ms, the 11 MLA+indexer layers ~16 ms, the 90
  hyper-connection prepare sites ~8 ms and the lm_head 2.7 ms; the per-layer
  host sync the previous pass blamed for the MoE (the selected expert ids are
  read back with `mx::eval` to build the grouped GEMV's buffer list) is ~0.10 ms
  per layer, i.e. ~4 ms/token, and routing the ids on the device instead is not
  available on this runtime (a Metal buffer's GPU address is not the host
  address MLX hands out, and a 288-expert arena is 98 GB of the checkpoint's own
  bytes). What binds is the trellis GEMV's own rate - 118-170 GB/s on this
  model's shapes, issue-bound rather than launch- or page-cache-bound, against
  ~6 GB of weights per token. See `docs/perf.md` for the measured attribution
  and the arithmetic behind the distance to the 14.3 ms/step (70 tok/s) target.

- EXL3 + plain Llama-family architectures only (no MoE/MLA/custom architectures, no
  `rope_scaling`).
- Weights are materialized as fp16 in unified memory (~4.5 GB for a 2.5B-parameter model),
  so this trades memory for simplicity versus decoding the trellis on the fly.
- One generation runs at a time; concurrent requests queue.
- The chat template's tool-call XML rendering (`tool_calls` on assistant messages) is not
  implemented; `role: "tool"` text messages are rendered.
- Apple Silicon only.

## Third party

Vendored single headers: [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT) and
[nlohmann/json](https://github.com/nlohmann/json) (MIT). The EXL3 format and the
procedural codebooks come from [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3).
