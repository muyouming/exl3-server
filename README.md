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
- Measured on an M4 Max (128 GB) with MiniCPM5-2B-EXL3: ~5 s to load 4.5 GB of weights,
  ~44 tokens/s decode, ~2k tokens/s prefill.

## Limitations

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
