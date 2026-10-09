# exl3-server

C/C++ inference engine + OpenAI-compatible HTTP API for EXL3-quantized LLMs
on Apple Silicon, backed by MLX C++.

- Loads EXL3 checkpoints (turboderp/exllamav3 serialized trellis format).
- Reconstructs quantized weights to f16 at load time (procedural mul1 codebook,
  128x128 Hadamard rotation, row/column sign-scale vectors).
- Llama-family forward pass (RoPE, GQA attention, SwiGLU, RMSNorm) on MLX.
- BPE tokenizer loaded from HF `tokenizer.json`.
- CLI generation and HTTP server with SSE streaming.

## Build

```
cmake -B build && cmake --build build -j
```

Requires Homebrew `mlx` (`brew install mlx`).

## Usage

```
build/bin/exl3-server --model /path/to/model --serve --port 8080
build/bin/exl3-server --model /path/to/model --prompt "Hello"
```

## Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions` (streaming SSE + non-streaming)
- `POST /v1/completions` (streaming SSE + non-streaming)
