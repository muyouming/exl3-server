# mlx-serve — how a native Zig/MLX server accelerates GLM-5.3-Flash-class decode

**Biggest single acceleration mechanism.** The dominant cost of these MoE decoders is the routed-expert
path, and mlx-serve attacks it with a *fused, per-architecture MoE decode kernel* that does the whole layer in
two dispatches — gate|up|shared-expert projections + SwiGLU in one, down + shared + routing-weighted reduce in
the next — taking the router's expert **indices and scores as device tensors** instead of reading them back to
the host. Everything else in the project (continuous batching, four flavours of speculative decoding, KV-cache
policies, prefix caching, dispatch-count discipline) is real and measured, but the fused MoE kernels are what
move the read of the layer's weights and the launch count at the same time, and they are the reason the same
project can serve our exact model through the EXL3 trellis path (`sushi_exl3.moe`) at 275–313 GB/s of effective
GEMV bandwidth where our own kernel sits at 118–170 GB/s (`docs/perf.md`, "Honest distance to 14.3 ms/step").

---

## 1. Identity, and how to get it

| Field | Value |
|---|---|
| Resolved name | **`ddalcu/mlx-serve`** — *"Native LLM inference server for Apple Silicon. OpenAI + Anthropic API compatible. No Python. Zig backend, Swift frontend macOS app"* |
| URL | https://github.com/ddalcu/mlx-serve (site: https://mlxserve.com/) |
| Commit read | `1b00c65a3cdf4a60a6e64709e150b75d4ecd8b02`, 2026-10-10 03:27 -0400 ("fix(cuda): media load preflight bills weights to RAM, the working set to the GPU") |
| Version | v26.10.2 "Another One" — **UNRELEASED - DEV** (`CHANGELOG.md:3`); last release v26.10.1, pinned 26.10.1 in `Formula/mlx-serve.rb:4` |
| Language | Zig (server + all kernels), Swift (macOS app); MLX and llama.cpp are pinned submodules |
| Companion module read | `ddalcu/sushi` branch `exl3-module`, commit `44315136ec59b70d44aa0bd82380b8d5cd686e30` (2026-09-28, "Make the EXL3 engine a module a host build can import"), CHANGELOG v1.0.5. This is the EXL3 trellis engine (`sushi_exl3`) that mlx-serve imports; it began as a fork of mlx-serve and was detached 2026-09-17 (sushi `CHANGELOG.md:3`) |
| Clone | `git clone --recurse-submodules https://github.com/ddalcu/mlx-serve` (submodules: `ml-explore/mlx`, `ml-explore/mlx-c`, `antirez/ds4`, `davidtai/mlx-stream`, `ddalcu/sushi`); `git clone -b exl3-module https://github.com/ddalcu/sushi` |

**Name ambiguity (resolved).** `mlx-serve` is used by at least three projects:

1. **`ddalcu/mlx-serve`** — 1824★, Zig, 506+ commits, own Metal kernels in-tree, native menu-bar app. **This is
   the one the task means**: it is the only candidate that is simultaneously "MLX-based LLM server",
   "OpenAI-compatible inference stack", kernel-level, and it *directly serves our model* —
   `CHANGELOG.md:15` "**Sushi's GLM-5.3-Flash and MiMo-V2.6-Flash packs load.** `GLM-5.3-Flash-Sushi-2.4bpw` and
   `MiMo-V2.6-Flash-Sushi-2.3bpw` serve text and tool calls with their EXL3 experts" and
   `CHANGELOG.md:16` "**GLM-5.3-Flash.** TensorFold's MLX packs of GLM-5-Next run with text, thinking … and the
   pack's own multi-token-prediction layer drafts ahead of the model". `docs/reference.md:582` documents
   `glm5_next` layer by layer against mlx-vlm's layout.
2. **`raspoli/mlx-serve`** (PyPI `mlx-serve` 0.2.0) — Python, MIT, 9 commits; it *is* an OpenAI-compatible MLX
   server, but it is a supervisor: each model runs in an `mlx_lm.server` / `mlx_vlm.server` subprocess
   (PyPI description: "text → `mlx_lm.server` subprocess"). No kernels, no scheduling, nothing to transfer.
   Rejected as the target, kept here only so the name is not confused later.
3. Other near-neighbours searched and rejected: `micah-klein/mlx-batch-server`, `mlx-openai-server`,
   `cyans/vllm-mlx`, `mlx-lm`'s own `mlx_lm.server`. None is called `mlx-serve`.

All citations below are to the two pinned checkouts above. Files marked `docs/gotchas/*` are the project's own
engineering post-mortems (the most honest measurements I found anywhere).

---

## 2. Architecture in one short section

* **One server process, one HTTP surface per protocol** (OpenAI chat/completions + Responses, Anthropic Messages,
  Ollama, plus media endpoints), a 7 MB Zig binary with no Python at runtime (`README.md:52`).
* **One inference thread.** `src/scheduler.zig:1-30`: the scheduler owns the only thread that calls into MLX
  ("mlx 0.31.2 made GPU streams thread-local so the model weights are bound to whichever thread first calls
  `useCurrentThreadStream`"); HTTP threads parse, tokenise, `submit()`, then drain a per-slot ring buffer.
  Continuous batching lives in that one thread (`runDecodeTick`).
* **MLX through C.** All ops go through `mlx-c` (`src/mlx.zig` is a hand-written `extern "c"` binding surface:
  `mlx_set_wired_limit` at `src/mlx.zig:483`, `mlx_quantize` at `:372`). Model code is per-architecture Zig in
  `src/transformer.zig` (74 k lines) plus `src/glm5_next.zig`, `src/kv_quant.zig`, `src/prefix_cache.zig`, …
* **Kernels are in-tree Metal sources, embedded with `@embedFile` and JIT-compiled by MLX**
  (`mlx_fast_metal_kernel_new(... @embedFile("kernels/glm5_moe_gate_up.metal"), @embedFile("kernels/glm5_qmv_header.metal"), ...)`,
  `src/glm5_next.zig`'s `moeKernel`). `src/kernels/` holds 19 `.metal` files (GLM-5 MoE, NAX attention tiles,
  gather-QMM for NAX, KDA decode, QSA/MoE prefill mixes). The EXL3 trellis kernels do **not** live here: they are
  the `sushi_exl3` module (`sushi/src/exl3/expert_exl3_kernels.zig`, 8383 lines) imported through an `mlx_host`
  indirection so another MLX host can serve EXL3 packs through the same code (`sushi/src/exl3/root.zig:1-7`).
* **Embedded foreign engines** for GGUF (llama.cpp) and DeepSeek-V4 (ds4). Irrelevant to us.

---

## 3. Acceleration mechanisms

Ordered roughly by decode impact. "Measured effect" is quoted with the machine: mlx-serve's bench box is an
**M4 Max 128 GB** (`benchmarks.md:5` "Apple M4 Max 128 GB ONLY"), sushi's is an **M5 Max 128 GB** — the same
peak class as our 546 GB/s M4 Max, but not the same silicon, so cross-project GB/s comparisons are indicative,
not head-to-head.

| Mechanism | What it does | Evidence | Measured effect |
|---|---|---|---|
| **Fused batched MoE decode for GLM-5.3 (2 dispatches/layer)** | `moeDecode`: one kernel computes gate+up for all `rows*(topk+1)` slots *and* the clamped SwiGLU into the intermediate; the second kernel computes down for every routed expert plus the shared expert, folds the routing score in, and writes the row sum. Expert ids and scores arrive as **device arrays**; `mlx_argsort_axis` sorts them on device so no host round trip is needed. | `src/glm5_next.zig:2084` (`MOE_DECODE_MAX_ROWS: c_int = 16`), `:2086` (`moeDecode`), `:2112-2122` (`many = rows > 1` → `mlx_argsort_axis` on device, "run the routed slots expert-sorted and slot-major, so a shared expert's rows are read from cache"), `:2121`/`:2129` (the two launches); `src/kernels/glm5_moe_gate_up.metal:1-8`, `src/kernels/glm5_moe_down.metal:1-8` | Per-component M5 Max ubench: whole decode forward **18.99 → 17.2 ms**; marginals KDA 5.3 ms (34 layers, "near the 8-bit weights' bandwidth floor"), **MoE 7.2 ms against a ~5.3 ms floor**, DSA 1.95 ms, mHC 1.25 ms (`docs/reference.md:598`) |
| **Slot-major / expert-sorted ordering so a shared expert's row block is read back to back** | `SM` (slot-major) puts *routes* fastest in the grid and the kernel indexes the expert-sorted `order` array, so several rows sharing one expert read its row block consecutively (L1 hits); the same trick is what the grouped EXL3 GEMVs use. | `src/kernels/glm5_moe_gate_up.metal:9-12` ("SM (slot-major): route slots vary fastest, and `order` lists them expert-sorted, so the slots of an expert that several rows share read its row block back to back (cache hits)"); `src/kernels/glm5_moe_down.metal:9-10` | Design rationale, no isolated number; slot-sharing frequency measured elsewhere: **20 / 28 / 31 % of routed slots repeat an expert at 2 / 3 / 4 rows, 41–47 % at 6–8** (sushi `docs/perf-baselines.md`, M2 Max decode attribution) |
| **The router can be replayed *inside* the gate/up kernel (compile-time `SELECT`)** | The kernel carries a `glm_router_topk<NE,TOPK>` arm that replays the selection from the biased sigmoid scores and publishes `sel_indices`/`sel_scores` from tile 0 / slot 0 for the down kernel to consume — i.e. a device-side route table, no select dispatch and no host readback. **Compiled out (`#define SELECT 0`) in the current GLM path**, which feeds host-computed `indices`/`order`; it is a shipped-worked design in the file. | `src/kernels/glm5_moe_gate_up.metal:5` (`#define SELECT 0`), `:72-99` (the arm: "One token: this simdgroup replays the router's selection on the biased sigmoid scores (no separate select dispatch)", "slot 0 / tile 0 publishes the routes and routing weights for the down kernel") | Not measured in-tree (off by default) — [UNVERIFIED] as a speed claim; the surrounding oMLX 0.7.0 port is credited in the file header |
| **EXL3 trellis decode chain: 3–4 dispatches per MoE layer, Hadamard + SwiGLU folded into the GEMVs** | `moeSwigluFused` = (1) pair-prepare, (2) split-K pair GEMV (gate and up in one kernel, f32 inner planes), (3) *fused mid+down* GEMV where the down-input Hadamard `suh` and the SwiGLU product are computed in the same kernel that reads the down trellis, (4) an f32 finish reduce. | sushi `docs/engine-exl3-experts.md:78-83`; `src/exl3/expert_exl3_kernels.zig:2654` (`moeSwigluFused`), `:2680` (`preparedMidOn`/group enable), `:1875-2050` (`DOWN_FUSED_SOURCE`, the SwiGLU + `suhd` Hadamard folded in), `:7933` (`PAIR_GEMV_GROUPED_SOURCE`); dispatch from `src/exl3/root.zig:18` (`moe`) | M5 Max Flash-Next K4: **pair GEMV ~313 GB/s, down fused ~275 GB/s**, 8-bit trunk qmv ~600 GB/s, against a measured 538 GB/s streaming-read peak; decode forward 18.35 ms (~56 tok/s live) (`sushi/docs/perf-baselines.md:44-45`, `:27`) |
| **Lane-funnel weight windows (64-bit index math) and 2 output tiles per threadgroup** | Below K4 every rate reads its weights through a "funnel": one lane's 8–16 codewords are extracted from a `ulong merged = words[i0]<<32 \| words[i1]` with a compile-time shift table derived from `n`; the lane funnel carries `OTPT = 2` output tiles per threadgroup, sharing the index math, the activation loads and the prepare. | `src/exl3/expert_exl3_kernels.zig:39-57` (`funnelReads(n) = n < 64`, `gemvLayout`, `FUNNEL`/`OTPT` template args), `:1720-1735` + `:1940-1998` (the merged 64-bit read and the 8-shift decode), `:516-519` (per-quad funnel, "above n50 a quad's codewords pass 32 bits, and the funnel reads 64 bits from three words") | "The decode GEMVs are bound by **fixed per-tile work, not DRAM** (64-bit index math, two word loads and a 64-bit shift, four input reads, loop control)"; a rate that misses the funnel decodes **~40 % slower per GEMV** with no other symptom; `OTPT=2` shares the loop skeleton (`sushi/docs/engine-exl3-experts.md:84-90`) |
| **Expert-grouped decode GEMVs: decode an expert's weights once per pair of slots** | `PAIR_GEMV_GROUPED_SOURCE` / `DOWN_PREPARED_GROUPED_SOURCE`: among an expert's slots, each even-ranked slot leads itself and the next, decodes each weight once and feeds both members, so every row's bytes equal a one-row decode tick's. Found with a 64-bit ballot mask (`DECODE_GROUP_MAX_SLOTS = 64`), `DECODE_GROUP_MEMBERS = 2` (4 spill the accumulators). | `src/exl3/expert_exl3_kernels.zig:61-75`, `:2680`; `sushi/docs/engine-exl3-experts.md:91-92` | Grouping is geometry-keyed; on MiMo verify rows it is on, and "A decode GEMV slot is bound by its own FMA and input path, not the weight decode or DRAM, so deduplicating shared experts recovers only **4–7 %** of the expert kernels at 3–4 rows". On Flash-Next it is declined (rows share too few experts). Also measured as *not* paying at the Qwen geometry: MiMo's grouped gate/up GEMV was **−0.8 %/−1.3 %** decode (sushi `docs/perf-baselines.md`, M2 Max decode attribution) |
| **Non-NAX prefill GEMM built on `simdgroup_matrix` (`D = Wᵀ Xᵀ`)** | Off NAX (M1–M4 — *our* hardware class) the EXL3 prefill GEMM uses an 8x8 `simdgroup_matrix` body laid out so each lane's eight-weight slot group lands directly in its A fragments, with a **compile-time** block count (a data-dependent bound over the fragment arrays spilled them). | `sushi/docs/engine-exl3-experts.md:70-77`; `src/exl3/expert_exl3_kernels.zig:1484-1502` (the `.simdmat` arm) | Against the scalar body, one process, paired: **3.2x–5.4x** at 17–64-token chunks, **17.8x–22.2x** at 2048 tokens (241.0 → 10.9 ms); end-to-end M2 Max prefill **4.0x/4.4x** (30.9 → 123.8–140.8 tok/s); the same lever later took Flash-Next prefill 1600 → 1850 tok/s (`sushi/docs/perf-baselines.md:256-300`, section "Flash-Next Sushi-3bpw on an M2 Max 64 GB") |
| **Continuous batching — single-threaded, one trunk forward per group** | `runDecodeTick`: `active.len == 1` takes the legacy single-slot path bit-identically (spec included); `≥ 2` partitions slots by model, runs one `forwardBatchedDecode` per model group, and caps the group by *padding waste* (one long stream would pad its short neighbours). Lazy-graph work is the point: N logits from one kernel pass. | `src/scheduler.zig:1-30`, `:7692-7700` (the legacy gate), `:7745-7790` (partition by model), `:7787-7840` (pad-waste cap: sort by `kv_len` ascending, `groupKeepCount` at `:2514`, "the tail decodes serially this tick"); GLM path: `src/transformer.zig:22272` → `:26401` (`forwardGlmBatchedDecode`) | **~1.6x total throughput at 4-way on dense models, at some per-request latency cost** (`docs/performance.md:43`) |
| **GLM-5.3-specific batched decode: 8 rows/forward, no KV padding** | Each slot's next token is a *row* of a `[1, n]` window, so `hcPre`, the joined KDA input projection, the routed experts and the head read their weights once; per-slot KDA recurrence + DSA attention run against each slot's own conv/recurrent state and latent/indexer cache. `MOE_DECODE_BATCH_MAX = 8` (the MoE decode kernel's row limit); larger groups run in 8-row chunks; **no KV padding**, so the pad-waste cap is skipped. MTP rounds stay solo (`mtpRoundsStaySolo`), so 3+ company switches the slots to plain batched ticks. | `src/transformer.zig:26385` (`MOE_DECODE_BATCH_MAX = 8`), `:26401-26450` (`forwardGlmBatchedDecode`, `bctx.batch_slots = group`), `:26302` (`glmBranchSlots`), `:26339` (`glmKdaSlots`); `docs/reference.md:588` ("8 rows per forward … no KV padding, so `groupPadsKv` skips the pad-waste cap") | No published tok/s cell for GLM-5.3-Flash batched decode; the mechanism is documented and the 4-way aggregate for the project is the ~1.6x above. [UNVERIFIED] for GLM specifically |
| **Speculative decoding, four flavours + adaptive gates** | (a) **native MTP** — the checkpoint's own head (for GLM-5.3 it is `mtp.0`, DeepSeek-NextN style: `eh_proj(cat[enorm(embed x_{p+1}), hnorm(h_p)])` through one plain pre-norm DSA+MoE block, then the trunk's lm_head); (b) **PLD** n-gram lookup over `prompt+generated`; (c) a model-shipped `drafter/` block companion; (d) Gemma-4-style cross-attention assistant drafter. Shared: adaptive prompt-time repetition gate + a runtime acceptance gate that disables speculation mid-decode below break-even. | `src/glm_mtp.zig:1-9`; `src/pld_index.zig:1-10`; `src/drafter.zig:1-15`; `docs/performance.md:9-24` | Release-over-release decode gains attributed to spec: Qwen3.6 35B-A3B **191 → 259 tok/s** (26.8.6 → 26.9.2, MTP), Qwen3.8 27B **70 → 104 tok/s** (MTP → DFlash2) (`benchmarks.md`, M4 Max); Gemma 4 E4B drafter **+52 %** (117 → 178 tok/s); sushi MTP on Flash-Next K3 47.2 → 70.2 tok/s at 3.69 tokens/step, verify = 89–92 % of a round's wall (sushi `docs/perf-baselines.md`) |
| **Prompt lookup *inside* the MTP round** | If the last 3 committed tokens + the draft seed match earlier in the prompt/output and agree for 8+ tokens, the drafts stand in for the chain — no head forward at all. A "line rule" prevents a draft that agrees to end-of-line from predicting the next line (which killed the unified-diff case). | sushi `docs/engine-mtp.md` ("Prompt lookup inside the round", ported from mlx-serve #523/#533); mlx-serve `src/pld_index.zig` | Sushi-3bpw: copying a file **+21 %**, renaming **+21 %**, bug-fix **+16 %**, `write_file` **+9 %**, long-context edits at 32k **+24 %** / 64k **+18 %**; diff/new-code/prose inside noise (sushi `docs/perf-baselines.md`, M5 Max) |
| **Draft-side cost cuts in the spec round** | Early dispatch (`mlx_async_eval` the draft chain while the CPU builds the verify graph), cross-round pre-draft, a coarse 3-bit shortlist lm_head copy with exact re-score of the top-32, deferred comp rows. | `docs/gotchas/engine-mlx.md:286`; `:4683` (coarse shortlist) | Coarse shortlist: **3.38 ms off a depth-5 round** (41.77 → 38.39); llmprobe predictable **141 → 153 tok/s** with acceptance unchanged (`docs/gotchas/engine-mlx.md:4683`) |
| **KV-cache quantization + a *fused packed-read* attention path** | Affine 4/8-bit KV (group 64) with the storage/dispatch contract isolated in `kv_quant.zig`; every read goes through `KVCache.denseView`, which hands the packed triple to a kernel or dequantizes. The fused path replaces dequant+SDPA with two `mlx_quantized_matmul`s + a softmax, reading the triples in place — GQA is folded by **reshaping Q down** to `[B, H_kv, g·T_q, D]` rather than expanding K/V. | `src/kv_quant.zig:1-20` (contract), `:168-214` (`quantAttention`, the GQA reshape rationale, "never the triples expanded up to H_q — that materializes the packed bank ×g per layer per token"), `:521`; engaged from `src/server.zig:521` (`resolveKvAttnFusedPure`, auto from an 8K prompt) | ≈ **−10 % decode** at 2–4K context; at long context the fused packed reads (auto from 8K) cut the penalty to ≈ **−18 % at 42K** (was −47 %) (`docs/performance.md:40`). Unit ubench in `src/kv_quant.zig:947-1095` |
| **Decode-only quantized copy of bf16 attention weights** | Dense bf16 attention weights get a decode-only quantized copy; prefill keeps the dense weights. | `docs/performance.md:47` | **−23 % per-token time** on models that ship bf16 attention (poolside Laguna); may change wording, so it is opt-out |
| **Prefix / prompt cache: hot RAM + SSD tier + SSD-first, with hybrid restore points** | Prompt-prefix KV matching; entries record their KV scheme so a `kv_quant=4` slot never restores an 8-bit entry; hybrid (SSM/QSA) candidates rank by *restorable checkpoint position*, not raw match; oversized candidates are trimmed to the longest restorable prefix; the hot-cache byte budget is clamped at load to what the weights leave and follows residency; an SSD tier spills and restores in chunks; spec state rides the cache. | `src/prefix_cache.zig:202-215` (the scheme filter and the crash it prevents), `:1015-1023`, `:1024-1080` (`findBestRestorableMatch`); `src/main.zig:874-899` (flags); sushi `docs/engine-prefix-cache.md` (full design + numbers) | M2 Max: prefix cache **6.8x** (4.1 s cold → 607 ms warm, 1509/1540 tokens cached). Long-session budget: unset budget 11 516 MB → turns 2–3 prefill in **0.35 s** (199.7k reused) vs 36.0/31.3 s under `--prefix-cache-mem 2GB`; a 150k-token disk restore **170–177 ms** warm (sushi `docs/perf-baselines.md`, `docs/engine-prefix-cache.md`) |
| **Long-context read trimming for sliding-window models** | Every attention read is trimmed to the model's window — prompt chunks and speculative steps included. | `docs/performance.md:22-30` | Muse-Glimmer 30B 4-bit: decode 16k **24.7 → 40.2 tok/s**, 64k **8.6 → 21.0**; Laguna prompt 64k 236 → 540 tok/s |
| **Memory policy: wired limit, load preflight, and the OOM semantics that make billing mandatory** | `mlx_set_wired_limit` + a policy (`mlx.applyWiredPolicy()` at boot); the GPU ceiling is `iogpu.wired_limit_mb` (read once per process, so the ceiling cannot move under a live request) with a floor when the user raised it; load-time preflight refuses with `InsufficientMemory` → 503 naming the compared number and the flag that would admit; **Metal OOM is uncatchable and the working-set edge returns ZEROS before it aborts** (all-zero logits from healthy inputs = a memory symptom); a freed Metal buffer unwires asynchronously (~0.5 s per 50 GB), so an unload/eviction waits for the unwire or the next preflight reads the bytes as taken; `mlx_clear_cache` at chunk boundaries because MLX parks freed buffers in a cache. | `src/mlx.zig:480-486`, `:736`; `src/main.zig:1552`; `src/server.zig:3436-3480`; `src/generate.zig:3673`; sushi `docs/engine-memory-admission.md:14`, `:34` | No speed number; this is correctness-of-accounting. Sushi quotes its own envelope tables (Sushi-3bpw requirement 51.35 GiB, load/warmup peak 49.91 GiB) |
| **Dispatch-count discipline: price a dispatch before fusing** | A probe injects N output-identical elementwise kernels into the MoE layer to price a dispatch *in the dependency chain*; null results are recorded so nobody retries them. | `docs/gotchas/engine-mlx.md:869-890` | Slope **~0.067 ms per dispatch-per-layer ≈ 1.5–1.7 µs per GPU dispatch** at 40 layers (`:886`); `MLX_SERVE_DISPATCH_PROBE` 0/4/8 = 11.07/11.61/12.13 ms GPU, ~2.8 µs per *chained* op (`:5502`); a bit-identical fused attention gate worth "~0.2 ms" by that slope measured **zero** (the chain is off the critical path); fused add+RMSNorm (on the critical path) also measured **zero**. `MLX_MAX_OPS_PER_BUFFER=200` read 10.82 vs 10.45 ms — **worse** |
| **Small-kernel / occupancy discipline** | Big projections run *at* the bandwidth floor while ~150 small dense GEMVs per step are launch-latency-bound; the fix is fewer, denser kernels, not better small kernels. A register-resident top-k, a split-K small GEMV and a vectorized affine-8 reader were all built, measured and reverted. "Rows per simdgroup is an occupancy decision." | `docs/gotchas/engine-mlx.md:5523-5535` (per-kernel profile via Metal System Trace), `:5554` (rows/simdgroup 4 → 1 = +4 %), `:4570-4585` | 996 kernels / 9.99 ms step on M5 Ultra; `affine_qmv_fast` 44 %, big GEMVs at the floor (lm_head 636 MB in 594 µs, out-proj 870 GB/s); waste = 150 small GEMVs at 6.8 µs each = **1.65 ms against a ~0.3 ms byte floor**; rows-per-simdgroup 4 → 1: 110.5 → 114.8 tok/s |
| **`mlx_compile` for elementwise chains** (the mechanism we already listed as lever #1) | Compiled functions replace chains of tiny ops; also used for GEGLU/GELU. | `src/main.zig:1556-1557` (`xfm.compileGelu()`), `docs/gotchas/engine-mlx.md:971` | The project's post-mortem is a *warning*, not a win: extending `mlx_compile` to the SiLU arm was rejected for intermediate-rounding reasons, and per-phase `mx::eval` for profiling "destroys pipelining: **47.4 ms/token against a real 13.1**" (`docs/gotchas/engine-mlx.md:855`) |

---

## 4. Quantization support and layout

**Weight formats the engine serves** (this matters because our engine is trellis-only):

1. **EXL3 trellis (the format we use).** Per MoE layer/projection, expert `e` on axis 0:

   | tensor | dtype | shape |
   |---|---|---|
   | `<prefix>.layers.{L}.mlp.switch_mlp.{P}.trellis` | U16 | `[E, in/16, out/16, n]` |
   | `…{P}.suh` | F16 | `[E, in]` |
   | `…{P}.svh` | F16 | `[E, out]` |

   `n` is **halfwords per 256-weight tile**, `K = n/16`; weight `t`'s codeword is the 16-bit window ending at
   `((t+1)*n)>>4`, so the fresh bits follow from `n` and are never stored. Even `n` in `[32,64]` is admitted
   (36 = K2.25, 48 = K3, 64 = K4); `k` in `config.json` may be fractional and names the *widest* rate a layer
   packs (a narrower layer is over-billed, never refused). `in`/`out` are tile-grid multiples of 16 and the
   Hadamard block is `H128`, so both are multiples of 128. The codeword **window** (8..16, absent = 16) and the
   **codebook** (`mul1`, `mcg`) are pack fields; the same bitstream decodes to different weights at each window,
   so an unreadable one is refused by name, never defaulted. The converter folds the per-expert global codebook
   scale into `suh`. Evidence: `sushi/docs/pack-format.md:15-56`, `:74-79`; `sushi/src/exl3/expert_exl3.zig:9-33`
   (`MCG_MULT`/`MUL1_MULT`, `Codebook`, `Window`), `:81-140` (`Rate`, `kFromPackedDim`), `:320` (`prepareInput`),
   `:364` (`finishOutput`); `sushi/src/exl3/root.zig:57-95` (`Spec`, `rateFromConfigK`, `windowFromConfig`,
   `parseExpertQuant`, refusal cases in its tests). Shard `__metadata__` stamps (`format`,`k`,`codebook`,`window`)
   are checked against `config.json` before bytes are uploaded (`sushi/docs/pack-format.md:120-140`).
   Top-k is capped at 32 by the decode reduce bank (`admitTopK`, `Exl3TopKExceedsReduceBank`).
2. **MLX affine** 2/3/4/5/6/8-bit, group 32/64/128, solved from *packed shapes* rather than a literal width at
   any call site (`sushi/docs/pack-format.md`, "Loader rules the engine applies to every pack"); MXFP4
   (`uint8` e8m0 scales, no biases) has its own gather-QMM arm (`src/gather_qmm_nax.zig:8`, `:173`); FP8 trunks and
   stored-affine trunk linears (`o_proj`, `lm_head`, `embed_tokens`) are served as stored, with **no load-time
   quantization** (`sushi/docs/pack-format.md`, "Stored-affine trunk linears").
3. **KV cache**: affine 4/8-bit, group 64, with the fused packed-read attention path above; schemes are
   extensible (one `Scheme` variant + one quantize/dequantize pair + two switch arms).
4. **Per-weight quant resolution** (`computeQuantParams`) and a "NEVER_QUANTIZE" class for gather-read tables
   (`src/dflash.zig:587`, `:1007-1013`) — codebooks read by a gather must stay bf16.
5. Decode-only quantized copy of bf16 *attention weights* (`docs/performance.md:47`).

For our engine the important structural point is that mlx-serve's EXL3 layout is **format-identical to ours**
(`[E, in/16, out/16, n]` with `n = 16*K` and the same `((t+1)*n)>>4` window rule — compare our
`src/trellis_gemm.h`, "one uint8 element per byte, shaped `(in/16, out/16, 16*K)`", 8*K u32 ring words per 16×16
tile). At K=4 (`n=64`) their engine takes the **N64 packed branch**, not the lane funnel
(`funnelReads(n) = n < 64`, `src/exl3/expert_exl3_kernels.zig:39-57`; the `N == 64u` branch merges
`words[lane]` with `words[(lane+31)&31]` at `:1937-1955`). So the funnel/OTPT=2 machinery applies to us only if
we ship a sub-4-bit pack; the *rest* of the chain (fused prepare, split-K partials, fused SwiGLU+Hadamard) is
rate-independent.

---

## 5. Transferable to exl3-server, ranked

Our current position, from `docs/perf.md`: 42 MoE layers with a host loop (`mx::eval` + ~10 tiny ops per selected
expert), 4000–5000 kernels per token, ~6 GB of trellis weights read per token at **118–170 GB/s** on an M4 Max
(~546 GB/s peak), reads alone 35–45 ms, per-layer host sync ~0.10 ms/layer (4.2 ms/token, ~3 %), grouped expert
GEMV capped at 29 members by Metal's 31-buffer limit, `trellis_linear` = 3 launches × ~1450 linears/token, and
the two "obvious" ways to drop the per-layer `mx::eval` (a pointer table, an expert arena) measured dead.
mlx-serve is the closest published engine to that exact position, so the ranking below is ordered by
(expected ms saved per token) ÷ (effort).

1. **Device-indexed expert launch — pass `inds`/`scores` into one kernel per projection class instead of
   gathering on the host.** *(Highest value, and it dissolves two dead ends at once.)*
   *Mechanism:* the routed banks stay stacked `[E, …]` on axis 0; the kernel takes the flattened `inds`
   (uint32, device) + `scores` and indexes `expert` per slot; `rows × (topk+1)` slots are one launch whose grid
   is `(tiles, slots)` with no member list and no 31-buffer ceiling. mlx-serve even sorts the ids on device
   (`mlx_argsort_axis`) so nothing round-trips.
   *Evidence:* `src/glm5_next.zig:2086-2135`; kernels `glm5_moe_gate_up.metal` / `glm5_moe_down.metal`;
   `sushi/src/exl3/root.zig:18` (`moe(s, x, bank, inds, scores, dec, verify_rows)`) and
   `expert_exl3_kernels.zig:2654` (`moeSwigluFused`) — all take ids as arrays.
   *Expected effect here:* removes 42 `mx::eval`s/token (the measured ~0.10 ms/layer, 4.2 ms ≈ 3 %), removes the
   per-expert `take`/`scatter_add` glue and the ~150 dispatches of the ungrouped form, and removes our
   "≤ 29 members, `B ≤ 14`" batch ceiling so request-level batching is not structurally limited.
   *Cost/risk:* the kernels must handle `rows ≤ 16` and `topk` as template args with the router output fed as a
   device buffer; our `G5MoePrep` scale stacks become a single `(E, in|out)` f16 stack read by the kernel
   (mlx-serve reads per-row scale/bias pointers with `row0*G`). Bit-identity is preserved by construction if each
   member keeps its own scale vector and k-order — the same contract our `tests/moe_test.cpp` (f) already hashes.
   Medium effort (kernel rewrite + parity), highest payoff.
2. **Fold the router (sigmoid top-k) into the gate/up kernel and publish routes on device.** Their `SELECT` arm
   does exactly this: replay `glm_router_topk` on the biased scores in the same dispatch, and have slot 0 /
   tile 0 write `sel_indices`/`sel_scores` for the down kernel (`glm5_moe_gate_up.metal:3`, `:76-99`).
   *Expected effect:* the router is 12 % of one MoE layer here (`docs/perf.md`: 0.10–0.14 of 0.83–0.95 ms, "~25
   small ops + the ids eval"), and it is the only host-visible dependency left after item 1. Whether it pays is
   decided by the same test mlx-serve used: it must shorten the *dependency chain* (`docs/gotchas/engine-mlx.md:867-903`).
   *Cost/risk:* it is compiled out in their shipped GLM path, and our router is group-limited top-k with a
   correction bias, so the in-kernel replay has to reproduce our `moe_topk_route` exactly — this is a parity risk
   (their arm uses plain top-k). Do item 1 first, then measure whether the router dispatch is on the critical
   path at all.
3. **Fuse the two Hadamards and the SwiGLU into the trellis GEMV ("prepare"), and share the prepare across
   output tiles.** Their down GEMV computes the SwiGLU product and the `suh` Hadamard in registers inside the
   same kernel that reads the down trellis (`DOWN_FUSED_SOURCE`, `expert_exl3_kernels.zig:1875-2050`), and
   `preparedMidOn` makes it once per (row, expert).
   *Evidence:* sushi `docs/engine-exl3-experts.md:78-83`; our `docs/perf.md` lever #2/#3 measures the alternative
   directly: `trellis_linear` = 3 launches × ~1450 linears/token, ~20 µs per dispatch visible when chained.
   *Expected effect:* ~1450 × 2 launch removals/token plus the Hadamard passes; bounded by our own measurement
   that the whole small-projection set is ~1.7 ms/step with ~7 of every 10 µs being launch overhead — so the win
   is of the same order as item 1, not larger, but it also removes two full passes over the activation.
   *Cost/risk:* at `n=64` their OTPT/2-tile sharing does not apply, so the sharing win is smaller for us;
   correctness rests on keeping the exact accumulation order (our `glm5_exl3_decode_test.cpp` byte-alignment gate).
4. **Adopt their `simdgroup_matrix` trellis GEMM body for the non-NAX (M1–M4) prefill path.** Our target box is
   an M4 Max — i.e. exactly the "off NAX" case where their scalar prefill body was replaced by an 8×8 fragment
   body computing `D = Wᵀ Xᵀ`, with the block count **compile-time** because a data-dependent bound spilled the
   fragment arrays (2.6x slower).
   *Evidence:* `sushi/docs/engine-exl3-experts.md:70-77`; `expert_exl3_kernels.zig:1484-1502`.
   *Expected effect:* 17.8x–22.2x over a scalar body at 2048-row windows, 4.0x/4.4x end-to-end prefill on an
   M2 Max, and 1600 → 1850 tok/s on Flash-Next from the follow-ups. Our own row-blocking change made the 26-row
   prefill ~2x cheaper, so this is complementary, not redundant.
   *Cost/risk:* highest implementation cost of the kernel items (a real GEMM), but the payoff is prefill, which
   we have not attacked at all. Do it after decode.
5. **Batched-decode design rules to copy** (we are implementing request-level batching now):
   (a) partition the batch by model, one trunk forward per group (`scheduler.zig:7745-7790`);
   (b) keep the `len == 1` path bit-identical to the serial path and gate on it (`:7692-7700`) — this is what
   lets them keep spec decoding and batching coherent, and it gives a free equality test at N=1 via
   `MLX_SERVE_FORCE_BATCHED`;
   (c) cap the group by *padding waste* when the arch pads KV, and skip the cap entirely for archetypes that
   don't pad (`docs/reference.md:588`: GLM "no KV padding, so `groupPadsKv` skips the pad-waste cap");
   (d) fix the batch row limit to the kernel's row limit (8 for their GLM MoE, 16 for the decode chain) and chunk
   larger groups;
   (e) do not mix speculative rounds with a crowded batch — 3+ company degrades to plain batched ticks that
   capture the hidden so the head stays resumable.
   *Expected effect:* their published aggregate is ~1.6x at 4-way; our own `moe_layer_rows` already measured the
   per-launch bookkeeping saving, and the remaining structural limits (29 members, B ≤ 14) are what items 1–2 fix.
6. **Memory policy.** `mlx_set_wired_limit` at boot with a read-once `iogpu.wired_limit_mb` ceiling, a load-time
   preflight that refuses by name (503) instead of thrashing, an explicit wait for asynchronous unwire after
   free, and `mlx_clear_cache` at boundaries. Our `docs/perf.md` already lists `mx::set_wired_limit` as lever #4;
   the transferable *details* are the ones with teeth: the ceiling must not move under a live request
   (`src/server.zig:3441-3445`), a re-load right after an unload must wait for the unwire or the preflight sees
   the freed bytes as taken (sushi `docs/engine-memory-admission.md`), and Metal at the working-set edge returns
   **zeros**, not an exception, so a bill must never go down except where the bytes are gone.
7. **Prefix cache (RAM + SSD) and its scheme bookkeeping.** Not a kernel; the single best measured win in the
   project for agent traffic (6.8x on warm turns; 170 ms restores of a 150k-token prefix from SSD). We serve
   single-sequence today with no prefix reuse. If we ever serve agents, the transferable *rules* are: record the
   KV scheme (bits+group, not just the scheme enum) on every entry, filter lookups on it
   (`src/prefix_cache.zig:1015-1023` explains the SDPA crash a scheme-only filter caused), clamp the cache budget
   to what the weights leave, and treat RAM-cache and disk-tier persistence as two decisions.
8. **MTP items that specifically apply to our GLM-5.3 head.** Keep verify rows inside the decode arm (rows ≤ 16)
   so a verify row costs a row's *own* experts and nothing else; group verify rows that share an expert so its
   weights are decoded once per pair; keep a measured per-(model,width,KV) round-cost table and an adaptive-depth
   controller rather than a fixed depth; and prompt-lookup inside the round is the biggest measured win on
   echo-shaped traffic (+16…+24 % on file copies/renames/edits) with an explicit line rule to avoid the
   diff-shaped failure.
9. **Measurement discipline worth importing wholesale** (all of it is corroborated independently in our own
   `docs/perf.md`): price a dispatch with an output-identical injected kernel *inside* the chain, not by timing a
   component twice; never judge a decode kernel from a per-phase-eval profiler (their 47.4 vs 13.1 ms is our
   274 vs 133–162 ms); run A/Bs interleaved in one process (their separate runs drift 15 %, our box drifts ~30 %);
   record null results so they are not retried (their list: fused attention gate, fused add+RMSNorm, split-K
   small GEMV, vectorized affine-8 reader, 4 rows/simdgroup, `MLX_MAX_OPS_PER_BUFFER`, bigger command buffers).

### What does NOT transfer

* **NAX/tensor-unit kernels.** `src/kernels/nax_tiles.metal`, `gather_qmm_nax*.metal`, `nax_attention.metal`,
  `qmv_nax2.zig` and the NAX prefill/attention arms are M5-generation. Our box is an M4 Max (no NAX), and their
  own rule is that a lever's *default* belongs to the engine that measured it. Read them for the shape discipline
  and nothing else.
* **The affine/FP8/MXFP4 kernel families** (`rowqmv.zig`, `moe_fp4.zig`, `moe_affine4.zig`,
  `simd_qmm.zig`, `dec_attn.zig`) work on MLX affine weight layouts, not trellis — a different weight format, so
  they are layout references, not code to port.
* **The Gemma-4 assistant drafter** (4 cross-attention layers that read the *target's* K/V, so no weights are
  duplicated) is target-model-specific; and their DFlash2 note is a warning for us: on this MoE "a verify row
  costs (about +8 ms for the second, +3.4 each after, one row 17 ms)", so spec depth must be priced, not assumed
  (`docs/reference.md:590`).
* **The whole non-LLM surface** (image/video/music/TTS/3D, FLUX/LTX/Hunyuan3D/ACE-Step, `lib/fqms` mesh
  simplification, `lib/xatlas`, the Ollama/Anthropic API surfaces, LAN model sharing, the Swift app) — no
  inference-acceleration content for us.
* **GGUF/llama.cpp and ds4 engines** — different runtime, different kernels, and their continuous-batching work
  there (`--llama-cache-entries`) is a wrapper around llama.cpp's own batching.
* **Python-level designs**: `raspoli/mlx-serve`'s subprocess-per-model isolation, and anything that only exists to
  work around MLX's Python ergonomics.
* **Their KV-quant fused attention in its exact form**: it targets standard `[B,H,T,D]` K/V with GQA. Our GLM-5.3
  attention is absorbed MLA over a 512-element latent plus a separate indexer cache; the *idea* (quantize the
  cached latent 8-bit and read the packed triples directly) transfers, the kernel does not.
* **`--decode-attn-quant`-style lossy decks**: it changes output wording; mlx-serve ships it default-on, we
  should not copy that default without a KLD gate.

---

## 6. Open questions / what I could not verify

* **No published decode tok/s for GLM-5.3-Flash on mlx-serve.** The changelog documents the pack loading and the
  MTP drafting, and `docs/reference.md:598` gives a per-component ubench (18.99 → 17.2 ms/forward, MoE 7.2 ms),
  but no llmprobe cell for this model appears in `benchmarks.md` or the benchmark directories
  (`docs/benchmarks/` covers Qwen3.8 only). So my "biggest mechanism" claim is a mechanism ranking, not a
  measured model-level number. Related gap: for the EXL3 pack of GLM-5.3 nobody publishes which `n` the layers
  actually pack (the `2.4bpw` label is a pack name, not the `k` field, and `k` must satisfy `16k` even and in
  `[32,64]`). [UNVERIFIED]
* **The `SELECT` router-replay arm is compiled out in the shipped GLM path.** I read the kernel source and the
  launch's input list (`src/glm5_next.zig:2121-2135` passes `indices`/`order`, not `sel_*`), so I can say the arm
  exists and what it would do; I cannot claim it is faster. [UNVERIFIED]
* **Batched GLM-5.3 decode numbers.** The design is documented (`docs/reference.md:588`) and the code exists
  (`forwardGlmBatchedDecode`), but I found no measured GLM batch cell; the ~1.6x is for "dense models" at 4-way
  (`docs/performance.md:43`). [UNVERIFIED for this arch]
* **Cross-machine comparability.** Their 275–313 GB/s EXL3 GEMV and 538 GB/s streaming-read peak are M5 Max;
  ours are M4 Max. The mechanism conclusions hold; the magnitudes are not a head-to-head.
* **Upstream MLX version pin.** The engine's docs reference "mlx 0.31.2" (`src/scheduler.zig:3`) and sushi's
  roofline table names mlx 0.32.2; I did not check out `lib/mlx-src` to confirm what `1b00c65` pins (submodule
  SHAs are not in a `--depth 1` clone).
* **I did not run anything.** Every number above is quoted from the projects' own docs/CHANGELOGs or read from
  their sources; no benchmark was executed (the checkpoint rule and the single-process rule forbid it), and the
  M5-vs-M4 gap means our port would have to re-measure each item.
