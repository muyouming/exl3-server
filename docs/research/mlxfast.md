# mlxfast — MLX inference-speedup engine (Layr-Labs / Yukon)

**Resolved identity.** `mlxfast` is *not* a library on PyPI (PyPI `mlxfast` → HTTP 404).
It resolves to **Layr-Labs' "mlxfast" (Yukon) benchmark program**: a family of
Swift/Metal MLX inference-speedup engines, plus the `mlxfast` Yukon CLI that submits
to them. The canonical, name-bearing repository is
[`Layr-Labs/mlxfast-challenge`](https://github.com/Layr-Labs/mlxfast-challenge)
(21★, Swift), whose README is literally titled `# mlxfast — Poolside Laguna XS 2.1`
(`README.md:1`) and which contains a complete MLX inference runtime plus vendored,
*editable* MLX Metal kernels.

**Version read:** commit `4ea72c3b28873fca23b12b6f33193a2eeb5042f8` (2026-08-11,
"Validate submission cdcd0918-…"), cloned to `/tmp/mlxfast` with `--depth 1`.
Model checkpoint pinned at `poolside/Laguna-XS-2.1-NVFP4-mlx@841778bda563a36104dd521e37d99218e46f4f25`
(13 files, 21,568,905,520 bytes). All `file:line` citations below are relative to that
clone; `/tmp/mlxfast/README.md`, `/tmp/mlxfast/TASK.md`, `/tmp/mlxfast/docs/` are the docs.

**Ambiguity, and why this pick.** GitHub's exact-name search for `mlxfast` returns 15
repos, all downstream of this program: `Layr-Labs/mlxfast-challenge` (the arena,
21★), `mlxfast-bonsai2-27b-engine` (ternary Bonsai 2 27B, 34★),
`mlxfast-gemma4-26b-a4b-engine`, `mlxfast-qwen38-125b-a6b-engine`, plus
`shikharpant/laguna-mlxfast-server` ("OpenAI-compatible local HTTP server … powered by
the optimized mlxfast runtime"). Apple's `MLXFast` Swift namespace
(`mlx-swift/Source/MLXFast/`) is a *different* thing — the custom-Metal-kernel API that
these engines **use**, not the project. I document `mlxfast-challenge` because it is the
flagship, the only repo whose README claims the bare name, and the one whose workload
(NVFP4 4-bit group-16 MoE, 40 layers, 256 experts top-8, bandwidth-bound serial decode)
is nearest to our GLM-5.3-Flash problem.

---

## Biggest acceleration mechanism (one paragraph)

mlxfast's single biggest lever is **replacing MLX's per-layer op graph with hand-written
fused Metal kernels driven by an explicit per-layer scheduling ladder**. The decode step
is bandwidth- *and* dispatch-bound, so the runtime attacks both at once: (1) it cuts bytes
read per token by re-encoding every hot weight/scale tensor into a narrower certified-equal
layout (group-32 affine INT8 or NVFP4 attention side copies at 0.5625–1.125 B/param vs 2 B
BF16, halved/narrowed scale planes, and a certified two-pass `lm_head` screen that replaces
a 411 MB BF16 GEMV with a 1344→1088 B/row int5 screen plus exact refinement); and (2) it
fuses whole dependency chains into one kernel (QK-norm+RoPE+KV-write+SDPA, residual+RMSNorm,
projection+gate, MoE gate/up+Swish+down+residual, MoE tail) so one dispatch replaces three
to five dispatches and their encoder-wide memory barriers. The scheduling half is the
largest *single* measured number in the tree: firing `asyncEval` at layer boundaries during
decode is worth **+9.7%** on the decode step vs. no overlap
(`Sources/MLXFastModel/LagunaRuntimeModel.swift:717`), and a one-rung-per-layer ladder takes
essentially all of the remaining 0.15 ms.

---

## 1. What it is, how to get it

| | |
|---|---|
| Repo | `https://github.com/Layr-Labs/mlxfast-challenge` (public, license "Other" / no SPDX) |
| Commit read | `4ea72c3b28873fca23b12b6f33193a2eeb5042f8` (2026-08-11) |
| Install | `git clone` + `./setup.sh` (downloads the pinned checkpoint to a shared HF cache) |
| Language | Swift only — "no Python harness path" (`TASK.md`, "Editable Surface") |
| Build | SwiftPM (`Package.swift`); `tools/build-mlx-metallib.sh` compiles the vendored AOT `.metal` sources; families with an `mlx-generated/*.cpp` twin build in JIT mode |
| Model | Poolside Laguna XS 2.1 text tower, NVFP4 (`README.md:"Why this challenge exists"`) |
| Hardware target | self-hosted **Apple M5 Max, 128 GB** (`README.md`) |

The program is a benchmark arena, not a reusable product: `benchmark.json` lists ~97
`editablePaths` (runtime, transform, vendored model, vendored MLX Metal kernels), and the
scored artifact is the `mlxfast-runtime-worker` binary. Score is
`decode_speedup^0.75 * prefill_speedup^0.25` with hard 0.95 floors per component and a
±5% two-sided acceptance band per submission (`TASK.md` "Score"). The serial track
**forbids speculative decoding of any kind** (`TASK.md` "Serial non-speculative rule") —
which is precisely why only byte-reduction and dispatch-reduction survive.

Sibling engines of the same program (same shape, different model): ternary Bonsai 2 27B,
Gemma 4 26B-A4B, Qwen 3.8 125B-A6B. A third-party serving layer,
`shikharpant/laguna-mlxfast-server`, wraps the runtime in an OpenAI-compatible HTTP server.

## 2. Architecture

- **Runtime:** Swift, linked against MLX core (C++) and `mlx-swift`; the model tree is
  reimplemented in `Sources/MLXFastModel/LagunaRuntimeModel.swift` (12,024 lines) as
  `Module`s over `MLXArray`, not by reusing the vendored `MLXLLM` model.
- **Bindings / extension point:** Apple's `MLXFast.metalKernel(name:inputNames:outputNames:source:header:…)`
  API (`Vendor/mlx-swift/Source/MLXFast/MLXFastKernel.swift:47-57`) JIT-compiles a bare
  Metal function body from a Swift string. The runtime defines **66** such kernels
  (60 in `LagunaRuntimeModel.swift`, 6 in `LagunaLmHeadPrune.swift`), 46 unique names.
- **Where the kernels live:** (a) inline in `LagunaRuntimeModel.swift` as `source:` string
  literals; (b) in the vendored MLX Metal sources under
  `Vendor/mlx-swift/Source/Cmlx/` — both the readable `.metal`/`.h` (e.g.
  `mlx/backend/metal/kernels/fp_quantized.metal`, `steel/gemm/kernels/steel_gemm_gather.h`,
  `steel/gemm/kernels/steel_gemm_gather_nax.h`) and the runtime-effective JIT twins in
  `Source/Cmlx/mlx-generated/`. Both are inside the editable surface (`README.md`).
- **Model geometry** (`docs/laguna-weight-contract.md`): hidden 2048, vocab 100 352,
  40 layers, 8 KV heads, head_dim 128, 48 Q heads on full-attention layers (every 4th
  layer), 64 on sliding layers, sliding window 512, 256 routed experts + 1 shared, top-8,
  expert intermediate 512, layer-0 dense MLP 8192, untied head, per-head gating.
- **Weight residency:** the whole 21.6 GB is loaded into RAM once; there is **no weight
  streaming** (`TASK.md` "Model Artifacts"). On <64 GiB machines a pure-memory-management
  profile caps the allocator cache at 6 GiB (`Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift:12-19`).
- **Caches:** `StandardKVCache` for full-attention layers, `RotatingKVCache(maxSize: 512)`
  for sliding layers (`LagunaRuntimeModel.swift:11692-11697`).

## 3. Acceleration mechanisms (the core)

Legend: **[E]** = exact/bit-identical substitution claimed and internally argued; **[N]** =
numerically different (re-quantized) but inside an accepted envelope.

| Mechanism | What it does | Evidence | Measured effect |
|---|---|---|---|
| **Per-layer `asyncEval` ladder (decode)** | Enqueue already-built work at layer boundaries instead of building a ~400-op graph with the GPU idle, then one final eval. Pure scheduling: "adds no operation, cache row, dtype boundary or token". Schedule is a bitmask over layer indices, default `at:0,1,7,15,23,31,39`. **[E]** | `LagunaRuntimeModel.swift:694-800`; firing sites `:11572,11584,11589,11654,11679` | `off` 10.3735 ms → previous `ladder8` 9.4533 ms = **+9.7%**; best `ladder1` (40 fires) 1.0178, `ladder2` 1.0169, `at:1,7,15,23,31,39` 1.0170 (`:717`). Lane 1 alone measures 0.9476 (worst). |
| **Prefill `asyncEval` ladder** | Same idea for the 512-token prefill graph; default stride 1 (after *every* layer). **[E]** | `LagunaRuntimeModel.swift:770-790` | +0.40% on the ranked 1.87782 s base (`:776`); rejected only because a larger win promoted mid-queue. |
| **`set_wired_limit` on the resident weight set** | Wire ~31.4 GiB (1.0× live bytes + 64 MiB) so weights cannot be evicted/paged during the timed window. Guards: ≥96 GiB physical, `DARKBLOOM_WIRED_ZH`. | `LagunaRuntimeWeights.swift:549-640` | Dose curve: 42 MiB **−4.2%** prefill, 350 MiB −8.2%, 0.10× −11.2%, 0.20× −17.2%, 0.35× −20.8%, 1.0× **−28.3%** prefill, −4.2% decode composite (seed-prefill share); chunk-1 (42 MiB) ranked **+1.24% score** (`:556-560`). |
| **Certified two/three-pass `lm_head` screen** | Replaces the full BF16 `[100352, 2048]` = **411 MB** final-row GEMV with: (1) a fused GEMV over an init-time planar **int5** copy (1344 B/row: 1024 B nibble + 256 B 1-bit + 64 B e8m0 scales) emitting per-row coarse logit *and a certified error bound Δ*; (2) stock two-pass argmax over coarse; (3) threshold just below the winner; (4) exact BF16 GEMV only on surviving 4-row blocks. Decode variant re-splits the planes so the nibble plane alone is a 2×-coarse code → **1088 B/row**, then refines with the residual bit plane inside the exact dispatch. **[E]** for the winner. | `LagunaLmHeadPrune.swift:8-62,235,700-830` | 1088 B vs 1344 B/row = **−25.7 MB per decode step** over the 100352-row vocabulary; ~135 MB extra resident memory (`:815`). |
| **Attention projection re-encoding (native affine side layouts)** | Keeps the BF16 parameter authoritative but quantizes a *decode-only side copy* of `q_proj/k_proj/v_proj/o_proj/g_proj` to group-32 affine INT8 (or a group-16 NVFP4 "tail" window for later layers) and dispatches a quantized matmul. `o_proj` is called out as ~**1.2 GB** of the decode token's BF16 weight traffic. **[N]** | `LagunaRuntimeModel.swift:2877-2990` (builder), `:336-470` (flags/coverage) | 0.5625 B/param (NVFP4) vs 1.125 (INT8) vs 2 (BF16); "decode is bandwidth-bound; local sweep monotone ~−0.5%/layer" (`:2905-2908`). Prefill keeps BF16. |
| **Fused sliding attention (QK-norm+RoPE+KV-write+SDPA in one dispatch)** | ONE dispatch replaces `[QK-norm+RoPE] → [K slice-assign] → [V slice-assign] → [sdpa_vector]`; computes the new Q/K in threadgroup memory, persists the K/V row into the ring slot `RotatingKVCache.updateInPlace` would write, attends over the 512-slot ring with the shipped `sdpa_vector` GQA-pair schedule. **[E]** | `LagunaRuntimeModel.swift:1399-1503` | Removes **3 dispatches + encoder-wide barriers per sliding layer per decode step**, ×30 sliding layers. |
| **Fused gated output projection** | Folds the per-head softplus gate into the `o_proj` GEMV, with one kernel variant per attention family; also hoists two k-blocks of weight/scale/bias loads into registers above the norm prologue (depth-4 prefetch), strictly "loads only" so the FP32 chain is unchanged. **[E]** | `LagunaRuntimeModel.swift:336-341,3479-3500,5050-5100` | Depth-4 prefetch: **−11.9%/−12.2% kernel-level** (48.61→42.84 µs and 39.34→34.55 µs), i.e. **554/549 GB/s vs 488/482 GB/s stock**; projected ~−171 µs/step. |
| **Fused QKV projection (one dispatch over 3 weights)** | Issues Q/K/V as one dispatch over the three stock weights; keeps *no* concatenated bank, so prefill is untouched. **[E]** | `LagunaRuntimeModel.swift:341-346` | (ablation flag; the alternative row-concatenated-bank variant measured "mild prefill cost with no decode gain" and ships opt-in, `:340`) |
| **Fused routed gate/up (`[gate;up]` bank) for decode** | After load, retains one row-concatenated NVFP4 `[gate; up]` bank per sparse layer so single-token decode's gate/up is one gather-QMM. **Decode-only** — deliberately declines for prefill. **[E]** | `LagunaRuntimeModel.swift:206,10502-10764` | **~+1.9% decode**; "badly hurts the M=512 sorted gather-GEMM prefill path" (`:206-215`). |
| **Fused routed down-QMV + router weighting + fixed-order expert reduction** | Custom kernel emits one 2048-wide branch instead of materialising eight expert rows. **[E]** | `LagunaRuntimeModel.swift:195-198,7931-8056` | — |
| **Merged routed+shared down+residual dispatch** | The eight routed down projections *and* the shared down projection share one 288-thread dispatch that also performs the exact router reduction, routed scale, and both BF16 residual adds. **[E]** | `LagunaRuntimeModel.swift:139-145` | — |
| **Prefill MoE tail fusion** | Weighted expert combine + routed scale + shared add + residual add in one elementwise dispatch, so the `[1, L, 8, 2048]` expert bank is read once; a "sorted twin" reads the expert-sorted output rows directly via `inverse_order` to **delete a 16 MiB copy** at the 512-token window. **[E]** | `LagunaRuntimeModel.swift:9525-9535,10194-10214,10251-10257` | deletes a 16 MiB copy per prefill |
| **Exact router top-8 tournament (decode & prefill)** | Replaces a full 256-element stable argsort with: phase 1 = eight independent 32-lane bitonic sorts (local top-8 per block), phase 2 = one cross-simdgroup bitonic merge of the 64 candidates (one barrier), then gather the 8 winners with `simd_shuffle`. Proof in the comment that the 64 local winners provably contain the global top-8 as a set. **[E]** | `LagunaRuntimeModel.swift:9159-9340` (decode), `:9640-9760,9832-9870` (prefill) | replaces 2–3 `argpartition` dispatches with 1. Note the counter-lesson at `:9515`: a per-lane predecessor-count prefill variant ranked **−0.68%** because at 512 rows the stock sort amortizes to µs/layer — "nothing to save, only kernel shape to lose". |
| **QMV `R1` schedule (one output row per simdgroup)** | Halves rows-per-simdgroup so the grid exposes **2× as many independent simdgroups** to cover memory latency; each row keeps its own K-block order and 32-lane reduction tree. **[E]** | `LagunaRuntimeModel.swift:291-320` | "Official submission `b56a6d9` passed all 1,344 exact-token checks" (`:292`); same trick for the shared expert, which runs concurrently with the router. |
| **Packed / halved scale planes** | (a) `PACKED_SCALES`: decode-only scale-interleaved side bank storing only the 32 scale bytes per row and K-block, in the kernel's exact walk order `[expert][tile 128][k-block 4][row-pair sub 8]`, so codes and scales are not read from two separate tensors (was 4 device streams per simdgroup iteration). (b) `SHARED_SCALE_HALVED`: serve group-32 halved planes, installing them only when every discarded odd byte is bitwise equal to its even partner (the quantizer wrote the value twice) → "nothing is requantized". Both **[E]**. | (a) `LagunaRuntimeModel.swift:152-178`; (b) `:306-335` | (a) +~16 MiB resident per sparse layer; (b) 196,608 B/layer, decode **−0.171% s/token** (3/3 order-alternated pairs). |
| **Narrow / lane-major attention scale planes** | Lossless re-encoding of the decode-only attention NVFP4 scale planes: 5-bit index + uint8 base reconstructs the byte (census: `max−min ≤ 31` for 100.00% of 2.78M attention 32-group blocks). Lane-major form puts the `blocks/2` nibbles a lane consumes adjacent: `groups/2 + 1` bytes/row. Escape rows (base `0xFF`) fall back to the stock plane. **[E]** | `LagunaRuntimeWeights.swift:680-740` | 21 B vs 32 B = **−34.4% attention scale traffic**; lane-major 65 vs 84 vs 128 B/row for fused QKV; full-row span ≤15 for 98.1–99.6% of rows. Pairwise halving stores one nibble per *pair* of groups (exact, same argument). |
| **Router threadgroup widening** | `rows_per_group` 64→8 raises the router GEMV from 4 to 32 threadgroups; at 64 "four threadgroups cannot cover the machine's cores". No wave quantization hole (`rows_out` stays 256). **[E]** | `LagunaRuntimeModel.swift:644-670` | 140.2 GB/s → against a 575 GB/s box. |
| **Kernel-cache warmup for the greedy path** | Runs the identical `reshape → last row → argMax` chain once at untimed init so the `argmax_bfloat16` pipeline state does not compile inside the scored window. | `LagunaRuntimeWeights.swift:524-545` | a timestamped PSO miss showed the compile firing ~0.23 s into the scored prefill and again in the decode seed, matching recurring ~17 ms MTLCompilerService intervals. |
| **Depthwise expert-aligned NAX prefill path** | M5-generation `_nax` variants of the steel gather-GEMM (`steel_gemm_gather_nax.h`) plus sidecar expert-prefix bounds so the two fused sorted prefill QMMs skip re-deriving bounds. | `LagunaRuntimeModel.swift:225-250,10516-10593`; `Vendor/mlx-swift/Source/Cmlx/mlx-generated/metal/steel/gemm/kernels/` | — |
| **Compiled decode (available, mostly used elsewhere)** | `compile(shapeless: true)` activation fusions, gated by `MLXHardwareInfo.isCompiledDecodeSupported`; `CompilableRotatingKVCache` makes the ring cache trace-able (pre-allocated buffer, `idxArray` in MLXArray space, wrap arithmetic in graph ops). Laguna itself only compiles the softplus gate (`LagunaRuntimeModel.swift:5412-5438`), preferring its own Metal kernels. | `Vendor/mlx-swift-lm/Libraries/MLXLMCommon/MLXHardwareInfo.swift:3-33`; `…/CompilableRotatingKVCache.swift:1-60` | prefill-side stride-1 ladder above is the engaged variant |
| **Self-instrumented per-dispatch / DRAM cost** | A sentinel-gated instrument injects N empty dispatches exactly one per layer boundary and sweep/matmul work into a never-read sink, so differencing two receipts yields `DRAM GB/s = (bytes_B−bytes_A)/(T_B−T_A)` and per-dispatch cost `(T_C−T_A)/(dispatch_C−dispatch_A)`. Documents the key machinery fact: MLX's encoder is `DispatchTypeConcurrent` and **inserts a memory barrier whenever a dispatch binds a buffer a previous dispatch wrote** — so dependent dispatches serialise. | `LagunaRuntimeModel.swift:11770-12023` (esp. `:11787-11810`) | in-tree, research-only; "deliberately SLOWS the tree" |

**Pinned baseline numbers for scale** (`Sources/MLXFastCore/Constants.swift:167-168`,
baseline commit `15852ee52858def42ddd4f32bca7e59d275e020e`):

```
officialBaselinePrefillSecondsPerToken = 0.00036751938916015625   (0.3675 ms/token)
officialBaselineDecodeSecondsPerToken  = 0.01385621216015625     (13.86 ms/token ≈ 72 tok/s)
```

An in-tree receipt records `4.905191 ms decode` for a tuned router variant
(`LagunaRuntimeModel.swift:9363`). That is ~2.8× the pinned baseline; because the
acceptance band caps a *single* submission at ~+5%, this is either a long cumulative
campaign or a different measurement basis — **[UNVERIFIED]**.

## 4. Quantization support and layout

- **Format contract** (`docs/laguna-weight-contract.md`, "Quantization"): exactly
  `{"group_size": 16, "bits": 4, "mode": "nvfp4"}`, no per-tensor overrides.
  `weight` is `U32 […, rows, in*4/32]`, `scales` is `U8 […, rows, in/16]` (NVFP4 **E4M3**
  group scales), and **there is no `.biases` companion**. Any affine checkpoint, group ≠16,
  quantized router, non-U8 NVFP4 scale, or non-BF16 dense projection fails closed before
  the first forward.
- **What is quantized in the shipped artifact:** *only* the routed and shared expert
  projections. Attention, embeddings, the untied `lm_head`, the layer-0 dense MLP and the
  routers stay BF16.
- **What the runtime adds on top (side layouts, decode-only):** group-32 affine INT8 and
  group-16 NVFP4 re-encodings of the BF16 attention projections
  (`LagunaRuntimeModel.swift:2877-2990`), plus the int5 `lm_head` screen. Effective
  bits/param: NVFP4 0.5625 B, affine group-32 INT8 1.125 B, BF16 2 B.
- **Registered MLX kernels:** dispatch goes through MLX's own quantized path —
  `fp_quantized.metal` for NVFP4, `gatherQuantizedMM` / `quantizedMM` in Swift — with `_nax`
  M5 variants selected at runtime.
- **layout note worth stealing:** the code is explicit that NVFP4 as shipped costs
  0.5625 B/param vs 1.125 for INT8 and that the choice is *purely* a bytes/token trade
  decided by a monotone per-layer sweep (`LagunaRuntimeModel.swift:2905-2908`).

## 5. Transferable to exl3-server (ranked)

Our context (`docs/perf.md`): GLM-5.3-Flash, 45 layers, 288-expert MoE top-8, ~4.6 GB/token,
**launch-bound** (4000–5000 kernels/token at ~30–50 µs), 546 GB/s machine peak, currently
118–170 GB/s effective. Levers already listed there: `mx::compile`, grouped expert GEMV,
fuse the two Hadamards into the trellis GEMV, `mx::async_eval`, `mx::set_wired_limit`.

| # | Steal this | Expected effect (mlxfast's own numbers) | Cost / risk |
|---|---|---|---|
| 1 | **`mx::async_eval` at layer boundaries in the decode loop** (their `at:0,1,7,15,23,31,39` ladder). Their finding: the *front* rung matters most (a lone layer-1 fire is the worst schedule, 0.9476; the full ladder 1.0178). | +9.7% decode step vs no overlap, +1.78% more from the ladder tail (`LagunaRuntimeModel.swift:717`) | Very low. Pure scheduling, bit-exact, no new kernel. Needs the equivalent of their `DARKBLOOM_DECODE_ASYNC_STAGE` bitmask so the boundary set is sweepable without a rebuild. |
| 2 | **`mx::set_wired_limit` for the resident EXL3 weight set** | −4.2% → −28.3% prefill on a dose curve; **+1.24% ranked score** at the smallest dose (`LagunaRuntimeWeights.swift:555-560`) | Low. Clamp to `GPU.maxRecommendedWorkingSetBytes() − 256 MiB` (their code throws otherwise) and gate on ≥96 GiB physical. Watch prefill/decode asymmetry. |
| 3 | **Fuse the Hadamards into the trellis GEMV + drop per-layer host `mx::eval`** — this is their "one fused dispatch replaces a 4-stage chain" pattern applied to our `trellis_linear` (3 launches × ~1450 linears/token). | Their fused sliding attention removes **3 dispatches + encoder barriers × 30 layers**; their fused MoE tail removes a 16 MiB copy | Medium. Our `src/trellis_gemm.cpp` already owns the Metal plumbing; the Hadamard is currently separate because it may need to stay fp16-exact — same "loads only / same FP order" discipline they document. |
| 4 | **Grouped expert GEMV with expert-aligned gather** (`gatherQuantizedMM` semantics): one kernel per layer over the top-8 routed experts, no host loop, no per-expert `mx::eval`. Their fused routed gate/up is decode-only *because prefill prefers the stock sorted gather-GEMM* — copy that split. | ~+1.9% decode for the fused gate/up alone (`:206`) | Medium-high. Biggest structural win for our 8 ms/token MoE host loop, but it is the largest code surface. |
| 5 | **Exact router top-k tournament instead of `argpartition`** — we currently issue 3 `argpartition` dispatches per MoE layer (`src/glm5_moe.cpp:30,37,47`); they replace a 256-way stable argsort with a per-block top-8 + one 64-element bitonic merge. | replaces 2–3 dispatches with 1, bit-exact | Medium. Their code is a directly adaptable Metal kernel; the exactness proof (local top-8 per block suffices) is in the comment and carries over to our 288 experts / top-8. |
| 6 | **One row per simdgroup (their `QMV_R1`)** to double independent simdgroups and cover memory latency — our trellis GEMV measured 258 GB/s at rows=1 vs 546 peak. | "2× as many independent simdgroups to cover memory latency", exact | Low-medium. Pure grid/thread mapping change to `trellis_gemm.cpp`; must keep each row's reduction tree. |
| 7 | **Register prefetch across k-blocks** (depth-4: hoist the first blocks' weight/scale/bias loads above the norm prologue, then consume in the exact order). | **−11.9%/−12.2% kernel-level**, 488→554 GB/s | Low-medium. "Loads only" is the whole trick; our GEMV already does aligned loads (`perf.md` notes 305 GB/s with aligned loads), so measure before assuming headroom. |
| 8 | **Cut bytes/token in the head and scales** — the `lm_head` screen pattern (coarse certified bound → argmax → threshold → exact refine only on survivors) is directly applicable to our dense `apply_lm_head` (`src/glm5_forward.cpp:265`), and their scale-plane narrowing shows per-group scale bytes are often highly redundant (they *prove* byte-equality before halving). | head: 411 MB → 1088 B/row; scales: −34.4% | Medium-high for the head (needs an int5/int8 coarse copy + a certified Δ bound — nontrivial but their algebra is written out in `LagunaLmHeadPrune.swift:8-40`); low for scale planes if our EXL3 `.suh/.svh` scales have comparable redundancy. |
| 9 | **Move one-time pipeline compiles out of the timed window** (warm the argmax/last-row path at init). | avoided a ~17 ms MTLCompilerService stall inside the scored window | Trivial. Our sampler's argmax/top-k path is the analogue. |

### What does NOT transfer

- **The whole NVFP4/affine re-quantization envelope.** Our weights are EXL3 **trellis**
  (`.trellis`/`.suh`/`.svh`, K=1..8 codebooks) — a *different* weight format. Their
  `fp_quantized`/`quantizedMM` NVFP4 group-16 kernels, `.scales`-tensor layout, E4M3
  group-scale decoding, and `quantized()`/`dequantized()` builders do not apply unless we
  add an NVFP4 path. Their "re-quantize a decode-only side copy" idea *does* transfer as a
  principle, but the concrete kernels and layout walk-orders (`[expert][tile 128][k-block 4]`)
  are group-16 NVFP4 specific.
- **Per-request serial-decode constraint.** Their serial track *forbids* speculation, and
  their entire result set is tuned for exactly one token per invocation. We deliberately
  support MTP (+30–50% measured), which their numbers cannot bound.
- **Their absolute performance numbers.** M5 Max (128 GiB, ~596 GB/s fabric ceiling cited
  at `LagunaRuntimeModel.swift:3466`) vs our M4 Max (~546 GB/s, 128 GB). Their local
  "driver rig" ablations and the ranked paired ratios are not portable; only the
  *relative* mechanism ordering is.
- **Their `compile(shapeless:)` compiled-decode path as the headline.** Laguna's own
  runtime uses `compile` only for a tiny gate; the wins come from hand-written Metal. In
  our C++ engine `mlx::core::fast` is not even exposed (`docs/perf.md`), so we must write
  kernels through our own Metal plumbing, not through `MLXFast.metalKernel`.
- **Swift/`MLXFast.metalKernel` string-JIT ergonomics** and the vendored `mlx-swift-lm`
  module tree: our engine is C++, links MLX 0.32 directly, and has no Swift binding.
- **Anything gated on the M5 `_nax` kernel variants** (`lagunaNAXAvailable`,
  `steel_gemm_*_nax.h`) — those pipeline variants are selected by OS/architecture and are
  M5-generation-only.
- **The benchmark harness itself** (`benchmark.sh`, `mlxfast-swift` trusted binary, Yukon
  upload): a scoring arena, not inference machinery.

## 6. Open questions / not verified

- **[UNVERIFIED] Cumulative vs. per-submission gains.** The repo is the *baseline scaffold*;
  many flags are default-ON, and comments record ablations and "promoted/ranked/rejected"
  verdicts, but the actual submitted champions (and the resulting leaderboard score) are not
  in this repo. So the true current-best speedup over `officialBaseline*` is unknown.
- **[UNVERIFIED] `notes/` and `research/` are referenced throughout but not committed**
  (e.g. `notes/68`, `notes/52`, `notes/47`, `research/frieren-pr35-scale-census.md`). Every
  measurement quoted above is from an in-source comment, not from the primary note.
- **[UNVERIFIED] The 4.905191 ms decode receipt's basis** (era, host, whether it is a
  driver-rig or ranked number) — it is inconsistent with the `off = 10.3735 ms` figure in
  the adjacent scheduling note, likely a different generation of the tree.
- **[UNVERIFIED] Whether the int5 `lm_head` certificate is still used as-shipped**: the
  `LagunaLmHeadPrune.swift` header cites notes 68/70 that are absent, so the bound algebra
  (`Δ_i = d_i(1+γ) + 2γ·m_i`, γ = 2⁻¹⁵) is reproduced here from the code comment only.
- **Not inspected:** `Sources/MLXFastTransform/Transform.swift` beyond its quantization
  contract, the trusted harness/timing layer, and the vendored Steel GEMM `.metal` bodies
  (I read the file inventory, not the kernel arithmetic).
- **Peer engines not read:** `mlxfast-bonsai2-27b-engine` (ternary weights — the most
  interesting bytes/token datapoint for us) and `mlxfast-qwen38-125b-a6b-engine` were
  identified but not cloned.
