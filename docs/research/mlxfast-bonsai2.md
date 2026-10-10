# mlxfast-bonsai2 — Ternary Bonsai 2 27B MLX engine (Layr-Labs / Yukon)

**Resolved identity.** `https://github.com/Layr-Labs/mlxfast-bonsai2-27b-engine`, read at
commit **`0c7c39233afa8aa3e22ac9f36f113b1d2389c3af`** (2026-10-02, "Accept submission
1cb74b93-5c70-49f5-94d4-71aac365ff38"), cloned to `/tmp/bonsai2/repo` with `--depth 50`.
It is the **engine (scaffold) repository for track `bonsai2-27b-mlx-v1`** of the same
Layr-Labs / Yukon benchmark program as `mlxfast-challenge` (see
[`mlxfast.md`](mlxfast.md)). 50 commits: a seed commit `9186633` (2026-09-29) followed by
49 bot merge/validate commits ("Accept submission …"), i.e. the arena's autoresearch loop
checkpointing accepted participant submissions into `main`. The scored artifact is that
tree; a separate verified binary (`benchd`) owns all timing and scoring.

**Corrections to the description, up front, because they change what is transferable.**

1. **The weights are not ternary.** The pack's own config says
   `"quantization": {"bits": 2, "group_size": 128, "mode": "affine"}`
   (`fixtures/bonsai2_27b_config.json`), and the inventory confirms
   `weight: U32 [rows, K/16]`, `scales`/`biases`: `F16 [rows, K/128]`, `signs`: `F32 [K]`
   (`fixtures/bonsai2_27b_tensor_inventory.json`; `language_model.lm_head.weight` =
   `U32 [248320, 320]`). "Ternary" is the checkpoint's marketing name
   (`Ternary-Bonsai-2-27B-mlx-2bit`), not a `{-1,0,+1}` format. The `ternary*` files in
   the vendored MLX core (`Source/Cmlx/mlx/mlx/backend/metal/kernels/ternary.metal`,
   `ternary.h`, `ternary_ops.h`) are Apple's **ternary *select*** op (`where`) — a red
   herring; they are not the weight decoder.
2. **It is a 2-bit engine plus a folded Hadamard transform**, and the Hadamard is the
   structural fact that shapes every kernel: 402 of the pack's linear modules ship
   `weight`/`scales`/`biases`/`signs` and a normalized block Walsh–Hadamard transform of
   width 1024 folded into the packed weight
   (`docs/bonsai2-27b-port-notes.md`, §3.2; `Vendor/mlx-swift/Source/MLXNN/Hadamard.swift:19`).
3. **The model is dense, not MoE.** 64 layers, 16 full-attention + 48 gated-delta-net,
   hidden 5120, intermediate 17408, 24 Q / 4 KV heads, head dim 256, vocab 248320
   (`fixtures/bonsai2_27b_config.json`). The shared tower has an MoE path
   (`Qwen35SparseMoeBlock`, `Qwen35.swift:4759`) but it is selected only when
   `numExperts > 0` (`Qwen35.swift:5028`), and this pack declares none.
4. It is **not a from-scratch engine**: the tree is a seed of the Nemotron 3.5 Lightning
   30B A3B engine (`648495f`, re-stamped by `tools/new-track.sh` at `5f4b611`), with two
   vendored forks: `Vendor/mlx-swift` = `Layr-Labs/mlx-swift` `70052b2` (nested
   `Source/Cmlx/mlx` `3fa8f25e`, `mlx-c` `02cf6f4d`) and `Vendor/mlx-swift-lm` cut from
   `Layr-Labs/mlx-swift-lm` `feat/qwen38-flash-next-runner` `859f4d9` + fork `main`
   `0607011`, with the Bonsai 2 files taken from fork `main` `fd0eaac` (PRs 154/155)
   (`README.md`; `docs/bonsai2-27b-port-notes.md` §2, §4.1).
5. **The track is armed at this commit, but nothing in the repository times itself.** The
   fixture is filled in: `official_scoring_enabled: true`, a pool of 11 recorded tapes,
   three live goldens (`pool-9/10/11`), a pinned `hidden_correctness_golden`, and the
   `mtp1..dflash16` per-depth oracles
   (`fixtures/bonsai2_27b_mlx_v1_track.json:5, :35-51, :131-134`). The prose has not caught
   up — `README.md:768-774`, `docs/participant-contract.md:682-683` and
   `docs/bonsai2-27b-port-notes.md` §5/§7 all still say scoring is off, the pool is empty
   and the sentinel is in place; the fixture's `baseline_reference_commit` is `88cf569e…`
   while the docs name `fffde01`; and `TASK.md:208` still states a 0.90 floor against the
   fixture's 0.95. Trust the fixture over the prose. A ranked *runner* is still
   unregistered (`.github/workflows/bonsai2-27b-ranked-benchmark.yml` is a fail-closed
   stub), and there is still no in-tree receipt of a scored run. **So every number below is
   an in-source measurement note taken on somebody else's box, not a scored result**, and
   none of it is reproducible from this repository. The port itself was never compiled on
   the author's laptop (`docs/bonsai2-27b-port-notes.md` §7).

**What it costs per token.** From the pinned inventory: trunk weights excluding the
embedding and `lm_head` = **6.96 GB**, `lm_head` = **357.6 MB**, embedding table =
357.6 MB (read only on lookup). So one decode step streams ≈ **7.3 GB** of a dense 27B at
2 bits — 0.28125 B/weight (2 bits + F16 scale + F16 offset per 128). At our M4 Max's
546 GB/s that is a **13.4 ms floor per step**, and every mechanism in this report is aimed at
the gap between that floor and what a kernel that decodes each weight actually achieves.

---

## Biggest mechanism (one paragraph)

**They take the per-weight work off the SIMD lanes entirely.** Every packed projection's
matmul is issued as a MetalPerformancePrimitives tensor op —
`mpp::tensor_ops::matmul2d` — whose right operand is *the packed 2-bit weight stream read
as such*: either natively, as `tensor<device uint2b_format, dextents<int,2>, tensor_inline>`
around the raw `U32` words (`Qwen35MTPTopTwo.swift:63-118`, `:120-127`), or, where that
operand format is unavailable, by expanding the words into **threadgroup int8 tiles with
three shifts and four ANDs per 16 weights** and reading them as an `int8_t` operand
(`Qwen35MTPTopTwo.swift:2792-2925`, the `sourceStaged8` body). Either way the cost per
weight is not a decode and not an FMA: it is 0 ALU (native form) or ~0.44 integer ALU ops
plus 1/16 of a 16-byte store (staged form), and the multiply–accumulate happens on the
tensor unit at its own rate. The affine correction is applied once per *output element per
128-group*, ~0.016 FMA per weight, with two of its three terms precomputed at load
(`uT = −128·s·colsum`, `Qwen35MTPTopTwo.swift:5474-5476`; `NEG`: the offset *is* the
negated scale, proven by comparing FP16 bit patterns, `Hadamard.swift:508-518`). The second
half of the mechanism is **keeping such a matmul from ever meeting MLX's scalar path**: the
core's `qmv*` kernels "pay a device load and an FMA per weight per row", so the runtime
*changes the shape* — row counts below the core's per-arch matrix threshold are zero-padded
up to it and the pads dropped (`matrixRegimeMinimumRows = 13`,
`Hadamard.swift:730`, `:840-856`, `:863`), and verify-width row counts are padded to 16 with
a cached zero buffer (`tensorRouteMaximumNarrowRows = 16`, `:941`, `cachedNarrowZeros :780`).
The third lever is the fold: because the Hadamard lives in the weights, the *activation*
must be rotated, so they fuse the rotation with its producer (SwiGLU / attention sigmoid
gate / GDN gated RMSNorm) and with the quantization the next matmul wants, in one kernel
(`bonsai_signed_hadamard_1024` family, `Qwen35.swift:7334-7544`; `FusedInputHadamardKernel`,
`Hadamard.swift:2325-2540`) instead of a three-launch chain. Result: every projection of every
layer runs one of a few fused, fully unrolled kernels, and no kernel in the trunk pays
per-weight instructions (an inference from the linear-module/launch structure, not a measured
dispatch count).

---

## 1. Weight format and decode

### 1.1 Format (all verified from the pinned inventory and config)

| Fact | Value | Evidence |
|---|---|---|
| Quantization | affine, 2 bits, group 128 | `fixtures/bonsai2_27b_config.json` `quantization` |
| Codes | `U32 [rows, K/16]` — 16 two-bit codes per word, little-end first | `language_model.lm_head.weight = U32[248320, 320]`; K=5120 |
| Scale / offset | `F16 [rows, K/128]`, both present | `lm_head.scales/biases = F16[248320, 40]` |
| Signed Hadamard | `signs: F32 [K]`, block 1024, `hadamard.json` manifest, 402 modules | inventory summary `packed_module_count: 402`; port notes §3.2 |
| Folded arithmetic | `y = Σ_g as[m,g]·(s[n,g]·(C[m,n,g] − 128·colsum[n,g]) + b[n,g]·rs[m,g])` | `Qwen35MTPTopTwo.swift:19-27` |
| Bytes per weight | 0.28125 (0.25 code + 0.015625 scale + 0.015625 offset) | computed from the inventory |
| Trunk stream per step | 6.96 GB + 357.6 MB head | computed: `Σ` tensor bytes over `language_model.` minus embed/lm_head |

`C` is the integer dot product of the activation's shifted codes with the weight codes,
`colsum` the per-group sum of the weight codes, `as`/`rs` the activation's per-group scale
and code sum. Two forms of the same identity are used, both claimed bit-for-bit equal:
signed int8 activation codes drop the `128·colsum` term entirely
(`Qwen35MTPTopTwo.swift:658-671`: "`fma(s, C, -128 s colsum)` and `s·(C − 128 colsum)` each
round once to the same number (the offset is exact in FP32)"), and when the offset is the
negated scale the epilogue becomes `s·(as·C − rb)`, "one FMA fewer per output element and
128-group (1–1.5% on the prompt-width matmuls on an M5 Max" (`:662-675`).

### 1.2 Instruction count per weight (the answer to the question)

Measured by reading the four decode paths in the tree. **Lane** below means one of the 32
threads of a simdgroup; **warp-instr** means one issued instruction for all 32 lanes.

| Path | Where | Per weight | Per 16 weights (one `U32` word) |
|---|---|---|---|
| **Native 2-bit operand** (prompt width) | `Qwen35MTPTopTwo.swift:63-118` | **0 ALU**; the tensor op reads the 2-bit words directly | 1 `op.run(tA,tB,cT)` per 64×64×128 tile-step; that single tensor instruction covers the MACs |
| **Staged int8** (prompt width, preferred) | `:2792-2925` | 7/16 = **0.4375 integer ALU** (3 shifts + 4 ANDs) + 1/16 of a 16-byte `uint4` threadgroup store | 3 shifts, 4 ANDs, one `*(threadgroup uint4*)` store; the word itself is ¼ of one `uint4` device load |
| **Staged int8, verify width (M≤16)** | `:813-955` (staging at `:836-848`), `:2661-2667` | same 7/16 ALU, same 1 `uint4` store | identical staging, but per **column** of the 32-column tile instead of per 16-column half |
| **Software-pipelined verify** (K2/K3) | `:959-1194` | same, with a register ring of PD∈1..4 groups and a constants ring, group loop unrolled so every ring index is a constant | the ring refills make the loads independent of the AND/shift work |
| **Few-row (4–16 rows) `simdgroup_matrix`** | `Hadamard.swift:2757-2856` | 1 OR + 1 SUB per **two** weights: `uint bitsv = (words[b][j/D] >> (BITS*(j%D))) & pair; half2 q = as_type<half2>(bitsv \| 0x64006400u) - half2(1024.0h)` (`:2806-2808`) | the classic mul1/bitcast-to-half idiom, one lane pair at a time, feeding `simdgroup_multiply_accumulate` fragments |
| **MLX core scalar (`qmv`/`qmv_wide`) — what they avoid** | `Cmlx/mlx/mlx/backend/metal/quantized.cpp:1962-2001` | **1 device load + 1 FMA per weight per row** | quoted verbatim in `Hadamard.swift:840-842` |

Epilogue cost, all paths: `CAP` accumulators per lane (`CAP = 32` in the prompt kernel,
`16` in the verify kernel), each getting **2–3 FMA per 128-group** depending on the form —
`acc = fma(b, rb, fma(as, fma(s, C, u), acc))` unfactored, `acc = fma(s, fma(as, C, −rb), acc)`
factored and signed (`Qwen35MTPTopTwo.swift:108-116` native, `:871-875` verify, `:2887-2891`
the factored branch), i.e. **≈0.02 FMA per weight** — plus per-lane loads of the
group's scales/offsets/code-sums — as `half4`/`float4`, four columns at a time
(`:87-93`). With the row-tiled constant layout (`MPERM`) a lane's four rows are adjacent in
a 64-row tile so those are one `float4` instead of eight scalar loads (`:101-107`,
`:700-712`).

**Contrast with ours**, measured in `docs/perf.md` ("Fourth pass"): the trellis GEMV retires
**~110 warp instructions per (16×16 output tile, k-tile) iteration**, i.e. per 256 weights —
a 16-bit sliding-window extract, a mul1 hash and a half-precision FMA per weight, plus the
FMAs, the packed-word loads, the 64-bit window shift and the loop's address math. That is
**0.43 warp-instructions per weight** (13.75 lane-slots), against their **0.0156
warp-instructions per weight** of staging (8 issued instructions — 3 shifts, 4 ANDs, one
16-byte store — per lane-word of 16 weights, so one warp instruction retires 512 weights)
plus one tensor-unit op per 512 weights. That is **≈27× fewer issued non-tensor instructions
per weight**, and the MACs are off the issue pipe. Their engine does not beat our problem by
being cleverer per weight — it removes the per-weight work.

### 1.3 What they do *not* do

- **No lookup table decodes weights.** Tables appear only for *constants*: the derived
  layout cache (`HadamardConstantLayoutCache`, `Hadamard.swift:465-520`), the folded
  code-sum plane (`codeSums`, `Qwen35MTPTopTwo.swift:3394`, folded at `:5474`), and the
  DFlash escape list (§5.2). The weight itself is either handed to the tensor unit raw or
  expanded by shifts/ANDs.
- No per-weight scale/offset load: those are per 128-group and read as vectors.
- No dequantise-then-FMA matmul anywhere in the trunk path; `dequantized(...)` is used only
  for the embedding rows (`Hadamard.swift:2052-2058`).

---

## 2. Kernel shape and scheduling

### 2.1 Prompt-width packed matmul (`M ≥ 128` and `M % 64 == 0`)

`Hadamard.swift:952` (`tensorRouteMinimumRows = 128`); bodies at
`Qwen35MTPTopTwo.swift:63-118` (native) and `:2792-2925` (staged).

- Grid `(N/64 × 128, M/64, 1)`, threadgroup `(128,1,1)`, `execution_simdgroups<4>`,
  descriptor `matmul2d_descriptor(64, 64, 128, false, true, false, multiply)`.
- `CAP = 32` FP32 accumulators per lane (a 64×64 tile over 128 threads), unrolled init and
  a fully unrolled epilogue with `float4` stores — the destination-layout index algebra is
  written out in the comment (`:69-77`).
- **Double-buffered weight staging**: two threadgroup tiles (`bs[2][64*128/4]` = 8 KB),
  `stage(g+1, cur^1)` issued before the current group's `op.run`, one
  `threadgroup_barrier` per group (`:2806-2862`).
- Weight reads are one `uint4` (four words = 64 codes) per thread per group; the device
  address is `n0+sc` row-major or, with `TILED`, the tiled copy so "the 64 threads read each
  block's 1 KB" contiguously (`:2833`, `:2845-2848`).
- Constants (scales, offsets, folded code sums) are read as `half4`/`float4` per lane per
  group from a transposed `[K/128, N]` layout, built once in the layout cache.

### 2.2 Verify-width packed matmul (`M ≤ 16`)

`Hadamard.swift:941` (`tensorRouteMaximumNarrowRows = 16`); bodies `Qwen35MTPTopTwo.swift:813-955`,
`:959-1194`, `:704-812`, `:1195-1420` etc.

- Grid `(N/TN × 128, 1, 1)`, threadgroup `(128,1,1)`; **`TN = 32`** by default, chosen
  because "32 (measured 1.5% faster on the decode window than 64, which the isolated kernel
  preferred)" (`:223-226`) — an explicit standalone-vs-in-situ reversal.
- **Four simdgroups split K into contiguous quarters** (each running its own
  `execution_simdgroup` 16×TN×128 op), and the K partials are summed "in simdgroup order"
  through threadgroup memory; the reduction reuses the staging buffers once the K loop is
  done, "16 KB of threadgroup memory in all, two threadgroups per core" (`:897-898`).
- `narrowDeepSplit`: eight simdgroups when `k ≥ 8192` and `(k/128) % 8 == 0` — i.e.
  `down_proj` (`:216-220`, `:5442-5443`).
- Software-pipelined variant for K2/K3 (`:959-1194`): a register ring of the next PD∈{1..4}
  groups' words plus a constants ring at depth `max(PD,2)` loaded that many groups minus one
  ahead; `KH ∈ {128,64}` sets K per tensor op, halving staging to 2 KB/simdgroup at 64 for
  four threadgroups per core. Ring indices are compile-time because the group loop is
  unrolled by the ring depth.
- The verify branch pads rows 1..15 up to 16 with a **cached resident zero buffer**
  (`cachedNarrowZeros`, `Hadamard.swift:780-793`; `MLXFAST_RIDER_NARROW_ZERO_PAD=0` restores
  a fresh zeros kernel per projection) and slices the real rows back out.

### 2.3 Escaping the scalar kernel (the shape hack)

- `matrixRegimeForward` (`Hadamard.swift:863`) zero-pads `rows` to
  `max(rows, 13)` (`:730`), calls the core's `quantizedMM`, and drops the pad rows; the
  chosen read dtype is `routeInputDType` (`:1636-1646`), which picks FP32 for narrow
  (split-K) shapes and for the vocabulary head, FP16 otherwise.
- Their own summary of the core dispatch is at `:840-862`: "fewer than 13 rows to the scalar
  `qmv_wide` kernel, which pays a device load and an FMA per weight per row, and 13 or more
  rows to the tensor `qmm_t_nax` kernel or, for a projection with at most 256 32×32 tiles,
  to the split-K kernel whose body is on the tensor unit for FP32 input". The vendored core
  confirms it: `get_qmv_batch_limit` returns 13/25/33 by arch and D×O
  (`Cmlx/mlx/mlx/backend/metal/quantized.cpp:89-120`), and `dispatch_qmv` routes below it to
  `qmv`/`qmv_wide` (`:1962-2001`).
- A vocabulary-head specialisation at verify width (`n ≥ 65536`, `n % 32 == 0`,
  `k % 256 == 0`, `m == 16`) runs the core's few-row body with an FP32 store and 8
  simdgroups; `headColumnBlocks` defaults to 1 because "on the drafter's 16 × 100,352 head
  one block per threadgroup runs about 4% faster on M5 Max (460 vs 482 µs, standalone)"
  (`Qwen35MTPTopTwo.swift:243-272`).

### 2.4 Weight copies and plane forms

- `narrowTiledWeight` / `narrowPlaneWeight` build per-projection re-layouts at load: the
  tiled copy turns "32 B from each of 32 rows" into "1 KB in one run" (`:841-849`), and the
  plane form gives each lane its own 32-byte group (`narrowTiledWeight :4470`,
  `narrowPlaneWeight :4507`). A `PlaneFormTrial`
  and a register-weight form (`sourceStaged8Reg`) are self-tested bitwise at load and only
  then offered.
- Sibling stacking: several packed projections that read the *same* rotated activation are
  concatenated along the output axis **byte for byte** — "the stacking copies packed rows; it
  does not unpack, requantize or re-scale anything" (`Hadamard.swift:563-567`,
  `HadamardFusedSiblings` `:571-605`) — one matmul replaces one per sibling, and the wide
  result is split back with `MLX.split` at the cumulative boundaries.
- DFlash 2's drafter weights get a **lossless 12-bit resident re-encoding** whose effects are
  measured per shape (§5.2).

### 2.5 Host-side graph build, eval cadence, batching

**Two scheduling layers, and both are ladders.**

- **Engine layer: one `asyncEval` per step, one host sync per step.** `EngineLoopV2` "does
  graph-build + `asyncEval` ONLY" (`ContinuousBatchingV2/EngineLoopV2.swift:5-6`); the whole
  mixed step is one submission (`:3437`; chained decode `:3009`; MTP round
  `EngineLoopV2+MTP.swift:64` plus a second, later submission for `lateEvalTargets` `:68`).
  Chained decode builds step N+1 on step N's still-lazy sampled tokens and submits it
  *before* blocking for stop detection — "Tokens are therefore inspected one step late"
  (`:13-17`), chain launch `:2219-2265`. The one host sync is at `:3599-3602` ("THE host
  sync — overlapped with the successor step's GPU work when chained … graph pipelining stays
  bounded at two steps"). This is what replaces mlxfast-challenge's layer ladder *at the
  engine level*.
- **Prompt chunking exists but the scored seed is deliberately ONE chunk.** Scheduler cut is
  `min(remaining, cap, budget, headroom)` with `prefillChunkSize = 512` and a 2048-token step
  budget (`SchedulerV2.swift:712-714`, `:386-394`; `CBv2Contracts.swift:820-822`), and each
  assignment runs as a `[1, chunk]` forward or as one packed rectangular cohort
  (`EngineLoopV2.swift:3301-3400`, `:3221-3300`). But the bench worker builds the engine with
  `prefillChunkSize = max(512, seedLength)` (`BenchWorker.swift:713-714`) — and the reason is
  a bit-identity finding: "a chunked prefill is **not bit-identical** to a single forward: …
  a ~5e-5 logit delta in float32 and the 125B shows as an argmax flip on the first close
  margin (a box saw the live golden diverge at step 1 with two 512-token chunks and
  reproduce through nine tokens with one)" (`BenchWorker.swift:703-712`).
- **Model layer: layer-boundary `asyncEval` inside the forward — this exists, and it is the
  same mechanism mlxfast-challenge measured at +9.7%.** `Qwen35.swift:226-260`: "at chosen
  layer boundaries the forward `asyncEval`s its hidden state, so the GPU runs the front of
  the tower while the host is still building the rest … **only command-buffer boundaries
  move, so every value is bit-identical**." Plans (`Qwen35TrunkSubmission`, `:365-375`):
  prompt when `rows >= 128` (`:262`), verify when `captureRecurrentWindow`. The verify plan's
  default boundaries are **after layers 8, 24 and 40** (`:315-321`), overridable with
  `MLXFAST_VERIFY_SLICE_LAYERS`, off with `DARKBLOOM_QWEN35_VERIFY_SLICES=0`; a queue-aware
  variant uses `[2,8,24,40]` ("unqueued", `:330-336`). Prompt plan = **stride 4**
  (`MLXFAST_PREFILL_PIPELINE`, `:338-341`); the fused prompt plan = `[1,2,4,8,16,32,48]`
  (`MLXFAST_PREFILL_PIPELINE_FUSED`, `:352-363`). Submission sites `:5858-5867`
  (`asyncEval([hiddenStates])` `:5864`) and `:5824-5833`. Extra rungs: a layer-0 split that
  submits the layer's projection on its own (`Qwen35FirstLayerSplit`, `:3745-3747`, rows ≥ 64)
  and a prompt-embedding pre-submission "so an idle GPU starts waking while the host builds
  layer 0" (`:5725-5729`). It is never used over paged KV (`:254-256`).
  **Their measured reason** (`:305-313`): "the host builds the verify at about **0.1 ms per
  layer**, so the 56 layers behind the first boundary outlast the GPU's **~3 ms** on the first
  8 and the GPU idled **~1.3 ms per round** in between" — and the counter-lesson, "slices
  every 2 layers had lengthened the ranked window" (`:245-246`).
- **Keep the engine thread off the efficiency cluster.** `CBv2EngineWorkInterval`
  (`ContinuousBatchingV2/CBv2EngineWorkInterval.swift:7-23`) sets an audio work interval per
  step with a 4 ms deadline after the readback because the scheduler otherwise runs the engine
  thread on an efficiency core: measured on M4 Max with per-thread counters, the **verify
  build takes 11.4 ms instead of 3.8 ms**, the committed recurrent state 5.3 ms instead of
  3.1 ms, and after the fix "the verify build took 3.1 ms in every round, and GPU idle per
  round fell from **2.0 ms to 0.6 ms** with unchanged GPU busy time". The prompt-side variant
  (`MLXFAST_PROMPT_WORK_INTERVAL`, default off) measured the seed step's engine CPU at 78% on
  the efficiency cluster, "**59 ms instead of 82 ms**". Both are host-side QoS only.
- **Keep the host's work off the GPU's critical path in other ways**: host-built
  `positionOffsets` replace one tiny on-device add per layer ("16 one-element launches per
  verify window of the 27B", `LayerCacheV2.swift:291-303`, `MLXFAST_HOST_POSITION_OFFSETS`,
  default off); the KV append is fused into the q/k prework kernel instead of two slice
  updates per attention layer, adopted only when a trial logs it ≥1% faster
  (`MLXFAST_INPLACE_KV_APPEND`, `:336-346`); a prompt row's tapped context moves into the
  step's late submission so the seed's readback does not wait for a 26 MB concat
  (`MLXFAST_PROMPT_TAP_LATE`).
- **GPU wake trick**: `CBv2GPUWake` commits one scalar add as engine assembly begins because
  "on an M4 Max after a 20 s idle, 13–15 ms after the commit (0.5 ms warm)" and "~4 ms of
  assembly and graph building" precede the first real submission
  (`SequenceKV/ContiguousKVBackend.swift:32-40`).
- **Respect the in-flight command-buffer cap**: "One `asyncEval` of the whole proposal holds
  this thread in MLX's in-flight cap (at most 10 command buffers; the proposal is ~30)"
  (`EngineLoopV2+MTPFinalize.swift:488-493`), so a DFlash proposal is submitted in
  `earlyDraftLeadingLayers = 3` parts (`:83-95`). The cap itself is only asserted in comments
  in this tree — **[UNVERIFIED]** where the core enforces it.
- **Kernel selection is in situ, per shape, with an orthogonal array.**
  `NarrowInSituTrial` (`Qwen35MTPTopTwo.swift:3730-3900`) runs candidates interleaved round
  by round at block boundaries (stage 1: 3 rounds per set; stage 2: 8 rounds per set for up
  to 3 candidates plus the record's pick), then estimates **per-shape effects with an
  `OA(18, 3^5)` array** over five production shape factors (`perShapeKeys`, `:3828-3832`) and
  installs the best kernel per shape. Adopt margin **1.5%** (`:3787-3790`), stage-2 slack 2%,
  rounds above 1.5× their candidate's median dropped, at least 4 kept rounds required
  (`:3791-3798`). The trial runs only in the deferred load warm, never inside a served
  request (`:3722-3729`). Every candidate is gated on a **bitwise self-test** first
  (`exact(kernel, .float32)`, `:5320-5345`). Their own critique of synthetic timing: an
  early shortlist "streams each kernel for 0.5–0.7 ms bursts in which the GPU never reaches
  the clock state of the decode window" (`:3690-3692`), which is why the trial moved in-situ.
- `compile()` fires only for small shapeless elementwise chains (`swigluSigned`,
  `gatedDeltaGates`, `gatedNormTail`, `sigmoidGateSigned`, … all `compile(shapeless: true)`,
  `Qwen35.swift:439-487`) plus the quantized-attention softmax block, whose comment says
  `compile(shapeless:)` "fuses those launches into a single graph and cuts the per-step launch
  overhead that dominates decode latency" (`KVCache.swift:2151-2154`, `:2178-2196`). Gate:
  `MLX_COMPILED_DECODE`, default on. **Whole-model compiled decode is NOT wired**:
  `Qwen35Runner.swift:51-55` declares `supportsCompiledDecode: false` and `CompilableKVCache`
  is documented as "presently UNWIRED" (`CompilableKVCache.swift:28-31`).
- **`setWiredLimit` is never called by this tree**: no call site in `Sources/`, `Tests/`,
  `MLXRunners` or `MLXLMCommon`; the wired budget defaults to 0 and only the
  `WiredMemoryManager` ticket API can set it. (`MLXFAST_GPU_WAKE` is the residency-adjacent
  thing they do use, plus `MLXFAST_SEEDGAP` residency touches off the engine thread,
  measured "~10–20 ms/GB after the gates' idle".)
- **Paged-KV write faults gate the whole forward build**: the latch is checked at every write
  and again *before and after* the forward build, and a fault retires the entire planned
  cohort (`EngineLoopV2.swift:2421-2480`). This is exactly why the model-side early
  submission plans are disabled whenever a paged cache is bound.
- **Warm everything that could compile inside a timed window**: `FewRowPackedMatmul.warm`
  compiles both row-block variants per weight shape at init "so no speculative round pays a
  Metal library build" (`Hadamard.swift:2698-2723`); the tensor-operand support is decided by
  running a tiny probe kernel with a scoped MLX error handler
  (`Qwen35MTPTopTwo.swift:571-604`); a load-time kernel trial gets a 9 s budget ("a cold
  toolchain compiles each body at 0.2–0.9 s", `:5163-5164`).
- The offline transform (`Sources/MLXFastTransform/`) re-emits the checkpoint into `weights/`;
  the runtime loader rejects any mismatch (`PrismHadamardCheckpointValidation`).
- Buffers: parameterised kernels with `ensureRowContiguous:
  !HadamardStridedInputs.enabled` so the strided-read variant is used only when the layout
  probe passes.

---

## 3. MoE / expert handling

**Absent — this track's target is a dense model.** `num_experts` is not in the pack config,
`Qwen35TextConfiguration.numExperts` defaults to 0 (`Qwen35.swift:130`), and the layer
chooses `Qwen3NextMLP` over `Qwen35SparseMoeBlock` at `numExperts > 0` (`Qwen35.swift:5028`).
There is no gather/scatter, no expert ordering, no router kernel, and no expert launch
counting in the trunk path. Anything in this repository about MoE is dead shared-tower code
for other model families (and the `Qwen35A3B*` files, which the port notes §6.2 record as
**unreachable and wrong for this pack**).

The nearest analogues, which *are* interesting next to our grouped expert GEMV:

| Their mechanism | Where | What it is |
|---|---|---|
| **Sibling stacking** | `Hadamard.swift:563-605`, `:1755-1805` | Several projections that share one rotated activation are row-concatenated byte-for-byte into one packed operand, one matmul, split back per sibling. Used for q/k/v and gate/up. "The stack has a wide output, so the core gives it the 32-row tensor kernel where a narrow sibling on its own (k, v, z) would have been split-K or, on this route, a 64-row gather tile. And one dispatch replaces one per sibling … with the rotated activation read once" (`:1762-1770`). |
| **Leading-rows submodule** | `Hadamard.swift:814-838` | The first `N` output rows of a packed projection as a module of their own (views of the same packed rows), used by the drafter to score a frequency-ranked vocabulary prefix. |
| **Fused top-2 in the head matmul** | `Qwen35MTPTopTwo.swift:6411-6470` | The verify head's logit stores are rewritten in place so the kernel emits each row's top two ids/values plus a butterfly merge, instead of a `[16, 248320]` logits tensor (`bonsai_head_top2_merge`, `:6472-6501`). |

No measured launch-count or tok/s number for any of these is published in the tree.

---

## 4. Attention, KV, speculation

### 4.1 Attention

- 16 full-attention layers (`full_attention_interval: 4`): 24 query heads, 4 KV heads,
  head dim 256, partial rotary 0.25, output gate (`attn_output_gate: true`). QK-norm + RoPE
  + the head transpose are **one kernel** (`bonsai_attn_prework`,
  `Qwen35.swift:6618-6800`, with the RMSNorm reduction, `precise::rsqrt`, the gate and the
  RoPE rotation written out inline), with `bonsai_attn_prework_lastq` specialising the
  decode's last-query row (`:6805`) and `bonsai_attn_qkrope_tables` writing the RoPE tables
  as part of the same pass (`:7034`).
- 48 gated-delta-net layers: `bonsai_gated_delta_chunk_{prep,scan,scan_fresh,scan_fresh_pf}`
  (`:1943`, `:1949`, `:2263`, `:2354`) and `bonsai_gdn_gated_norm_signed` (`:9551`); the
  recurrence state is FP32 (`mamba_ssm_dtype: float32`, `gdn_activation_layout: "grouped"`),
  16 key heads × 48 value heads × 128 with a conv kernel of 4.
  **A negative result worth keeping**: for the chunked verify scan, "Double-buffered staging
  (one barrier per chunk) and v / decay-factor prefetch were measured and lose (M4 Max: +28%
  and +0-7%), so they are not offered" (`Qwen35A3BTargetVerify.swift:4289-4294`). The same
  file records that a per-lane predecessor-count variant of another kernel lost because at
  that width "the stock sort amortizes".
- The Hadamard rotations around attention are fused into the attention-side producers rather
  than run as their own launch: `rotatedSigmoidGate` / `applyAfterSigmoidGate`
  (`Hadamard.swift:317-330`, `:1567-1620`), including a strided
  `[B,S,heads,headDim]` read that avoids materialising a head-concat copy
  (`HadamardStridedInputs`, `:2233`).
- **The decode path specifically**: `bonsai_attn_prework_kv` (defined in
  `Qwen35MTP.swift`, `Qwen35AttentionPreworkKV`) does Q/K RMSNorm + partial RoPE **and writes
  the KV row in place**, keys in FP32; MLX SDPA then runs the Metal `sdpa_vector` kernel at
  `qL = 1`. The GDN layers keep a conv state of `[1,3,10240]` in activation dtype and an SSM
  state of `[1,48,128,128]` in FP32, and run about **4 dispatches per token** (input
  projection, prework, `qwen35_gated_delta_v3` recurrence, gated output projection). The pair
  of facts that matter for us: the KV write is *inside* the prework kernel (no slice update),
  and the recurrence is one kernel with FP32 state.
- Their measured per-layer wins inside this machinery: 32-bit index/offset text in the prework
  launch "-8% on the kernel alone …, -3 us per call (about -4%) in the one-op census of the
  prompt forward", and "M4 Max, 48 dependent layers: **-20 us per layer**" for a fused prep
  launch (`Qwen35A3BTargetVerify.swift:1707`, `:1766`, `:3871-3875`); fusing the Hadamard sign
  into the elementwise chain measured "**SwiGLU 132 → 121 us, attention gate 97 → 88 us, GDN
  gated norm 95 → 70 us per launch**" on an M4 at 512 rows (`Qwen35.swift:7941-7942`).

### 4.2 KV cache

- The **scored** backend is contiguous: `kvBackend = contiguous`
  (`fixtures/bonsai2_27b_mlx_v1_track.json` / `docs/participant-contract.md` §5.3).
  `CBv2FullSequenceKV` is a per-sequence contiguous `[1, kvHeads, capacity, headDim]` K/V
  pair, sized `promptLength + 256` initially, grown by doubling, appended by slice update,
  with offset-only rollback; `CBv2ContiguousKVBackend` adds lazy rows and an admission
  reservation so concurrent same-step admissions cannot collectively overrun the byte budget
  (`SequenceKV/FullSequenceKV.swift`; `ContiguousKVBackend.swift:45-60`). The in-place kernel
  append (`MLXFAST_INPLACE_KV_APPEND`) is offered as the destination but is adopted only when
  a logged trial shows it ≥1% faster (`Qwen35MTP.swift:1195`, `:1516-1517`).
- A paged backend exists behind the same protocol
  (`ContinuousBatchingV2/Paged/...`, `PagedBackendBenchmark.swift`): 16-token pages with a
  window+8 ring. Its measured cost against contiguous: "paged trails contiguous at B == 1
  (**88.5 vs 101.8 tok/s**) while matching it at B ≥ 4", a paged verify round costs
  "~16.4 ms against contiguous ~14.2 ms (**+15.5%**)", and a page-size overshoot was "the whole
  of the measured **1.10x** per-sequence KV regression versus the contiguous backend at 10k
  context" (`PagedAttentionKernel.swift:202-223`; `PagedDecodeProfiler.swift:424-425`;
  `PagedKVPool.swift:492-494`).
- The fork's own research note documents the trap the paged backend hit and the fix
  direction: with multi-GiB shared slabs, a custom kernel that *reads* the slab pins the
  `Data` for the command buffer's lifetime, the next `slice_update` fails `is_donatable()` and
  falls back to a full-slab copy — "measured at ~**370 GiB of copies per decoded token** on
  GPT-OSS-20B", fixed by writing the slabs in place, "measured **25.5 s → 13.6 ms** per
  24-layer step at the 16 GiB pool" (`Vendor/mlx-swift-lm/docs/engine-v2/kernel-research.md:236-300`;
  `Paged/pagedattention.metal:94`; `PagedAttentionKernel.swift:18-25`).
- KV dtype: the backend adopts the dtype of the first appended K/V
  (`ContiguousKVBackend.swift:12-20`); the pack promotes norm outputs and KV rows to FP32
  (`Vendor/mlx-swift-lm/docs/bonsai2.md`, "Artifact contract").

### 4.3 Speculation

Two declarable decoders, both separate pinned exports, one armed per submission
(`mtp-head.manifest.json`, `docs/bonsai2-27b-port-notes.md` §3.4–3.5):

| | MTP head | DFlash 2 drafter |
|---|---|---|
| Export | `EigenLabs/Qwen3.8-27B-MTP-4bit` @ `329261c5` | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c` |
| Shape | `qwen3_5_mtp`, 1 layer full attention, 31 tensors | `DFlash2DraftModel`, 5 sliding layers, window 2048, 32 Q / 8 KV heads, d128, hidden 5120, intermediate 17408 |
| Precision | affine 4-bit group 64; **re-quantizable on load in memory** (`Qwen35MTP.swift`) | BF16 as published |
| Depth | 1–7 | 1–16 (block size − 1) |
| Proposal | one token per round | **a whole block in one forward** ("One `propose` call takes the last committed token followed by `depth` mask tokens, runs the five layers once, and returns `depth` draft tokens") |
| Cost per round | 238.9 MB of head weights | 3.85 GB per proposal, one forward regardless of depth |
| Target access | borrows the pack's packed `embed_tokens`/`lm_head` | reads the output hidden state of target layers `[5, 19, 33, 47, 61]` |

- Verification is **one target forward at M = depth + 1 rows**; because "a speculative round
  verifies its draft in one forward at M > 1, and MLX dispatches a different kernel there",
  the free-run tape can fork from the serial tape at a near-tie, so the track records a
  **per-depth oracle** (`mtp1..mtp7`, `dflash1..dflash16`) rather than demanding the serial
  tape (`docs/bonsai2-27b-port-notes.md` §5.1). The served DFlash depth is **15**, i.e. a
  16-row verify — which lands exactly on `tensorRouteMaximumNarrowRows = 16`, so **the whole
  speculative verify runs the ≤16-row packed matmul path of §2.2**, not the prompt path.
- The **token gate is a 10% budget, not equality**:
  `timed_token_tolerance_per_thousand: 100` with a near-tie exemption at
  `timed_token_near_tie_relative_gap: 0.005`
  (`fixtures/bonsai2_27b_mlx_v1_track.json:7-8`; `docs/participant-contract.md` §5.4). This is
  the single most important structural difference from our engine: this track *permits*
  numerics to move. That permission is what pays for the FP16 activation reads, the int8
  activation quantization and the tensor-unit accumulation orders throughout §1–§2.
- **Accept rate: unmeasured for this pack**, stated three times (`docs/bonsai2-27b-port-notes.md`
  §7; `README.md` "The DFlash 2 accept rate is not measured";
  `Vendor/mlx-swift-lm/docs/bonsai2.md`). Both drafters were trained against a BF16 Qwen3.8-27B
  trunk; this trunk is the same architecture at 2 bits with a folded transform. The only
  acceptance figure anywhere in the tree is an inherited MTP-head A/B from 2026-08-13 on an
  M4 Max — "K=256 acceptance **0.79 → 0.58**, 95.5 tok/s (full head: 116.5); K=2048
  acceptance 0.68, 110.3 tok/s; K=16384 acceptance 0.77, 115.2 tok/s", against "the ~1.3 ms
  full **286 MB** 4-bit head read it replaces" (`Qwen35MTP.swift:850-858`) — which is a
  *negative* result for shortlisting, not an acceptance rate for this pack.
- Also: the batch path **deliberately disables speculation** (`submit_batch` falls back to
  the serial path when a request asks for the draft), and the scored stream count is 1.

---

## 5. Measured numbers, and the hardware they were measured on

### 5.1 Hardware

| Box | Used for |
|---|---|
| **M5 Max** | the tensor-matmul numbers, the head rates, the tensor-unit TF/s, the prompt/verify in-situ percentages (`Qwen35MTPTopTwo.swift`, `Hadamard.swift:1444`) |
| **M5 Pro** | the DFlash 2 drafter weight-stream probe (`DFlash2Draft.swift:3191-3286`) |
| **M4 Max, 40-core GPU, 128 GB** | all numbers in the vendored fork's own kernel research and the GDN scan-form negatives (`Vendor/mlx-swift-lm/docs/engine-v2/kernel-research.md:28-30`; `Qwen35A3BTargetVerify.swift:4292`) |
| Ranked box | unprovisioned; the stub workflow targets `[self-hosted, macOS, bonsai2-27b-mlx-v1]` |

The fork's hardware table also gives the fleet facts we care about:
`simdgroup_matrix` present from M1 (Apple7) onward, `mpp` tensor ops only on Apple10/M5;
32 KB threadgroup-memory cap everywhere; M4 Max 546 GB/s DRAM at ~80% achievable
(`kernel-research.md:38-66`). Our box is an M4 Max, so **only the `simdgroup_matrix` family of
their kernels is assumed to run on our hardware**; whether
`mpp::tensor_ops::matmul2d` / `uint2b_format` exist on M4 is **[UNVERIFIED]** — the engine
*runs a probe at init and falls back* (`Qwen35MTPTopTwo.swift:571-604`), and its preferred
form was chosen partly because it "needs no packed format" (`:664-668`).

### 5.2 Published measurements (all from in-source comments; none reproducible from the repo)

**Tensor unit rates** — "int8 × int8 → int32 ops … run at the tensor unit's int8 rate (about
twice its FP16 rate): **116 against 58 TF/s on an M5 Max**" (`Qwen35MTPTopTwo.swift:660-663`).

**Decode-window (whole-step) effects**

| Claim | Where |
|---|---|
| int8-staged verify form beats the native 2-bit operand with FP16 activation by **−3%**, and the record's verify kernels by **−5%** | `:616-618` |
| the vocabulary head on the tensor route: **1.2% faster on the decode window** | `:675` |
| excluding `gate\|up` (n = 34816) from the verify route: **3% slower in situ** | `:671-674` |
| verify tile `TN = 32` vs `64`: **1.5% faster on the decode window** (the reverse of the isolated preference) | `:223-226` |
| factored epilogue (`s·(as·C − rb)` instead of `as·(s·C) + (−s)·rb`): **1–1.5% on the prompt-width matmuls** | `:662-666` |
| super-tile order on: the box read the timed prefill **+1.45% slower** on every pair, against a 0.05% spread (rejected) | `:3053-3054` |
| head kernel `CB = 1` vs `4`: **460 vs 482 µs standalone** on the drafter's 16 × 100 352 head (~4%) | `:269-272` |
| verifier's own window on a GDN scan form: **5.9% longer** than its parent's | `Qwen35.swift:1847` |

**Bandwidth**

| Claim | Where |
|---|---|
| the verify head body's rows stream "about **544 GB/s on M5 Max**, against about **305** for the FP16-read head kernel" | `Hadamard.swift:1444-1446` |
| on the M5 Pro, "every class streams at the device's read rate, and the projections are ~**95%** of a block forward's GPU time" | `DFlash2Draft.swift:3191-3194` |

**DFlash 2 drafter lossless 12-bit weight copies** (sign+mantissa byte + a 4-bit exponent
offset per weight, escapes in a side list; ~99.98% of BF16 exponent fields fall in 15
consecutive values; **2.49% of weights escape**; ~**11.23 bits/weight** with side data vs
12.19 for the 12-bit copy) — chained launches over the real weights, `[32,64]` tiles,
**packed vs stored: gate|up −20%, down_proj −3%, fc +1%, q|k|v +2%, o_proj +7%, tap +63%**;
with `[16,64]` tiles: **gate|up −20%, down_proj −20%, fc −19%, q|k|v −17%, o_proj −17%, tap
−5%** — which is why "every eligible weight is packed"
(`DFlash2Draft.swift:3274-3286`, `:3944-3945`).

**Host-side / scheduling numbers** (all M4 Max unless stated)

| Claim | Where |
|---|---|
| the engine thread lands on an efficiency cluster by default: "The verify build then takes **11.4 ms instead of 3.8 ms**, and the committed recurrent state takes 5.3 ms instead of 3.1 ms" | `CBv2EngineWorkInterval.swift:7-13` |
| after the work-interval fix: "the verify build took **3.1 ms in every round**, and GPU idle per round fell from **2.0 ms to 0.6 ms** with unchanged GPU busy time" | `CBv2EngineWorkInterval.swift:22-23` |
| prompt-side variant: "the seed step's engine CPU ran 78% on the efficiency cluster, **59 ms instead of 82 ms**" | `CBv2EngineWorkInterval.swift:37-41` |
| verify slicing: "the host builds the verify at about **0.1 ms per layer**, so the 56 layers behind the first boundary outlast the GPU's **~3 ms** on the first 8 and the GPU idled **~1.3 ms per round** in between" | `Qwen35.swift:305-313` |
| layer-0 split: "on an M4 Max the first layer's buffer waited **7.4 ms** after its commit with the GPU awake (**1.1 ms** in the timed prefill, which reuses its warm-up's buffers)" | `Qwen35.swift:2850-2851` |
| prompt-embedding gather: "M4 Max, measured per cold start: **4.8–8.5 ms of idle GPU** before the gather" (0.7 MB for 512 tokens) | `Qwen35.swift:5341-5343` |
| GPU wake: "on an M4 Max after a 20 s idle, **13–15 ms** after the commit (0.5 ms warm)"; "~4 ms of assembly and graph building" | `ContiguousKVBackend.swift:33-35` |
| residency wiring stalls: "**~10–20 ms/GB** after the gates' idle, ~3.8 GB here" | `ContiguousKVBackend.swift:405` |
| a chunked prefill is not bit-identical: "a **~5e-5** logit delta in float32 … an argmax flip on the first close margin" | `BenchWorker.swift:703-712` |
| command-buffer budget: "One `asyncEval` of the whole proposal holds this thread in MLX's in-flight cap (at most **10 command buffers**; the proposal is ~30)" and "~11 of the proposal's ~32 command buffers" | `EngineLoopV2+MTPFinalize.swift:488-493`, `:81-82` |
| DFlash first-layer-ahead: "commits the block's first command buffer **69 us sooner** after the readback (median of 39 rounds)" | `DFlash2Draft.swift:7257-7259` |
| early drafter submission missing its warm: "that round's early drafter submission took **143–208 ms instead of 36–41 ms** and the round **464–534 ms instead of ~358 ms**" | `Qwen35DFlash2Assistant.swift:304-307` |
| DFlash pack12 prefetch effect: "ran in **89.2 ms against 89.4 ms** for the chains alone" | `DFlash2Draft.swift:6366-6370` |
| a parallel walk vs serial: "here all 3,600 run at once (about **80 us → 13 us per proposal**)" | `DFlash2Draft.swift:5631-5632` |
| verify slicing as a batch: "the record plus this batching alone, **4.27–4.28 against the record's 4.23–4.26 ms per token**" — basis unclear (**[UNVERIFIED]**: a 27B step cannot read 7.3 GB in 4.3 ms, so this is not this model's decode step) | `Qwen35.swift:981-982` |

**Standalone kernel timings**

| Claim | Where |
|---|---|
| the verify head: "v0 runs the 16 × 248 320 head in **761 us**, k32pd1 in **621** (the words' plain read takes 568)" | `Qwen35MTPTopTwo.swift:5035-5036` |
| per-shape verify launches, k32pd1 vs k32pd2 on M5 Max: "down **52.2 vs 54.1 us**, gate\|up 100.7 vs 104.5, qkv\|z 56.2 vs 57.7, attn 49.7 vs 51.6, o 26.0 vs 27.5" | `:5027-5031` |
| head logits store: "[16, 248320] FP32 logits, **15.9 MB a round**" | `:6103-6104` |
| GDN v3 row layout: "on the M5 Max this layout runs the 512-row prefill recurrence **2.4x faster** with a ~1e-7 relative deviation of y and state" | `Qwen35.swift:634` |
| GDN chunked scan forms (M4 Max): "record 324 us, `kt/4` 300 us (−8%), `kt/8` 289 us (−11%)" at 32 cores; "record 264 us, `kt/4` 250 us (−6%), `kt/8` 287 us (+9%)" per-core at 40-core part-load | `Qwen35A3BTargetVerify.swift:4281-4284` |
| fused prep launch: "M4 Max, 48 dependent layers: **−20 us per layer**"; prework index text "135 → 108 us" | `:1766`, `:1707` |

**Calibration constants (pinned, and *not* the ranked denominator)**

`Sources/MLXFastCore/Constants.swift:255-256`:
`officialBaselinePrefillSecondsPerToken = 0.0003276219582519531` (0.328 ms/token) and
`officialBaselineDecodeSecondsPerToken = 0.012374741210937498` (12.375 ms/token ≈ 80.8 tok/s),
labelled "a cached Gemma 4 26B A4B 2026-08-25 calibration, NOT the ranked denominator". The
per-box calibration behind the bands is four cool-gated runs with
**CV 0.2712% (prefill) / 0.0937% (decode)** and spreads 0.581% / 0.2084% (`:241-244`), from
which the bands `prefillBandUp/Down = 0.03`, `decodeBandUp = 0.01`, `decodeBandDown = 0.025`
are derived (`:157-188`, `:209-219`). Both local modes and the ranked pairs are gated by a
quiescence gate (1-minute load average < 2.0, GPU utilisation < 10%, polled every 15 s,
refused after 900 s) and a cool gate (GPU ≤ 40 °C, refused after 900 s; a stuck sensor is a
refusal) — `docs/participant-contract.md:707-710`, `tools/ranked-box-preflight.sh:157,178-196`.

**Assertions that are not measurements** (separate them out): "sixteen rows cost about what
eight do" (`Hadamard.swift:2674-2677`, an argument from the fragment geometry); "the core
splits K whenever the 32×32 tile count is at most 256" (`:737-739`, a reading of the core);
"1.5% … the trial's picks can be slower than such a base" (`:3787-3790`, a policy that encodes
a prior reading, not a fresh measurement).

### 5.3 What is *not* measured anywhere in this tree

- No end-to-end tok/s, ms/step or GB/s for the track, and no scored result: the fixture is
  armed (`official_scoring_enabled: true`) but no runner is registered and there is no in-tree
  receipt of a ranked run.
- No accept rate for either drafter on this pack (the only acceptance figure is the inherited
  MTP-shortlist A/B of §4.3).
- No compile-or-not evidence for the HEAD tree: the port notes say the port was never built,
  and later commits are bot "Accept/Validate submission" merges with no receipts
  **[UNVERIFIED]**.
- Which zoo member is finally installed: the trial decides at load time on the box, so the
  tree's stated defaults are the *starting point*, not the outcome **[UNVERIFIED]**. In
  particular, `statistics` such as "k32pd1 ahead of k32pd2 on all five shapes on M5 Max" are
  the *record* the trial starts from, not a post-trial result.
- The ±% in-situ numbers' session discipline (the "1.5%" figures give no spread; the
  super-tile rejection is the only one that quotes a spread, 0.05%; the trial's own constants
  imply a floor of ~1–1.5% adoptability).
- The in-flight-command-buffer cap of 10 is asserted in comments only; no enforcement was
  found in the vendored C++/Metal tree **[UNVERIFIED]**.

---

## 6. Ranked — transferable to `exl3-server`

Our binding constraint (`docs/perf.md`, "Fourth pass"): the trellis GEMV is **issue-bound**
at the machine's full issue width — ~110 warp instructions per (16×16 tile, k-tile), 2.4e9
tiles/s, so **bytes-per-tile decides GB/s**: K4 (128 B) ≈ 300 GB/s, K3 (96 B) ≈ 215, K2
(64 B) ≈ 134–153. 22 of our 43 MoE layers are K2. Hard gate: **the single-sequence path must
stay bit-identical, and the trellis byte-granularity contract must not be broken**; the
k-split count is part of the f32 association and may not change.

| # | Item | What it changes for us | Expected effect | Risk | Bit-identical? |
|---|---|---|---|---|---|
| **1** | **Cut the per-weight instructions: pre-expanded operand planes instead of per-weight decode+FMA.** Their staged form costs 3 shifts + 4 ANDs + one 16-byte `uint4` store **per 16 weights** (`Qwen35MTPTopTwo.swift:2836-2856`), and their MACs are on a matrix/tensor unit, not the lanes. | Our hot loop retires 0.43 warp-instr/weight (≈13.7 lane-slots): 8 extracts + 8 mul1 hashes + 8 half-FMAs + 8 FMAs + 4 loads + window shift + addressing, per 256 weights. Their decode is ≈27× cheaper in issued non-tensor instructions *and* the MAC is off the pipe. | If the per-weight hash+extract collapses into a wide bit-parallel op or a table lookup, the K2/K3 classes stop paying 1.7 instr/byte: 134 → toward the DRAM-bound 400–450 GB/s. Bounded by our own 546 GB/s; the decode+FMA is ~60% of our K2 byte cost. | **High on correctness, medium on payoff.** Our codes are a **16-bit sliding-window trellis codeword with a mul1 hash**, not a 2-bit field — a shift/AND plane split does *not* reproduce it; the honest version is an offline decode table (or a re-expressed codeword layout) built and **byte-checked at load**. A table is cache-hostile; the reference's own notes "rule out ALU trims in the MUL1 decode" (`docs/perf.md`, "Fourth pass"). If the *tensor-unit* half is taken, its accumulation order is not ours. | **Only if** the decoded value per weight is unchanged **and** the accumulation order is unchanged: a pure re-layout + table building is bit-exact; moving the MAC to a simdgroup/tensor unit is **not** and is blocked by our gate. |
| **2** | **Fire `asyncEval` at layer boundaries inside the forward, not only once per step.** Their ladder: `Qwen35TrunkSubmission` (`Qwen35.swift:226-260`, `:365-375`) submits the hidden state after layers **8, 24 and 40** on the verify path, stride 4 on the prompt path, and `[1,2,4,8,16,32,48]` on the fused prompt path; a layer-0 split and a prompt-embedding pre-submission are extra rungs. | Our decode step builds its whole graph on the host; our own instrumented loop shows ~360 stage barriers at ~0.3 ms each of host time. Their ladder exists because "the host builds the verify at about **0.1 ms per layer**, so the 56 layers behind the first boundary outlast the GPU's ~3 ms on the first 8" (`:305-313`). | From their box: GPU idle per round fell with the slices in place; the counter-lesson is that "slices every 2 layers had lengthened the ranked window" (`:245-246`), i.e. the rung set matters and must be swept. mlxfast-challenge independently measured **+9.7%** for the same mechanism on Laguna. For us the win is bounded by our host build time per step, which must be measured first — our GPU has no idle slots to fill today (the GEMV is at full issue width, not waiting on DRAM). | Low. Pure scheduling; the failure mode is choosing rungs that add command-buffer round trips. | **Yes, by construction** — "only command-buffer boundaries move, so every value is bit-identical" (`:245`). The one scheduling change our gate does not touch. |
| **3** | **Put the graph-building thread on a performance core.** `CBv2EngineWorkInterval` sets an audio work interval per engine step precisely because the OS otherwise schedules the engine thread on an efficiency cluster. | Our step is `mx::eval`-bound at 0.11–0.2 ms per call plus real host graph-build time; this is the same failure mode, purely at the scheduler's discretion. | One of the largest host-side numbers in their tree: "The verify build then takes **11.4 ms instead of 3.8 ms**"; after the fix "the verify build took 3.1 ms in every round, and GPU idle per round fell from **2.0 ms to 0.6 ms** with unchanged GPU busy time" (`CBv2EngineWorkInterval.swift:7-23`). Prompt variant: "**59 ms instead of 82 ms**" (`:37-41`). | Trivial: an API call and a policy. | Yes (host-side QoS only). |
| **4** | **Do not chunk the prefill on the bit-identical path — and treat it as a gate, not a preference.** They chunk by default (512) but the bench worker forces `prefillChunkSize = max(512, seedLength)` because "a chunked prefill is **not bit-identical** to a single forward: … a ~5e-5 logit delta in float32 … an argmax flip on the first close margin" (`BenchWorker.swift:703-714`). | We already keep the single-sequence path unchunked; this is the *evidence* that the rule is load-bearing, and a warning that any future chunked/streamed prefill must be excluded from the bit-identity path rather than merely "close". | Prevents a whole class of silent gate failures; no speed claim. | Low. | N/A — it is a rule that *protects* bit-identity. |
| **5** | **Write the KV row inside the QK-prework kernel instead of as separate slice updates.** `bonsai_attn_prework_kv` does Q/K RMSNorm + partial RoPE **and** the in-place KV row write, keys FP32; the in-place append is adopted only when a logged trial shows it ≥1% faster (`Qwen35MTP.swift:1195`, `:1516-1517`; `MLXFAST_INPLACE_KV_APPEND`). | Our MLA/KDA cache append is a concat at a common index plus the attention-side ops. This folds the write into the kernel that already computes the row, deleting two dispatches per attention layer. | Their baseline for the alternative (`MLXFAST_INPLACE_KV_APPEND` off) is two slice updates per attention layer; the adoption threshold is 1%. On our side the analogous per-layer machinery is ~58 ms/token across 45 layers, so removing two dispatches per layer is in the same class as our measured `hc_prepare` win. | Low–medium: the write must be made race-free against the row's readers in the same dispatch (their design does this by always loading the newest row from the kernel's own inputs — "bit-identical bytes", `kernel-research.md:250-262`). | Yes if the bytes written are the bytes the slice update would have written; verify by hash, as they do. |
| **6** | **Fuse our two Hadamards into the trellis GEMV, and fold the producer into the rotation.** They give a complete, citable recipe: one kernel, grid `(64, rows·width/1024)`, 64 threads, threadgroup `buf[1024]`, the producer (`silu(a)·b`, `a·sigmoid(b)`, or per-head RMSNorm+silu) formed **in the read**, then radix-16 + radix-4 butterflies, the store scale `0.03125` and one cast (`Hadamard.swift:2382-2503`); "it must compute exactly what the op chain computes (FP32 sign multiply, the FP32 block transform with the stock kernel's butterfly order and scale, then one cast)" (`:53-62`). They also choose the store dtype so the consumer never casts again (`fusedInputStoreDType`, `:1481-1503`). | Our `trellis_linear` is 3 launches (input Hadamard, GEMV, output Hadamard) × ~1450 linears/token. This removes the two Hadamard launches and one activation round trip per linear. | Our own estimate: ~1 of 3 dispatches per linear, i.e. ~2 µs × 1450 ≈ **3 ms/token**, up from the smaller win if only the Hadamard is fused. Their analogue (fusing a 4-stage chain into one dispatch, ×30 layers) is their largest structural saving. | Medium: our Hadamard may need to stay fp16-exact; their kernel is FP32-internal, and the producer fusion (SwiGLU / gate / RMSNorm) is real work. | **Yes**, if we reuse our butterfly order and scale and keep the FP32 sign multiply — exactly the discipline their doc comment pins. The producer sub-item is bit-exact only for the same fp32 expression order (`Hadamard.swift:299-302` spells out "each product rounded to FP32 in that order"). |
| **7** | **Re-lay the packed weights and the per-group constants into the kernel's own lane order at load.** Their `TILED` copy turns "32 B from each of 32 rows" into "1 KB in one run" (`:836-849`); `MPERM` makes a lane's four rows adjacent so its per-group scales and code sums are **one `float4`** instead of eight scalar loads (`:101-107`, `:700-712`); the folded code-sum plane `uT = −128·s·colsum` is built once (`:5474-5476`). | Our `trellis_pack` already materialises an aligned engine-owned copy. This generalises it: tile-major/consumer-major order keyed to the lane walk, and per-member constant planes in the same order. Directly attacks the part of our 110 instructions that is address math and the 4 packed-word loads. | Our doc attributes **0.071 ms of 0.162 ms** at K4 to the loop skeleton (no memory) and 0.052 ms to the packed-word reads — the address/load portion is real. Their measured payoff for the analogous change is the in-situ 1–3% class, plus it is what makes the `uint4` loads possible. | Low on correctness, **medium on memory**: their constant-reuse feature retains an extra **~1.60 GB** (`Vendor/mlx-swift-lm/docs/bonsai2.md`); a tiled weight copy doubles storage for those tensors unless it *replaces* the raw buffer. | **Yes** — copies and index maps, no arithmetic. Their own wording: "it moves bytes and changes no bit" (`docs/bonsai2-27b-port-notes.md` §4.1). |
| **8** | **Prove-derived per-group constants so the epilogue loses work.** `NEG`: the offset *is* the negated scale, proven by XOR-comparing FP16 bit patterns (`.view(.uint16) ^ 0x8000`) (`Hadamard.swift:508-518`) ⇒ no offset load; signed codes ⇒ the `128·colsum` term and its two 16-byte loads per lane per group disappear (`:658-671`). | Our GEMV carries per-member scale vectors and per-block constants. Any constant that is provably derivable or provably duplicated can be loaded once per group instead of per element, and any term provably exact in fp32 can be folded. | Their measured effect: **1–1.5% on prompt-width matmuls** for one fewer FMA, and it is part of why the int8-staged form beats the native 2-bit form by 3%. | Low–medium: the "exact in FP32" arguments are specific to their algebra and precision; we would have to re-prove each one for our f16/f32 mix, or fall back. | **Only if re-proved per constant.** Where the algebra holds exactly it is bit-identical (they argue: "each round once to the same number"); where it does not, it is not. |
| **9** | **Adopt kernel variants on the scored window, in situ, with a margin.** Their trial: interleaved rounds per candidate, ≥4 kept rounds, drop rounds >1.5× the candidate's median, adopt only above a 1.5% margin, quantify per-shape factors with an `OA(18,3^5)` array and install the best body per shape (`Qwen35MTPTopTwo.swift:3730-3900`), and they document the case where the isolated winner lost in situ (3% slower) while another won (1.2% faster) (`:671-675`). | Our box's session-to-session spread is ±30%, so a 5–10% change is unresolvable end-to-end (`docs/perf.md`, "Third pass"); we currently fall back to component-level benches. Their rotation + per-shape OA estimation is exactly the instrument for our many-shape kernel matrix (K2/K3/K4/K6 × rows × member counts). | Methodology only: prevents landing noise and standalone-only wins. Our `docs/perf.md` already records one such trap (nt=8/16 measured 476 → 342/218 GB/s standalone but was adopted on a *rule*). | Low. It is a measurement harness, not arithmetic; it must never run inside a served request (they run theirs in a deferred load warm). | N/A (no numerical change). |
| **10** | **Bitwise self-test every fused or side-copy route at load, and refuse silently-degenerate ones.** `HadamardEmbeddingGather.verified` compares 4 id patterns (including negative ids and the vocab edges) bitwise before the one-launch gather is used, and falls back otherwise (`Hadamard.swift:2203-2270`); `DFlash2PackedWeights.prepare` checks *every* rebuilt weight and the products of random 16/32-row inputs against the live kernel (`DFlash2Draft.swift:3207-3219`); the tensor-operand support is decided by running a probe kernel (`Qwen35MTPTopTwo.swift:571-604`); the zoo gates candidates on `exact(kernel, .float32)` (`:5320-5345`). | Our hard gate needs exactly this. Our own `docs/perf.md` records a gate that "passed while the grouped MoE's member order was wrong" because the fixture was degenerate and decoded to NaN. | Makes items 1–4 landable; costs a one-off load-time pass. Removes the class of failure where a route is silently wrong but type-checks (their port notes §3.2: `HadamardQuantizedLinear IS a QuantizedLinear`, so a cast succeeds and skips the transform). | Low. Keep the raw path as the authority and the self-test on the real tensors. | Yes — it *asserts* bit-identity; it does not change arithmetic. |
| **11** | **Lossless narrow resident copies for the non-trellis F16 tensors.** Their 12-bit/11-bit copy: census the stream, prove a bounded exponent window + an escape list reproduces every element bit-for-bit, then rebuild the operand in registers (`DFlash2Draft.swift:3186-3300`); adopted per shape with a `worthPacking` predicate. | Our trellis bytes are already 2–6 bits, so nothing to gain there — but our *F16/f32* tensors are: the router's 288×4096 F16 (our doc measured 14 MB of re-cast traffic per layer), the `hc_prepare` stream (a (24, 16384) f32 matrix = 1.57 MB/site read at ~8 GB/s), `f_a`/`g_a`/`b_proj`, and the scale vectors. | Their measured per-weight wins are −17% to −20% on the shapes that paid (16-row tiles) and they explicitly *decline* shapes where it loses (+63% for the tap at 32 rows). The transferable number is the method, not the ratio. | Medium: extra resident bytes, a load-time verifier, and it only pays for weight-stream-bound shapes — so it must be per-shape and de-adoptable. | **Yes, if** the copy is proven bit-exact per element at load (their contract), with the raw tensor kept as the authority. |
| **12** | **Micro-idioms from the hot loops.** (a) One vector load per lane per group (`uint4` words, `half4`/`float4` constants) and `float4`/`half4` output stores (`Qwen35MTPTopTwo.swift:2839-2848`, `:120-131`); (b) re-use the K-loop's staging buffers as the reduction scratch — "16 KB of threadgroup memory in all, two threadgroups per core" (`:882-895`); (c) a resident cached zero buffer per pad shape instead of a fresh zeros kernel per projection (`Hadamard.swift:765-793`); (d) decide the vector/matrix path by asking what the dispatcher would pick, then force the better one by padding rows (`:840-862`); (e) `CBv2GPUWake`: one committed scalar add to start the GPU's command-buffer wake before the host's ~4 ms of assembly (measured 13–15 ms late after a 20 s idle on M4 Max vs 0.5 ms warm) (`ContiguousKVBackend.swift:32-40`). | (a),(b) shave address math and threadgroup traffic from our hot loop; (c) kills a per-projection tiny dispatch; (d) is the general shape-forcing test; (e) is free at process start. | Small individually, additive: our own micro-list already measured −20% for vectorised aligned loads and −9 ms/token for issuing four loads before their dependent FMAs in `hc_prepare` (`docs/perf.md`). | Low. (d) and (e) are host-side only; (a)–(c) are per-kernel and must be re-benched. | Yes for all five (copies, scheduling, and a zero-filled buffer that is never written). |
| **13** | **Stack more members per launch.** Their sibling stack row-concatenates projections that share one rotated activation into one operand and splits the wide result (`Hadamard.swift:563-605`, `:1755-1805`); stacking makes a narrow sibling wider so the dispatcher picks a better kernel ("The stack has a wide output, so the core gives it the 32-row tensor kernel where a narrow sibling on its own would have been split-K"). | We already have `trellis_linear_group` for the 8 routed experts and the merged gate\|up. The transferable part is *why* they stack: it changes the dispatched kernel class, not just the launch count. | Bounded: our doc measures the whole small-projection set at ~1.7 ms of a ~150 ms step, 70% of which is launch overhead, and the grouped path already exists. | Low–medium (we hit Metal's 31-buffer limit at K·B members per launch, `docs/perf.md` "Fifth pass"). | Yes — our grouped form "keeps its own scale vectors, split geometry and k-order" and member order is verified by hash (`moebench` identity line). |
| **14** | **Two-launch-free concepts worth borrowing for the *verify* path only.** `FewRowPackedMatmul` (`Hadamard.swift:2680-2856`): 4–16 rows, `simdgroup_float8x8`, lane decodes two codes into a `half2` with the mul1 bitcast idiom (`as_type<half2>(bitsv \| 0x64006400) − half2(1024.0h)`), every fragment multiplied against all rows ("sixteen rows cost about what eight do"), split-K chosen by an `occupancy = 2048` rule, FP32 accumulation, one split-sum at the end. Uses only `simdgroup_matrix`, i.e. M1+ hardware. | Our MTP verify step is exactly 4 rows, and our k-split cannot change — but this shows a shape in which 4–16 rows share one weight read **with FP32 accumulation and no change to the reduction association**. | Their argument is geometric (fragment reuse), not measured: cost per row falls with row count. If we could reuse a 4-row verify's weight read across its rows, the verify step's bytes-per-weight-*read* fall 4× — this is the one place our k-split contract is not in the way (rows, not k). | Medium–high: our decode carries `acc[mb][nt][8]` with `mb = min(rows,4)` already; the open question is whether `simdgroup_matrix` fragments can carry our trellis decode without changing the f32 tree. Their own lesson says the isolated win may not survive in situ. | **Only if the f32 accumulation order per row is preserved.** Their split is chosen so "the integer sum of a group is exact under any split"; ours is not an integer sum, so a new split changes the association → **not bit-identical** unless the k-order is kept and only the row blocking changes. |

**Ordering rationale.** 1 is the only item that attacks our binding constraint head-on
(instructions per weight) and is therefore first despite the highest risk. 2–5 are the
scheduling layer: they are bit-exact by construction, were measured on an **M4 Max** (the box
we use), and their biggest single number is a host-side effect that our own `docs/perf.md`
already suspects (host build time, `mx::eval` cost). 6–8 are the kernel changes that are
bit-exact by construction (one fusion, one re-layout, one provable epilogue trim). 9 and 10
are the measurement discipline that makes 1–8 landable on a box whose session spread is ±30%.
11–14 are shape- and idiom-level, worth keeping but bounded by our own measurements. Note the
dependency: 9 (in-situ adoption) is what should gate 1, because 1 is the one item that can
silently change arithmetic.

---

## 7. Does NOT transfer (with reasons)

- **The tensor-unit operand itself (`mpp::tensor_ops::matmul2d` with `uint2b_format`).** It is
  a *hardware unit that ingests sub-byte weight operands*, i.e. the one thing that makes
  their decode free. Two blockers: (a) the vendored fork's own hardware table lists
  `mpp` tensor ops only for M5/Apple10 (`kernel-research.md:44-54`), and the engine probes at
  init rather than assuming — on our M4 Max it is **[UNVERIFIED]** whether the op exists at
  all; (b) it cannot carry our decode — our operands are trellis codewords requiring a mul1
  hash, not 2-bit affine fields, so a "raw packed operand" route has nothing to hand the unit.
- **Their numeric envelope.** FP16 activation reads (`matrixRouteInputDType`,
  `Hadamard.swift:722-729`: "a tower projection reads its rotated activation in FP16"),
  int8 symmetric quantisation of the activation per 128-group
  (`round(x/scale)+128`, `:77-82`), and tensor-unit accumulation orders. Their own text prices
  the FP16 read as a change "rather than the product precision" and relies on the track's
  **10% token budget** to absorb it (`docs/participant-contract.md` §5.4). Our gate is
  bit-identity on the serial path — every one of these routes is out.
- **The folded-Hadamard weight format itself.** Rotating the activation instead of unfolding
  the weight is their central structural idea, but it is a property of the *checkpoint*
  (402 packed modules with `signs`), and their own contract forbids re-quantising the target
  (`README.md`, "The target quantization is frozen"). Our EXL3 checkpoints are frozen too, and
  our byte-granularity contract fixes the stream, so we can copy the *fusion* (item 6) but not
  the *fold*.
- **Sibling stacking as a substitute for expert grouping.** Their stack groups projections
  that share an input *within one layer*; it is not expert selection, has no router, no
  gather, no per-expert scale vector. Our `trellis_linear_group` already does the analogous
  thing and our `docs/perf.md` already knows the real MoE lever is *expert dedup across rows*,
  which has no counterpart here.
- **All MoE machinery.** There is none in the trunk path (§3). Their `Qwen35SparseMoeBlock`
  is dead code for this pack, and the `Qwen35A3B*` exact-verify lane is documented as
  unreachable *and wrong for this pack* (`docs/bonsai2-27b-port-notes.md` §6.2) — a trap, not
  a reference.
- **The absolute rates.** M5 Max (their head kernel 544 GB/s, tensor unit 116/58 TF/s) vs our
  M4 Max (546 GB/s, and no `simdgroup_matrix`→tensor-op equivalence assumed). Only relative
  mechanism ordering transfers; their 1.2%/3%/5% in-situ numbers were taken on a different
  chip with a different model.
- **Speculative decoding as a lever for us.** Their drafters only *propose* and their accept
  rates are unmeasured; the batch path disables speculation entirely; the depth and decoder
  are free levers in a single-stream arena. We already have MTP with measured +30–50%, and
  their numbers cannot bound it.
- **The MX/MLX dispatch machinery**: `get_qmv_batch_limit`'s per-arch thresholds
  (`quantized.cpp:89-120`), `max_ops_per_buffer` (`device.cpp:657-678`) and `quantizedMM`
  routing. We write our own Metal kernels through `src/trellis_gemm.cpp` and do not get to
  choose a kernel by changing the row count, so the 13-row padding trick (§2.3) has no
  analogue for us — only the *question it answers* ("which kernel will this shape get?") does.
- **`compile()` / compiled-decode and `mlx::core::fast`**: not this engine's headline either
  (this fork uses hand-written `MLXFast.metalKernel` JIT for everything fused), and
  `mlx::core::fast` is not exposed in our install (`docs/perf.md`).
- **The "one `asyncEval` per step" *engine* cadence as a rule.** Their engine submits the
  whole mixed step in one graph (`EngineLoopV2.swift:5-6`, `:3437`) *and* the model fires a
  layer ladder inside the forward (§2.5) — so both cadences coexist and neither is the whole
  story. Note this engine's counter-lesson directly: "slices every 2 layers had lengthened the
  ranked window" (`Qwen35.swift:245-246`), i.e. more rungs is not monotonically better. For us
  the relevant facts remain the measured costs we already have (`mx::eval` ≈ 0.11–0.2 ms,
  custom kernel dispatch 4–10 µs); the *rung set* has to be swept on our box.
- **The paged-KV backend and its slab-donation trap** (`PagedAttentionKernel.swift`,
  `PagedKVPool.swift`, `kernel-research.md:236-300`). It is a solution to MLX's
  `Data`-retention/donation semantics — 370 GiB of copies per token, fixed by writing slabs in
  place with fences. We own our cache buffers and do not version them through functional
  updates, so there is nothing to donate and nothing to fix. (The *measurement* — 25.5 s →
  13.6 ms — is a warning about what MLX's lazy graph does to a shared multi-GiB buffer, not a
  mechanism for us.)
- **Their measurement values as targets.** No scored run is recorded in this repository and
  no ranked runner is registered; the port was never compiled (port notes §7). Treat every
  number above as a claim on hardware we do not share, measured by someone else.

---

## 8. Open questions / not verified

**Documentation that contradicts the code at this commit** (the fixture is the authority):

| Fact | Docs say | Tree says |
|---|---|---|
| Arm state | `README.md:768-774`, `docs/participant-contract.md:682-683`, port notes §5/§7: scoring off, pool empty, sentinel | fixture: `official_scoring_enabled: true`, 11 pool tapes, live goldens `pool-9/10/11`, pinned hidden golden |
| Reference commit | `README.md:599`, `participant-contract.md:545`: `fffde01` | `fixtures/…track.json:135`: `88cf569e885019ffe637192efcbda4aaf2c32f1c` |
| Speedup floor | `TASK.md:208`: 0.90 | fixture:22-23 and `README.md:736-743`: 0.95 |
| Checkpoint size | `docs/ci-coverage.md:23`: "about 17.3 GiB across 4 safetensors shards" | fixture:185-186: 1 shard, 8,595,477,990 B; `setup.sh:46-53`: ~8.1 GiB over 19 records |

- **[UNVERIFIED]** Whether the HEAD tree builds and what the current installed kernel
  selections are. The port commit was never compiled; later commits are bot merges with no
  build receipt, and the zoo's picks are made at load time by a trial.
- **[UNVERIFIED]** Whether `mpp::tensor_ops::matmul2d` / `uint2b_format` are available on
  M4 Max. The engine's init probe decides this at runtime; the tree only tells me it is
  probe-gated and that the preferred form was chosen to avoid needing the new operand format.
- **[UNVERIFIED]** The exact per-shape in-situ medians: the trial prints them at load
  (`NarrowInSituTrial.roundBoundary` doc, `Qwen35MTPTopTwo.swift:3718-3730`) and nothing in
  the repository records a run. The same applies to the plane-forms and head top-two trials,
  and to the DFlash drafter's own trials.
- **[UNVERIFIED]** Whether the tensor route's GPU-time share and dispatch counts are as I
  infer. I ran nothing (no build, no weights); the dispatch-count picture is read from the
  code's grid/template structure.
- **[UNVERIFIED]** `Qwen35.swift:981-982`'s "4.27–4.28 against 4.23–4.26 ms per token" — the
  basis cannot be this model's decode step (7.3 GB cannot be read in 4.3 ms on any of these
  boxes), so it is either another model, a partial graph, or a different measurement basis.
- **[UNVERIFIED]** Where MLX enforces the asserted 10-command-buffer in-flight cap; the
  comment appears three times and no enforcement was found in the vendored C++/Metal tree.
- **Not inspected in depth:** the DFlash 2 drafter's own kernel zoo (`DFlash2Draft.swift` is
  8,946 lines; I read the pack12/11-bit encoder, the trials and the scheduling seams, not all
  8 kernel families), `Qwen35A3BTargetVerify.swift` (the unreachable exact lane, ~9 k lines),
  `Qwen35A3BOptimization.swift`, the transform pipeline beyond its contract, the CLI/trusted
  harness, and the MLXLMCommon engine internals past the eval-cadence question.
