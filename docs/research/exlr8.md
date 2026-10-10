# exlr8 — what it is, and whether it lowers our per-weight instruction count on an M4 Max

**The answer, first.** No. exlr8 is not a new codec and not a new kernel: it is exllamav3's EXL3
trellis code (the codec we already run — same 16×16 tiles, same tail-biting 16-bit state, same
tensor-core placement permutation, same 128-block Hadamard) plus four *deployment* changes. Its
decode loop must do exactly the bit extractions ours does, and the only arithmetic it changes is
the codebook: cheaper (mcg instead of the mul1 our checkpoint uses — but that is a codebook choice,
not a format feature) **and** more expensive (every decoded value must be rounded to FP8 E4M3,
which is a software routine on Apple9-class lanes: MLX's own Metal fp8 emulation is ~10 ops per
float→fp8 conversion). The FP8 payoff is real but it is a *hardware* payoff: it exists only where
the lanes are removed from the loop by fp8 matrix units — MLX gates every fp8 quantized matmul to
its `*_nax` kernels, and this box reports `MTLGPUFamilyApple10 = false` (measured, §5).

Consequences for our decision, in one place:

- **Per-weight work: 0.43 → ≥0.50 warp-instructions** if exlr8's format is honoured literally on
  M4 Max lanes [DERIVED, §3]. Nothing in the format, at any batch size, goes *below* 0.43 here.
- **The premise does not hold for us.** "Decode speed is bound by how many bytes of expert weights
  you read per token" is true on GB10 (220 GB/s streaming, tensor-core MACs) and false in our
  engine: our issue ceiling is ~596 Gweights/s per row, i.e. `74.5·K` GB/s at the issue wall, and
  the DRAM wall (546 GB/s) is only reached at K ≈ 7.3. For the K2/K3 expert classes that dominate
  the model we are issue-bound at *every* row count, so a format whose only lever is bytes/token
  cannot buy us time.
- **The one lever that does work is not theirs:** tabulate the codebook (state → value) in the
  kernel. Our own `EXL3_QMV_NODECODE` control bounds its payoff at the whole codebook-arithmetic
  share, 0.039/0.168 ≈ **23% of kernel time (measured, §3)**. It is bit-exact by construction and
  works on EXL3 as it stands.
- Adopting the format itself (new encode, per-half widths, E4M3-codebook, shared signs) would cost
  a full requantisation + calibration we do not have, and would **not** beat 0.43 on this hardware.

Everything below is read from source, with the commit named; §9 lists what could not be verified.

---

## 1. Resolved identity

`https://github.com/kindlingai/exlr8`, cloned to `/tmp/exlr8` and read at commit
**`d56e3ecf7aa3104be0b77b59b076406562fd20c5`** (2026-10-09 18:13 −0600, *"loader README: config
and tokenizer now ship with the checkpoint (v0.4)"*). 9 commits, 34 files, **no tags in the repo**
(the HF model card carries the v0.1→v0.4 changelog instead). The link is what it seems — this is
the repo the model card points at for the reference loader — but the *format* is what matters:

| in the repo | what it is |
|---|---|
| `README.md` (275 lines) | the explainer: the trellis/rotation/LDLQ math, the four deltas from EXL3, the TP sharding rule |
| `loader/exlr8.py` (340 lines) + `loader/README.md` | a **pure-PyTorch reference decoder**, "a reference for correctness, not for speed" (`loader/README.md:8`) |
| `calibration/` | the **public part** of `calib-2m-v1`: 12 public sources (byte-identical, sha256-pinned), `calib-2m-public.npy` (658×2048 int32), `rows.json`; 37% of the mix (Claude Code session transcripts) is not published (`calibration/README.md:7-9`) |
| — | **no kernels, no CUDA/Metal/C++, no engine, no encoder.** A `grep -in 'metal\|cuda\|kernel'` over the tree matches only README prose and the loader's `--device` help string |

The README is explicit that the codec is borrowed: *"the codec itself is borrowed: each 16×16 weight
tile is EXL3's trellis code (QTIP-style)"* (`README.md:12-13`), and the loader's header names the version —
*"The trellis codebook, the bit packing, the tile order and the 128x128 Hadamard rotation reproduce
exllamav3 0.0.43"* (`loader/exlr8.py:1-8`, `loader/README.md:11-18`). exlr8 v2 = exllamav3 0.0.43 +
four changes (`README.md:17-22`):

1. **E4M3-rounded codebook** — rounding applied *inside* the Viterbi search, so decoded weights are
   exactly FP8 and (on NVIDIA) prefill can feed FP8 tensor cores.
2. **Three widths stored** (K2/K3/K4) per routed expert; a manifest picks the served width **per
   128-channel half** (gate/up) or per expert (down) **at load time**, no re-encode.
3. **Rotation shared, not per-expert** — one sign vector per layer on the hidden side, plus
   block-constant scales, so the activation is rotated once for all 8 active experts.
4. **TP-native layout** — the sharding rule `(16e + h + L) mod N` serves TP=3–7, and halves are
   individually HTTP-range-fetchable.

**Corrections up front, because they change what is transferable.**

1. **There is nothing here to benchmark.** No kernel, no engine, and the HF card says
   *"No public serving runtime loads this format yet"*; their engine (kindlingai's `gr8`/`g6`) is
   not public. Their own status line is *"Status: draft / preview. End-to-end gates (TP agreement,
   held-out KL against BF16, **decode speed**) are running."* So the README's "decode speed is bound
   by bytes" is a design premise, not a measurement, and no exlr8 decode number exists yet.
2. **It is a different model.** exlr8 v2 ships GLM-5.3 (744B: 256 routed experts × 75 MoE layers,
   hidden 6144, expert intermediate 2048) as ~819 GB of trellis plus int4/fp8 dense; `home` = 294.7 GB
   of routed experts. Ours is GLM-5.3-Flash (121 GB, 288 experts top-8, 45 layers, per-`docs/glm5.md:12-13`).
   Nothing here is a drop-in.
3. **The format is licensed CC BY 4.0 for its own additions; the three exllamav3 primitives stay
   MIT** (HF card, "Licence"). No licence obstacle to reading it.

---

## 2. Format spec, byte for byte

The authoritative byte-level statement is the model card's *Loader* section (shipped as v0.4 with the
weights, referenced by `loader/README.md:18`); the loader is a second, executable statement of it. I
checked the two against each other; they agree.

**Container.** Little-endian safetensors; `__pad.*` names are skippable. Expert files are per layer:
an index (`exlr8-layer-LLL.index.safetensors`, holding `kK.index`, side vectors, checksums, the width
list in metadata) plus one file per width (`…kK.safetensors`). Matrices are (in, out) internally;
`*.weight` is the transpose.

**Tile.** A 16×16 tile at width K is `32·K` bytes = 16K int16 words. The rule (card, *Tile decode*;
`loader/exlr8.py:106-119`):

| step | rule | where |
|---|---|---|
| bits | swap each *adjacent pair* of int16 words, then read words MSB-first as one 256K-bit stream; `vᵢ` = bits `[iK, iK+K)` | card; `exlr8.py:108-111` |
| state | `sᵢ = (vᵢ + 2^K·vᵢ₋₁ + 2^2K·vᵢ₋₂ + …) mod 2^16`, indices mod 256 (**tail-biting**) | `exlr8.py:113-115` |
| codebook | `x = ((sᵢ·0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60`; add the low and high 16 bits **as two fp16 values, in fp16**; round the sum to **E4M3 (RNE, saturating)** | `exlr8.py:94-103`, `MCG` at `:24` |
| placement | stream index `j = 8t + n` → tile element `(r₀+…, c₀+…)`, `r₀ = 2(t mod 4)`, `c₀ = ⌊t/4⌋` (the tensor-core permutation) | `exlr8.py:82-92` |

That is our format, step for step: the same `SWAP16` word order, the same ring, the same
`tensor_core_perm`, the same procedural codebook (`src/exl3.h:11-27`, `src/exl3.cpp:21-52`,
`src/trellis_gemm.cpp:87-101`). **The only codec deltas are the codebook identity (mcg vs our mul1)
and the final E4M3 rounding.** Our checkpoint uses `.mul1` (`cb = 2`) — `src/model.cpp:250`,
`src/glm5.cpp:303`, `src/exl3.h:33` — while exlr8 mandates mcg (`cb = 1`).

**Side vectors and rotation.** `W = diag(suh)·Hb·T·Hb·diag(svh)`, `Hb` = 128×128 Sylvester Hadamard
blocks scaled 1/√128 (`loader/exlr8.py:133-147`), i.e. exactly our `src/trellis_gemm.h:1-11` /
`src/exl3.cpp:reconstruct`. The delta is *which* vectors exist (`exlr8.py:170-196`):

| | gate/up half `h` (6144×128) | down half `h` (128×6144) |
|---|---|---|
| `suh` | `signs.in ⊙ repeat₁₂₈(su_block[e])` — a **per-layer** sign vector shared by all experts/widths, times **48 block-constant** fp16 scales | `down.suh[e, 128h:128h+128]` — 128 per-channel fp16 |
| `svh` | `…svh[e, 128h:128h+128]` — 128 per-channel fp16 | `signs.out ⊙ repeat₁₂₈(down.sv_block[e])` |

So one side is per-channel and the other is block-constant + a shared sign vector. Upstream EXL3 (and
our loader) carry **both** sides as full per-channel f16 vectors (`src/exl3.h:39-40`).

**Widths and granularity.** Every routed expert is stored at K2, K3 and K4 (one file set per width;
codes do not nest, so each is an independent encode). Storage unit = a 128-channel *half*: 16 halves
per expert for gate/up, per-half widths; down per expert. `half_bytes = 98304·K` = `6144·128·K/8`.
The manifest (JSON, on HF) gives `layers[L].gate[e][h]`, `.up[e][h]`, `.down[e][h]`. Tiles are
`(k16, c)` for gate/up halves and `(b, r, c)` for down halves, with the offsets in the card's table.

**Dense and vocabulary.** Same index/side-vector machinery plus non-trellis formats: `record` =
`(format, out, in, K, side, …)` with formats 1 trellis, 2 int4-g32, 3 MXFP8, 4 fp8 block-128,
5 fp8 row, 6 bf16, 7 fp32 (`exlr8.py:26-28`, `:200-230`; card table). `lm_head`/`embed_tokens` are
**fp8 per-row** in the shipped checkpoint; dense q_a/kv_a/indexer are MXFP8, kv_b block-fp8.

**Integrity.** xxhash64 (seed 0) over the trellis bytes + the half's 128 per-channel values + the 48
block scales (`exlr8.py:189-193`), per half; per 128-channel block for dense (`exlr8.py:232-250`).
Cheap to implement, and the kind of check our own gate (§6, item R6) should have.

---

## 3. The decisive question: instructions per weight

### 3.1 Our baseline, as measured

`docs/perf.md:466-478`: a (16×16 output tile, k-tile) simdgroup iteration — **256 weights** — retires
**~110 warp-instructions**: 4 packed-word loads (2 per group of four weights), the 64-bit window
shift, 8 sliding-window extracts, 8 codebook decodes, 8 FMAs, the activation path (one `half2` load
+ two shuffles + 4 casts) and the loop's address math. Measured: 393216 iterations in 0.168 ms =
**2.34e9 tiles/s = 256e9 warp-instructions/s, the machine's entire issue width**. Therefore

- **0.43 issued warp-instructions per weight** (13.75 lane-slots), per row;
- **596e9 weights/s per row** at the issue wall, whatever K is;
- bytes/s at that wall = `596e9 · K/8` = **74.5·K GB/s** → K2 149, K3 224, K4 298, K6 447 GB/s,
  against a 546 GB/s DRAM wall (`docs/perf.md:33`). Measured points agree: exp8 K2 134 GB/s,
  exp8 K3 149, 4096×24576 K4 289–305, lm_head K6 450 (`docs/perf.md:496-507`).

**The two walls only meet at K ≈ 7.3.** Below that, *time is set by instructions, not bytes*: halving
K halves bytes per tile and leaves the instruction count per tile unchanged, so it halves GB/s at a
constant weight rate. That is the fact the whole comparison turns on.

The instrumented split of that kernel on the K4 shape (`docs/perf.md:93-95`): 0.071 ms of loop
skeleton (no memory at all) + 0.052 ms of packed-word reads + **0.039 ms of codebook arithmetic**, of
0.168 ms total. So the entire codebook arithmetic — the only part of our decode a format could
change — is **≈23% of kernel time**; the other ~77% is loads, window/extract, FMAs, the activation
path and address math, all of which exlr8 keeps *identical*.

### 3.2 What exlr8 changes in that budget

| term | ours (mul1, K4, 1 row) | exlr8, honoured literally | exlr8 with a state→value table |
|---|---|---|---|
| word loads + window + 8 extracts + FMA + activation + address math | ~85 of 110 (measured by difference: 110 − codebook share) | **identical** (identical packing) | **identical** |
| codebook | mul1: `mul` + 4-op byte sum + ~4-op combine/scale + `HFMA` + `F2F` (`src/trellis_gemm.cpp:185-192`) | mcg: `mul` + `lop3` + `HADD2` + `F2F` (`src/trellis_gemm.cpp:194-201`; exllamav3's `decode_3inst<1>`, `quant/codebook.cuh`) — then **+ the E4M3 rounding the format requires** | 1 gather (65536-entry fp16 table = 128 KiB working set) + 1 `F2F`, replacing the arithmetic |
| total | **110 / tile** (measured) | 110 − (mul1−mcg) + (E4M3 round-trip) | ~110 − codebook arithmetic |

The E4M3 term is the problem. On this class of hardware FP8 is not a lane type: MLX ships its own
software `fp8_e4m3` struct — `float→fp8` is a saturating RNE emulation of ~10 ops and `fp8→half` is
5 more (`mlx/backend/metal/kernels/fp8.h`, in the installed MLX 0.32.3 headers here). A kernel that
must *produce* weights that are exactly E4M3 therefore pays roughly one extra codebook's worth of
work per weight. Literal accounting [DERIVED from the source op counts, anchored on the measured 23%
codebook share]: codebook arithmetic 23% → ~21% (mcg is cheaper) **+** ~25% for the rounding →
**~0.50–0.54 warp-instructions per weight**, i.e. the K2/K3 classes would *fall* from 134/149 GB/s to
~110/120. **exlr8 is a slowdown on M4 Max unless the rounding is removed from the lanes.**

Two ways to remove it from the lanes, and neither is available here:

1. **FP8 matrix units.** Then the decoded value never enters a lane FMA. This is what the format is
   for, and it needs NAX (Apple10/M5) or NVIDIA. Measured on this box: `.apple10 = false` (§5).
2. **An FP16 table of the 65536 possible states** (`state → the exact fp16 pattern the codebook
   produces`). Because exlr8's values are constrained to 254 E4M3 patterns, the *value* space is
   tiny — but the *state* space, which is what you must index by, is still 16 bits, so the table is
   128 KiB whether the alphabet is fp8 or fp16. Then per weight it is 1 gather + 1 cast instead of
   ~5–10 ALU ops. **This is bit-identical by construction** (it is the same function; we already
   build exactly this mapping on the host for `reconstruct`, `src/exl3.cpp:21-52`). It is not an
   exlr8 optimisation — it applies to our mul1 trellis unchanged — and its payoff is bounded by our
   own control: the NODECODE variant (`src/trellis_gemm.cpp:166-176`) removes the codebook arithmetic
   and takes 0.168 → 0.123 ms, i.e. **at most ~27%** (134 → ≤170 GB/s on the K2 class). Whether a
   128 KiB random gather beats 5–10 ALU ops on Apple9 is **unmeasured** [UNVERIFIED].

### 3.3 Where the per-weight work can actually go below 0.43

Only two mechanisms exist on this hardware, and exlr8 supplies neither:

- **Move the MACs off the lanes.** exllamav3's own m=1 GEMV does this with `mma.m16n8k16` behind a
  lane guard (`quant/exl3_gemv_kernel.cuh:13-22, :34-78, :405-406`) — that is why the same format
  reaches 313 GB/s of K4 tiles on a 538 GB/s NVIDIA part (`docs/perf.md:456-459`). On Apple GPUs the
  equivalent hardware is `simdgroup_matrix` (M1+, needs ≥8 rows) or MPP `matmul2d` (fp8 operands are
  in the SDK's type table, `MPPTensorOpsMatMul2d.h:74-83`); MLX's own dispatcher only leaves the
  scalar GEMV at 13 rows (`get_qmv_batch_limit`, quoted at `docs/research/mlxfast-bonsai2.md:222-226`),
  so **a 1-row decode is on the scalar path by construction** in every route we know of, and neither
  hardware route is unlocked by changing the codec.
- **Remove the per-weight extraction itself**, i.e. a packed-operand hardware path. Their own
  survey recorded that this is exactly the thing the tensor unit does and that our codewords are not
  what it ingests (`docs/research/mlxfast-bonsai2.md:§7`).

---

## 4. EXL3 (ours) vs exlr8, side by side

| | EXL3 in our engine | exlr8 v2 |
|---|---|---|
| codec | EXL3 trellis, 16-bit state, tail-biting ring; `.mul1` codebook | **same codec**; mcg codebook, values rounded to E4M3 inside the Viterbi search |
| tile packing / placement | 16K little-endian i16, tensor-core perm — byte-granular contract (`src/trellis_gemm.h:14-27, :62-70`) | identical, per their own loader/card |
| bytes/weight | K/8 (K2 0.25 … K6 0.75) | K/8, K ∈ {2,3,4}, chosen per half at load from a manifest |
| per-weight lane work | **0.43 warp-instr** measured | same extraction + cheaper codebook + an E4M3 round-trip → **≥0.50** [DERIVED] |
| MAC location | lanes (FMA in registers, `mb` rows per threadgroup) | lanes for decode; **fp8 tensor cores** for prefill on GB10 |
| scale vectors | both sides full per-channel f16 (`suh[in]`, `svh[out]`) | one side per-channel, other side block-constant (48) × a **per-layer shared sign vector** |
| rotation | fused per launch, per member (`trellis_linear_group`) | shared by construction across experts/widths |
| hardware needs | SIMD only; M1+ | fp8 matrix units for the advertised win (NAX/M5+, or NVIDIA GB10); ≥13 rows for any matrix route; GB10-class 128 GB unified memory; TP 3–7 across nodes for the shipped layout |
| published numbers | our own measurements (`docs/perf.md`) | **none for decode** ("gates running"); one claimed GB10/TP=4 prefill delta (layer-40 MoE 23.07 → 18.67 ms) and claimed quality ratios (2.3–33× vs RTN, −14% layer error) |
| status | shipping, measured | draft/preview, no public runtime |

---

## 5. Measurements I made for this report (not from exlr8)

- **This box**: `MTLCreateSystemDefaultDevice().name == "Apple M4 Max"`,
  `supportsFamily(.apple9) == true`, **`supportsFamily(.apple10) == false`** (Metal probe, compiled
  and run here).
- **MLX 0.32.3, as installed here**: every fp8 quantized matmul it ships is NAX-suffixed —
  `strings mlx.metallib` yields `mxfp8_gather_qmm_rhs_{nax_,}nn_*` families in which 240 of 336
  `mxfp8_*` names carry `_nax`; the non-NAX ones are the `steel` path that dequantizes fp8 to the
  compute type with the software `fp8_e4m3` struct (`fp_quantized.h:34-55, :141-144, :2279`).
- **The gate**, from MLX's source: `is_nax_available()` requires
  `__builtin_available(macOS 26.2, …)` *and* `get_architecture_gen() >= 17` (18 for `'p'`)
  (`mlx/backend/metal/device.cpp:952-971`, tag v0.32.3). M5-era, not M4.
- **exllamav3 has no Apple backend at all**: the tree has no Metal source and no CMake path for it;
  its GEMV/GEMV-int8/GEMM are CUDA/HIP (`exllamav3/exllamav3_ext/quant/*.cu*`).

So on this machine, exlr8's one genuinely new codec idea — "decoded weights are exactly E4M3" — has
no consumer, and exlr8's decode would have to be emulated on the lanes we are already bound on.

---

## 6. Ranked: what is transferable to `exl3-server`

Gate restated: the single-sequence path must stay **bit-identical**, the byte-granularity contract in
`src/trellis_gemm.h:14-27` must not be broken, and the k-split count is part of the f32 association
(`src/trellis_gemm.cpp:403-421`) and may not change.

| # | Item | What it would change for us | Expected effect | Cost / risk | Bit-identical? |
|---|---|---|---|---|---|
| **R1** | **Tabulate the codebook in the GEMV: `state → fp16 value`** (not exlr8's idea — it is the only route under 0.43 that needs no hardware — but this read makes the size question concrete: their E4M3 alphabet is 64 KiB as bytes, though on M4 that costs more to expand than it saves). | Replaces the mul1 hash (`src/trellis_gemm.cpp:185-192`) with one gather + one cast. The extraction, the window, the loads and the FMA are untouched; the table is exactly the function `src/exl3.cpp:21-52` already computes. | Bounded by our own control: the codebook arithmetic is **0.039 of 0.168 ms (≈23%; measured, `docs/perf.md:93-95`)**, so ≤ ~27% on the K2/K3 classes: **134/149 → ≤170/190 GB/s** (DRAM-capped at 546, so no wall). | Medium: a new 128 KiB input buffer bound to the kernel, built per codebook at load; the risk is L2 latency on a random gather (unmeasured) and register/occupancy pressure. Keep the arithmetic path behind an env switch and hash-compare the two. | **Yes, exact by construction** — the table holds the bits the arithmetic produces. Needs a test hashing kernel output against the arithmetic path (the pattern of `tests/glm5_gemv_nt_test.cpp`). |
| **R2** | **Requantise the K2/K3 expert classes with the mcg codebook** (the codebook exlr8 chose: `exlr8.py:24, :94-103`). | Our checkpoint is `.mul1` (`src/model.cpp:250`); mcg is `mul`+`lop3`+`HADD2`+`F2F` (`src/trellis_gemm.cpp:194-201`) against mul1's byte-sum, which is 4 ops on NVIDIA's `dp4a` but ~10 on Metal. | Small at the kernel level: the codebook is ≤23% of kernel time and mcg saves only part of it — **single-digit %** (134 → ~145 GB/s) [DERIVED]. The real exlr8 evidence is *quality viability at 2–3 bpw with mcg*, which we currently assume needs mul1. | High: needs an encoder + the 2M-token calibration (Hessians per expert) we do not have, a new checkpoint, and a re-baselined single-sequence golden. | **No** — different weights. It is a new model variant, not an optimisation of the current one. |
| **R3** | **Width manifests as a *quality* lever, not a speed lever** (option 2 of their four deltas). | Reproduces what we already do per layer (one K per projection); adds per-half heterogeneity, which would break the uniform-config invariant of the grouped launch (`src/trellis_gemm.cpp:566-576`) and add per-chunk address math. | **Zero decode benefit on M4 Max** and it costs a third variant per expert (their `home`+`down1`+`up1` = 819 GB). Since bytes are not our constraint for K ≤ 6 (§3.1), picking a lower K cannot reduce time. Reject as a speed item; adopt only if a future quality/disk trade needs the per-unit *error tables*. | High (requantise, 3× disk, kernel complexity). | No (needs requantise) — and it would change which k-tiles each simdgroup accumulates, i.e. the f32 association, unless the split boundaries are redefined. |
| **R4** | **Their shared-rotation accounting, checked against ours.** | Their point is that per-expert input-side rotations force 8 activation rotations per token. We already avoid that: `trellis_linear_group` does **one** stacked Hadamard per launch with a per-member scale vector (`src/trellis_gemm.cpp:628-640`, `:645`). The residual delta is *constant traffic*: a per-member `suh[6144]`+`svh[out]` f16 pair vs their 48 block scales + one shared per-layer sign vector. | Prep traffic only (their own claim: layer-40 MoE prefill 23.07 → 18.67 ms on TP=4 GB10, *claimed*); nothing in the GEMV's 0.43. | Low to copy the *idea* of shrinking constants (needs a scale-broadcast path in `hadamard_run`), high to adopt the encoding (requantise). | Not for the encoding (values change); the fused-rotation structure is already ours and already bit-identical. |
| **R5** | **Their calibration/LDLQ recipe as the reference for any future requantise** (`README.md:111-140`, `:213-233`): per-expert `H = Σxxᵀ`, damping λ = 0.025·mean-diag, shrinkage for thin experts, `H = LDLᵀ` and error feedback over 16-column blocks, 2M tokens, 64 held-out rows. | Nothing today. If R2 or R3 ever happens, this is the method and these are the guardrails (RTN is 2.3–33× worse on their near-rank-one Hessians). | Quality only. | — | N/A (it is a recipe) |
| **R6** | **Their integrity and addressing plumbing** (option 4): per-half xxhash64 (`exlr8.py:189-193`), a pointer table per (expert, projection, half) instead of contiguous ranges, 64 KiB-aligned files so a slice is range-fetchable, and `set_id` computed from the hash list. | We already repack to an aligned engine-owned buffer (`trellis_pack`) and index members per launch; nothing here is a decode change. | Zero decode effect. Useful only if we ever shard one model across Macs (fetch-your-own-slices), which is outside this task. | Low to borrow the checksum style. | Yes (it protects bytes, does not change them). |

**Ranking rationale.** R1 is first because it is the only item in this space that lowers the per-weight
count on this hardware, and it is bit-exact. R2 is second because it is the only *codec* thing exlr8
demonstrates (a shipped 744B K2/K3/K4 checkpoint on mcg) and it is cheap in kernel terms — but it is
a requantisation, so it is a new model, not a change to this one. R3/R4 are recorded as *rejected for
speed* with the reason (we are issue-bound, not byte-bound), R5/R6 as material for a future
requantise or a multi-machine deployment.

---

## 7. Does NOT transfer — with reasons

- **The headline (FP8 codebook → prefill on FP8 tensor cores).** Needs fp8 matrix units: NAX
  (`is_nax_available()` = macOS 26.2 + GPU gen ≥ 17, `mlx/backend/metal/device.cpp:952-971`; this box
  is `apple10 = false`, measured), or NVIDIA GB10. On Apple9 lanes the requirement is a *cost*, not a
  benefit (`fp8.h` software conversions; §3.2).
- **The E4M3 rounding itself, as a decode step.** Extra per-weight work with no consumer on M4 Max.
  It only pays together with the item above.
- **MPP `matmul2d` with fp8 operands.** The SDK's matmul2d *does* list
  `metal_fp8_e4m3/e5m2_format` operand pairs (`MPPTensorOpsMatMul2d.h:74-83`), but (a) whether the op
  runs on M4 Max at all is not verifiable here — the Metal toolchain is not installed on this box
  (`xcrun metal` fails), so it stays [UNVERIFIED] exactly as recorded in
  `docs/research/mlxfast-bonsai2.md:§7`; and (b) MLX's dispatcher keeps any row count below 13 on the
  scalar GEMV, so a 1-row decode cannot use it even if it exists. Our sibling survey's conclusion
  stands: our operands are trellis codewords requiring a hash, not fields a packed-operand unit can
  ingest.
- **Three-width storage / per-half manifests.** Bytes are not our binding constraint for K ≤ 6 (§3.1);
  3× disk (819 GB for their model); per-half heterogeneity breaks our one-config-per-launch grouped
  path; and it would move k-tile→simdgroup membership, i.e. the f32 association our gate forbids.
- **Their container and file layout** (safetensors index + per-width files + manifests + TP slices,
  with half-major tile runs). Adopting it means a loader rewrite for no decode gain; our
  `.trellis/.suh/.svh/.mul1` shard reader is already zero-copy (`src/safetensors.cpp:145`, `src/model.cpp:242-252`).
- **Their side-vector encoding** (shared per-layer signs + block-constant scales). The *structure*
  (rotate the activation once per fan-out) is already ours; adopting the *encoding* requires
  requantisation and touches nothing in the hot loop.
- **TP sharding, pointer tables, HTTP range fetch, `set_id` bookkeeping.** There is no tensor
  parallelism on a single M4 Max, and no remote node to fetch from. Pure deployment machinery.
- **Their quality claims (2.3–33× vs RTN, −14% held-out layer error, NMSE 0.9996–1.0003 for shared
  signs) and the 23.07 → 18.67 ms prefill delta.** GB10/TP=4, claimed, not reproducible from this
  repo, and not measured by us. They cannot be used to justify a format change here; and their own
  decode-speed gates are still running.
- **Their encoder.** `exlr8quant` is referenced (`calibration/README.md:82`) but not published;
  nothing about the E4M3-in-Viterbi or LDLQ encode is reproducible from this repo.
- **Reading their checkpoint as a substitute for ours.** Different model (744B vs our GLM-5.3-Flash),
  ~819 GB of trellis, needs 4×128 GB GB10 nodes and their unpublished runtime.

---

## 8. If we ever wanted to adopt the format: migration cost

**What would have to be re-quantised: everything on the trellis path.** exlr8 changes (a) the codebook
(mcg + E4M3 rounding *inside the search*), (b) the side-vector structure (shared per-layer signs,
block-constant scales on one side), (c) the storage granularity (per-half widths, three encodes).
None of the three is a decode-side change we can apply to bytes we already have — the E4M3 constraint
is a property of the *search*, and while a stock decoder can apply the rounding after the fact, the
checkpoint's own README prices that at +2.3% error ("a stock EXL3 decoder … reads measurably worse
weights from the same bytes"). So: all routed + shared experts, the trellis dense tensors and the
vocab path.

**What that needs that we do not have:** an encoder (exllamav3's quantiser is CUDA-only and has no
Apple path; E4M3-in-search and LDLQ are theirs), a calibration run over ~2M tokens of *our* base
model with per-expert Hessians, three encodes per expert (or a decision to store one width), and a
new golden for the single-sequence gate (the weights change, so bit-identity with today's checkpoint
is impossible by definition — it would have to be re-established modulo a new reference).

**What code would change:** a second loader path (`src/exl3.cpp` format declaration → safetensors
index/manifest reader), the kernel's width handling (per-chunk K → either three launch families or a
runtime byte-stride table, with extra address math per tile), the E4M3 decode (either the software
round trip — slower — or the R1 table, which happens to be the *only* place the format's alphabet
could be cashed in cheaply, and even then only because a table indexed by the 16-bit state is
available to any codebook).

**Could it ever beat 0.43 / 86–452 GB/s on M4 Max?** Not through anything exlr8 does. Its decode
floor is our decode floor. The three levers that do work here are (i) a tabulated codebook (R1, ours,
bit-exact, ≤27% measured bound), (ii) matrix units — `simdgroup_matrix` from 8 rows, MPP/NAX from the
dispatcher's 13-row crossover and only on M5+/NVIDIA for fp8, (iii) choosing a mechanically cheaper
codebook at encode time (R2, single-digit %). Its own wins are real on the machine it targets: GB10 has fp8 tensor cores,
which is the one place the E4M3 constraint stops being a lane cost and becomes the point of the
format. On an M4 Max, the format is *readable* (that is all the loader proves) but not adoptable as a
speed lever.

---

## 9. Claims, controls, and what I could not verify

**Measured here** (this box, this session): GPU identity and `apple10 = false`; MLX 0.32.3's NAX
kernel inventory and its gating source; exllamav3's lack of any Apple backend; the exlr8 repo's
contents, history, tags and licence headers; every format statement above, cross-checked between the
repo loader and the model card's byte-level spec.

**Measured earlier by us, quoted** (not re-run — the checkpoint is in use by another agent, and this
task needs no weights): 110 warp-instr / 0.43 per weight / 2.34e9 tiles/s / 256e9 instr/s; the
0.071/0.052/0.039 ms split; the K-class GB/s table (`docs/perf.md:93-95, :466-507`).

**[DERIVED]**, not measured: the per-weight op counts for mcg/E4M3 on Apple9 lanes; the exlr8-literal
figure of ~0.50–0.54 warp-instr/weight; the single-digit-% effect of R2; the ≤27% bound on R1 (that
one is anchored on a measurement, but the *realised* win depends on gather latency vs ALU issue).

**[UNVERIFIED]**: whether MPP `matmul2d` executes at all on M4 Max (no Metal toolchain installed
here to compile a probe; our sibling survey left it open too); whether a 128 KiB state→fp16 table
gather beats the arithmetic; exlr8's quality claims and any decode-speed number (none exists); the
claim that the shipped exlr8 checkpoint decodes into a BF16 model that passes their own end-to-end
gates; and whether their per-half width mixes would hold up on our model's routing distribution.
