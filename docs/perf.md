# Decode performance: profile, budget, and levers

Target: **70 tok/s effective decode** for GLM-5.3-Flash (121 GB, 45 layers,
288-expert MoE top-8, mhc 4 streams) on an M4 Max / 128 GB.

## Where the time goes (measured, ~317 ms/step at the start)

| component | time | notes |
|---|---|---|
periphery | ~165 ms | by subtraction: 2 hyper-connection sites per layer x 45, each a 20-iteration Sinkhorn on a 4x4 matrix plus a small linear, two RMSNorms and host-side DSA mask/indexer bookkeeping |
MoE | 77 ms | 42 layers, host loop: `mx::eval` + ~10 tiny ops per selected expert, 8 experts x 3 projections |
KDA | 47 ms | 34 layers |
MLA + indexer | 24 ms | 11 layers |
dense MLP | 3 ms | 3 layers |

Roughly 4000-5000 kernels per token at ~30-50 us each: the step is
**launch-bound, not bandwidth-bound**.

## Bandwidth budget (the real ceiling)

Bytes actually touched per token:

```
qkv_proj      34 layers x 4096x24576 at K=4   ~1.7 GB
o_proj        34 layers x 8192x4096  at K=4   ~0.6 GB
MLA + index   11 layers                       ~0.3 GB
experts       42 layers x 8/288 experts, K=2  ~2.1 GB
shared/dense  3 + 42 shared experts           ~0.05 GB
                                              --------
                                              ~4.6 GB/token
```

An M4 Max sustains ~546 GB/s, so the floor is **~8.4 ms/step (~119 tok/s)** with
zero launch overhead. 70 tok/s = 14.3 ms/step = ~320 GB/s effective, i.e. 59% of
peak - reachable only by getting the kernel count down to a few hundred per
token *and* keeping the GEMVs bandwidth-efficient. After each optimisation step,
say whether a component became launch-bound or bandwidth-bound; that decides
what is left to do.

## Levers, in order of expected payoff

1. **`mx::compile`** (available in the C++ API: `mlx/compile.h`) fuses
   elementwise chains and reductions into far fewer kernels. The
   hyper-connection sites are the ideal target: their shapes are identical on
   every decode step (4x4 comb, `hc_mult` 4), so one compiled function per layer
   per site replaces ~40 tiny ops. Same for the per-layer norm/residual glue.
   Note compiled functions are traced per shape/dtype - build them once, not per
   token.
2. **Grouped expert GEMV** for MoE: one kernel per layer over the routed
   experts instead of ~24 launches plus a host loop. `mlx::core::fast` is not
   exposed in this install, so this goes through the repo's own Metal plumbing
   (see `src/trellis_gemm.cpp`), which is also where the next item belongs.
3. **Fuse the two Hadamards into the trellis GEMV**: `trellis_linear` is 3
   launches (input Hadamard, GEMV, output Hadamard) x ~1450 linears per token.
4. **`mx::async_eval`** in the steady-state decode path (evaluation batching),
   and `mx::set_wired_limit` to keep the 121 GB working set from being evicted
   (a refill from the external SSD runs at ~2.6 GB/s, i.e. 400 ms per GB).

### Speculation is a launch-overhead multiplier

While the step is launch-bound, extra rows in a verify batch are nearly free
(their kernels are dispatched together), which is exactly why the MTP drafter
measured +30-50%. It is *not* a bandwidth win: at the floor, each extra row
costs its own ~4.6 GB, so speculative gains vanish once the bandwidth wall is
reached. Report both the serial step and the effective rate with
`--draft-mtp N`.

## Measured facts (2026-10-10, layer-0 tensors, real checkpoint)

| item | value |
|---|---|
`mx::eval` | ~0.2 ms per call (thread handoff + command-buffer wait) - fatal inside a decode loop; the MoE host loop alone spent ~8 ms/token |
MLX elementwise dispatch | ~5-6 us |
custom metal_kernel dispatch | ~4-10 us |
trellis GEMV, 4096x24576 K=4 (50.4 MB) | **258 GB/s at rows=1** (305 with the aligned loads below), raw QMV with 64 iterations per graph - see the correction below |
trellis GEMV, small tensor (0.5 MB) | 0.010 ms => ~50 GB/s: launch-bound, the kernel itself is ~7 us of the 10 |
scaling with rows | with the old `mb` rule 1/2/4/8 rows cost 0.20/0.30/0.68/1.83 ms; with `mb = min(rows,4)` and `mb*nt <= 4`: 0.17/0.20/0.36/0.70 ms |
f16 matmul 4096x24576 (reference) | 0.48 ms => 420 GB/s at rows=1, 200 GB/s at rows 2-8 |

**Methodology correction (important).** The `123 GB/s` this file used to record
for the 4096x24576 tensor came from `tests/glm5_trellis_bw.cpp` at its default
8 iterations per graph, where the host-side dispatch of the 3 kernels per
iteration dominates the measurement. With 64-256 iterations the same kernel
measures 258 GB/s, i.e. the "3x gap to MLX's f16 matmul" was 1.35x in time at
rows=1. The harness's default is now 64 iterations; anything below ~32 still
measures the dispatch of the graph, not the kernel. Also note `mx::eval` costs
0.11-0.2 ms, which exceeds a small GEMV outright.

## Second pass (2026-10-10): aligned word loads, packed buffer, row blocking

What the kernel actually spends its time on at rows=1 (4096x24576 K4, raw QMV
with `EXL3_QMV_NODECODE` and the (now removed) skip-load variant as controls):
0.071 ms of loop skeleton (no memory at all), +0.052 ms of the packed-word
reads, +0.039 ms of the codebook arithmetic. The kernel is issue/latency-bound,
not DRAM-bound (50 MB in 0.16 ms is 305 GB/s of a 546 GB/s machine).

- **Aligned word loads** (+20% at rows=1): a tile's 8*K words are addressed at
  byte 4*j, so with a 4-byte-aligned buffer each word is one `ld.global.b32`
  instead of four byte loads. Fetching the neighbour word with a simdgroup
  shuffle instead of the second load measured 19% *slower*, so the second load
  (an L1 hit in the same line) stays; a per-lane 8-byte "window" layout would
  have doubled the checkpoint and was not viable on 128 GB.
- **`trellis_pack`** materialises the words into a 16-byte-aligned engine-owned
  buffer for every tensor whose file offset is not a multiple of 4: 11 of the 13
  shards have `header_len % 4 != 0`, i.e. 96 of the 102 GB of trellis weights.
  Without it the zero-copy view can only use the byte path, which is what the
  kernel did before.
- **Row blocking**: `acc[mb][nt][8]` was unbounded (mb=8/nt=2 at 8 rows = 128
  registers, so the kernel spilled). `mb = min(rows, 4)` with `mb*nt <= 4` makes
  the 4-row MTP verify step and the 26-row prefill ~2x cheaper.
- **Grouped expert GEMV** (`trellis_linear_group`): the 8 routed experts of a
  decode step share the hidden row, so gate/up/down each become ONE multi-buffer
  launch with `grid.z = split + N_SPLITS*member` instead of 8 single-tensor
  linears. Bit-identical by construction (each member keeps its own scale
  vectors, split geometry and k-order) and checked in `moe_test`.

Measured back to back on the real checkpoint (a 16-token greedy run, best of 3
repeats, no other load on the box; the box drifts ~30% between sessions, so only
same-session numbers are comparable):

| config | serial ms/step | serial tok/s | `--draft-mtp 4` ms/step | effective tok/s |
|---|---|---|---|---|
| kernel at HEAD (byte loads, old `mb`) | 234.1 | 4.27 | 141.8 | 7.05 |
| this pass | 152.8 | 6.54 | 116.8 | 8.56 |

The greedy text is identical between the two rows (and the pixel-identical
"first token 785"), which is the end-to-end bit-identity check for the whole
change set.

What did *not* pay off and is therefore not in the tree: the simdgroup-shuffle
word fetch (19% slower), a duplicated per-lane window layout (2x memory), and
`mb=8`/`nt=4` variants (register pressure). `f_a`-sized tensors (4096x256,
0.5 MB) are latency-bound at ~45 GB/s and did not improve - they need batching
across layers, not a better kernel.

Honest distance to the 14.3 ms/step target: the step is no longer GEMV-bound.
The MoE stage is still 123 of ~153 ms (42 layers, 8 routed experts each) while
the 8 experts' weights are only ~0.6 ms of GEMV per layer, so what remains is
per-layer machinery (the router's ~25 small ops plus one `mx::eval` per layer to
read the selected expert ids on the host, the per-expert `take`/`scatter_add`,
and the ~150 dispatches of the ungrouped form) and the 2 hyper-connection sites
per layer.

Consequences: at ~300 GB/s the 4.6 GB/token means a **~15 ms/step floor**, so
the remaining work is the per-layer machinery and the launch count, not the
weight reads:

1. **Kill the per-layer eval** in the router: 42 host round-trips per token. The
   grouped kernel makes the expert *selection* the only host-visible dependency,
   so a device-side indirection table over the expert weights is the way out.
2. **Fuse the two Hadamards into the trellis GEMV**: `trellis_linear` is 3
   launches (input Hadamard, GEMV, output Hadamard) x ~1450 linears per token,
   and ~20 us per dispatch is visible when the ops are chained rather than
   batched.
3. Merge the small per-layer projections (f_a/g_a/b_proj and the MoE's shared
   expert) into grouped GEMVs. Within one layer they share the hidden row, so
   `trellis_linear_group` already covers them; *across* layers it cannot, because
   the layers are a sequential dependency chain (layer i+1's activation is
   layer i's output) and only their weights are independent. Measured headroom:
   the whole small-projection set is ~1.7 ms of a 153 ms step, and 7 of every
   10 us of it is launch overhead rather than the kernel, so the win here is
   bounded by ~1 ms - it is the mechanism (multi-buffer launches) that was the
   expensive part, and that now exists.
4. **`mx::async_eval`** in the steady-state decode path (evaluation batching),
   and `mx::set_wired_limit` to keep the working set from being evicted (a
   refill from the external SSD runs at ~2.6 GB/s, i.e. 400 ms per GB).

Methodology warnings, both learned the hard way here: (a) measuring with an
`mx::eval` per sample adds 0.11-0.2 ms, which exceeds the op itself for small
tensors - subtract a no-op baseline or batch correctly; (b) a lazy-evaluation
loop that drops the previous result each iteration measures graph construction
only (it reported 2259 GB/s for the 50 MB tensor, i.e. work that never ran) -
keep every intermediate alive and treat anything above ~546 GB/s as a sign the
work was optimised away; (c) an 8-iteration batch measures the *dispatch* of the
graph, not the kernel (see the correction above).

## Measurement discipline

- `tests/glm5_bench.cpp` reports the per-component breakdown (evaluate after
  each stage) and end-to-end tok/s; use it before and after every change.
- Correctness gates that must stay green: the trellis byte-alignment regression
  (`tests/glm5_exl3_decode_test.cpp`), the prefill-vs-decode logits invariant
  (~1.4 f16 ulp) and every other test in `tests/` - `tools/run_tests.sh` builds
  and runs them with full parallelism for the light tests while serialising the
  ones that map the 100 GB checkpoint (two of those would evict each other's
  pages on a 128 GB machine and be slower than running them one at a time).

## Third pass (2026-10-10): the per-layer machinery, measured stage by stage

This pass set out to remove the MoE's one-host-sync-per-layer and fuse the two
Hadamards, and it starts by *measuring* rather than reasoning about the sync.
Two things came out of that, one of which changes the whole picture:

**The profiled breakdown is not the step.** `tests/glm5_bench.cpp`'s per-stage
breakdown evaluates at every stage boundary, and in the real loop each of those
~360 barriers costs ~0.3 ms: the instrumented run reports 274 ms/step where the
uninstrumented one is 133-162 ms, and its component numbers are inflated by
roughly the same factor (MoE 138 "ms" of which ~130 is not the MoE). It is still
the right *relative* instrument, but absolute attribution needs a different
one: `tests/moe_bench.cpp` times one component per (build graph -> eval) cycle
with `reps` repetitions, and subtracts a "null" row (one elementwise op plus the
same sync). Run it on the real checkpoint:

```
/tmp/moe_bench 20 1 <model_dir>   # a fresh process is up to 2x slower than the
                                  # second one (Metal shader cache + DVFS), so
                                  # every number here is from the second run
```

| component | ms per call | calls/token | ms/token | notes |
|---|---|---|---|---|
| `moe_layer` (MoE layers) | 0.83-0.95 | 42 | ~36 | layer 4 (K=3) and layer 12 (K=2) |
| `kda_layer` | 0.73-0.80 | 34 | ~25 | qkv_proj alone is 50.3 MB |
| `mla_layer` + `indexer_topk` + `mla_q_resid` | 1.45-2.0 | 11 | ~16 | the indexer is host CPU code plus two syncs |
| `hc_prepare` (one site) | 0.194 -> **0.089** | 90 | 17.5 -> **8** | this pass |
| `rms_norm` | 0.025 | 90 | 2.3 | |
| `hc_mix` | 0.006 | 90 | 0.5 | |
| `lm_head` (4096 -> 154880, 317 MB) | 2.68 | 1 | 2.7 | 118 GB/s: one kernel, one big tensor |

The null row itself is 0.12-0.13 ms, i.e. one `mx::eval`, which is why the
per-call numbers only mean anything after subtracting it.

**Inside one MoE layer** (real weights, decode shape):

| stage | ms | share |
|---|---|---|
| router: logits matmul + 25-op top-k + the ids eval | 0.10-0.14 | 12% |
| 3 grouped expert launches (gate, up, down; 75 MB) | 0.49-0.51 | 60% |
| shared expert (3 trellis linears, 12.6 MB) | 0.12-0.13 | 15% |
| weighted sum + residual + the elementwise glue | ~0.1 | 12% |

So the layer is ~72% weight reads at the QMV's own rate, and the machinery is
the rest - not the other way round. The grouped expert launches read 25 MB in
0.147-0.166 ms, i.e. **150-170 GB/s**; the same launch over eight *different*
layers' experts (cold, 400 MB of distinct bytes) costs 3% more, and eight such
launches in one eval scale linearly, so this is the kernel's issue-bound rate
for these shapes, not the launch or the page cache. One big tensor (`lm_head`,
317 MB, 9680 output tiles) gets 118 GB/s from the same kernel.

**What the host sync actually costs.** With the ids handed over by the host
(the shipped path minus the `mx::eval` and the host-side buffer list), the
routed part of a layer drops from ~0.60 to ~0.50 ms: **~0.10 ms per layer,
4.2 ms per token, ~3%**. Removing it is worth having, but it is not where the
step goes.

**Why the two suggested mechanisms for removing it are not viable here:**

1. *A pointer table* (the kernel reading each member's weight address from a
   device buffer) is impossible: a Metal buffer's GPU address is not the host
   address MLX hands out, so a raw pointer stored in a buffer dereferences to
   zero (`/tmp/ptr_probe.cpp`: an mmap-backed view *and* a `malloc`'d
   `trellis_pack` buffer both read 0 through the table; MLX exposes no
   `MTLBuffer.gpuAddress`).
2. *An expert arena* needs 288 experts x 3 projections x 42 layers = 98.3 GB
   contiguous, and the checkpoint's routed experts are exactly that much: an
   anonymous copy is the 2x-loss case this file already measured (MoE
   123 -> 257 ms), and a page-remap arena (MAP_FIXED over the shard pages, one
   offset table per class, a per-member aligned/byte word-path split in the
   kernel) is a large, high-risk change for those 4 ms.

What *is* in the tree from this lever, all bit-identical by construction:

- `G5MoePrep` (glm5.h + glm5_moe.cpp): per-layer, built once on the first
  decode step - the router weight cast to f32 (the reference re-casts the whole
  288x4096 f16 matrix per layer, 14 MB of traffic), and one (E, in|out) f16
  stack per projection class (gate|up merged, in that member order) so the
  step's scale vectors are one device-side `mx::take(stack, ids)` instead of two
  eight-way concatenates.
- the merged gate|up launch, so the two 4096->2048 QMVs and their two input
  Hadamards become one launch each (member k of the merged launch keeps its own
  weight buffer, scale vector, split geometry and k-order - the scale stack is
  indexed at `E + id` for the up half, which is what the first version of this
  change got wrong: the real-checkpoint identity check below is what caught it).
- the shared expert is dispatched *before* the ids readback, so the GPU has that
  work queued while the CPU waits for the router instead of idling.
- `hc_prepare`'s kernel: four loads are issued before their four dependent FMAs
  in pass 1 and pass 2. Same accumulation order, four requests in flight
  instead of one: 2.2x on the site (0.194 -> 0.089 ms), which is 9 ms/token of
  the step. The kernel is one threadgroup per position streaming a
  (24, 16384) f32 matrix = 1.57 MB, i.e. it was ~8 GB/s of latency-bound
  single-SM traffic; 8-deep gave nothing more.

**Gates for this pass** (all green, same session as the numbers):

- `tests/moe_test.cpp`'s (f) section hashes the grouped, per-expert and general
  MoE against each other on quantized experts. It was silently vacuous before
  this pass - the new path returned before consulting `EXL3_MOE_GENERAL`, so the
  "general" hash was the new path again - and the three now genuinely differ.
- `tests/moe_bench.cpp` prints an identity line on the *real* layer 4 (2.7 GB of
  weights): `prepared == old-grouped == general`, byte for byte.
- the greedy text of the end-to-end bench is unchanged, and `hc_test` /
  `moe_test` / the rest of `tools/run_tests.sh` stay green.

**Honest distance to 14.3 ms/step.** A token reads ~6 GB of trellis weights
(the routed experts are 2.7 GB of it, qkv 1.7, o_proj 0.57, shared 0.53, the
lm_head 0.32) and the trellis GEMV delivers 118-170 GB/s on the shapes this
model uses, so the reads alone are 35-45 ms. 14.3 ms would need ~420 GB/s
*and* the ~58 ms of per-layer machinery above to disappear; the machinery is
per-layer *launches* (a `hc_prepare` site was 0.19 ms for one kernel launch,
`lm_head` 2.7 ms), which is a different problem from the one this pass set out
to solve. What this pass moved is the ~9 ms of `hc_prepare` and ~1-4 ms of
per-token re-layout; the per-layer sync it was aimed at is worth 4 ms.

### End to end, same session, interleaved, warm

Three rounds of `before`/`after` alternating in one session (16 tokens per take,
24 warm-up steps, best-of-3 repeats inside each run; every run's greedy text is
identical, first token 785):

| run | serial ms/step | serial tok/s | `--draft-mtp 4` ms/step | effective tok/s |
|---|---|---|---|---|
| before (HEAD) #1 / #2 / #3 | 125.8 / 172.3 / 146.0 | 7.95 / 5.80 / 6.85 | 88.0 / 134.4 / 119.5 | 11.36 / 7.44 / 8.37 |
| after #1 / #2 / #3 | 140.8 / 116.0 / 161.0 | 7.10 / 8.62 / 6.21 | 110.1 / 87.8 / 128.8 | 9.08 / 11.39 / 7.77 |
| **best** | **125.8 -> 116.0** | 7.95 -> 8.62 | **88.0 -> 87.8** | 11.36 -> 11.39 |
| median | 146.0 -> 140.8 | | 119.5 -> 110.1 | |

The spread between two runs of the *same* binary here is +-30% (125.8 to 172.3),
so a 5-10% change is not resolvable end to end on this box; the changes are
neutral-to-better on both the best and the median, and the component-level rows
above (sub-second, single process) are the instrument that actually resolves a
change of this size. (Both runs in the ticket's baseline came from a colder
page cache: the earlier session measured 152.8/116.8 ms/step for the same binary
that reads 125.8/88.0 here.)

## Fifth pass (2026-10-10): request-level batching, and the ceiling it hits

The 70 tok/s target is unreachable on ONE stream: a token reads ~6 GB of trellis
weights and the trellis GEMV delivers 118-170 GB/s on this model's shapes, so a
14.3 ms step would need ~420 GB/s sustained. The remaining route to the number is
**aggregate** throughput - several concurrent sequences sharing each weight read.
This pass builds that and measures where it stops.

### What was built

* `Glm5Model::batch_join / batch_step / batch_leave` (`src/glm5_forward.cpp`):
  B sequences advance by one token each in ONE fused forward pass, every tensor
  of the step carrying the slot on dim 0, so one step's weight reads serve the
  whole batch while each slot keeps its own state.
  * The KDA half needed **no new layer code**: `kda_layer`'s conv window and
    delta-rule matrix are already dim-0-batched and its recurrence is elementwise
    over them, so a `(B, k-1, C)` window and a `(B, H, kd, vd)` matrix are the
    entire change. (The state is 142 MB per slot - 34 layers x 64 heads x
    128 x 128 fp32 - which is the batch's irreducible memory cost.)
  * The MLA latent cache and the packed DSA indexer cache are `(B, T, ...)` with
    slot b's rows as the **last** `pos[b]` rows of its own block - blocks are per
    slot, so padding never aliases - which makes a step's append one concatenate
    at a common index and the per-slot mask/selection a host-side range
    (`mla_layer_batch`, `indexer_topk_batch` in `src/glm5_attn.cpp`).
  * `batch_join` prefills through the ordinary single-sequence path and copies
    its state in, so the prefill is untouched and a batch of one is
    bit-identical to serial decoding (gate below).
* `moe_layer_rows` (`src/glm5_moe.cpp`): the routed experts of a batch as K
  grouped launches per layer with B members each (the merged gate|up launch has
  2B). One Metal kernel takes 31 buffers - measured, 30 members + x + out fails
  to compile ("buffer attribute parameter is out of bounds") - so a single launch
  cannot hold K*B work items, and the per-rank split is what fits.
  **The rows of a batch do not share experts**: with 8 rows and top-8 of 288, the
  expected number of distinct experts is 288*(1-(1-8/288)^8) = 57.6, so the
  expert bytes (2.4 GB of the 6 GB) barely move per token. The batch amortises
  the launches, the host bookkeeping and the *dense* weights; it does not amortise
  the expert weights unless the rows route alike.
* `Engine::start_batch / submit_batch` + one worker thread (`src/engine.cpp`): a
  single inference thread owns the model, admits requests continuously, prefills
  a joining request through the single-sequence path **while the batch is
  mid-flight** (the batch keeps its own state, so the prefill cannot disturb it),
  and advances every live slot once per step. The per-request emission machinery
  (think routing, stop strings, streaming deltas) was extracted into one
  `Emitter` shared verbatim with the serial loop.
* `--serve --batch N` (default 8 for GLM-5-Next, `0` = one request at a time)
  routes every request through the scheduler; `Engine::submit_batch` falls back
  to the serial `generate()` whenever batching is off, the model is not
  GLM-5-Next, the request asks for the MTP drafter, or the worker died.

### Gates (real checkpoint, one binary, one session)

| gate | result |
|---|---|
| batch of one vs the serial decode (prefill row + 3 steps) | max abs logit diff = **0** (bit-identical) |
| 4 slots, same prompt/tokens: rows against each other | **0** (no cross-slot coupling) |
| 4 slots, same prompt/tokens: rows vs the serial row | 0.51 = **1.99%** of max logit, argmax identical |
| 4 slots, 4 different prompts, greedy (argmax per step + token sequence) | **identical to 4 serial runs** |
| `moe_layer_rows` vs per-row `moe_layer` vs the general gather/scatter path | **exact** in both comparisons (light fixture, `tests/moe_test.cpp` (g)) |

The 2% logit delta is a *shape* effect, not an algorithmic one: the light gate
compares rows=1 against rows=4 through the same code path and gets bit-equality,
so the trellis GEMVs do not move; what moves is MLX's batched fp16 reduction
order in the MLA/attention matmuls and the head. Token identity is the hard gate
and it holds for every prompt for the whole measured window.

Two traps this pass walked into, both worth keeping:

* the first version of the light gate **passed while the grouped MoE's member
  order was wrong**. Its fixture was 32-wide (below the Hadamard's 128-block, so
  `blocks_per_row = 0` - degenerate) and its random trellis words decoded through
  the mul1 codebook to NaN, so every comparison agreed vacuously (`NaN - NaN`,
  and `std::max(0, NaN) == 0`). The fix was a 128-wide fixture with byte-uniform
  codewords whose decode is finite (computed from the kernel's own formula), plus
  an `isfinite && |out| > 0` gate in front of the comparison;
* the *test* of the batch was off by one row (a step's logits follow the token it
  was fed), which reads exactly like a wrong kernel.

### Measured

`tests/glm5_batch_test.cpp <model> 4 --bench`, two interleaved runs of the
pre-dedup binary (best of 3 rounds inside each run; the same binary's two runs
differ by up to 10% in the light cells and by 4x in the mixed-8 cell, which is
the page-cache state - see the noise note below):

| case | run A ms/step | run B ms/step | best | aggregate tok/s |
|---|---|---|---|---|
| B=1 serial (one stream, reference path) | 229.99 | 244.60 | 229.99 | 4.35 |
| B=1 batch | 118.85 | 117.36 | 117.36 | 8.52 |
| B=2 | 218.84 | 184.14 | 184.14 | 10.86 |
| B=4 | 314.09 | 302.84 | 302.84 | 13.21 |
| B=8 | 495.11 | 453.05 | 453.05 | 17.66 |
| B=4, 1 prompt (shared routing) | 178.69 | 214.59 | 178.69 | 22.38 |
| B=8, 1 prompt | 284.52 | 276.47 | 276.47 | 28.94 |
| B=8, 2 prompts | 466.58 | 350.53 | 350.53 | 22.82 |
| B=8, 8 prompts | 2270.35 | 589.52 | 589.52 | 13.57 |
| B=16, 1 prompt | 538.83 | 442.95 | 442.95 | 36.12 |
| B=16, 4 prompts | 560.91 | 498.25 | 498.25 | 32.11 |

The four correctness gates in the same runs: one-slot batch vs serial
max|logit diff| = **0**; four identical slots vs serial 0.0166-0.0176
(0.065% of max|logit|) with rows bit-identical to each other; four different
prompts vs four serial runs 0.3945-0.4453 (1.29-1.52%), token identity exact.
The `<= 2%` identical-slots gate is green at 0.065% in both runs (and at both
`steps 4` and `steps 6`) - the 1.99% this file recorded earlier is not
reproducible on today's build, so both numbers are reported and the gate is
left exactly as it was.

The "prompts" column is the number of *distinct* prompts the rows are drawn
from, i.e. how many distinct routings (and so experts) the step touches: see
the sixth pass for what that gradient means.

### The ceiling, from the byte model

Split the per-token bytes of the budget above by whether a row shares the read:

* **shared by every row** - qkv 1.7 GB, o_proj 0.6, the shared expert 0.53, the
  lm_head 0.32, MLA + indexer ~0.3 - is **~3.4 GB per STEP**, not per token;
* **routed experts** are ~0.34 GB per expert-set (2.7 GB / 8) times the number of
  **distinct** experts the B x 8 selections touch. For B independent sequences
  that is `288 * (1 - (1 - 8/288)^B)`: 8 at B=1, 15 at B=2, 28 at B=4, 57.6 at
  B=8. The expert bytes per *token* therefore fall from 2.7 GB to only ~2.4 GB at
  B=8 - the batch does not amortise them.

So a step costs `3.4 + 0.34*D(B)` GB and the aggregate is `B / step_time`. At the
trellis GEMV's own measured rate (150-170 GB/s on these shapes) and with the
machinery cost ignored, that is:

| B | distinct experts D(B) | GB/step | ms/step at 160 GB/s | aggregate tok/s |
|---|---|---|---|---|
| 1 | 8.0 | 6.1 | 38 | 26 |
| 2 | 15.2 | 8.6 | 54 | 37 |
| 4 | 28.0 | 12.9 | 81 | 49 |
| 8 | 57.6 | 23.0 | 144 | 56 |
| 16 | 104.8 | 39.0 | 244 | 66 |
| 32 | 171.5 | 61.7 | 386 | 83 |

Those are ceilings with *zero* machinery: 70 tok/s aggregate needs ~20 concurrent
sequences **and** a step whose non-weight work is free. If the rows shared their
routing (identical prompts, or experts deduplicated across rows), the expert term
collapses to 2.7 GB and the same table reads 4.3/5.0/6.1/6.1/6.1/6.1 GB per step
- 210 tok/s aggregate at B=8. That is why "expert dedup / expert-order pair
execution" is the mechanism both reference engines lean on
(`docs/research/mlx-serve.md` §5, `docs/research/omlx.md` T3), and why the
measured numbers below are reported for both prompt regimes.

### What did NOT pay off

* **A single grouped MoE launch per projection class**: blocked by Metal's
  31-buffer limit (`kMaxGroupMembers = 29`); the per-rank split costs K launches
  per layer instead of 1. mlx-serve's device-indexed expert bank is the way
  through it (`docs/research/mlx-serve.md` §5) and needs a descriptor table for
  our 288 scattered expert tensors.
* **Deduplicating experts within a rank** (the other half of the launch story):
  at 8 rows the expected saving is ~11% of the expert bytes (57.6 distinct of 64
  selections), which is not worth the padded-row waste it costs.
* **Speculative decoding inside the batch**: it is deliberately not enabled -
  `submit_batch` falls back to the serial path when a request asks for the MTP
  drafter, because the draft rows (and the rewind bookkeeping) are per sequence.

## Fourth pass (2026-10-10): the GEMV's geometry, checked against the same-format reference

`ddalcu/sushi`'s `exl3-module` (branch `exl3-module`, commit 4431513) runs the
same EXL3 trellis format on the same class of machine and its docs quote **pair
GEMV ~313 GB/s, down fused ~275 GB/s** against a 538 GB/s streaming peak, with
the decode GEMVs described as *index-ALU bound, not DRAM bound*. This pass
checked what of that applies to us. The instrument is `tests/glm5_trellis_bw.cpp`,
which now measures the checkpoint's **real** shape classes (read from the shard
headers, visual tower excluded), reports the raw QMV kernel separately from the
two Hadamard dispatches, prints the row-block weight-pass count (so a multi-row
cell's GB/s is not read as a DRAM rate) and reports the median of 5 repeats.

### What the kernel actually is

Per (16x16 output tile, k-tile) simdgroup-iteration the kernel retires ~110
warp-instructions: 8 codeword decodes per lane (a 16-bit sliding-window extract,
the mul1 multiply-hash and a half-precision fma), their 8 FMAs, 4 packed-word
loads (2 per group of four weights), the 64-bit window shift and the loop's
address math. On the 4096x24576 K4 tensor 393216 of those iterations run in
0.168 ms, i.e. **256e9 warp-instructions/s - the machine's entire issue width**.
The kernel's tile rate is therefore ~2.4e9 tiles/s (300 GB/s of 128-byte K4
tiles, and 2.4e9*K/2 bytes/s for any K), which is the reference's 313 GB/s of K4
tiles at 2.44e9 tiles/s to within 2%. **The GB/s difference between shapes is
bytes per tile, not a kernel defect:** at the same tile rate a K2 tensor reads
64 bytes/tile and delivers ~150 GB/s, and 22 of this model's 43 MoE layers are K2.

### The fix: the tile/threadgroup geometry rule

`pick_geom` (src/trellis_gemm.cpp) replaces the old auto rule, which picked
`nt=16` for the lm_head's 9680 output tiles (mb*nt = 16 -> 128 accumulator
registers, a spill) and `nt=1` for every tensor with <=1024 output tiles (where
the per-iteration activation load, its two shuffles and the loop's address math
are a third of the work). The new rule takes the largest power of two that stays
inside the register budget (mb*nt <= 4) and leaves ~2048 simdgroups in flight.
**Only nt and mb moved. The k-split count remains a function of (k_tiles,
n_tiles) and nothing else, because it is part of the f32 association the caller
sums with `mx::sum` - changing it would not be bit-identical.**

Raw QMV, same session, interleaved builds (old rule vs new), 64 iterations per
graph, median of 5, best of 2 rounds:

| shape (real checkpoint geometry) | rows | before ms | after ms | speedup | before GB/s | after GB/s |
|---|---|---|---|---|---|---|
| lm_head 4096x154880 K6 | 1 | 2.44 | 1.05 | **2.32x** | 193 | 450 |
| lm_head 4096x154880 K6 | 2 | 13.28 | 1.27 | **10.5x** | 34 | 362 |
| exp8 gate\|up+down (8 members, one launch each) | 1 | 0.308 | 0.241 | **1.28x** | 160 | 202 |
| exp8 gate\|up+down K2 | 1 | 0.297 | 0.228 | **1.31x** | 109 | 134 |
| exp8 gate\|up+down K3 | 2 | 0.400 | 0.315 | **1.27x** | 119 | 149 |
| dense gate/up 4096x12288 K3 (nt 1 -> 2) | 1 | 0.1193 | 0.1130 | 1.06x | 150 | 163 |
| mla q_b 1536x16384 K4 (nt 1 -> 2) | 1 | 0.0565 | 0.0485 | 1.16x | 210 | 239 |
| kda qkv_proj 4096x24576 K4 (nt was already 2) | 1 | 0.1690 | 0.1726 | 1.00 | 283 | 289 |
| mla o_proj 16384x4096 K4 (nt unchanged) | 1 | 0.1424 | 0.1424 | 1.00 | 215 | 215 |
| exp gate 4096x2048 K3, single tensor (nt unchanged) | 1 | 0.0237 | 0.0237 | 1.00 | 133 | 133 |
| f16 matmul 4096x24576 (MLX's own GEMV, same bytes) | 1 | 0.60 | 0.60 | - | 327 | 327 |

The cells marked 1.00 are shapes whose geometry the new rule leaves alone; their
apparent 1.08-1.19x in the first cut of this table was box drift.

### The component-level end-to-end effect (tests/moe_bench.cpp, real checkpoint)

Two interleaved rounds of `moe_bench 20 1 <model>`, best and median ms per call,
minus the `null` row (one elementwise op + sync: 0.126/0.140 before,
0.114/0.124 after, i.e. the "after" runs were ~10% faster on the box, so the
raw ratios overstate every win by about that much):

| component | before (best/median) | after | after, null-adjusted |
|---|---|---|---|
| lm_head (trellis, real vocab) | 2.792 / 2.882 | 1.494 / 1.510 | **1.98x** |
| 3 grouped linears gate/up/down (one MoE layer's GEMVs) | 0.601 / 0.662 | 0.535 / 0.560 | 1.20x |
| full moe_layer (shipped path, + ids eval) | 0.876 / 0.949 | 0.813 / 0.853 | 1.11x |
| 2nd MoE layer (layer 12, its own K) | 0.992 / 1.090 | 0.926 / 0.964 | 1.13x |
| 8x grouped QMV over 8 layers (real bytes) | 1.347 / 1.421 | 1.231 / 1.264 | 1.12x |
| 8x raw QMV in one eval (independent) | 1.226 / 1.285 | 1.009 / 1.043 | 1.25x |
| shared expert (K4; geometry unchanged) | 0.234 / 0.283 | 0.231 / 0.250 | ~1.1x (noise) |
| kda_layer (geometry unchanged) | 0.813 / 0.890 | 0.796 / 0.835 | 1.05x (noise) |
| mla_layer (geometry unchanged) | 1.581 / 1.690 | 1.545 / 1.589 | 1.06x (noise) |

That is **~5 ms/token** (lm_head 1.4 ms + 42 MoE layers x ~0.08 ms), i.e. ~3-4%
of the 116-160 ms step - and the whole-step instrument cannot resolve it: the
four interleaved `glm5_bench` runs of this window read 237.7/147.6 ms/step for
the old build and 208.8/202.7 for the new one (serial BEST 4.21/6.78 vs
4.79/4.93 tok/s; `--draft-mtp 4` 104.6/168.4 vs 120.1/144.7 ms/step), i.e. the
same binary's two rounds differ by 1.6x. Report the component rows, not the
step, for a change of this size. `moe_bench`'s real-layer identity line stays
BIT-IDENTICAL (`prepared == old-grouped == general` on layer 4's 2.7 GB).

### Bit-identity gates (all green this pass)

| gate | command | result |
|---|---|---|
| decode vs the reference decoder + the byte-alignment contract, real tensors (lm_head K6 634M elements, qkv K4, q_b K4, expert gate K3) | `/tmp/glm5_exl3_decode_test <model>` | `trellis decode: PASS`, 0 mismatches everywhere, rc=0 |
| forward / prefill-vs-decode invariant + greedy text | `/tmp/glm5_forward_test <model>` | rc=0, first token 785 (the recorded one) |
| grouped == per-expert == general, hashes | `/tmp/moe_test` | rc=0, all three `eebe4e4e51c6c683` |
| real-layer MoE identity on 2.7 GB | `moe_bench 20 1 <model>` | `60ad94aefeb44530` for prepared / old-grouped / general |
| greedy text across the old and new builds | 4 interleaved `glm5_bench` runs | 1 distinct text (identical) |
| kernel output hashes over the 25 real tensor classes x rows {1,2,3,13,26} (raw QMV + trellis_linear, 250 hashes) | `/tmp/glm5_gemv_nt_test` | byte-identical to the pre-change kernel (HEAD~1) |
| nt/mb scheduling inertness (8 shapes x 6 row counts x 12 configs + grouped/scale-stack forms) | `/tmp/glm5_gemv_nt_test` | PASS |

### Honest distance to the reference, and what is left

- The reference's 275-313 GB/s is a K4-class figure and we now match it: qkv 308,
  lm_head 450-476, mla/kda o_proj 300-325 GB/s. On the K2/K3 tensors that
  dominate this model the same 2.4e9 tiles/s yields 134-202 GB/s, and that is
  arithmetic, not memory: 8 decodes x ~11 instructions per 8 weights at rows=1,
  plus one FMA each. Halving that would be needed to approach DRAM peak, and the
  reference's own notes rule out "ALU trims in the MUL1 decode".
- Rows > 1 amortise the decode over mb rows and are the only lever on bytes per
  weight at fixed format: at rows 13/26 the kernel runs at ~200-215 GB/s of
  actual DRAM traffic (the printed x4/x7 pass count), which is what a verify
  batch pays per row block.
- The remaining gap is not in this kernel: a 0.83-0.95 ms MoE layer is ~0.5 ms of
  GEMV and the rest is the Hadamard + dispatch + reduce tail. Folding the input
  Hadamard into the QMV (the reference's `pair GEMV prepare`: the same 128-thread
  butterfly, so bit-identical by construction) is the next kernel-side step,
  worth ~1 of 3 dispatches per linear, i.e. ~2 us x ~1450 linears/token.

### Did not pay off (measured this pass; do not retry blind)

- **Raising the k-split count together with nt** (to buy parallelism for small
  tensors): not bit-identical, because it changes the split-sum association, and
  it measured no faster.
- **nt=8/16** (mb*nt = 64/128 accumulator registers): grouped K2 148 -> 105 GB/s,
  lm_head 476 -> 342/218 GB/s.
- **nt=4 on the big single tensors**: 4096x24576 loses 17% (266 against 317 GB/s)
  - 384 threadgroups is one wave.
- **Forcing nt>1 on the small single tensors** (f_a 4096x256, mla kv_a 4096x512,
  a *single* 4096x2048 expert shape): within +-1% of nt=1. Those grids are
  latency/dispatch bound (`f_a` is 0.5 MB and its kernel is ~9 us, of which the
  tile loop is ~2 us), so they need the `multi_qmv` treatment - several small
  projections of the same layer in one launch - which is plumbing, not kernel
  work.
- **The codebook as a device table** (`EXL3_QMV_LUT=1`, tried and removed in this
  pass): a 65536-entry fp16 table of `codebook_value` (src/exl3.cpp's own
  builder) with the mul1 multiply-hash replaced by one gather, the one
  codebook-level lever that had a measurable ceiling (the whole codebook is 23%
  of the kernel under the `EXL3_QMV_NODECODE` control). It is **bit-exact** -
  `tests/glm5_gemv_nt_test.cpp` compares both paths in-process over the 25 real
  classes x rows (250 hashes, ok) and `glm5_exl3_decode_test` with the knob on
  reports 0 mismatches over the real checkpoint's lm_head (634,388,480 elements)
  and three more tensors - and **1.5-3.4x slower** at rows 1: lm_head 1.03 ->
  3.46 ms, qkv 0.167 -> 0.527, exp8 gate|up+down K3 0.238 -> 0.759, single exp
  gate K3 0.0240 -> 0.0525, dense 0.104 -> 0.277, f_a 0.0087 -> 0.0131. The tell
  is the tile rate: 2.2-2.4e9 tiles/s for the arithmetic becomes a flat
  ~0.6-0.75e9 for every K, i.e. the kernel turns latency-bound on the table. A
  128 KiB table cannot be cached where ~10 ALU ops live, and each of the eight
  gathers per lane-tile is an L2 round trip on the critical path, so no table can
  pay for itself against this codebook's arithmetic.
- Prior passes' negatives still stand and were not revisited: the simdgroup
  shuffle word fetch, the per-lane window layout, mb=8/nt=4 variants.

## Sixth pass (2026-10-10): expert-order pair execution, and the serving thread

Two independent pieces of work: the MoE batch's work items now execute in expert
order (the mechanism oMLX's `moe_verify_gather` transcribes, `docs/research/
omlx.md` §B), and the HTTP serving path - which served *nothing* - was
reproduced, explained and fixed.

### 1. Expert-order pair execution (`src/glm5_moe.cpp`)

A step has B*K (row, expert) pairs. `moe_layer_rows` used to walk them in routing
order, one grouped launch per rank, so an expert that two rows selected had its
weights streamed once per row. It now sorts the pairs by expert id and executes
that list: pairs sharing an expert run back to back and its tile is read from
cache for all but the first of them.

Only the ORDER changes, which is what keeps it in the bit-exact layer: each pair
is still one `rows=1` member of a grouped launch with its own weight buffer and
its own slice of the prepared scale stacks, so the kernel's tile geometry,
k-order and f32 split sum are the ones that pair had before, and a `take` (a pure
move) puts the per-pair results back into the rank-major `(K, B, 1, H)` stack the
weighted sum consumes exactly as it did. The chunking is over the *pair list*
(14 pairs per merged gate|up launch, 29 per down launch - the 31-buffer limit),
not over B, so the old `2*B <= 29` bound is gone and the path now covers any
batch width, including the 13-16 rows a matrix-unit kernel would need.

Measured on the real checkpoint, `tests/glm5_batch_test.cpp <model> 4 --bench`,
four interleaved processes (old, new, new, old; best of 3 rounds inside each
process - the reported number - and both processes shown because the box's
page-cache state moves the heavy cells by up to 10% and one old cell by 4x):

| case (distinct prompts) | before ms/step | after ms/step | before tok/s | after tok/s | change |
|---|---|---|---|---|---|
| B=1 serial (reference path) | 230.0 / 244.6 | 225.6 / 272.2 | 4.35 | 4.43 | - |
| B=1 batch (4) | 118.9 / 117.4 | 118.3 / 113.7 | 8.52 | 8.80 | +3% |
| B=2 (4) | 218.8 / 184.1 | 197.8 / 179.3 | 10.86 | 11.15 | +2.7% |
| B=4 (4) | 314.1 / 302.8 | 303.1 / 310.6 | 13.21 | 13.20 | ~0 |
| B=8 (4) | 495.1 / 453.1 | 468.6 / 414.9 | 17.66 | **19.28** | **+9.2%** |
| B=4 (1, shared routing) | 178.7 / 214.6 | 180.6 / 176.6 | 22.38 | 22.65 | +1.2% |
| B=8 (1, shared routing) | 284.5 / 276.5 | 405.6 / 269.5 | 28.94 | 29.69 | +2.6% (cell noisy: 405.6) |
| B=8 (2) | 466.6 / 350.5 | 434.3 / 321.4 | 22.82 | 24.89 | +8.3% |
| B=8 (8) | 2270.4 / 589.5 | 510.8 / 461.9 | 13.57 | **17.32** | **+27.6%** |
| B=16 (1, shared routing) | 538.8 / 443.0 | 567.3 / 472.4 | 36.12 | 33.87 | -6.7% |
| B=16 (4) | 560.9 / 498.3 | 558.5 / 507.3 | 32.11 | 31.54 | -1.8% |

**The honest reading of that table.** The mechanism can only remove *duplicated*
expert reads, and at B=8 with 8 distinct routings there are 64 pairs over 57.6
distinct experts: 10% of the expert reads, i.e. at most ~4% of the step. The
measured +27.6% in that cell is therefore NOT the ordering - it is the launch
change that came with it (16 launches per layer in the old per-rank split, 8 in
the sorted chunks, with 28 members instead of 16 per gate|up launch), and the
cell is the noisiest one in the table (the page-cache state at the highest expert
count). The B=16 cells, where the launch count did not change as much, are within
noise or slightly worse. So: expert-order execution is neutral-to-better and
cheap, but it is not what closes the mixed/same-prompt gap - and the gradient
below says what is.

**The sizing measurement: what the mixed/same-prompt gap actually is.** The
added "distinct prompts" axis separates expert-set size from everything else. At
B=8, after the change (best of two processes):

| distinct prompts | expected distinct experts | ms/step | aggregate tok/s |
|---|---|---|---|
| 1 (all rows share one routing) | 8 | 269.5 | 29.69 |
| 2 | 15 | 321.4 | 24.89 |
| 4 | 30 | 414.9 | 19.28 |
| 8 (one per row) | 58 | 461.9 | 17.32 |

Each extra ~15 distinct experts costs 65-95 ms/step at B=8: that is the expert
term, and it is *linear in the number of distinct experts*, not in the number of
pairs. Row blocking of a duplicated expert (one member with `rows=r` per expert,
ceil(r/4) weight passes instead of r - `mb = min(rows, 4)` - so 16 rows of one
expert are 4 passes instead of 16) is the only thing that can cut it, and it pays
off exactly in the shared-routing regime. Ordering cannot: it removes reads, not
distinct experts.

Row count: at a fixed routing (1 prompt) the aggregate is 22.4 (B=4) -> 29.7
(B=8) -> 33.9 (B=16) tok/s, i.e. the marginal row costs ~23-25 ms and the
marginal gain is falling. The best aggregate measured anywhere in this pass is
**33.9 tok/s (B=16, one prompt)** and 31.5 tok/s (B=16, 4 prompts).

### 2. The serving thread: the server served nothing

Reproduction, on the tree before the fix, on the real checkpoint (engine-level,
no httplib, so nothing but the engine can explain it):

* `start_batch(0)` + 4 concurrent `submit_batch`: **4/4 requests threw**
  `There is no Stream(gpu, 0) in current thread.` after 22-30 ms - stream 0 is
  the thread that loaded the model.
* `start_batch(4)` + 4 concurrent `submit_batch`: the scheduler thread died on
  its first `batch_join` (`engine: batch worker stopped: There is no Stream(gpu,
  2) in current thread.`), its catch returned without answering anything, and all
  4 submitters waited forever - the probe's watchdog had to kill the process
  after reporting 4/9 requests completed. This is the deadlock the ticket's run
  hit: the next request after a failed one blocks in `submit_batch`'s `p->cv.wait`
  on a scheduler that no longer exists.
* the same failure through HTTP was the 500 body in
  `/tmp/exl3_batch_server/w_0.json` (arriving in 0.029 s).
* and the run produced *nothing* even after the requests were answered: the
  bench script's `wave()` ended in a bare `wait`, which in bash waits for every
  child of the shell - the server included, since `start_server` launches it in
  the background from the same shell. The server never exits, so the first wave's
  `wait` blocked forever with its response files already complete (reproduced:
  `/tmp/exl3_sweep/w_0.json` finished, 24 tokens, `w_0.time` 34.08s, no wave
  summary ever printed). Two independent bugs, one symptom.

Mechanism:

* MLX binds a `Stream` to the thread that created it (`mlx/backend/
  metal/device.cpp`: `get_command_encoder` looks the stream up in a thread-local
  encoder map and throws if it is not there). The loader's trellis weights were a
  lazy `slice -> view -> reshape` chain over each shard's page-aligned mapping,
  deliberately left unevaluated - every step of it is metadata-only
  (`slice()` is `shared_buffer_slice`, `View`/`Reshape` `copy_shared_buffer`, no
  allocation, the words stay a zero-copy view) - so the weights carried the
  loading thread's stream into every request served on an httplib worker.
* the hang: the scheduler thread died on its first request (the exception
  escaped `batch_join`), and `Engine::batch_worker`'s catch set `batch_stop_` and
  returned without answering the live requests or the queue, so every later
  `submit_batch()` waited forever on `p->cv` that nobody would notify.

After the fix, the same probe: **9/9 requests** complete (4 through `--batch 0`'s
serial mode, 4 fused through `--batch 4`, one stopped mid-flight), rc=0 - and the
fused four take 4.5-4.7 s where the serial mode's four take 13.6-27.9 s.

Fix:

* `mx::eval()` the trellis view where it is built: resolves it on the loading
  thread, zero-copy and zero-cost (MLX active memory after load is 101 GB for
  104.7 GB of used weights, before and after).
* one inference thread: the scheduler thread serves every request now, including
  `--batch 0` (its serial mode, one request at a time through `generate()`) and
  the requests that cannot be batched (MTP draft rows, empty prompts), instead of
  whichever HTTP worker picked the request up.
* the scheduler's exit path answers everything it holds, releases the live batch
  slots (without that reset the next user of `batch_step()` sees a stale slot
  count - this pass's own test caught it) and clears its liveness flag under the
  queue lock, so a submit that races the exit either enqueues into a live
  scheduler or is served on its own thread; a failed cycle answers live + queued
  requests and keeps the worker alive.

After: 4 concurrent requests from fresh threads all complete; 9/9 requests across
`--batch 0/4/8` complete; `stop_batch()` with a request in flight returns in
0.02-1.1 s. `tests/glm5_batch_test.cpp` section 4 is the permanent gate: two
concurrent requests through the scheduler must equal their serial runs, and
stopping the scheduler with a request in flight must return.

### 3. The N-sweep through HTTP

`MAXTOK=8 REPS=1 CONC_LIST="1 4 8" WARM=1 tools/batch_server_bench.sh`, two
interleaved rounds (the sweep FINALLY produced numbers - the bare `wait` in
`wave()` was the reason it never had):

| round / setting | conc | wall s | completion | aggregate tok/s | per-request latencies s |
|---|---|---|---|---|---|
| 1, `--batch 0` | 1 | 22.76 (warm-up) | 8 | 0.35 | 22.72 |
| 1, `--batch 0` | 1 | 1.61 | 8 | 4.96 | 1.58 |
| 1, `--batch 0` | 4 | 24.65 | 32 | 1.30 | 1.60 8.72 15.79 24.60 |
| 1, `--batch 0` | 8 | 49.68 | 64 | 1.29 | 5.79 ... 49.62 |
| 1, `--batch 8` | 1 | 20.40 (warm-up) | 8 | 0.39 | 20.36 |
| 1, `--batch 8` | 1 | 1.56 | 8 | 5.12 | 1.53 |
| 1, `--batch 8` | 4 | 28.19 | 32 | 1.13 | 25.82 28.14 28.14 28.14 |
| 1, `--batch 8` | 8 | 54.26 | 64 | 1.18 | 50.44 ... 54.19 |
| 2, `--batch 0` | 1 | 20.76 (warm-up) | 8 | 0.39 | 20.72 |
| 2, `--batch 0` | 1 | 1.75 | 8 | 4.57 | 1.72 |
| 2, `--batch 0` | 4 | 26.21 | 32 | 1.22 | 1.69 8.98 16.90 26.15 |
| 2, `--batch 0` | 8 | 50.31 | 64 | 1.27 | 3.87 ... 50.24 |
| 2, `--batch 8` | 1 | 21.15 (warm-up) | 8 | 0.38 | 21.11 |
| 2, `--batch 8` | 1 | 1.60 | 8 | 5.00 | 1.57 |
| 2, `--batch 8` | 4 | 21.71 | 32 | 1.47 | 20.96 21.65 21.66 21.66 |
| 2, `--batch 8` | 8 | 50.14 | 64 | 1.28 | 46.00 ... 50.08 |

Three things in that table, and only the first is about batching:

* **the server works now**: every wave returns a real completion, and `--batch
  8` fuses - the 8 requests of a conc-8 wave finish *together* (46-54 s, six of
  them exactly equal) where `--batch 0` serializes them (3.9 -> 50.2 s, strictly
  increasing). The engine-level numbers match the in-process instrument
  per-request: a warm single request is 1.56-1.75 s for a 26-token prefill plus 8
  decode tokens, i.e. ~0.2 s/step, which is this box's serial decode rate;
* **the box, not the server, sets the level**: the same request costs 20-23 s as
  the wave's warm-up (first pass over the step's weight pages) and 1.6 s
  immediately afterwards - a 13x spread on identical work. With a 105 GB
  checkpoint on a 128 GB machine the step's expert pages are evicted between
  waves, so a served step whose experts are cold runs at the external SSD's
  2.6 GB/s instead of the GPU's 130-200 GB/s (docs/perf.md's own warning, and the
  reason `WARM=1` exists). The conc-4/8 aggregates (1.1-1.5 tok/s) are
  therefore *SSD-bound end-to-end latencies*, not the fused step's rate: they are
  not comparable with the in-process table above, which measures a warm step with
  the prefill excluded and takes the best of 3 rounds;
* **the prefill is a real part of the served latency and was never in the
  in-process numbers**: 26 prompt tokens + 8 decode tokens = 1.6 s warm, and the
  prefill of a joining request is run on the scheduler thread (one chunked
  forward per join), so a wave of N requests pays N prefills end to end. Nothing
  in this pass changed that; it is the next thing the served path would need if
  the HTTP rate rather than the step rate is the target.

An honest note on the sweep's precision: with one wave per cell and a page-cache
state that the 4 server starts keep changing, the conc-4/8 cells move by ~30%
between rounds; the conc-1 cells (1.56-1.75 s) and the warm-up/steady ratio are
the parts that reproduce.

### Gates (real checkpoint, one session)

| gate | where | result |
|---|---|---|
| one-slot batch == serial decode | `glm5_batch_test` §1 | max abs logit diff = **0** |
| four identical slots == serial | §2 | 0.0166-0.0176 = **0.065%** of max\|logit\|, rows bit-identical to each other; the `<= 2%` gate is GREEN (the 1.99% recorded in the fifth pass is not reproducible at `steps 4` or `steps 6`; the gate itself is untouched) |
| four different prompts == four serial runs | §3 | token identity at every step and argmax identity every step; max abs logit diff 0.3945-0.4453 = 1.29-1.52% (gate 5%) |
| serving thread | §4 | see above |
| grouped == per-expert == general (hash) | `moe_test` (f) | `eebe4e4e51c6c683` for all three, as recorded |
| rows path == N single-row calls, and == the general path | `moe_test` (g) 4 rows, (h) 11 rows = 33 pairs over 4 experts (three gate\|up chunks, two down chunks, every expert selected several times) | max \|row diff\| = **0** in both, vs single-row and vs general |
| nt/mb scheduling + 25 real shapes x rows {1,2,3,13,26} (250 hashes) | `glm5_gemv_nt_test` | `nt/mb invariance: PASS` rc=0; `src/trellis_gemm.cpp` and the harness are byte-identical to the run the hashes were recorded from (`git diff a6b8d51 -- src/trellis_gemm.cpp tests/glm5_gemv_nt_test.cpp` is empty), so the table cannot have moved |
| prefill vs incremental decode | `glm5_forward_test <model>` | max\|diff\| = 0.0352 on max\|logit\| 23 (relative 0.00153), argmax 785 vs 785, top-5 overlap 5/5 |
| greedy text + first token | `glm5_forward_test` | first token 785, greedy 32-token text identical to the recorded run; rc=0 |
| trellis decode vs the reference decoder + byte alignment (lm_head K6 634M elements, qkv K4, q_b K4, expert gate K3) | `glm5_exl3_decode_test <model>` | `trellis decode: PASS`, 0 mismatches |
| the light suite (routing, grouped MoE, KDA rewind/recurrence, MLA incremental, indexer, hc, chat template, pretokenizer) | `tools/run_tests.sh --fast` | all PASS, `== done (rc=0) ==` |

### Honest distance to 70 tok/s

The best aggregate measured in this pass is 33.9 tok/s (B=16, shared routing) and
31.5 tok/s (B=16, 4 prompts) - 45-48% of the target, and B=8 mixed-prompt moved
17.7 -> 19.3 tok/s. Two things stand in the way, and the first is not about
bandwidth:

1. **Per-weight ALU.** The trellis GEMV spends 0.43 issued warp-instructions per
   weight, which is the machine's entire issue width (fourth pass) - the mixed
   case is bound by *work*, not by the weight bytes (a mixed B=8 step's ~5.8 GB
   at the kernel's own 134-202 GB/s would be ~30-45 ms of its ~460).
   `docs/research/mlxfast-bonsai2.md`
   measures the escape (packed stream -> int8 threadgroup tiles -> simdgroup
   matrices, or a native packed tensor op), and it needs 13-16 rows: that is why
   this pass made the pair layout free of any B bound and measured B=16. It is
   the order-of-magnitude lever.
2. **A real dedup, not an ordering.** One member with `rows=r` per expert turns
   r duplicate pairs into ceil(r/4) weight passes - the only way to cut the ALU
   work a duplicated expert costs, and the exact regime where the shared-routing
   row is 1.7x the mixed one. It has no gate yet:
   `trellis_linear_group_rows(rows=R)` has never been checked per row against R
   single-row calls, and the MoE path must not use it before that gate exists
   (its per-row k-order is presumably the same, but "presumably" is not the
   standard here).
3. **The per-row machinery.** At a fixed routing the marginal row costs ~23-25 ms
   (B=4 -> B=16: 176.6 -> 472.4 ms), so the aggregate saturates near 34 tok/s
   long before the expert term (or 70 tok/s) is reached.

## Seventh pass (2026-10-11): the staged matrix path for the batched rows

The sixth pass left one lever named and unbuilt: the classes whose weights do not
diverge per row (kda `qkv_proj`, kda/mla `o_proj`, the shared expert, the dense
MLP, mla `q_b`) pay their decode once per *row block* (`mb = min(rows,4)`), so at
the row counts a request-level batch reaches they stream the same packed words 4
times at 16 rows and 7 at 26 - and the decode (a mul1 multiply-hash and its half
fma, ~10 issued lane-instructions per weight) is what the machine's issue width
is spending. This pass moves those classes onto a staged fp16 + simdgroup_matrix
kernel and measures it per class and at B=16.

### What was built

* `src/trellis_gemm.cpp`: a second kernel in the QMV family (`exl3_qmv_mat_*`,
  built by the same `make_qmv_kernel` factory so the codebook prologue, the
  member head and the buffer list cannot drift apart). It decodes each (16k x 16n)
  packed tile ONCE per threadgroup into a threadgroup fp16 tile and then
  multiplies with `simdgroup_matrix`; the tile is staged (n, k)-major with 16
  halves per row, which is both what a lane's k pairs want (two `half2` stores
  per group of four weights) and what the B fragment's `simdgroup_load(...,
  transpose = true)` reads as W[k][n]. One warp per output tile, one barrier per
  k-tile, the next tile's words loaded a tile ahead of their decode.
  * Gate: `TrellisTuning.mat_min_rows` (default 16, `EXL3_QMV_MAT_MIN_ROWS`) AND
    a decode-shaped call - one whose sequence axis is 1, i.e. (B, 1, hidden) from
    the request-level batch (or a flat 2-D (rows, in) benchmark call). A prefill
    chunk is (1, L, hidden) with L > 1 and keeps the byte-exact QMV, so the
    single-sequence path is bit-identical *including its prefill* (the ticket's
    hard gate) and a wide forward does not pay the staged kernel's rows/MAT_MB
    weight passes - for a prefill the answer is a K-blocked GEMM over the batch,
    not this kernel. Below the threshold every call is on the QMV anyway.
  * Scope: the single-tensor entry points (`trellis_linear`, `trellis_matmul_raw`)
    only. The grouped forms keep the QMV - their activation is (n, rows, in) with
    the sequence axis already flattened into `rows`, so the decode-shaped test
    cannot be made there, and the two callers that reach rows >= the threshold are
    the model's routed experts (rows = 1 per member) and moe_layer's general path,
    whose (K, B*L, H) activation includes the single-sequence prefill.
  * The staged path picks its own k-split count (`mat_split_count`): the QMV's is
    tuned for `n_tiles * 4` warps and the staged kernel runs one warp per tile, so
    it aims at ~4096 threadgroups instead. This matters: at the QMV's count the
    narrow-output classes (mla `o_proj`, kda `o_proj`, the shared expert, kv_a)
    ran at 25-35 GB/s, i.e. no better than the QMV - they are latency-bound, and
    the split is the only parallelism axis the staging leaves.
* `trellis_path_info()` (`src/trellis_gemm.h`): which path, row block, padded rows
  and pass count a tensor will take at a given row count - so the benchmarks and
  the batch test report the path they measured instead of re-deriving the rule.

### Measured, per class (`tests/glm5_trellis_bw.cpp`, same session, interleaved,
64 iterations per graph, median of 3, warmed)

`qmv` is the kernel alone; the pair is (QMV / staged). GB/s is the packed weight
bytes over the kernel's own time, so the QMV's figure has to be multiplied by its
pass count to read as a memory rate (the staged figure needs no correction at 16
and 32 rows - one pass).

| class (real geometry) | rows 16 | rows 32 |
|---|---|---|
| lm_head 4096x154880 K6 | 8.742 / 2.387 ms (3.7x) | 17.45 / 3.68 ms (4.7x) |
| kda qkv_proj 4096x24576 K4 | 1.368 / 0.418 ms (3.3x) | 2.706 / 0.794 (3.4x) |
| mla o_proj 16384x4096 K4 | 0.931 / 1.126 ms (**0.83x**) | 1.827 / 0.884 (2.1x) |
| kda o_proj 8192x4096 K4 | 0.469 / 0.567 ms (**0.83x**) | 0.916 / 0.495 (1.9x) |
| dense gate/up 4096x12288 K3 | 0.693 / 0.242 ms (2.9x) | 1.361 / 0.599 (2.3x) |
| dense down 12288x4096 K3 | 0.719 / 0.844 ms (**0.85x**) | 1.398 / 0.724 (1.9x) |
| shared gate 4096x2048 K4 | 0.128 / 0.099 ms (1.3x) | 0.245 / 0.209 (1.2x) |
| shared down 2048x4096 K4 | 0.128 / 0.130 ms (0.98x) | 0.245 / 0.123 (2.0x) |
| mla q_b 1536x16384 K4 | 0.359 / 0.117 ms (3.1x) | 0.704 / 0.190 (3.7x) |
| exp gate/up 4096x2048 K3 (8 members) | 0.129 / 0.092 ms (1.4x) | 0.247 / 0.205 (1.2x) |
| exp gate/up 4096x2048 K2 | 0.130 / 0.088 ms (1.5x) | 0.246 / 0.191 (1.3x) |
| mla kv_a 4096x512 K4 | 0.043 / 0.018 ms (2.3x) | 0.076 / 0.027 (2.9x) |
| kda f_a 4096x256 K4 | 0.027 / 0.013 ms (2.0x) | 0.046 / 0.017 (2.7x) |

**The isolated kernel's optimum and the step's optimum disagree, and the shipped
split target is the step's.** With `EXL3_QMV_MAT_TG=4096` (what the isolated
kernel wants: thousands of threadgroups) every one of those 15 classes is a win
at both row counts - 2.6x to 4.8x, the narrow-output ones included (`mla o_proj`
0.937 / 0.268 ms, `kda o_proj` 0.479 / 0.141, `dense down` 0.717 / 0.209 at 16
rows). With the shipped 512 the same cells read 0.83-0.98x on those four classes
while the aggregate at B=16 follows the opposite way - see the pair below, where
the 16-row same-prompt cell reads **0.83x at 4096 and 1.19x at 512**. The knob is
`EXL3_QMV_MAT_TG`, and the reason the two disagree is that a split costs the
isolated kernel nothing (it runs alone, and more 32-thread threadgroups is more
latency hiding) and the step a great deal (`splits` f32 output planes per linear,
each summed by its own dispatch, and thousands of tiny threadgroups competing
with the MoE's expert kernels for the SMs).

Every other row count is unchanged by construction: rows 1-13 are below the
threshold, and a prefill - any (1, L, hidden) call, whatever L, and the general
MoE's flat (tokens, hidden) per-expert calls - is out of the staged path's scope
entirely, so the two columns are the same kernel there (`trellis_path_info`
prints the path; the rows 8/13 cells measure within 1%).

The staged kernel's own ceiling is visible in the table: ~120-195 GB/s at K4/K6,
i.e. the same per-weight instruction count the scalar QMV spends (the decode is
~10 of its ~13 issued lane-instructions per weight) - it does NOT decode faster.
What it removes is the row multiplicity: one decode pass serves 16-32 rows and
the MACs leave the issue pipe for the tensor unit, so the per-(weight,row) cost
falls by the pass ratio (3-5x measured, growing with the row count).

### Exactness: a named, gated deviation (not a silent loosening)

The staged path decodes the *same* fp16 weights (the decode is the QMV's own
`qmv_decode_h`, and the staged value is the same bits) and accumulates in fp32,
but the tensor unit sums the k range in its own tree and the split count is the
staged path's own. Reproducing the QMV's order is not possible while using the
matrix units: its per-output reduction is `quad_sum(a0+a1) + quad_sum(a2+a3)` per
eight stream positions plus a 4-lane interleave, and a `simdgroup_matrix` k
reduction is the hardware's.

So the deviation is *bounded and reported* rather than eliminated:

* `tests/glm5_gemv_nt_test.cpp`'s `matrix_path()` section measures the staged
  path against the QMV on the raw f32 output for 8 shapes x rows {16, 32, 64}:
  **max |diff| is 1.7e-5% to 8.9e-5% of max|value|** (2e-7 to 9e-7 relative) -
  f32 association scale, four orders of magnitude below what a wrong decode, a
  wrong fragment layout or a wrong tile mapping would show. It also checks that
  the staged path is inert to the QMV's nt/sg/mb knobs, that the grouped form is
  unchanged by the staged path being on (and still equals n single-tensor QMV
  calls), and that both scale-stack forms equal it.
* The byte-exact table over the 25 real classes x rows {1,2,3,13,26} is unchanged:
  `TrellisTuning{0, 0, 0}` (what the table is recorded with) pins the QMV, and the
  staged hashes are printed alongside it.
* The hard gates on the checkpoint are the token-identity ones: the single-slot
  batch == serial decode (rows = 1, byte-exact by construction), the four-slots-
  same-prompt row comparison, the four-different-prompts token identity, and the
  greedy text of `glm5_forward_test <model>`. See the gate table below.

### Measured, aggregate (`tests/glm5_batch_test.cpp <model> 6 --bench`)

The bench now measures every case twice per round - once with the default tuning
and once with the staged path forced off (`mat_min_rows = 0`) - in one process,
with the order alternating by round: the box drifts more than the lever, so only
a back-to-back pair is a measurement. The report prints both.

| case | staged ms/step (tok/s) | QMV ms/step (tok/s) | staged/QMV |
|---|---|---|---|
| B=1 serial | - | 151.70 (6.59) | - |
| B=1, 4 prompts | 105.36 (9.49) | 106.10 (9.42) | 1.01x |
| B=2, 4 prompts | 146.40 (13.66) | 144.10 (13.88) | 0.98x |
| B=4, 4 prompts | 370.95 (10.78) | 330.95 (12.09) | 0.89x |
| B=8, 4 prompts | 476.16 (16.80) | 466.21 (17.16) | 0.98x |
| B=4, 1 prompt | 169.88 (23.55) | 170.03 (23.53) | 1.00x |
| B=8, 1 prompt | 263.67 (30.34) | 260.06 (30.76) | 0.99x |
| B=8, 2 prompts | 480.31 (16.66) | 444.85 (17.98) | 0.93x |
| B=8, 8 prompts | 494.28 (16.19) | 446.25 (17.93) | 0.90x |
| **B=16, 1 prompt** | **381.55 (41.93)** | 454.40 (35.21) | **1.19x** |
| **B=16, 4 prompts** | 598.93 (26.71) | 606.08 (26.40) | 1.01x |

Best aggregate measured anywhere in this pass: **41.93 tok/s (B=16, one prompt)**
- the same process's byte-exact pass reads 35.21, the sixth pass's recorded best
was 33.87, and the target is 70. The B=8 cells, which cannot involve the staged
path at all (rows = 8 < the threshold), read 0.89x to 1.01x around 1.0: that is
the box's noise floor for this instrument, +/-10%, and the B=16 pair should be
read against it. It says the lever works, not how much of a 60-70 ms prediction
it delivers.

The same pair with `EXL3_QMV_MAT_TG=4096` (the isolated kernel's split optimum,
one run earlier in the same session) reads **0.83x (550.70 vs 458.17 ms) and
1.11x (540.71 vs 599.02)**, which is what moved the shipped target to 512: the
split count is a step-level knob, not a kernel-level one, and this is the one
place in the pass where the isolated kernel misled.

### Gates (real checkpoint, one session)

| gate | where | result |
|---|---|---|
| one-slot batch == serial decode | `glm5_batch_test` §1 | max abs logit diff = **0** |
| four identical slots == serial | §2 | 0.0166-0.0176 = **0.065%** of max\|logit\|, rows bit-identical to each other; the `<= 2%` gate is GREEN (the 1.99% recorded in the fifth pass is not reproducible at `steps 4` or `steps 6`; the gate itself is untouched) |
| four different prompts == four serial runs | §3 | token identity at every step and argmax identity every step; max abs logit diff 0.3945-0.4453 = 1.29-1.52% (gate 5%) |
| serving thread | §4 | see above |
| grouped == per-expert == general (hash) | `moe_test` (f) | `eebe4e4e51c6c683` for all three, as recorded |
| rows path == N single-row calls, and == the general path | `moe_test` (g) 4 rows, (h) 11 rows = 33 pairs over 4 experts (three gate\|up chunks, two down chunks, every expert selected several times) | max \|row diff\| = **0** in both, vs single-row and vs general |
| nt/mb scheduling + 25 real shapes x rows {1,2,3,13,26} (250 hashes) | `glm5_gemv_nt_test` | `nt/mb invariance: PASS` rc=0; `src/trellis_gemm.cpp` and the harness are byte-identical to the run the hashes were recorded from (`git diff a6b8d51 -- src/trellis_gemm.cpp tests/glm5_gemv_nt_test.cpp` is empty), so the table cannot have moved |
| prefill vs incremental decode | `glm5_forward_test <model>` | max\|diff\| = 0.0352 on max\|logit\| 23 (relative 0.00153), argmax 785 vs 785, top-5 overlap 5/5 |
| greedy text + first token | `glm5_forward_test` | first token 785, greedy 32-token text identical to the recorded run; rc=0 |
| trellis decode vs the reference decoder + byte alignment (lm_head K6 634M elements, qkv K4, q_b K4, expert gate K3) | `glm5_exl3_decode_test <model>` | `trellis decode: PASS`, 0 mismatches |
| the light suite (routing, grouped MoE, KDA rewind/recurrence, MLA incremental, indexer, hc, chat template, pretokenizer) | `tools/run_tests.sh --fast` | all PASS, `== done (rc=0) ==` |

### Honest distance to 70 tok/s

The best aggregate measured in this pass is 33.9 tok/s (B=16, shared routing) and
31.5 tok/s (B=16, 4 prompts) - 45-48% of the target, and B=8 mixed-prompt moved
17.7 -> 19.3 tok/s. Two things stand in the way, and the first is not about
bandwidth:

1. **Per-weight ALU.** The trellis GEMV spends 0.43 issued warp-instructions per
   weight, which is the machine's entire issue width (fourth pass) - the mixed
   case is bound by *work*, not by the weight bytes (a mixed B=8 step's ~5.8 GB
   at the kernel's own 134-202 GB/s would be ~30-45 ms of its ~460).
   `docs/research/mlxfast-bonsai2.md`
   measures the escape (packed stream -> int8 threadgroup tiles -> simdgroup
   matrices, or a native packed tensor op), and it needs 13-16 rows: that is why
   this pass made the pair layout free of any B bound and measured B=16. It is
   the order-of-magnitude lever.
2. **A real dedup, not an ordering.** One member with `rows=r` per expert turns
   r duplicate pairs into ceil(r/4) weight passes - the only way to cut the ALU
   work a duplicated expert costs, and the exact regime where the shared-routing
   row is 1.7x the mixed one. It has no gate yet:
   `trellis_linear_group_rows(rows=R)` has never been checked per row against R
   single-row calls, and the MoE path must not use it before that gate exists
   (its per-row k-order is presumably the same, but "presumably" is not the
   standard here).
3. **The per-row machinery.** At a fixed routing the marginal row costs ~23-25 ms
   (B=4 -> B=16: 176.6 -> 472.4 ms), so the aggregate saturates near 34 tok/s
   long before the expert term (or 70 tok/s) is reached.

## Seventh pass (2026-10-11): the staged matrix path for the batched rows

The sixth pass left one lever named and unbuilt: the classes whose weights do not
diverge per row (kda `qkv_proj`, kda/mla `o_proj`, the shared expert, the dense
MLP, mla `q_b`) pay their decode once per *row block* (`mb = min(rows,4)`), so at
the row counts a request-level batch reaches they stream the same packed words 4
times at 16 rows and 7 at 26 - and the decode (a mul1 multiply-hash and its half
fma, ~10 issued lane-instructions per weight) is what the machine's issue width
is spending. This pass moves those classes onto a staged fp16 + simdgroup_matrix
kernel and measures it per class and at B=16.

### What was built

* `src/trellis_gemm.cpp`: a second kernel in the QMV family (`exl3_qmv_mat_*`,
  built by the same `make_qmv_kernel` factory so the codebook prologue, the
  member head and the buffer list cannot drift apart). It decodes each (16k x 16n)
  packed tile ONCE per threadgroup into a threadgroup fp16 tile and then
  multiplies with `simdgroup_matrix`; the tile is staged (n, k)-major with 16
  halves per row, which is both what a lane's k pairs want (two `half2` stores
  per group of four weights) and what the B fragment's `simdgroup_load(...,
  transpose = true)` reads as W[k][n]. One warp per output tile, one barrier per
  k-tile, the next tile's words loaded a tile ahead of their decode.
  * Gate: `TrellisTuning.mat_min_rows` (default 16, `EXL3_QMV_MAT_MIN_ROWS`) AND
    a decode-shaped call - one whose sequence axis is 1, i.e. (B, 1, hidden) from
    the request-level batch (or a flat 2-D (rows, in) benchmark call). A prefill
    chunk is (1, L, hidden) with L > 1 and keeps the byte-exact QMV, so the
    single-sequence path is bit-identical *including its prefill* (the ticket's
    hard gate) and a wide forward does not pay the staged kernel's rows/MAT_MB
    weight passes - for a prefill the answer is a K-blocked GEMM over the batch,
    not this kernel. Below the threshold every call is on the QMV anyway.
  * Scope: the single-tensor entry points (`trellis_linear`, `trellis_matmul_raw`)
    only. The grouped forms keep the QMV - their activation is (n, rows, in) with
    the sequence axis already flattened into `rows`, so the decode-shaped test
    cannot be made there, and the two callers that reach rows >= the threshold are
    the model's routed experts (rows = 1 per member) and moe_layer's general path,
    whose (K, B*L, H) activation includes the single-sequence prefill.
  * The staged path picks its own k-split count (`mat_split_count`): the QMV's is
    tuned for `n_tiles * 4` warps and the staged kernel runs one warp per tile, so
    it aims at ~4096 threadgroups instead. This matters: at the QMV's count the
    narrow-output classes (mla `o_proj`, kda `o_proj`, the shared expert, kv_a)
    ran at 25-35 GB/s, i.e. no better than the QMV - they are latency-bound, and
    the split is the only parallelism axis the staging leaves.
* `trellis_path_info()` (`src/trellis_gemm.h`): which path, row block, padded rows
  and pass count a tensor will take at a given row count - so the benchmarks and
  the batch test report the path they measured instead of re-deriving the rule.

### Measured, per class (`tests/glm5_trellis_bw.cpp`, same session, interleaved,
64 iterations per graph, median of 3, warmed)

`qmv` is the kernel alone; the pair is (QMV / staged). GB/s is the packed weight
bytes over the kernel's own time, so the QMV's figure has to be multiplied by its
pass count to read as a memory rate (the staged figure needs no correction at 16
and 32 rows - one pass).

| class (real geometry) | rows 16 | rows 32 |
|---|---|---|
| lm_head 4096x154880 K6 | 8.742 / 2.387 ms (3.7x) | 17.45 / 3.68 ms (4.7x) |
| kda qkv_proj 4096x24576 K4 | 1.368 / 0.418 ms (3.3x) | 2.706 / 0.794 (3.4x) |
| mla o_proj 16384x4096 K4 | 0.931 / 1.126 ms (**0.83x**) | 1.827 / 0.884 (2.1x) |
| kda o_proj 8192x4096 K4 | 0.469 / 0.567 ms (**0.83x**) | 0.916 / 0.495 (1.9x) |
| dense gate/up 4096x12288 K3 | 0.693 / 0.242 ms (2.9x) | 1.361 / 0.599 (2.3x) |
| dense down 12288x4096 K3 | 0.719 / 0.844 ms (**0.85x**) | 1.398 / 0.724 (1.9x) |
| shared gate 4096x2048 K4 | 0.128 / 0.099 ms (1.3x) | 0.245 / 0.209 (1.2x) |
| shared down 2048x4096 K4 | 0.128 / 0.130 ms (0.98x) | 0.245 / 0.123 (2.0x) |
| mla q_b 1536x16384 K4 | 0.359 / 0.117 ms (3.1x) | 0.704 / 0.190 (3.7x) |
| exp gate/up 4096x2048 K3 (8 members) | 0.129 / 0.092 ms (1.4x) | 0.247 / 0.205 (1.2x) |
| exp gate/up 4096x2048 K2 | 0.130 / 0.088 ms (1.5x) | 0.246 / 0.191 (1.3x) |
| mla kv_a 4096x512 K4 | 0.043 / 0.018 ms (2.3x) | 0.076 / 0.027 (2.9x) |
| kda f_a 4096x256 K4 | 0.027 / 0.013 ms (2.0x) | 0.046 / 0.017 (2.7x) |

**The isolated kernel's optimum and the step's optimum disagree, and the shipped
split target is the step's.** With `EXL3_QMV_MAT_TG=4096` (what the isolated
kernel wants: thousands of threadgroups) every one of those 15 classes is a win
at both row counts - 2.6x to 4.8x, the narrow-output ones included (`mla o_proj`
0.937 / 0.268 ms, `kda o_proj` 0.479 / 0.141, `dense down` 0.717 / 0.209 at 16
rows). With the shipped 512 the same cells read 0.83-0.98x on those four classes
while the aggregate at B=16 follows the opposite way - see the pair below, where
the 16-row same-prompt cell reads **0.83x at 4096 and 1.19x at 512**. The knob is
`EXL3_QMV_MAT_TG`, and the reason the two disagree is that a split costs the
isolated kernel nothing (it runs alone, and more 32-thread threadgroups is more
latency hiding) and the step a great deal (`splits` f32 output planes per linear,
each summed by its own dispatch, and thousands of tiny threadgroups competing
with the MoE's expert kernels for the SMs).

Every other row count is unchanged by construction: rows 1-13 are below the
threshold, and a prefill - any (1, L, hidden) call, whatever L, and the general
MoE's flat (tokens, hidden) per-expert calls - is out of the staged path's scope
entirely, so the two columns are the same kernel there (`trellis_path_info`
prints the path; the rows 8/13 cells measure within 1%).

The staged kernel's own ceiling is visible in the table: ~120-195 GB/s at K4/K6,
i.e. the same per-weight instruction count the scalar QMV spends (the decode is
~10 of its ~13 issued lane-instructions per weight) - it does NOT decode faster.
What it removes is the row multiplicity: one decode pass serves 16-32 rows and
the MACs leave the issue pipe for the tensor unit, so the per-(weight,row) cost
falls by the pass ratio (3-5x measured, growing with the row count).

### Exactness: a named, gated deviation (not a silent loosening)

The staged path decodes the *same* fp16 weights (the decode is the QMV's own
`qmv_decode_h`, and the staged value is the same bits) and accumulates in fp32,
but the tensor unit sums the k range in its own tree and the split count is the
staged path's own. Reproducing the QMV's order is not possible while using the
matrix units: its per-output reduction is `quad_sum(a0+a1) + quad_sum(a2+a3)` per
eight stream positions plus a 4-lane interleave, and a `simdgroup_matrix` k
reduction is the hardware's.

So the deviation is *bounded and reported* rather than eliminated:

* `tests/glm5_gemv_nt_test.cpp`'s `matrix_path()` section measures the staged
  path against the QMV on the raw f32 output for 8 shapes x rows {16, 32, 64}:
  **max |diff| is 1.7e-5% to 8.9e-5% of max|value|** (2e-7 to 9e-7 relative) -
  f32 association scale, four orders of magnitude below what a wrong decode, a
  wrong fragment layout or a wrong tile mapping would show. It also checks that
  the staged path is inert to the QMV's nt/sg/mb knobs, that the grouped form is
  unchanged by the staged path being on (and still equals n single-tensor QMV
  calls), and that both scale-stack forms equal it.
* The byte-exact table over the 25 real classes x rows {1,2,3,13,26} is unchanged:
  `TrellisTuning{0, 0, 0}` (what the table is recorded with) pins the QMV, and the
  staged hashes are printed alongside it.
* The hard gates on the checkpoint are the token-identity ones: the single-slot
  batch == serial decode (rows = 1, byte-exact by construction), the four-slots-
  same-prompt row comparison, the four-different-prompts token identity, and the
  greedy text of `glm5_forward_test <model>`. See the gate table below.

### Measured, aggregate (`tests/glm5_batch_test.cpp <model> 6 --bench`)

The bench now measures every case twice per round - once with the default tuning
and once with the staged path forced off (`mat_min_rows = 0`) - in one process,
with the order alternating by round: the box drifts more than the lever, so only
a back-to-back pair is a measurement. The report prints both.

| case | staged ms/step (tok/s) | QMV ms/step (tok/s) | staged/QMV |
|---|---|---|---|
| B=1 serial | - | 151.70 (6.59) | - |
| B=1, 4 prompts | 105.66 (9.46) | 104.09 (9.61) | 0.99x |
| B=2, 4 prompts | 152.11 (13.15) | 144.62 (13.83) | 0.95x |
| B=4, 4 prompts | 348.38 (11.48) | 316.54 (12.64) | 0.91x |
| B=8, 4 prompts | 480.13 (16.66) | 454.04 (17.62) | 0.95x |
| B=4, 1 prompt | 168.40 (23.75) | 167.27 (23.91) | 0.99x |
| B=8, 1 prompt | 261.58 (30.58) | 259.31 (30.85) | 0.99x |
| B=8, 2 prompts | 455.70 (17.56) | 490.05 (16.32) | 1.08x |
| B=8, 8 prompts | 491.42 (16.28) | 478.22 (16.73) | 0.97x |
| B=16, 1 prompt | 550.70 (29.05) | 458.17 (34.92) | 0.83x |
| B=16, 4 prompts | 540.71 (29.59) | 599.02 (26.71) | 1.11x |

### Gates (real checkpoint, one session)

| gate | where | result |
|---|---|---|
| single-slot batch == serial decode | `glm5_batch_test` §1 | max abs logit diff = **0** (bit-identical) |
| four identical slots == serial | §2 | 0.01758 (0.065% of max\|logit\|), rows bit-identical; <= 2% gate GREEN |
| four different prompts == four serial runs | §3 | token identity at every step, argmax every step; logits 0.4453 (1.52%) vs the 5% gate |
| serving thread | §4 | 9/9 requests, `stop_batch()` with a request in flight returns in 0.05 s |
| greedy text + prefill-vs-decode | `glm5_forward_test <model>` | max\|diff\| = **0.0352** on max\|logit\| 23 (relative 0.00153), argmax 785 vs 785, top-5 5/5, first token 785, 32-token text identical to the recorded run - **exactly the recorded numbers**, because the staged path never runs in that test (its forwards are (1, L, hidden) prefills and single rows; `EXL3_QMV_MAT_DEBUG=1` traces 0 staged dispatches) |
| trellis decode vs the reference decoder | `glm5_exl3_decode_test <model>` | PASS, 0 mismatches (lm_head K6 634M elements, qkv K4, q_b K4, expert gate K3) |
| nt/mb invariance, the byte-identity table over 25 classes x rows {1,2,3,13,26}, the staged deviation bound, the grouped contracts | `glm5_gemv_nt_test` | 781 checks PASS, rc=0 |

### Honest distance to 70 tok/s

The staged path does what it was meant to do *per class* - 2.5-4.5x on every one
of the 15 real tensor classes at 16 and 32 rows, by removing the row multiplicity
(one decode pass for MAT_MB rows) and the per-row FMAs - and it is byte-identical
in every context except the one it was scoped to. What it does not do, on this
box's evidence, is move the B=16 aggregate outside the noise: the pair that was
supposed to show +15% showed -17% and +11% in the two B=16 cells.

The next levers, in order of what the numbers now say:

1. **Re-tune the staged path's geometry against the step, not the kernel.** Its
   split target and one-warp threadgroups were chosen from isolated kernel
   measurements; a step saturated by expert kernels may want few, large
   threadgroups instead. This is a one-knob sweep (`EXL3_QMV_MAT_TG`,
   `EXL3_QMV_MAT_NT`, `EXL3_QMV_MAT_MB`) with the paired bench as the instrument.
2. **The expert term, which is still untouched and still the largest.** At B=16
   with one prompt the 16 rows share 8 experts, so the pairs (row, expert) are 16
   members with rows = 1 each: 16 passes over each expert instead of 4 (rows = 16
   in one member). That is ~43 GB of the step's reads against 3.4 GB of shared
   weights, and the same staged machinery would serve it if the MoE built
   expert-grouped members with rows = B (its `rows` would be a member's rows, so
   the scope rule needs a second, equally explicit signal from that caller).
3. **A prefill-width GEMM.** The staged path is out of scope for prefills by
   design; a chunked-over-k GEMM with the weight tile in threadgroup memory is
   the shape that serves (1, L>=64, hidden) - and it is what a longer prompt
   actually needs (the current QMV streams the weights L/mb times).
