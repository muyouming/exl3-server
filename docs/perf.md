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
