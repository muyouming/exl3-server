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
trellis GEMV, 4096x24576 K=4 (50.4 MB) | 0.522 ms per row with an eval per sample (0.113 ms no-op+eval baseline) => **~96-123 GB/s, ~20% of the M4 Max's ~546 GB/s** |
trellis GEMV, small tensor (3 MB) | 0.070-0.086 ms => 40-58 GB/s (latency/parallelism-bound, not bandwidth-bound) |
scaling with rows | 1/2/4/8 rows cost 0.52/0.85/0.82/2.05 ms: ~4x for 8 rows, i.e. the weights are re-read per row instead of being blocked in registers/shared memory |

Consequences: at ~120 GB/s the 4.6 GB/token means a **~27 tok/s ceiling** today
regardless of dispatch fusion, so the remaining work is memory layout and kernel
blocking, not launch count:

1. **Repack the trellis weights at load time** into a GEMV-friendly layout (the
   on-disk `(in/16, out/16, 16*K)` form with per-16-element scales gives
   strided, poorly coalesced loads). It is a pure byte permutation, so it can be
   RAM-neutral if done per tensor (materialise, repack, unmap) or in place; a
   slow load is acceptable when decode is the goal.
2. **Reuse the packed weights across batch rows inside the kernel** (register or
   shared-memory blocking) - this is what makes speculative batches and prefill
   pay, and today they do not.
3. Only then the dispatch items: kill in-loop `mx::eval`, fuse the
   hyper-connection/Sinkhorn work (`mx::compile` exists in the C++ API:
   `mlx/compile.h`), fuse the two Hadamards into the trellis GEMV, and batch the
   small per-layer projections (34 layers x ~3 MB) into single large GEMVs.

Methodology warnings, both learned the hard way here: (a) measuring with an
`mx::eval` per sample adds 0.11-0.2 ms, which exceeds the op itself for small
tensors - subtract a no-op baseline or batch correctly; (b) a lazy-evaluation
loop that drops the previous result each iteration measures graph construction
only (it reported 2259 GB/s for the 50 MB tensor, i.e. work that never ran) -
keep every intermediate alive and treat anything above ~546 GB/s as a sign the
work was optimised away.

## Measurement discipline

- `tests/glm5_bench.cpp` reports the per-component breakdown (evaluate after
  each stage) and end-to-end tok/s; use it before and after every change.
- Correctness gates that must stay green: the trellis byte-alignment regression
  (`tests/glm5_exl3_decode_test.cpp`), the prefill-vs-decode logits invariant
  (~1.4 f16 ulp) and every other test in `tests/` - `tools/run_tests.sh` builds
  and runs them with full parallelism for the light tests while serialising the
  ones that map the 100 GB checkpoint (two of those would evict each other's
  pages on a 128 GB machine and be slower than running them one at a time).
