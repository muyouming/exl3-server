# oMLX — how a Python/MLX serving stack accelerates GLM-5.3-Flash-class decode

**Identity resolved:** `omlx` = **oMLX**, `https://github.com/jundot/omlx` ("LLM inference server, optimized for your Mac",
`pyproject.toml:2`), Apache-2.0, python >= 3.11 < 3.14. Read at commit
**`013bc3f3a48fa9a4b7dffbfeb6cdc7c1face1c5d`** (2026-10-10, "fix(server): lease the decision model before encode in
/v1/systemone (#4413)"), in-tree version **`0.7.1.dev1`** (`omlx/_version.py:1`). Clone:
`git clone --depth 1 https://github.com/jundot/omlx`. Not on PyPI (`https://pypi.org/pypi/omlx/json` → HTTP 404,
checked 2026-10-10); distribution is a `.dmg` release, the Homebrew tap `jundot/omlx`, or `make install` from source
(`README.md:57-95`).

**Ruled out:**
- *An ONNX→MLX converter named omlx.* Does not exist. The real projects in that space are `ml-explore/mlx-onnx` and
  `skryl/mlx-onnx`; web hits pairing "omlx" with ONNX/OminiX were unrelated repositories, not the `omlx` package.
- *The generic "OpenAI-compatible MLX server" reading.* That describes `madroidmaq/mlx-omni-server` and
  `mlx_lm.server`, but the artifact actually named `omlx` is jundot/omlx. Its own README also claims OpenAI **and**
  Anthropic compatibility (`README.md:259-266`) — but the project's identity is the tiered-KV/continuous-batching
  Mac server, not an API shim.
- *Fork ambiguity.* Search surfaces many same-named mirrors (`opencolin/omlx`, `nikolay-bratanov/omlx`,
  `byLuYuanZhi/omlx`, `Yukon/omlx` — itself labelled "oMLX (forked)"). Picked `jundot/omlx` because pyproject
  `[project.urls]` declares it Homepage/Repository, `README.md:29` points at `omlx.ai`, and the file layout matches
  the published docs (`docs/`, `omlx/`, `apps/omlx-mac`, `packaging/`).
- *`nireson/omlx-tools`* is a third-party CLI (`omlx-add`, `omlx-top`) **for** oMLX, not the project.

## Biggest acceleration mechanism (one paragraph)

oMLX's single biggest lever is **collapsing the GLM-5.3-Flash decode step's op chain into a handful of JIT Metal
kernels, one per architectural stage, and submitting the graph in a pipelined cadence instead of per-op**. Its
vendored model code states the problem in our terms: a decode step "is ~800 dependent dispatches whose Python graph
build takes ~2.7 ms; mlx keeps at most ~10 command buffers in flight and, with its default per-buffer size budget …
a step commits ~170 buffers, so the GPU drains the previous step's last few buffers long before the next graph is
built and encoded (~1.3 ms idle per token)" (`omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py:71-79`).
Against that, `omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py` provides single-dispatch replacements for
the router (`moe_router`, :2135), the routed+shared expert block (`moe_gate_up_swiglu` :795, `moe_down_combine`
:933), the whole KDA linear-attention step (`kda_decode_step` :1843), the DSA indexer score/top-k/gather trio
(:1227, :1443, and the gather at :1506), the hyper-connection prologue/epilogue (`hc_pre_fused` :2709,
`hc_post_mm` :2752), and small per-layer projections merged into one launch (`multi_qmv` :2960). Every kernel
replays the reference reduction order, and the model flushes with `mx.async_eval` every 8 layers so encoding of the
first layers overlaps the building of the rest (`language.py:1915-1922`). The second mechanism in importance is
row/token amortization: batched and speculative verify rows are run by kernels written so that a block of rows costs
one pass over the weights (see §3-B).

---

## 1. What it is, and how to get it

| | |
|---|---|
| Name / package | `omlx` (import package `omlx`, entry point `omlx = "omlx.cli:main"`, `pyproject.toml:248-249`) |
| Repo | https://github.com/jundot/omlx |
| Commit read | `013bc3f3a48fa9a4b7dffbfeb6cdc7c1face1c5d` (2026-10-10) |
| Version | `0.7.1.dev1` (`omlx/_version.py:1`) |
| Install | Homebrew `jundot/omlx/omlx --with-custom-kernel`; DMG release; source `make install` (`README.md:57-95`) |
| Pinned runtime | `mlx==0.32.3`, `mlx-lm @ 94cdcae1` (main after 0.32.0), `mlx-vlm @ 4f4634bb`, `dflash-mlx @ jundot fork 71f7c2c`, `nanobind==3.0.1`, `transformers>=5.14,<5.18` (`pyproject.toml:1-12,43-45,198-215`) |
| Lineage | "oMLX started from vllm-mlx v0.1.0 and evolved significantly with multi-model serving, tiered KV caching, VLM with full paged cache support …" (`README.md:459`) |
| Also in tree | native SwiftUI menu-bar app + web admin UI (`README.md:413-441`) |

Scale: 1,055 Python files / 622,669 lines in the clone (including `tests/` and `apps/`), plus five native Metal
kernel packages (`omlx/custom_kernels/{bonsai,decode_fast,glm_moe_dsa,minimax_m3,qwen35_prefill}`) with 5,180
lines of `.metal` and 24,919 lines of C++/ObjC++ host code under `custom_kernels/`.

**GLM-5.3-Flash is a first-class target here** (`model_type: glm5_next`), via a vendored copy of the mlx-vlm
implementation (`omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/`, based on
`Blaizzy/mlx-vlm#2030` at `fa27a9a6`, per that package's `README.md:1-12`), plus DFlash and native-MTP adapters for
the same family. That makes oMLX the closest published sibling of this repo's workload that I found.

## 2. Architecture in one short section

Python 3.11-3.13; FastAPI/uvicorn server (`omlx/server.py`) exposing OpenAI + Anthropic-compatible endpoints, with
a scheduler/engine split: `omlx/engine_core.py` drives an asyncio loop that hands work to a **single MLX executor
thread**, `omlx/scheduler.py` (~14 k lines) owns admission, chunked/packed prefill, paged cache and fairness, and
the token step itself is mlx-lm's `BatchGenerator` ("The key insight is that mlx-lm's BatchGenerator already
implements continuous batching at the token level, so we use it as the backend", `omlx/scheduler.py:1929-1930`).
Model support is delivered by **monkey-patch packages** under `omlx/patches/` (74 entries): they register
optimized/ vendored model modules into `sys.modules` (e.g. `mlx_lm.models.glm_moe_dsa`, `mlx_vlm.models.glm5_next`),
install `mx.QuantizedLinear.__call__` / `mx.gather_qmm` wrappers, and attach fused kernels per family.

Kernels live in two places: (a) **precompiled** Metal + C++ per package, built only with
`OMLX_WITH_CUSTOM_KERNEL=1` and full Xcode, compiled to `<pkg>.metallib` loaded next to the extension dylib
(`glm_moe_dsa/csrc/CMakeLists.txt:70-117`, `dsa_indexer.cpp:24-31,208`), exposed through nanobind with
`NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx` (`glm_moe_dsa/csrc/CMakeLists.txt:124-133`) — a hard ABI coupling
to the exact `mlx` wheel that is checked at import with an `abi_probe` canary (`glm_moe_dsa/fast.py:52-78`); and
(b) **JIT** kernels written as `mx.fast.metal_kernel` source strings in `omlx/patches/*.py` (the GLM-5.3 decode
kernels, the KDA kernels, the NAX tiles). Absent the native extensions the server silently degrades to far slower
generic MLX paths — the GLM-5.2 DSA-prefill claim is "roughly 30x faster with the kernels (measured 845 vs ~29 tok/s
on an M3 Ultra)" (`README.md:84-88`).

## 3. Acceleration mechanisms

### A. Decode-path dispatch and machinery (the launch-bound killer)

| mechanism | what it does | evidence | measured effect |
|---|---|---|---|
| Per-stage fused decode kernels for `glm5_next` | One dispatch per architectural stage: `moe_router` (logits GEMV + sigmoid/bias + top-k with the reference's stable-sort tie order), `moe_gate_up_swiglu` (selected + shared experts, clamped SwiGLU), `moe_down_combine` (routing-weighted sum), `kda_decode_step` (short conv, SiLU, l2norm, gate projections, vector-gated delta rule, RMSNormGated), `dsa_decode_scores` / `dsa_expand_topk` / `dsa_gather_selected`, `mlp_gate_up_swiglu` | `decode_kernels.py:1-23` ("Single-token decode and short verify blocks (L <= 8) are dominated by thousands of tiny dependent dispatches, which cost both GPU time and host encode time"); `:2135`, `:795`, `:933`, `:1843`, `:1227`, `:1443` | no tok/s figure committed; the stated motivation is the ~800-dispatch / ~2.7 ms host-build / ~1.3 ms GPU-idle-per-token budget (`language.py:71-79`) |
| Exactness contract that makes fusion safe | "every reduction replays the order of the MLX kernel it replaces (qmv/qmv_quad lane mapping, gemv shuffle ladders, row_reduce orders, Steel/NAX MMA fragments)"; bitwise tests per kernel and end to end | `decode_kernels.py:19-23`; `tests/test_mlx_vlm_glm5_next_compat.py` (4,790 lines, `test_…_bitwise`, `test_…_matches_*`) | enables A/B by token identity, not by perplexity |
| Layer-grouped async flush | Encode the first layers while the rest of the step is still being built: `_DECODE_EVAL_EVERY = 8`, gated to `B==1, L<=8` | `language.py:70-79`, `:1915-1922` | "~1.3 ms idle per token" recovered (stated, not tabled) |
| Raised command-buffer caps | Reaches MLX's **private** `metal::Device::max_ops_per_buffer_` / `max_mb_per_buffer_` via an explicit-instantiation friend trick and sets `(200, 512)` for the duration of a speculative step | `custom_kernels/qwen35_prefill/csrc/qwen35_prefill.cpp:879-896`; `fast.py:74-82`; `patches/mlx_lm_mtp/batch_generator.py:367-405` | "Measured on M5 and M3 Ultra, so other GPUs keep MLX's caps" (`batch_generator.py:389-392`) |
| `mx.compile` for the elementwise fallback | HC glue (`_hc_split_sinkhorn_ops`, `_hc_expand_op`) is compiled when the fused kernel is unavailable/CPU | `patches/deepseek_v4/hyper_connection.py:185,259` | — |
| Hyper-connection (Sinkhorn) fusion | One kernel does collapse + mix GEMV + branchless Sinkhorn: lanes 0-3 each own a comb row, column norms via `simd_sum`, `float4` loads, FMA collapse — "Fused sinkhorn + collapse: eliminates one dispatch per HC cycle" | `patches/deepseek_v4/hyper_connection.py:12-27,165-183,221-256` | — |
| HC deferral across layers | `hc_pre_fused(…, deferred)` recomputes the previous half-layer's expand inside the next layer's HC prologue kernel; `hc_post_mm` moves post/comb off the dependent chain | `decode_kernels.py:2709-2760`; `language.py:1924-1931` | — |
| `multi_qmv` | 1..4 affine quantized linears of the same (K, bits, group) run as ONE dispatch for 1..8 rows — for our `f_a`/`g_a`/`b_proj`-class small projections | `decode_kernels.py:2946-3004` | — |
| One-token MoE in two launches | gate+up(+SwiGLU epilogue, shared expert, shared gate row) in launch 1 with the top-k folded into it, down+combine in launch 2, simdgroup `j` = expert `j`; selection stays on device | `patches/qwen35_moe_routed_decode.py:4-36,289-291,400-404` | "three dependent launches instead of four"; the whole stacked-expert bind costs a 50 MB command-buffer commit "~10-20 us of host CPU and a GPU gap per commit" (`:67-74`) |
| Fused MoE router | "the composed argpartition chain is ~2 ms/token on the 256-expert 35B-A3B" → one launch | `omlx/engine/batched.py:438-440`; `patches/qwen35_moe_router.py` | ~2 ms/token removed on 35B-A3B |

### B. Amortizing weight reads across rows and tokens

| mechanism | what it does | evidence | measured effect |
|---|---|---|---|
| Verify-shape QMM kernels | Stock MLX qmm is tuned for M=1 and large M; at M=3..6 it pays a per-row penalty. oMLX writes split-K / multi-simdgroup kernels for M=3..6 and an `mma` tile kernel for 7..24 rows "which dequantizes 16-column weight tiles into threadgroup memory once and multiplies them against every row with simdgroup matrices, **so a batch of rows costs one pass over the weights**"; `sg8` handles 4-8-row blocks | `patches/qwen35_verify_qmm.py:9-26,38-42,57-62` | "measured on Qwen3.6-27B/M3 Ultra the lm_head scales linearly (1.2 ms at M=1 → 4.3 ms at M=4) and the summed MLP/GDN projections add ~15 ms per verify call" (`:24-26`) |
| Row-exact multi-row projection | One threadgroup computes one `2*RPS`-column tile for `ROWS` rows: "each weight block is decoded once and applied to all of them"; each row still sums K in `qmv_fast` order, so row `r` equals a serial 1-row decode bit for bit; `MAX_ROWS = 16`; the one-row call uses the same tile | `patches/row_exact_qmv.py:1-21,36` | — |
| Weight stream read once for M rows (`qmv_wide`) | 2-bit weights at M>=3: "amortises the weight stream across all M vectors (1 read vs M reads)" | `custom_kernels/bonsai/fast.py:264-278` | "yielding 1.3-1.5x on large projections (benchmarked: gate/up/down_proj M=5 → 71→104 GB/s on g16s)" |
| Expert-ordered verify gather | Replaces `gather_qmm`'s one-matvec-per-(row,expert)-pair in routing order — "the same expert weights are fetched from DRAM once per row that selects them" — with one launch in expert order so "pairs that share an expert then run back to back and read the expert tile from cache"; one threadgroup per pair+tile, rank-based pair discovery in-kernel; bit-identical transcription of MLX 0.32.2 `quantized.h` | `patches/moe_verify_gather.py:4-15,205-221,340-341` | — |
| Gate/up fusion | Concat the expert gate and up weights along the output axis (`[E, 2*inter, hidden]`): one dispatch, one pass over the gathered activations, twice the output tiles per expert; bit-identical to the two calls | `patches/moe_gate_up_fusion.py:1-19,49,84-137` | "measured: MiMo 4096-token chunk 16.1 → 15.1 ms per layer, **GLM-5.3 2048-token chunk 8.5 → 8.0 ms**" (`:11-12`) |
| Lazy sorted routes | `sort_routes` = mlx-lm's `_gather_sort` without the gather: returns a `row_map` so the replicated `[T*k,1,K]` copy is never materialised; kernels read token rows in place through the row map | `patches/moe_routes.py:9-24`; consumed by `m5_gather_qmm_nax.py:70-77` ("0.4-0.5 GB per layer at 8192-token chunks … the copy is never computed") | — |
| Continuous batching | Rows from up to `max_concurrent_requests` (default 8) are stacked into ONE backbone forward per step; per-request caches are merged into batched row caches at insert | `patches/mlx_lm_mtp/batch_generator.py:3528-3530`; `scheduler.py:12669-12677`; `settings.py:1908-1910`, `:338`; `README.md:188-190` | no committed batch-vs-throughput table (the admin benchmark computes it at runtime; `omlx/admin/benchmark.py:44` uses batch sizes 2/4/8) |
| Lightning MTP depth-k chain | Verify `[next_main, d1..dk]` in a single backbone forward, greedy acceptance computed in-graph, next draft chain dispatched with `mx.async_eval` — **exactly ONE host sync per cycle**; stochastic path also one sync; batched rows share the verify | `patches/mlx_lm_mtp/batch_generator.py:4027-4035`, `:4145-4150`, `:2996-2999` | A/B on a REAP-pruned mixed-precision Step-3.7-Flash via oMLX's own benchmark, mtp on/off: pp4096/tg128 52.7→64.4 tok/s (1.22x); pp8192 51.2→63.3 (1.24x); pp16384 48.8→61.1 (1.25x); pp32768 42.6→55.2 (1.30x); +0.4 GB peak (`step3p7_model.py:54-64`) |
| MTP prompt priming | Feed the head the committed-history `(hidden[t], token[t+1])` pairs computed by the prefill forwards so its first drafts are not context-starved | `patches/mlx_lm_mtp/prompt_priming.py:3-10` | "MTPLX measured committed-history priming at 0.90 acceptance vs 0.26 unprimed on depth-1 real-code prompts" |
| Depth control that parks | Adaptive depth uses the measured cost of the whole active batch; depth 0 runs the cycle as a plain 1-token step | `batch_generator.py:2509-2515`, `:2445-2446` | "the whole depth menu can be worse than standard decoding — measured 0.67x on gemma4 26B story/16k" |
| Rollback bound as a depth cap | GLM-5.3 native MTP clamps at `_MAX_CHAIN_DEPTH = 7` because the indexer `PoolingCache` undo log only covers 8-row updates; DFlash blocks are capped at 16 | `patches/mlx_vlm_mtp/glm5_next_vlm_runtime.py:50-52`; `patches/deepseek_v4/cache_extras.py:21-22`; `patches/dflash_glm5.py:543-548` | structural, not a perf number |
| Marginal cost of one verify row on a sparse-MoE target | The adaptive depth controller's prior measures what an extra verify row costs when routing is 8-of-256: "each extra verify row pulls an almost disjoint expert set (~55% of step bytes)" | `patches/mlx_lm_mtp/glm_moe_dsa_model.py:361-365` (`self._omlx_mtp_marginal_ms = 35.0`) | "**measured ~35 ms per row on GLM-5.2** vs the dense-backbone 7 ms default" — i.e. 5x the dense row cost, which is exactly the ceiling that expert-dedup/verify-shape kernels exist to lower |

### C. Attention / sparse / linear-attention layouts

| mechanism | what it does | evidence | measured effect |
|---|---|---|---|
| Fused DSA indexer | `sum_h relu(q_h·k^T)·w_h` computed inside a Steel GEMM and written straight out as bf16 scores, with the pooled-ratio mask folded in (`finfo.min`) — the portable path "materializes `[B,M,H,N]` fp32 … before the ReLU/head reduction" | `custom_kernels/glm_moe_dsa/csrc/dsa_indexer.metal:22-27`; `kernels/steel_dsa_indexer_score.h:214-218`; `patches/glm_moe_dsa/sparse_mla.py:323-327` | ~13 TFLOPS on classic SIMD units (`patches/glm_moe_dsa/indexer_nax.py:7-10`); from-scratch MMA variant "~1.37x over the Steel kernel on M2 Ultra, bit-exact" (`kernels/mma_dsa_indexer_score.h:28-29`) |
| In-kernel radix top-k | "Four byte-wise radix passes identify the cutoff, then a deterministic segmented scan writes the selected set without materializing an O(K) index sheet" — replaces `argpartition` + index sheet | `kernels/steel_dsa_indexer_score.h:362-366`; `dsa_indexer.metal:34-79` | — |
| Decode-shaped indexer scores | One kernel for the single-position indexer with fp32 accumulation, replacing "the decode chain q@k^T → relu → *w → head-sum that materializes four S-sized tensors per layer per token" | `dsa_indexer.metal:95-168`; `dsa_indexer.h:80-83` | — |
| Sparse MLA kernel | FlashMLA-style attention over each query's own top-k latent rows (NoPE 512 + rope 64), online softmax in fp32, indices consumed in-kernel; the fallback materialises multi-head K/V over every cached position via `embed_q`/`unembed_out` | `sparse_mla.metal:5-33`; `patches/glm_moe_dsa/glm_moe_dsa_model.py:35-38` | BK sweep on M3 Ultra: 100.3 → 83.2 ms per layer-call at S=131k (`csrc/sparse_mla.cpp:171-181`) |
| Shaped dispatch gating (anti-pattern guard) | Prefill-shaped kernels are *slower* at tiny L, so call sites gate them: sparse MLA "prefill-shaped grid runs ~8x slower than this at tiny L (4.2 ms vs 0.5 ms per layer at K≈4.7k)"; fused scores "~5x slower … (~3 ms per indexer layer at 128k)"; `_FUSED_SCORES_MIN_S = 16` | `glm_moe_dsa_model.py:270-276`; `patches/glm_moe_dsa/deepseek_v32.py:153-161` | as quoted |
| DSA index reuse across layers | `IndexCache`: layer `i % freq == 0` is "Full" and computes/stores top-k indices, others are "Shared" and run only the indexer K path — "Q and Q*K and argpartition skipped" | `patches/index_cache.py:5-6,54-96,217` | "freq=4 means 75% of indexers are skipped"; paper claim "Adjacent layers … share 70-100% of selected tokens" (arXiv:2603.12201, quoted in-file, not oMLX-measured) |
| KDA (GLM-5.3 linear attention) prefill kernels | Recurrence with the gate `exp(lb·sigmoid(exp(A_log)·(a+dt_bias)))` evaluated during staging — "the fp32 gate tensor never round-trips through memory"; fp32 state resident in registers; per-core partition (one threadgroup per GPU core, ~102 rows on an 80-core M5 Ultra, 12-token blocks, four steps' `q·S` partials reduced together) with a blocked fallback | `patches/glm53_kda_recurrence.py:1-38`; `patches/glm53_kda_prework.py:37-146` | "per-core measured ~1.5x slower on an 80-core M3 Ultra" (i.e. per-core is default where NAX exists, blocked elsewhere) |
| KDA decode as one kernel | "One kernel (`decode_kernels.kda_decode_step`) replaces the conv, SiLU, l2norm, gate projections, delta rule and RMSNormGated ops between the fused input projection and `o_proj`" for S<=8, capturing inputs for exact speculative replay | `vendor/.../glm5_next/language.py:553-570`; `decode_kernels.py:1843` | — |
| Decode-mode SDPA port | `qL<=8` tiled online-softmax decode attention with vectorized KV loads + a context-scaled 2-pass split for long context, ported from a closed MLX PR | `custom_kernels/decode_fast/csrc/sdpa_decode.metal:16-48`, `sdpa_decode_kernels.h:294-296` | — |
| TurboQuant KV (2-8 bit) | Codebook-vector-quantized K/V with fused decode/multi-row/prefill attention paths | `turboquant_kv.py:1-7`; `patches/turboquant_attention.py:28-53,239-242` | codec-level tests only (8-bit "near-lossless", rel err <5%); **excluded for GLM-5.3-Flash** — `engine/vlm.py:2567-2572` |

### D. Bytes per token, memory placement, cache

| mechanism | what it does | evidence | measured effect |
|---|---|---|---|
| MoE expert offload — positional reads | A resident slot cache holds a configurable fraction of each layer's experts; misses are read with positional `pread` on a thread pool from the checkpoint's own safetensors (no converted copy). Explicitly *not* `mmap` gather: "a slab faulted in 16 KiB pages on the compute thread reads at 0.4 GB/s" | `docs/experimental/MoE_Expert_Offload.md:1-19,164-167,296-310`; `patches/glm_moe_dsa/moe_offload.py:58-66` | synthetic oQ3e geometry (M5 Max): decode one token per MoE layer 362 ms → 9.1 ms; fetch throughput 0.23 GB/s → 9.3 GB/s; sorted prefill 33 → 568 token-layers/s (`MoE_Expert_Offload.md:178-184`) |
| Expert-major prefill chunking | Over-capacity prefill is chunked on expert boundaries so each expert is read at most once per layer per model call, instead of once per chunk that touches it | `MoE_Expert_Offload.md:57-70,90-98` | 32,768-token prompt on Qwen3.8-Flash-Next-oQ4e (M3 Ultra): 50% residency SSD 396→636 tok/s (+61%), reads 233→122 GiB; page-cache 517→751 (+45%) |
| Decayed-count eviction, not LRU | On a miss evict the resident expert with the lowest decayed routing count (×0.7 every 4 calls), and protect every expert the current call routes to | `MoE_Expert_Offload.md:73` | "misses 5-7% less often than LRU at 10-25% residency" (replayed Qwen3.8-Flash-Next routing traces) |
| Decode/read overlap | If the first miss has not arrived after 0.5 ms, the resident routes' gather is dispatched while the reads continue | `MoE_Expert_Offload.md:100-102` | 24 GB-emulated host, 10% residency: decode 6.53→8.17 tok/s, TTFT 47.5→19.7 s (6.2 K prompt) |
| Hot/cold tiered KV cache | 256-token blocks; SHA-256 **chain** hash (`b'omlx-root'` seed, "each block's hash depends on all previous blocks (similar to vLLM)"); refcounted copy-on-write; one safetensors file per block on SSD written without touching Metal from a background thread; RAM holds metadata only plus an optional byte-budgeted hot tier; LRU block pool | `scheduler.py:1707,1711-1713`; `cache/paged_cache.py:84-123,209-211,1355-1410`; `cache/paged_ssd_cache.py:942-997,1698-1702,1983-1989`; `cache/prefix_cache.py:242,251-253` | "load_block() … hit hot cache (~0ms) instead of SSD (~2ms per block)" (`paged_ssd_cache.py:4367-4368`); no end-to-end hit-rate/TTFT delta committed |
| Non-sliceable recurrent state | Recurrent/KDA/rotating layers cannot be block-sliced, so boundary snapshots are stored (and superseded snapshots dropped), with a walk-back that finds the last block where every non-sliceable layer carries real state; split-GDN keeps one durable recurrent sidecar at the exact terminal | `cache/prefix_cache.py:1683-1685,3692-3695,4724-4740`; `cache/boundary_snapshot_store.py:4-6,52-53` | "GLM-5.3 KDA: 34 layers × 4.1MiB per block" (`prefix_cache.py:4727`) |
| Chain-tip supersede | Drop superseded tip blocks or "those blocks fill the hot cache after ~10-20 turns and LRU eviction breaks the prefix chain (multi-turn cache hit collapses to 0%)" | `prefix_cache.py:1601-1604` | as quoted |
| Memory tiers + wired limit | Tier reserves/watermarks (safe 0.20/0.85/0.95 … aggressive 0.02/0.95/0.98 of RAM), a ceiling = `min(static, dynamic, metal_cap)`, and `mx.set_wired_limit` clamped to `iogpu.wired_limit_mb` — deliberately a no-op when that sysctl is unset | `process_memory_enforcer.py:59-129,240-330` | wrong placement of one table: "35.3 → 14.1 tok/s" (`:850-855`) |
| Keep-warm probe | One trivial kernel per period stops the GPU idle-state stall | `engine_pool.py:75-84` | "156 GB model on an M5 Ultra: +1.0 s after 2 s idle, +1.7 s after 6 s" |
| Chunked/packed prefill fairness | Chunks clamped to the next cache-block boundary; under contention chunk size is derived from *time* (a 500 ms decode-stall target, snapped to the 64-token grid with a 256 floor), prefill is blocked while decode "owes" time (debt = 0.5 × chunk wall time), and up to 4 requests' chunks share one packed `(1,T)` forward | `prefill_boundaries.py:4-24`; `scheduler.py:1563-1580,1688-1693,5918-5943`; `prefill/packed.py:30-38` | decode-fairness share 0.5 gives contended decode ~3x the unthrottled rate at parity on batch completion (M3 Ultra, `scheduler.py:1559-1563`) |
| ANE/GPU split prefill | Private `AppleNeuralEngine.framework` programs on ANE instances 1/2 compute disjoint output-channel slices of gate/up while Metal computes the rest, merged natively | `docs/experimental/qwen35_ane_prefill.md:1-20,52-66` | real-server 32 K cold/cache-hit pair: ~507 vs matched 416 tok/s GPU baseline (+22%), "specific to the tested M3 Ultra" (`:221-224,237-241`) |

### E. Quantization-time levers (see §4)

oQ's mixed-precision bit planning, imatrix-weighted rounding and least-squares refit buy quality per byte, not
speed; the only byte-reducing mechanism in this family is the t5 base-3 repack (§4).

## 3-Z. Correction (source re-check, second pass)

- The NAX/M5 non-transferability is NARROWER than first written: most of the cited family (GEMV/row-amortisation/expert-gather: `glm_qmv_rows`, `multi_qmv`, `mla_head_qmv`, `moe_verify_gather`, `moe_router_logits`, the KDA decode step and the 8-row DSA scores/topk) is CLASSIC Metal (plain simdgroup/qdot arithmetic), so it IS portable to our M4 Max. Only the hyper-connection comb-product/expand paths (`hc_post_mm`, `hc_pre_fused`'s MPP parts) are NAX-gated.
- The portable layer is `omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py`: pure `mx.fast.metal_kernel` source strings (no nanobind, no metallib, no MLX-private symbols except where it transcribes MLX's own qmv/qmv_wide traversal arithmetic) - i.e. the same mechanism we already use in src/trellis_gemm.cpp. Stage map: glm_qmv_rows 191-195, DSA scores/topk 1137-1450, KDA decode step 1537-1947, moe_router_logits 2088-2173, hc_pre_fused 2352-2750, hc_post_mm 2680-2765 (NAX), mla_head_qmv 2776-2874, multi_qmv 2946-3004.
- The exactness contract everywhere is transcription of the reference reduction order followed by token-identity tests.
- WITHDRAWN FACT: any 'lm_head M=1 1.2 ms -> M=4 4.3 ms' row-amortisation measurement is NOT attributable to commit 013bc3f3 (the mechanism is present, the number was not found in the clone); it must not be cited. Treat the row-amortisation *design* as evidence-backed and the *number* as unverified.

## 4. Quantization support and layout

- **Runtime weight formats.** MLX **affine** (group sizes 32/64/128, bits 1/2/3/4/5/6/8 — 1-bit needs the Bonsai
  fork of `quantized.h`, `custom_kernels/bonsai/csrc/bonsai_quantized.metal:1-12`), **MXFP4** (uint32 words +
  uint8 per-32 scales; `deepseek_moe.metal:204-207`), **MXFP8/FP8** (e4m3 codes with one `fp8_e8m0` scale per 32 K,
  `dspark_qmv.metal:31-40`), and **t5 ternary** (base-3 packed uint8).
- **oQ "oMLX Universal Dynamic Quantization"** (`omlx/oq.py:1-9`) produces ordinary MLX safetensors (loadable by
  mlx-lm/mlx-vlm, `docs/oQ_Quantization.md:1-3`). Levels oQ2…oQ8 with fractional variants; the pipeline measures
  per-layer sensitivity on calibration prompts, then allocates a **byte budget** with boost tiers (+4/+2/+1) to the
  most sensitive tensors (`oq.py:921-924,1043-1065`); routers stay full precision, MoE routers/norms/recurrent
  params/vision encoders are protected, GLM DSA indexers are pinned to affine 8-bit as a "format invariant"
  (`docs/oQ_Quantization.md:60-66`; `oq.py:1065`). **oQe** adds an importance matrix (imatrix, llama.cpp-style) plus
  a weighted least-squares refit of scale/offset, kept only when it lowers error (`docs/oQ_Quantization.md:38-44`).
- Published quality (not speed) numbers: oQ4e vs mlx-lm 4-bit on 1,600 questions — gemma-4-26B-A4B 81.62 vs 79.88
  (15.8 vs 15.4 GB); Qwen3.5-9B 72.38 vs 70.12; Qwen3.6-35B-A3B 81.00 vs 81.00 at +1.6 GB
  (`docs/oQ_Quantization.md:118-131`); refit pass (#4385) at batch 32: Qwen3.5-9B oQ4e 75.11 → 77.05, KL to the
  original down 12-15% (`:134-141`).
- **t5 ternary packing** (`scripts/repack_ternary_t5.py:1-38`): pack 5 trits per byte (base 3) → 1.585 bpw instead
  of 2.0 bpw slots, "~20% fewer weight bytes", lossless, with matching `affine_qmv*_t5` / `affine_qmm_t5` kernels
  that dequantize inside the simdgroup MMA ("Reads each t5 weight byte exactly once per matmul",
  `custom_kernels/bonsai/fast.py:435-465`).
- **KV quantization**: TurboQuant 2/2.5/3/3.5/4/6/8 bits (default 4 when enabled, off by default), implemented in
  mlx-vlm, per-token codebook states, with a skip-last-layer guard. Explicitly unavailable for GLM-5.3-Flash's
  composite latent/indexer cache (`engine/vlm.py:2567-2572`; `scheduler.py:3630`).
- **What oMLX does *not* have:** any trellis/codebook weight format (EXL3-style) and any VQ weight format. Every
  GEMV/GEMM loader in the tree reads MLX affine packing, MXFP4/8, or the t5 trit packing.

## 5. Transferable to exl3-server, ranked

Our position (from `docs/perf.md`): GLM-5.3-Flash, 45 layers, 288-expert MoE top-8, EXL3 trellis weights, ~4.6 GB
read per token at ~300 GB/s of a 546 GB/s machine → ~15 ms/step floor, measured step 153 ms with the MoE stage at
123 ms, one `mx::eval` per layer to read routed expert ids (`src/glm5_moe.cpp:127,283,320`), ~150 dispatches per
layer in the ungrouped form, and the 2-hyper-connection-per-layer periphery costing ~165 ms of a 317 ms step. Every
item below is ranked by expected effect *given that*.

**T1 — Fuse each decode stage into one dispatch, GLM-5.3-specific first.**
The measured shape of our problem (per-layer machinery + launch count, not the GEMV) is exactly what
`decode_kernels.py` was written for. Ordered by our own cost table: (i) hyper-connections — our single largest
component; oMLX's `hc_pre_fused`/`hc_post_mm`/`hc_mix` + branchless Sinkhorn kernel + cross-layer deferral is a
direct template for `src/glm5_hc.cpp`, and even the NAX-free subset (one dispatch instead of ~40 tiny ops, 4×4 comb
in `float4`, `simd_sum` column norms) attacks the 165 ms; (ii) the router — one kernel replacing ~25 small ops plus
the per-layer host round-trip; (iii) MoE gate/up/down+combine as 1-2 dispatches (oMLX measured 8.5→8.0 ms per
GLM-5.3 layer for gate/up alone); (iv) `kda_decode_step` for our 34 KDA layers (47 ms/step); (v) `multi_qmv` for
the `f_a`/`g_a`/`b_proj` set.
*Cost/risk:* medium-high — the *loaders* must be rewritten for trellis, but the fusion boundaries and the
exactness discipline ("replay the reference reduction order"; a product never contracted into an add that consumed
it in a different reference kernel, `decode_kernels.py:19-23`) port unchanged. Adopt their bitwise test harness
shape: fuse, then assert token identity against the unfused path.

**T2 — Device-side expert indirection: delete the per-layer `mx::eval`.**
ours: `mx::eval(indices)` per MoE layer (42 host round-trips/token at ~0.2 ms each ≈ 8 ms, plus graph
serialization). oMLX never reads routing back: `mx.gather_qmm` gets lazy device indices; the per-expert block table
is *built on the GPU* by a tiny kernel (`_build_mxfp4_blocks` → `grid=(num_experts,1,1)`,
`patches/deepseek_v4/switch_layers.py:149-165`) and consumed by `deepseek_affine_gather_qmm_blocks`; the A8 path
states the guarantee outright — "Stage A is left lazy (no mx.eval) … nothing is read back to the host and the
launch never synchronises" (`m5_gather_qmm_a8.py:555-562`). For us this means: pass routed ids **and** a
device-side expert table (offsets/pointers into a packed weight slab) into our group GEMV, and keep the only host
dependency optional (sampled token id).
*Cost/risk:* medium. `trellis_linear_group` already launches multi-buffer; the change is to select members from
device data instead of a host-built vector, plus a pointer/offset table built with a small Metal kernel. Risk: the
kernel must tolerate arbitrary index order (our current fast path assumes all rows share one hidden row).

**T3 — Amortize the verify pass over rows properly.**
Our MTP gives +30-50%, but per `docs/perf.md` the verify rows currently cost "their own ~4.6 GB" at the floor.
oMLX's numbers show *why* stock multi-row kernels fail (lm_head 1.2 ms at M=1 → 4.3 ms at M=4), and its fixes are
portable as designs: a tiled kernel that dequantizes a weight tile once per threadgroup and multiplies all rows
against it; and — for MoE — **expert-ordered pair execution** so that at rows×top_k pairs the same expert's tile is
read once instead of once per pair (`moe_verify_gather.py:4-15`), plus `MAX_ROWS=16` row-blocked tiles
(`row_exact_qmv.py:36`). Their row-exactness trick (`one_row_qmv` per row inside a multi-row tile) is also the
cheapest way to keep our verify bit-identical to serial decode. The GLM-specific number to keep in view: on
GLM-5.2 (8-of-256 routing, same geometry as our 288-expert top-8) one extra verify row costs **~35 ms** — 5x the
dense-backbone 7 ms default — because each extra row "pulls an almost disjoint expert set (~55% of step bytes)"
(`patches/mlx_lm_mtp/glm_moe_dsa_model.py:361-365`). That measurement is the budget statement for this whole item:
a verify batch only becomes cheap once the extra rows stop re-streaming expert weights.
*Cost/risk:* medium; we already have `mb=min(rows,4)` blocking, so this is an extension of existing structure. The
expert-ordering change is the one with a real bytes/token payoff (fewer expert weight re-reads per verify step).

**T4 — Pipeline submission: async-eval cadence and command-buffer caps.**
(i) The `_DECODE_EVAL_EVERY = 8` pattern with its stated diagnosis (~800 dispatches, ~170 buffers, 1.3 ms GPU idle
per token) is directly applicable: submit partial graphs so encoding of early layers overlaps building of later
ones (`language.py:70-79,1915-1922`). (ii) `set_command_buffer_caps(200, 512)` via the private-field friend trick
(`qwen35_prefill.cpp:879-896`) — the repo's own allowlist is M5/NAX **and `applegpu_g15` (M3 family)**
(`batch_generator.py:367-392`); our M4 Max is neither, so treat the benefit as **[UNVERIFIED] on M4** but the hook
is cheap to test (it prints the previous pair, so an A/B is mechanical).
*Cost/risk:* low for (i) — but note our third pass measured that adding `mx::eval` boundaries *inflates* a
step by ~0.3 ms per barrier, so the cadence must be `async_eval`, not `eval`. Low for (ii) except the private-API
risk (an MLX bump can break it; oMLX accepts that trade and degrades gracefully).

**T5 — Expert-level read minimization inside our existing group kernel.**
Independent of offload: our decode reads 8 routed experts per layer per token. oMLX's sorting/row-map machinery
(`moe_routes.sort_routes` — keep the sort lazy and read token rows *in place* rather than materialising the
replicated `[T*k,1,K]` copy; 0.4-0.5 GB per layer at 8192-token chunks is what the copy costs them) is cheap and
directly applicable to our prefill and verify paths, where we currently build per-expert work lists on the host.
Combine with gate/up fusion (already measured by them for exactly our family) so gate and up are one launch per
layer.
*Cost/risk:* low-medium, bit-identical by construction in their implementation.

**T6 — MTP/draft acceptance engineering (design, not code).**
Four transferable practices: one host sync per verify cycle with in-graph greedy acceptance and the next draft
dispatched by `async_eval` (`batch_generator.py:4027-4035`); **prompt priming** of the draft head from the prefill
forwards (0.90 vs 0.26 acceptance, `prompt_priming.py:3-10`) — our DSpark head takes target hidden taps and we
already do a prefill, so the same trick may apply; a depth controller that parks when the measured step cost says
so (0.67x on gemma4 story/16k); and using the rollback mechanism's row limit as the depth cap (GLM-5.3 native MTP
depth 7 from an 8-row pooling undo log, `glm5_next_vlm_runtime.py:50-52`).
*Cost/risk:* low for priming if our DSpark drafter can consume prefill hidden states; the sync structure is
already the single-sync shape in our engine (our step's only host dependency is the sampled token, plus the MoE
evals that T2 removes).

**T7 — KV cache: tiered prefix cache + non-sliceable state handling.**
Our workload is agentic (Claude-Code-style clients), which is what the tiered cache targets: 256-token blocks,
chain-hash prefix sharing with copy-on-write, SSD safetensors per block, hot RAM tier (~0 ms vs ~2 ms per block).
The parts worth copying are the ones we would hit with GLM-5.3's structure: **KDA/`ArraysCache` layers are not
block-sliceable**, so they are snapshotted at block boundaries with a walk-back that only accepts a prefix where
every non-sliceable layer carries real state, and superseded snapshots are dropped (34 layers × 4.1 MiB per block
for GLM-5.3 KDA is a real per-block cost). Also copy the tip-supersede rule, whose absence they measured as
multi-turn hit rate collapsing to 0%.
*Cost/risk:* high (a subsystem, not a kernel) and it buys TTFT/prefill, not decode tok/s. Rank this below T1-T6
unless multi-turn KV reuse is the current product bottleneck.

**T8 — Sparse-attention and indexer dispatch shape.**
Our MLA+indexer stage is 24 ms/token across 11 layers, and oMLX's shaped-dispatch lesson is directly applicable:
their prefill-shaped sparse-MLA kernel is ~8x *slower* than the per-row top-k gather at tiny L (4.2 vs 0.5 ms per
layer at K≈4.7k), and the fused indexer score kernel ~5x slower at tiny L (~3 ms per indexer layer at 128k), so
they gate by token count (`_FUSED_SCORES_MIN_S = 16`). Our indexer currently runs at rows=1 decode and rows=4
verify; any fused/sparse indexer kernel we adopt needs the same two regimes. Their in-kernel radix top-k (no O(K)
index sheet) and the `IndexCache` reuse of a Full layer's top-k on "Shared" layers (75% of indexers skipped at
freq=4, on a 70-100% overlap paper claim) are the two other ideas here; the reuse idea needs a per-layer overlap
measurement on GLM-5.3 before it is trusted.
*Cost/risk:* medium; measurement first (their claim is a paper number, not oMLX's own).

**T9 — `mx::compile` for the glue.**
Trivially confirmed practice: oMLX wraps its non-kernel HC math in `@mx.compile`
(`hyper_connection.py:185,259`), and our own `docs/perf.md` already lists `mx::compile` as lever #1. If T1's fused
kernels land, this shrinks to the residual elementwise chains.

**T10 — Quantization-side ideas (mostly do not apply to trellis).**
oQ's *method* — measure per-tensor sensitivity, then spend bits under a byte budget, keep routers/norms/recurrent
state unquantized, keep MTP heads — is a quality-per-byte policy we could apply to which tensors stay at K=4 vs
K=2 in our EXL3 layout. But: oQ consumes unquantized/FP8/QAT sources and cannot take an already-quantized model
(`docs/oQ_Quantization.md:109`), and the t5 base-3 packing (1.585 vs 2.0 bpw) is an entropy-optimal trick for
a **3-level** alphabet; our trellis codebooks are 2^K-level, so K=4 is already 4.0 bpw and there is nothing to
recover. **Do not port the packers.**

### What does NOT transfer

- **NAX / M5 tensor units.** The whole `*_nax` family (MPP `matmul2d`, 16×16 fragments, `nax_tiles.py`'s ported
  `BaseNAXFrag`/`NAXTile`) requires M5-class hardware *and* macOS ≥ 26.2, and the fused GLM-5.3 HC paths gate on
  `nax_relaxed_fp32_matmul()` (`decode_kernels.py:2273`, `hc_defer_supported` at `:2690`). M4 Max has no tensor
  units; also worth knowing, `bonsai/fast.py:127-138` documents that gen-17 M5 computes *wrong results* with NAX
  qmm/gemm. The *structures* (tile scheduling, "one threadgroup per expert tile", SwiGLU epilogue) transfer as
  design ideas, the kernels do not.
- **The MLX-affine / MXFP weight loaders.** `affine_qmv*`, `affine_gather_blocks_rhs_*`,
  `deepseek_mxfp4_gather_*`, `oq_a8_*` (which decodes affine Q4/Q5/Q8 codes into INT8 tensor fragments,
  `qwen35_oq_a8_nax.metal:1-78`) and the A8 Stage-A activation quantizer all read MLX's packing. EXL3 trellis is a
  codebook layout with no scales, so every one of these loaders must be rewritten. Only the launch/fusion structure
  ports.
- **`mx.gather_qmm` itself.** We cannot call it: it requires MLX-quantized weights. The transferable part is the
  architecture (device indices + a device-side block table consumed by a gather GEMM), which we must implement on
  top of the trellis group kernel.
- **Continuous batching via mlx-lm's `BatchGenerator`.** oMLX's engine is "a thin vLLM-shaped scheduler wrapped
  around mlx-lm's BatchGenerator" with external prefill and insert-last-token. We have our own C++ engine; the
  transferable content is the policy, not the code: prefill outside the batch, cap decode rows by admission,
  keep singleton caches un-batched (`scheduler.py:816-819`), and grow row banks on a capacity ladder instead of
  reallocating every 256 tokens (`patches/vlm_batch_kv_capacity.py:4-6`).
- **Expert offload as a *speed* mechanism.** It is a fit-the-model tool, not a throughput tool: on
  `mlx-community/GLM-5.2-4bit` at 20% residency, 213 misses/token × 20 MiB = 4.2 GiB/token read at ~8 GiB/s from
  SSD → **1.8 tok/s** (`MoE_Expert_Offload.md:290-310`), and the `mmap` variant measured 646 s to first token and
  0.09 tok/s. Their own "why not pin hot experts" section shows pinning is worse than useless for accuracy
  (~91% of gsm8k lost by zeroing the unpinned half). If our 121 GB model has to exceed RAM, copy the read *path*
  (positional `pread`, expert-boundary chunking, in-call protection, decayed-count eviction) — not the idea that
  it will be fast.
- **ANE/GPU hybrid prefill** (private `AppleNeuralEngine.framework`, dual-die M3 Ultra tuning), **multi-Mac
  pipeline/RDMA cluster**, **TurboQuant KV for GLM-5.3-Flash** (excluded upstream), and **oQ/t5 quantization
  tooling** (wrong format family, §4) are all out.
- **Python-level monkey-patching as the integration model.** 74 patch packages over three pinned git deps mean
  oMLX's maintenance surface is enormous and ABI-fragile (the `abi_probe`/nanobind-domain warnings exist because
  this breaks in the field: `glm_moe_dsa/fast.py:41-78`). We should take the kernel and scheduling ideas and keep
  our C++ ownership boundary.

## 6. Open questions / what I could not verify

1. **The 845 vs ~29 tok/s (GLM-5.2 fused DSA prefill) figure exists only as README prose** (`README.md:84-88`,
   `README.ko.md:86-87`). No benchmark script, prompt, batch size or checkpoint reference in the repo ties to it;
   grepping for `845` finds only the READMEs. Attribution and reproducibility: **[UNVERIFIED]**.
2. **The dflash-mlx "single-host-sync decode cycle" is not in the clone.** It appears only in the dependency
   comment `pyproject.toml:119-122` ("one D2H transfer per cycle instead of four syncs plus a hard eval"), pinned at
   `dflash-mlx @ jundot fork 71f7c2c`; there is no vendored copy and no local install to read. Which syncs were
   removed and their measured effect are **[UNVERIFIED]**. The *pattern* is independently documented in-tree for
   oMLX's own Lightning MTP loop (§3-B), which is what I based the transferability claim on. The docs also lag the
   pin (`docs/experimental/dflash_mlx_integration.md:72,202` still cite `omlx.4`/`8e1df22`).
3. **No committed batch-size-vs-throughput curve exists** for any model. The admin benchmark computes 2/4/8-row
   throughput at runtime (`omlx/admin/benchmark.py:44,1507-1522`); no result files are in the tree. So "batching
   raises effective bandwidth" is supported here only by kernel-level numbers (M=5: 71→104 GB/s), not by an
   end-to-end table.
4. **No GLM-5.3-Flash decode tok/s or kernel A/B is published anywhere in the repo.** The GLM-5.3-specific numbers
   I found are: the gate/up fusion ms/layer, the ~800-dispatch/2.7 ms/1.3 ms-idle diagnosis, the tiny-L dispatch
   penalties, and the `_MAX_CHAIN_DEPTH = 7` rollback clamp. All GLM-5.3 DFlash/MTP steps are stated as validated
   for correctness, not measured for speed.
5. **Per-kernel speedups for most native packages are absent** — `decode_fast` SDPA, the DSA gather kernels,
   `glm_moe_weighted_sum`, `dspark_*`, Minimax MSA top-k, the qwen35 qmm family: mechanisms and exactness
   contracts, no throughput. The NAX JIT kernels state their baselines (~10-13 TFLOPS sparse MLA; ~13 TFLOPS
   indexer) but never a paired NAX-vs-native measurement.
6. **TurboQuant KV has no measured accuracy/latency effect in-tree** (codec-level MSE/bias/parity tests only), and
   it is refused for GLM-5.3-Flash anyway.
7. **The tiered prefix cache has no end-to-end hit-rate/TTFT/TPS result committed.** The ~0 ms/~2 ms hot-vs-SSD
   figures are docstring claims with no stated method; the only measured hit-rate tables are for MoE expert offload.
   The "GLM-5.3 KDA 34 layers × 4.1 MiB per block" figure has no stated measurement basis.
8. **Whether raising MLX's command-buffer caps helps on M4.** Their allowlist is NAX (M5) or `applegpu_g15` (M3);
   M4 is unmeasured. Also **[UNVERIFIED]**: whether any of `decode_kernels.py`'s fused paths engage at all on a
   non-NAX GPU (`fused_decode_supported()` / `nax_relaxed_fp32_matmul()` gates), i.e. how much of the GLM-5.3
   decode win is available on M4 rather than M5.
9. **oQ bit plans are not appliable to a trellis engine**, and I found no statement about oQ interacting with any
   engine whose weights are not MLX-packed; that whole avenue is closed by format, not by measurement.
10. **MLX-lm internals were not readable here** (`mlx_lm` is pinned to a git commit and not installed in the
    clone), so statements about `BatchGenerator`/`GenerationBatch`/`_merge_caches`/`BatchKVCache` come from oMLX's
    own descriptions and call sites, not from upstream source. **[UNVERIFIED]** at the upstream level.

### Method note

All file:line citations above were read from the clone at commit `013bc3f3` (five parallel read-only passes plus
direct reads of the GLM-5.3 decode kernels, the MTP generator, the hyper-connection kernel, the MoE gather
dispatch, the KDA recurrence/prework kernels, the tiered-cache and enforcer code, and the two experimental docs);
every quoted sentence was re-read from the file. Nothing here was taken from a blog post. Where a number exists
only in README prose or a dependency comment, it is marked as such rather than presented as measured.
