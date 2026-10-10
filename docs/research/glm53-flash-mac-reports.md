# GLM-5.3-Flash at 2-3 bits on M4 Max / M5 Max: what the wild reports, with what software, and what that software does

Research date 2026-10-11. Scope: every credible report I could find of
**GLM-5.3-Flash** (and, where no GLM report exists, the closest same-format or
same-hardware sibling) decoding on M4 Max / M5 Max at 2-4 bits, the software
behind each number, that software's hot path with `file:line` where source
exists, and a calibration of our own measured numbers against them.

Our numbers, from `docs/perf.md` (same session, warmed, second process, real
checkpoint):

| ours | value | source |
|---|---|---|
| single stream (one request) | **8.66-8.70 tok/s**, ~115 ms/step | `docs/perf.md:325-470, 851-877`; B=1 batch 118.9/117.4 ms/step, 8.52 tok/s at `docs/perf.md:410-413` |
| aggregate B=8 | 19.28 mixed @B=8 (4 prompts), 28.94-29.7 @B=8 same prompt | `docs/perf.md:670`, `:414-418` |
| aggregate B=16 | **33.9 same prompt**, 31.5 four prompts (36.12 in the earlier run) | `docs/perf.md:851-877`, `:423-424` |
| byte cost | 6.1 GB/step at B=1, split 3.4 GB shared + 0.34 GB × distinct experts | `docs/perf.md:436-470` |
| kernel ceiling | trellis GEMV 258-305 GB/s rows=1 on 4096x24576 K=4; 118-202 GB/s on this model's shapes; 0.43 issued warp-instr per weight = the issue wall | `docs/perf.md:68-88`, `:436-470`, `:857-860` |
| M4 Max achievable streaming (reduction kernel) | ~425 GB/s of the 546 GB/s spec | `docs/dspark.md:8-13` |

Checkpoint under test: `MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3`
(98.123 GiB on disk, ~121 GB resident working set), EXL3 2.51 bpw, h6 codebook,
from `orcarouter/GLM-5.3-Flash-Uncensored-FP8`; 288 experts top-8, 45 layers
(34 KDA + 11 sparse-MLA), 1M context, MTP head (`layers.45`) present in the pack.

### Headline answers

| question | answer |
|---|---|
| Is anyone faster than our 8.66 tok/s single stream **on M4 Max at 2-3 bits**? | **No.** There is no M4 Max 2-3 bit report of this model at all. The only two M4 Max numbers are 6.21 t/s (4-bit Q4_K GGUF, SSD-streamed, ds4) and 2.7 t/s (4-bit MLX `oQ4e`, expert offload, oMLX) — both below us, both memory-plumbing regimes. |
| Are the fast numbers people quote for this model on our hardware? | **No — they are M5 Max/Ultra.** M5 Max 128 GB, 2-bit: **~38.7 t/s plain, 42.0 with MTP** (oMLX v0.7.0). M5 Ultra 4-bit: 141 t/s. Those are different silicon (per-core Neural Accelerators via the Metal 4 tensor API, 614 GB/s vs our 546) and must not be transferred to M4. |
| Which software is fastest at 2-3 bits, and how? | **oMLX v0.7.0** on M5 Max at 2-bit, and **Kyojin** (ExLlamaV3-based, ROCm) on a 128 GB Strix Halo for our exact format (EXL3 2.05 bpw → **26.7-31.6 t/s**). Both win by *fewer issued instructions per weight* (Kyojin ~4-6 vs our 13.75), *decode-once-per-R-rows* for verify rows, fused per-stage kernels, and MTP. See §2.5. |
| Are we above or below the wild? | **Above every same-hardware report; below the same-format state of the art by ~3×.** Our 33.9 tok/s B=16 aggregate has no wild counterpart on M4 Max at all. |
| What is the single biggest transferable mechanism? | Decode once per R rows ≥ 8 — measured **2.4-4.8×** per tensor class in this tree today (`docs/perf.md:879-974`) — then grading the 19-50 GB/s small/expert tensors, then MTP in the serving path. Details and bit-identity posture in §4. |

---


## 1. Every credible report

Evidence classes: **[MEASURED]** = a named operator or harness publishing its own
run with hardware, quant and engine; **[CLAIM]** = attributed but thin (no
methodology, or the reporter's own machine quoted loosely); **[UNVERIFIED]** =
screenshot/marketing/estimate-aggregator, not evidence.

### 1a. GLM-5.3-Flash on M4 Max 128 GB — our exact hardware class

| # | tok/s (gen) | prefill | quant / bpw | software + version | ctx | class | source |
|---|---|---|---|---|---|---|---|
| A1 | **6.21** | 2.60 t/s | Q4_K GGUF, 177.77 GiB, **streamed from SSD** | ds4 (DwarfStar) | 4k | [MEASURED] | [llamaperf report 1864](https://llamaperf.com/model/glm-5-3) |
| A2 | **2.7** | 63.5 t/s | MLX oQ4e (4-bit), MoE expert offload on (resident 0.42) | oMLX **v0.7.0** | 32k | [MEASURED] | [omlx.ai/benchmarks/performance/nzjoog4a](https://omlx.ai/benchmarks/performance/nzjoog4a) |
| A3 | "low-to-mid tens" | not given | oQ2e (2-bit, target 2.8 bpw) 103 GiB; offload on (resident 0.825); MTP accept 8/9 (88.9%) | oMLX + oQ quantizer | 262k-capable | [CLAIM] — the quantizer's own machine, explicitly "no systematic benchmark was run" | [Cropduster69/GLM-5.3-Flash-oQ2e-mtp card](https://huggingface.co/Cropduster69/GLM-5.3-Flash-oQ2e-mtp) |

Notes that matter for comparison:

* A1 is the only careful, instrumented M4 Max run of this model I found. It says
  the GPU accounts for only **~24 ms of ~161 ms per token (15%)**; host-side
  per-layer expert-cache bookkeeping dominates. MTP made it **15% slower**
  (5.28/5.22 t/s) at 4-bit. It is a *4-bit, SSD-streamed* configuration, so it is
  not weight-matched to us — but it is the same chip, same model, same "does not
  fit in 121 GB free" problem.
* A2 is 4-bit with offload and is RAM-starved; its 63.5 t/s prefill / 2.7 t/s
  decode pair is a disk/offload number, not a kernel number.
* A3 is the closest same-bit-width datapoint on M4 Max in existence, and it is
  the pack author's own machine — but it is vague, offload-on, and unmeasured.

**No wild M4 Max report of GLM-5.3-Flash at 2-3 bits exists that reports a
number above our 8.66 tok/s single-stream, and no M4 Max report of any kind
reports an aggregate number for this model.**

### 1b. GLM-5.3-Flash on M5 Max 128 GB (40-core) — **different silicon, do not transfer**

| # | tok/s (gen) | prefill | quant | software + version | ctx | MTP | class | source |
|---|---|---|---|---|---|---|---|---|
| B1 | **42.0** | 880.9 t/s | oQ2e-mtp 2-bit (imatrix `oQe`), 103-110 GB resident | oMLX v0.7.0 | 4k | **on** (adaptive depth 3) | [MEASURED] | [omlx.ai/…/ibt32whi](https://omlx.ai/benchmarks/performance/ibt32whi) |
| B2 | **38.7 / 38.6 / 37.9** | 777.7 / 778.8 / 509.3 | MLX 2-bit + MTP pack, 109 GB resident | oMLX v0.7.0 | 1k | **off** (`mtp_enabled false` in the recipe) | [MEASURED] | [omlx.ai/…/0k222mu6](https://omlx.ai/benchmarks/performance/0k222mu6) (+2 siblings) |
| B3 | 22.0 and 9.6 | 233.4 / 155.0 | same oQ2e-mtp pack, same day; same settings except `moe_expert_offload_resident_fraction` = 0.85 and 0.25 | oMLX v0.7.0 | 4k | on | [MEASURED] | omlx.ai xv65dcfe, 3w2oor7e |
| B4 | 4.2-5.8 | 56.7-180 | oQ4e 4-bit (~150 GB → does not fit 128 GB) | oMLX v0.7.0 | 1k-16k | off | [MEASURED] | omlx.ai rows j8e2oqcc, 8yl6c0en, w8i6k3j4, hcfsddpv |
| B5 | 35.4 | "127" (internally implausible) | not stated | not stated | not stated | ? | [UNVERIFIED] — X blocks automated reads, so only the search snippet is available | [x.com/moelkholy95/status/2094586969921552649](https://x.com/moelkholy95/status/2094586969921552649) |
| B6 | 27 | — | "IQ3_XXS" | "MLX (recommended)" | — | — | **not evidence** — LLMCheck's own page says the figure is "estimated" and its methodology line describes *Q4_K_M*, contradicting its own header | [llmcheck.net](https://llmcheck.net/models/glm-53-flash-on-m5-max) |

Reading B1+B2+B3 together gives the cleanest wild datapoint in existence: on M5
Max 128 GB, a 2-bit GLM-5.3-Flash pack decodes **~38-39 tok/s plain**, **42.0 with
native MTP** (+9%), and **22.0 / 9.6 once expert offload is switched on at 0.85 /
0.25 residency** — i.e. on this model, on this silicon, offloading experts costs
1.9× and 4.4× respectively. That is the same lesson as the M4 Max ds4 row (A1)
from the opposite direction: expert-table plumbing, not arithmetic, is the
dominant cost when the model does not sit fully in memory.

A full synthetic sweep of the oMLX database (all quant and context filters, full
cursor pagination) shows the shape of the whole M5 Max 2-bit set — **20 rows,
16.0-42.0 TG**, and the spread is explained almost entirely by residency, not by
kernel choice:

* **42.0** — `oQ2e-mtp`, MTP **on**, expert offload **off**, 4k, 108.3 GB peak;
* **38.7 / 38.6 / 37.9** — the `MLX-2bit-MTP` pack, MTP **off**, offload **off**;
* **16.0-24.6** — 13 rows with MTP on *and* offload on (residency 0.78-0.92);
* **16.0** — two rows MTP off + offload on;
* **4.2-5.8** — the 4-bit rows (83.2 GB peak, residency 0.4).

Two external cross-checks on the same machine, neither from oMLX: **ds4 with a
90 GiB GGUF Q2 pack fully resident = 33.69 TG at 32k** [MEASURED], and
**llama.cpp Metal with Unsloth UD-IQ3_XXS on an M5 base ≈ 31 t/s low-ctx / 25
t/s high-ctx** after the Lightning-Indexer fix
([ggml-org/llama.cpp#29825](https://github.com/ggml-org/llama.cpp/issues/29867)
closure). **There is no 3-bit GLM-5.3-Flash row on any M5 Max anywhere** — the
only 3-bit GLM rows in the oMLX database are M5 Ultra 64c (51.4 / 41.5).

### 1c. GLM-5.3-Flash on other Apple Silicon (context only, not transferable)

| tok/s | hardware | quant | software | ctx | class | source |
|---|---|---|---|---|---|---|
| 141.0-141.8 | M5 Ultra 256 GB | oQ4-mtp | oMLX v0.7.0 | 16k | [MEASURED] | omlx.ai iuwlxla7/xm00pe4x/o2kwgd4x… |
| 141.0 | M5 Ultra 256 GB | MLX 4-bit | oMLX | 128k | [MEASURED] | omlx.ai d86mgd6n, nsfevu9x |
| 76.7 (55.4 on 0.7.0rc1) | M5 Ultra 256 GB | oQ4 + TurboQuant KV 4-bit | oMLX 0.7.0 | 4k-200k | [MEASURED] | llamaperf report 1535 |
| 71.1 @131k (72.7 @32k) | M5 Ultra 512 GB | MLX 4-bit | mlx-vlm, MTP on | 131k | [MEASURED] | llamaperf report 1028 |
| 64.3 / 53.1 (DFlash2) | M5 Ultra 64c 256 GB | oQ4e-mtp | oMLX | 64k | [MEASURED] | omlx.ai (UNCENSORED-oQ4e-mtp / -DFlash2) |
| 51.4 / 41.5 | M5 Ultra 64c 256 GB | JK-oQ3.5e 3-bit | oMLX | 64k | [MEASURED] | omlx.ai bzhhxabr, k2idc003 |
| 37.4 end-to-end @300k (serial 29→40 short ctx) | M3 Ultra 256 GB | Q4 | ds4 branch, no drafter; optional dflash +20-50% | 300k | [MEASURED] | llamaperf report 756 |
| 58 (from 27) | M3 Ultra | MLX-class + drafter | TensorFold | — | [CLAIM] (aggregator page, no log) | vramcalculator.com/tensorfold |
| 24 | M5 Max + Strix Halo over TB4 | IQ1_S 1-bit GGUF | llama.cpp | — | [CLAIM] | llamaperf report 788 |
| 1.82 | M4 Pro 48 GB | JANG (streaming) | oMLX | — | [MEASURED] | llamaperf report 779 |
| 0.085 | M1 Pro 32 GB | FP8, expert streaming | apus | 1M | [MEASURED] | llamaperf report 1369 |

One more row belongs in spirit: **TensorFold's `--ssd-experts` mode on an M3
Ultra, held to a 128 GB budget, decodes at 8.9-10.5 TG with prefill 144 vs 451
resident, and 36/36 identical tokens** (vramcalculator/TensorFold notes). That is
the closest published analogue to our configuration — a 2-3 bit GLM-5.3-Flash
with the expert tables partly outside memory — and it lands in the same 9-10 tok/s
band we are in, with prefill 3× worse. It is the best available sanity check that
our 8.66 t/s single stream is a *normal* number for a streaming/expert-tiered
regime, and it is also the reason to take the "no M4 Max 2-3 bit report" finding
seriously rather than as a search failure: the people who run this model at
2-3 bits on a 128 GB machine are running it *with* streaming, and they are not
faster than us.

### 1d. Same format (EXL3, 2.05 bpw), same memory class, **not Apple hardware**

These are the numbers that actually constrain "how fast can an EXL3 2-bit
GLM-5.3-Flash single stream go on a ~128 GB unified-memory machine", because
they use the same turboderp EXL3 trellis format we do.

| tok/s (gen) | machine | quant | software | class | source |
|---|---|---|---|---|---|
| **26.7 plain greedy / 26.2 prose, 31.4 chat, 33.4 code with MTP(2)**; 31.1-32.3 across 8K-256K; an earlier 82 GB pack measured 32.1-35.8 | Ryzen AI Max+ 395 (Strix Halo), 128 GB LPDDR5X, 256 GB/s, gfx1151 | EXL3 2.05 bpw + experts 29-44 from 3.05 bpw = 99.73 GB | **Kyojin** (ExLlamaV3-based, ROCm), MTP 2 drafts | [MEASURED] prefill 634 t/s @8K, 637/613/589/567 @32/64/128/256K; KLD 0.151, top-1 89.31% vs FP8 | [yamz-labs card](https://huggingface.co/yamz-labs/GLM-5.3-Flash-EXL3-Yamz), llamaperf report 1357 |
| 25 (prose) to 64 (structured); 25-34 short-ctx in a second run, 12.97 at 250K | 1× DGX Spark GB10 (121 GiB, 273 GB/s) | EXL3 2.05 bpw, 85.23 GB | ExLlamaV3 + DFlash2 K7 (block drafter), MTP acceptance 88.24% | [MEASURED], disputed in-thread (crashes; 50 t/s before the fixes, 40 after) | [NVIDIA forum 382140](https://forums.developer.nvidia.com/t/60-tok-s-glm-5-3-flash-on-a-single-dgx-spark/382140) |
| 28.8 @8k EXL3 (30.1 NVFP4) | 2× DGX Spark | EXL3 / NVFP4 | vLLM + DFlash2 k=7 | [MEASURED] | llamaperf report 1368 |
| 28.6 | RTX 3090 24 GB + host RAM | EXL3 3.05 bpw, all-RAM fast mode, CPU offload | ExLlamaV3 | [MEASURED] 989 t/s prefill | llamaperf report 1874 |
| 69.9 short-ctx (52.7 @8k, 12.2 @128k) | RTX PRO 6000 96 GB | IQ2_XXS GGUF, fully resident | llama.cpp | [MEASURED] | llamaperf report 1866 |
| 179.5 | 2× RTX PRO 6000 96 GB | EXL3-K3 + FP8 KV | vLLM, TP2/EP2/DCP2, DFlash2 | [MEASURED] 1,045 t/s aggregate @c16 | llamaperf report 1367 |

### 1e. The calibration block: **other MoEs on our own M4 Max**

This is what lets us interpret our own number without crossing hardware.

| tok/s | model (active params) | quant | software | class | source |
|---|---|---|---|---|---|
| **33.2 serial**; 73.8 with native MTP depth-4 (**2.22x**) | Qwen3.8-125B-A6B Flash-Next (6B) | Q4 affine experts, BF16 verify head | mlx-serve | [MEASURED] | llamaperf report 1648 |
| 60 serial; 78 with MTP | same (6B) | mixed 4/8-bit MLX | mlx-serve 26.8.11 | [MEASURED] 730 t/s prefill | llamaperf report 1655 |
| 57.4 (50-70 @128k) | same (6B) | oQ4e + MTP | oMLX | [MEASURED] 586 t/s prefill | llamaperf report 1653 |
| 45.01 MTP off / 90.07 MTP on | same (6B) | MLX pack | MLX | [CLAIM] — author states quants are *not* matched | llamaperf report 1813 |
| 27.3 | same (6B) | UD-Q3_K_XL GGUF, no MTP | llama.cpp main | [MEASURED] | llamaperf report 1813 |
| **31.7** | DeepSeek V4 Flash 285B (13B active) | IQ2_XXS/Q2_K GGUF | ds4 (DwarfStar) + DSpark spec @conf 0.6 | [MEASURED] DSpark adds ~10-20% | llamaperf report 1880 |
| **34.5** | Bonsai 2 27B dense ternary (1.58-bit, 7.2 GB) | PQ2_0 GGUF | llama.cpp (Prism fork), no spec | [MEASURED] | llamaperf report 1744 |
| 127.7 | Qwen3 30B-A3B (3B) | 4-bit | vllm-mlx | [MEASURED] | llamaperf M4 Max page |

Derived effective streaming rates on the same chip (bytes/token ÷ step time):

| run | weights read per token (est.) | effective GB/s |
|---|---|---|
| ours (2.51 bpw, 18B active) | **6.1 GB** (measured, `perf.md:436-470`) | **53** |
| Bonsai 2 27B ternary, llama.cpp | 7.2 GB file | **~248** |
| DeepSeek V4 Flash 13B-active @2-bit, ds4+DSpark | ~3.3-4 GB per *step*; ~28 steps/s at +15% tokens/step | **~95-110** |
| Qwen3.8-125B-A6B, 4-bit affine, mlx-serve serial | ~4-5 GB | **~135-165** |
| our own trellis GEMV, isolated big shape | — | **258-305** — and this is *the K4 issue wall* (298 GB/s per §2.5), not a bandwidth figure (`perf.md:68-88`) |

The last two rows are the whole story: **our kernel is at its own issue wall
(~298 GB/s on K4, 289-305 measured), and at model level we deliver 53 GB/s
because of per-row decode passes, small-tensor launches and per-token machinery —
not because the kernel is slow or the DRAM is busy.**

---

## 2. The software, and how each one does the hot path

### 2.1 oMLX (github.com/jundot/omlx) — the source of the M5 Max 2-bit numbers

The engine behind B1-B4 and most of 1c. Verified in this pass against
`github.com/jundot/omlx` at commit `013bc3f`, plus the earlier in-repo pass
(`docs/research/omlx.md` §3, §4, T3). The decisive facts:

* **It is a fused per-stage decode engine.** `omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py`
  (3004 lines) holds every `glm5_next` decode/verify kernel as inline Metal:
  `moe_router` (`:2135`, logits `:1964-2015` + select `:2016-2087`),
  `moe_gate_up_swiglu` (`:795`, source `:405-530`; `gate_up_proj` fused via
  `ESTRIDE`/`UP_OFF` at `vendor/.../language.py:1653-1660`),
  `moe_down_combine` (`:933`), `kda_decode_step` (`:1843`, one 1024-thread
  threadgroup per head over the whole KDA body for T≤8),
  `dsa_decode_scores`/`dsa_topk_rows`/`dsa_expand_topk`/`dsa_gather_selected`
  (`:1227/:1383/:1443/:1506`), `mlp_gate_up_swiglu` (`:573`), and the
  hyper-connection/MLA-head fusions (`hc_mix :1106`, `hc_pre_fused :2709`,
  `hc_post_mm :2752`, `mla_head_qmv :2835`, `multi_qmv :2960` — the last fuses up
  to 4 same-format projections of one input into one dispatch,
  `language.py:1437`). The rationale is in the file itself and matches our own
  diagnosis: *"Single-token decode and short verify blocks (L <= 8) are dominated
  by thousands of tiny dependent dispatches"* (`decode_kernels.py:1-23`).
* **Its GEMV is MLX's, not a new arithmetic core.** `glm_qmv_rows`
  (`decode_kernels.py:195`) reproduces MLX `qmv_fast` (2 simdgroups × 4 rows per
  8-row tile, `qdot` per K block, `simd_sum`), and `glm_qdot :121` /
  `glm_load_vector :66` are *"verbatim copies of MLX quantized.h"* with the same
  unpack masks, and *"every reduction replays the order of the MLX kernel it
  replaces"* (`decode_kernels.py:22-30`). The win is launch/host-encode
  reduction, not instructions per weight.
* **The fused GEMV path is affine-only and declines 2-bit.** `_qmv_fast_ok`
  (`decode_kernels.py:774`) requires bits ∈ {3,4,5,6,8}; `_affine_parts :785`
  rejects any mode ≠ `affine`. **So on B1's `oQ2e` pack the 2-bit routed experts
  fall back to stock `mx.quantized_matmul` / `gather_qmm`** — only the pack's
  4-bit MTP-head experts and 8-bit attention/shared/head projections ride the
  fused kernels. The M5 Max 2-bit number is therefore *stock MLX affine 2-bit
  GEMV plus oMLX's machinery and MTP*, not an exotic kernel result.
* **`oQ` is not a format.** `_mode_for_bits` returns `affine` for every width
  (`omlx/oq.py:4615`), group size 64 (`oq.py:43`), `_pack_affine_codes`
  (`oq.py:5667`) whose docstring says the layout *"intentionally matches
  mx.quantize(mode=affine)"*; levels 2/2.5/2.7→2-bit, 3/3.5→3, … (`oq.py:50`,
  `:105`). `oQ2e`/`oQ4e` are the imatrix ("enhanced") variants at those levels.
* **Some of its `glm5_next` kernels are M5-only.** oMLX's GLM PRs describe
  *"fused MoE gate/up with NAX gather tiles"* and a *"tensor-unit indexer and
  sparse MLA"* — "NAX" being the community name for the M5's per-core Neural
  Accelerators (§2.3). That is a second reason the M5 numbers cannot be read as
  an M4 target even after discounting the affine-vs-trellis difference.
* **There is no EXL3 loader anywhere in the oMLX tree**, so our checkpoint is not
  loadable by oMLX at all — a hard comparability blocker, and the same statement
  holds for PonyExl3 (its `model_type` list is `qwen3_5`, `qwen3_5_moe`, `llama`,
  `gemma4`; `glm5_next` is unmapped) and for TensorFold (its EXL3 reader is
  CUDA-only, with no GLM native-CUDA EXL3 backend).
* **MTP is a real adaptive engine, not a fixed k.** `patches/mlx_vlm_mtp/glm5_next_vlm_runtime.py`
  attaches the checkpoint's DeepSeek-V3-style nextn head as `mtp.<i>` (depth
  capped at 7 because the cache undo log only covers ≤8-row updates), with a
  `_DepthController` (`patches/mlx_lm_mtp/batch_generator.py:2477-2563`) choosing
  1..max per sequence from an EMA of per-depth acceptance and per-depth cycle
  cost (ALPHA 0.08, TAU 400 ms, hysteresis 1.03, exit margin 1.15, exit streak
  16), a depth-0 escape to plain decode, greedy equality / Leviathan–Chen
  verification (`:1645-1690`), acceptance logging
  (`MTP[uid] … accept=A/C (rate%) depth[d1=…]`, `:3719-3775`), and **exact
  rollback** by replaying the accepted prefix through the same kernel because the
  KDA `ArraysCache` cannot be trimmed (`glm5_next_vlm_runtime.py:26-34,640-730`).
  DFlash2 exists as a pinned fork (`dflash-mlx`, `pyproject.toml:119-132`,
  `patches/dflash_glm5.py`) pointed at `incoai/GLM-5.3-Flash-DFlash2`.
* **The published numbers come from a shipped harness** (`omlx/admin/benchmark.py`,
  2319 lines, uploads to `omlx.ai/api/benchmarks` at `:1285`) that measures
  single-request prefill/generation at prompt lengths 1024…200000
  (`:41`) and continuous batching at batch sizes **[2,4,8]** (`:44`), with a
  unique prompt UUID to defeat prompt-cache hits (`:543-561`). So the
  community leaderboard's aggregate axis only goes to B=8.

What I verified first-hand, from the oMLX benchmark pages themselves:

* the runs are **oMLX v0.7.0**, macOS 27.0/27.0.1, and each row publishes its
  full 40-42 setting recipe (`mtp_enabled`, `mtp_adaptive_max_depth`,
  `moe_expert_offload_enabled` + `resident_fraction`, `dflash_*`,
  `turboquant_kv_*`) plus peak-footprint traces every 0.25 s. That is unusually
  good provenance and is why I count these as [MEASURED] rather than CLAIM;
* the fast M5 Max row (B1) is `mtp_enabled=true` with `mtp_adaptive_max_depth=3`;
  the ~38.7 rows (B2) are `mtp_enabled=false` on a pack whose name says MTP — so
  MTP buys ~+9% on this model on M5 Max, far less than the 1.7-2.2× it buys on
  dense or smaller-MoE targets;
* B1/B3 differ only in `moe_expert_offload_resident_fraction` (B1 has offload
  *off* at 110.4 GB peak; B3's 22.0 has it **on at 0.85** with a 98.4 GB peak, and
  the 9.6 row has it **on at 0.25** with a 44.4 GB peak and only 48% GPU
  utilisation). So the whole 42.0 → 22.0 → 9.6 spread on the same machine, pack
  and day is explained by *how much of the expert table is resident* — a direct
  measure of what expert offload costs on this model, on this silicon;
* the `oQe` packs are imatrix-calibrated **streamed from the FP8 checkpoint**
  (one calibration round, 128×512 tokens, 671 entries, expert coverage
  38 688/38 688), with the MTP head preserved *and* calibrated, 2-bit on routed
  experts and 8-bit on attention/shared/head, 4-bit on the MTP head's experts.

### 2.2 PonyExl3 (beamivalice/PonyExl3) — EXL3 on Apple Silicon, in MLX/Metal

The closest thing in the wild to what we built, and the only public
implementation of the **same format and same kernel strategy** on Metal. It does
not yet support `glm5_next` (its published tables are Qwen3.6-27B/35B-A3B,
Gemma4-26A4B, MiniCPM5-1B), so there is no GLM number from it — but its kernel
work is directly readable, cloned to `/tmp/ponyexl3`:

* **M=1 GEMV** — `ponyexl3/mlx/gemv_metal.py:109` `_gemv_simd_source` ("v12:
  simdgroup-cooperative GEMV (M=1), ds4/llama.cpp mat-vec pattern"): one out-tile
  (16 cols) per threadgroup, `NSG 4` simdgroups striding in-tiles with **zero
  barriers in the hot loop**, lane `l` owning 1-2 trellis words plus the x value
  for row `l & 15`, x fetched by `simd_shuffle` from registers (`:110-118`).
* **The decode is the CUDA EXL3 3-instruction codebook**, inlined:
  `dq_cw = cw * 89226354u + 64248484u`, then `(dq_cw & 0x8FFF8FFF) ^ 0x3B603B60`
  (a `lop3`, forcing the exponent so it can never be NaN/inf), then one `half2`
  add — `gemv_metal.py:36-58` (`_decode_expr`), with the comment that this
  *"matches CUDA `__hadd`"*, i.e. the same numerics we inherit.
* **`dq4` window trick** — `gemv_metal.py:166-232`: a 4-aligned group of 4
  consecutive codewords spans `3K+16` bits, which for K∈{1..6,8} fits **one**
  64-bit window of exactly two words, so 4 weights are extracted from 2 loads +
  1 shift/or + 4 shifts + 4×(3-op decode) + 4×FMA + 4×`simd_shuffle`, with 8
  register accumulators per lane. Compared with per-pair funnels this halves the
  loads/merges.
* **Fused Hadamard** — `gemv_metal.py:77-107` `_had_prologue`: a cooperative
  7-stage 128-point butterfly *inside* the GEMV kernel, with the rationale
  *"Replaces the separate prep dispatch (and its serial GPU gap) … The GPU is
  idle between dependent dispatches anyway, so the extra ALU is free."* Same
  insight as our `docs/perf.md` lever 3.
* **Small-batch verify path** — `gemv_metal.py:234-300` (`MT > 1`): x is staged
  *transposed* in ping-pong threadgroup buffers so a lane reads its row's MT
  values as `float4`, accumulation becomes vector fma and *"the per-row cost
  collapses to ~1 op per 4 rows and decode dominates again."*
* **The v20 small-batch GEMM** — `gemv_metal.py:1316` `_gemm_devx_source`, whose
  header records the measured working points: staged design was
  threadgroup-bandwidth bound at mt=8 (201 Gw/s vs a ~308 slot model), device-
  direct transposed loads reach **mt=8 309-318 Gw/s (+58%), mt=4 456-486 (+33%,
  i.e. at the mt=1 GEMV's rate)**, bit-identical outputs, zero hot-loop barriers.
* **MoE** — `ponyexl3/mlx/exl3_moe.py:1-16`: all selected experts' trellises are
  stacked along out_tiles into one device buffer (gate+up interleaved, down
  separate) and decode runs **one mapped GEMV per projection group** covering all
  selected experts, with a device-side `tile_map` indirection and `tile_sub`
  selecting each (expert, proj)'s rotated activation row. Prefill instead routes
  flattened (token, slot) pairs through `mx.gather_mm`, **sorted by expert first**
  so each expert's weights stream once per chunk, not once per pair.
* **Dispatch ladder** — `ponyexl3/mlx/exl3_linear.py:8-9`: huge layer (lm_head)
  M=1 → fused GEMV with no weight materialisation; M≤144 → fused GEMM;
  `:47` "above this batch, decode-once (v13) + native matmul beats the fused
  GEMM"; `:141` the fused GEMM re-reads the trellis once per 8 batch rows.
* **Sibling fusion** — `ponyexl3/mlx/exl3_fused.py:1-15`: sibling projections that
  share an input (q/k/v, gate/up) become views onto one `FusedEXL3Group` that
  computes all outputs in one prepare + one GEMV + one finish, *"Per decoder
  layer this collapses ~9 dispatch chains into ~3."* Members' trellises
  concatenate along out_tiles (`model.py:376-430`, `exl3_fused.py:93-233`).
  Env knob `EXL3_FUSE_MIN_MB` (default 64 MiB, `exl3_linear.py:46`,
  `model.py:363-373`). **Direction matters:** the repo measures fusion as a
  *throughput* trade — it pays off when the kernels are big (27B: +prefill,
  +decode) and *regresses decode ~2×* on small/small-tensor models
  (`model.py:380-382`, README:45-46). It is a dispatch-collapse lever for
  big-MoE models, not a free win.
* **Tensor units are absent from the decode path entirely.**
  `simdgroup_matrix` appears only in the v19c segmented *prefill* MoE GEMM
  (`gemv_metal.py:1519-1543`, `:1597`, `:1721`), gated to `M > 8` with 64-row
  blocks and routed at `exl3_moe.py:701-760` (`_MM_MAX_ROWS = 9216`). Dense
  decode goes v18/v12 at M=1, v16/v20 at rows 2-16, and decode-all + MLX native
  matmul above 64 rows (`exl3_linear.py:150-165`).
* **Per-weight decode cost** (counted off the emitted Metal for K4/MCG/MT=1):
  ~76 ops per lane-iteration for 8 weights plus ~6-8 loop overhead ⇒ **~10.5
  issued lane-instructions per weight**, against our QMV's measured **13.75**
  (110 warp-instructions per 256-weight tile, `docs/perf.md:496-508`,
  `docs/research/exlr8.md:141-150`). Theirs is *cheaper per weight* — but on the
  same box, same shape, same format it is **2.5-3.5× slower per tile than our
  QMV** (their v12: 0.42-0.59 ms for 393,216 tiles of K4 4096×24576; ours:
  0.168 ms), at only ~23-30% issue occupancy. Their M=1 GEMV is
  latency/occupancy-bound, not ALU-bound. See §2.5.
* **Speculation is verify-gated four ways** — `ponyexl3/mlx/spec_sampling.py:59-110`:
  greedy acceptance at temp 0 (token-identical), Leviathan–Chen rejection
  sampling at temp>0 (distribution-exact), with MTP (`--mtp auto`), DFlash,
  EAGLE-3 and n-gram lookup. README numbers, M5 Max, Qwen3.6-27B 4.15 bpw:
  plain 16.6 / EAGLE-3 24.2 / MTP 28.3 / **DFlash 37.8** (4.80 accepted
  tokens/cycle); MoE Qwen3.6-35B-A3B 4.00 bpw: 68.5 plain, and speculation is
  *net-negative* there — "the host-coupled accept overhead makes speculation
  net-negative" because plain decode is already fast per token.

### 2.3 BaseRT (Base Compute, arXiv 2607.19438) — what M5's matrix units do and do not buy

This is the cleanest public statement of the M5 mechanism, and it directly
bounds what an M5 number can and cannot explain:

* M1→M4 expose `simdgroup_matrix`, which *"runs on the GPU's general-purpose
  arithmetic units: matrix multiplication competes with every other instruction
  for the same execution ports"* (§2.1). M5 adds a **Neural Accelerator in each
  GPU core**, reached only through the **Metal 4 tensor API** (`mpp::tensor_ops`
  `matmul2d`, compiled with `-std=metal4.0`); M5 still advertises
  `MTLGPUFamily.apple9`, so there is no capability flag to probe (§2.1-2.2).
* Their verdict on decode: *"Decode is memory-bandwidth-bound, and the M5's
  unified memory bandwidth, not its matmul throughput, is the binding
  constraint. A tensor core cannot accelerate a phase that is waiting on
  memory."* On M5 Pro, decode leads over mlx-lm are only 1.02-1.33× and MLX wins
  2 of 15 configurations; the tensor path is used for prefill GEMM, MoE expert
  GEMM and prefill attention only (§4.2, §3.2).
* And the forward-looking hint we should take seriously (§5.2): *"speculative
  decoding turns part of the decode workload into a batched verification step
  whose matmuls are compute-bound and therefore amenable to the tensor-core
  path."*

### 2.4 Others, briefly (mechanism named; full treatment already in-repo)

* **mlx-serve** (ddalcu, Zig) — batched decode with 8 rows/forward and **no KV
  padding**, native MTP plus PLD n-gram plus model-shipped drafters, and a
  runtime acceptance gate that disables speculation below break-even
  (`docs/research/mlx-serve.md` §3, cites `src/transformer.zig:26385-26450`,
  `src/glm_mtp.zig:1-9`, `src/pld_index.zig:1-10`). Its own `benchmarks.md` is
  M4 Max-only and contains **no GLM-5.3-Flash row at all** — only Qwen and
  Gemma4 — confirming `docs/research/mlx-serve.md:276`'s note that mlx-serve
  publishes no GLM-5.3-Flash decode number.
* **mlxfast / mlxfast-bonsai2** (Layr-Labs) — the packed-stream → int8
  threadgroup tiles → **`simdgroup_matrix`** escape, which needs **13-16 rows**;
  `docs/research/mlxfast-bonsai2.md` §2.3, §6 T-list. This is the same lever our
  `docs/perf.md:857-860` names as "the order-of-magnitude lever".
* **TensorFold** (ashxhart, Zig since 1.0) — a drafter-first engine whose whole
  pitch is *"a draft is accepted only when it equals the token the same engine
  would produce serially"*, i.e. losslessness by construction; on GLM-5.3-Flash
  it is qualified only on two M5 Ultras (91 t/s solo, 199 across 8 requests).
  Its Mac numbers for this model are M5-class or M3 Ultra — none on M4 Max.
* **ds4 / DwarfStar** — the engine behind both M4 Max GLM numbers (A1) and the
  31.7 t/s DeepSeek number. Its M4 Max GLM report is the most instructive single
  measurement in this whole file: **85% of the step is host-side per-layer
  expert-cache bookkeeping**, and removing a synchronous readback changed
  nothing (6.22/6.24) while MTP cost 15%.
* **llama.cpp on Metal** — not a speed source but a warning: GLM-5.3-Flash
  decode *stalled 1-9 s per token* because the fused Lightning Indexer Metal
  kernel only supports exactly 64 indexer heads and GLM-5.3-Flash has 32, so the
  fused op silently fell back to the CPU; ~31 t/s once the probe was fixed
  ([ggml-org/llama.cpp#29867](https://github.com/ggml-org/llama.cpp/issues/29867)).
  Our DSA/indexer path (`indexer_topk_batch`, `src/glm5_attn.cpp`) must stay on
  the GPU; the failure mode looks like "mysterious 1-9 s stalls", not like a
  wrong answer.
* **Kyojin** (Yamz-Labs, ExLlamaV3 + ROCm for gfx1151) — the reference for our
  format on a 128 GB unified-memory machine: turboderp's 2.05 bpw pack with
  layers 29-44 taken from the 3.05 bpw pack, `su`/`sv` scale vectors of layers
  21-44 tuned against FP8 on 526 rows, 99.73 GB, and 26.7 t/s plain / 31.6 with
  MTP-2 on a 256 GB/s part.

---

## 2.5 The crossover arithmetic: why our trellis GEMV is on the wrong side of it, and what moves it

This is the section that answers the actual question ("what mechanism explains a
faster number"), because at 2-3 bpw *every* engine in this file is doing the same
thing and the differences are arithmetic, not architectural.

| implementation | per-weight cost | matrix units on the decode path? | class |
|---|---|---|---|
| **exl3-server (ours, M4 Max)** | **13.75 issued lane-instr/weight** (measured: 110 warp-instr per 256-weight tile; 0.43 warp-instr/weight = the whole 256e9 warp-instr/s issue width) | no at rows≤13; staged `simdgroup_matrix` kernel from rows≥16 (seventh pass) | **issue-bound** |
| PonyExl3 (Metal, MLX) | ~10.5 (counted off emitted source) | no on the decode path; v19c prefill only | latency/occupancy-bound (23-30% issue) |
| exllamav3 (CUDA) | one word/lane/tile + 8 extracts + 8 decodes + **0.25 `mma.m16n8k16`/weight** | **yes at every m**, incl. m=1 via a 4-lane guard and 2-8 via MMODE 1 | memory-bound |
| Kyojin (ROCm, gfx1151) | **~4-6** (one lane per 16×16 tile, compile-time windows, fully unrolled, no shuffles/LDS, `v_dot2_f32_f16` = 2 MACs/instr) | WMMA on the GEMV path at all m (`EXL3_GEMV_ROCM_MAX_M=16`) | **stall/DRAM-bound** (55-65% of its 236 GB/s) |

The consequence, from our own instruction budget: with `K` bits/weight the issue
wall is `74.5·K` GB/s — K2 149, K3 224, K4 298, K6 447 GB/s — against a **546 GB/s
DRAM wall**. The two lines cross at **K ≈ 7.3**, i.e. above our entire working
range. So at 2.5 bpw the GEMV is **~2.9× on the instruction side of the
crossover**, and "more GB/s" can only mean *fewer issued instructions per weight*
or *more rows served per decode pass*. Everything else follows:

* **Kyojin is faster for exactly one reason:** ~4-6 lane-instr/weight instead of
  ~13.75 moves its crossover to where DRAM is the wall (its own profile: 36% VALU
  issue, 17% LDS, 144 GB/s, "stall-bound, not issue- or bandwidth-bound"). The
  ingredients are not exotic: one lane per tile so every bit window is a
  compile-time immediate (`exl3_dec.cu:88-105`), SGPR constants for the decode
  (`:112-140`), a 2-wide fp16 dot, and no shuffles/LDS staging in the inner loop.
* **Its second reason is row amortisation:** `pair_dot_r`
  (`exl3_dec.cu:288-295`) decodes each weight pair **once** and applies it to
  every row — they measured that re-decoding per row cost *~7% of the dense GEMV
  time per extra verify row* in pure ALU. `moe_dedup_gemv_kernel`
  (`exl3_gemv.cu:225`) does the same per expert run (≤8 rows), and the MTP verify
  batches are routed into the 16-row grouped MoE path via
  `EXL3_HIP_PREFILL_MIN_ROWS=2`.
* **Our own seventh pass closed this on M4 Max, today** (`docs/perf.md:879-974`):
  the staged fp16 + `simdgroup_matrix` kernel is a win on **every** tensor class
  at rows 16 and 32 — 2.4-4.8× per class, e.g. lm_head K6 8.855 → 2.441 ms (3.6×,
  194.9 GB/s), kda qkv_proj K4 1.380 → 0.409 ms (3.4×, 123.1 GB/s), mla o_proj
  0.937 → 0.268 ms (3.5×, 125.2 GB/s) — with rows 1-13 byte-identical by
  construction and a named, gated deviation above that (max |diff| 1.7e-5% to
  8.9e-5% of max|value|, f32 association scale). Note the doc's own framing: the
  staged path **does not decode faster** (same ~10 of ~13 lane-instr/weight); it
  removes the row multiplicity, so the win is the pass ratio.
* **The largest untouched lever is not arithmetic at all.** The small
  per-expert tensors run at **19-50 GB/s** against 289-305 GB/s for the big
  shapes (`docs/perf.md:570-634`, and the scout's independent K3 4096×2048
  single-expert measurement of 19.4 GB/s on the same box), because they are
  launched once per expert with far too few threadgroups to hide latency. Expert
  *grouping* (PonyExl3's `tile_map`/`tile_sub`, Kyojin's grouped/union/dedup,
  our `trellis_linear_group`) is the fix and needs no new math.
* **What is provably not the answer:** the codebook. The mul1 hash and its half
  add are only **23%** of our kernel (0.039 of 0.168 ms), and the no-decode
  ceiling is ≤27%; a device LUT was measured **1.5-3.4× slower** at rows=1 in our
  engine (`docs/perf.md:600-634`) and PonyExl3 measured the same
  (`gemv_metal.py:1016-1018`: "Computed 3INST decode beats the 128 KB LUT
  end-to-end on M5 Max"). The **load/addressing skeleton** is the bigger in-kernel
  term — our ablation puts it at 0.071 of 0.168 ms (**42%**) with no memory
  traffic in play at all.
* **And EXLR8-style FP8 weights are not a shortcut on M4:** they need an in-lane
  round-trip costing ~one more codebook per weight (~0.50-0.54 warp-instr/weight,
  `docs/research/exlr8.md:159-180`) unless real FP8 matrix units exist — M5/NA
  only.

---

## 3. Calibration verdict

### 3.1 Single stream: 8.66 tok/s

**Against the same hardware (M4 Max 128 GB): we are the fastest published run of
this model, full stop.** The only two M4 Max GLM-5.3-Flash numbers that exist are
6.21 t/s (4-bit, SSD-streamed) and 2.7 t/s (4-bit, expert offload), both of which
are memory-plumbing numbers, not kernel numbers. There is **no M4 Max 2-3 bit
report to compare against at all** — the closest is one unattributed "low-to-mid
tens" claim from the 2-bit pack's own author, on an offload-on machine, with no
benchmark.

**Against same-format siblings on other unified-memory machines: we are ~3x
below.** EXL3 2.05 bpw GLM-5.3-Flash single-stream: 26.7-31.6 t/s on Strix Halo
(256 GB/s), 25-50 t/s (prose/structured) on one DGX Spark (273 GB/s). We have
546 GB/s and do 8.66 — and that gap is **not** bandwidth, because at 2-3 bpw the
trellis GEMV is not bandwidth-bound at all (§2.5): with `K` bits/weight the issue
wall is `74.5·K` GB/s (K2 149, K3 224, K4 298) against a 546 GB/s DRAM wall, and
the lines only cross at K≈7.3. We sit ~2.9× on the instruction side of the
crossover; Kyojin and exllamav3 sit on the DRAM side because their per-weight
instruction count is 4-6 (Kyojin, one lane per tile with compile-time windows,
`v_dot2_f32_f16`) or 0.25 mma/weight (exllamav3), not 13.75. **So the mechanism
behind every materially faster EXL3 run in the wild is: ~2-3× fewer issued
instructions per weight, plus decode-once-per-R-rows for the verify rows.** Both
are things we can implement; neither is hardware.

**Against the same hardware on adjacent models: we are ~2-3x below a plausible
M4 Max ceiling.** On our exact chip, same 2-bit class: DeepSeek V4 Flash
285B/13B-active at ~2-bit does **31.7 t/s** (ds4 + DSpark) and a dense ternary
27B does **34.5 t/s** with llama.cpp, i.e. ~110-248 GB/s effective vs our 53. So
the chip can stream 2-4x more weight-bytes per second than our model-level
number, with non-trellis formats and with speculation.

**Against M5 Max (38-42 t/s at 2-bit): we are 4.4-4.8x below, but that number is
not ours to compare with.** M5 Max is different silicon: a Neural Accelerator in
every GPU core exposed through Metal 4 (BaseRT §2.1-2.2), ~614 GB/s vs our
546 GB/s, and Apple's own "over 4x the peak GPU compute for AI vs the previous
generation" ([Apple newsroom, M5 Pro/Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)).
Note the honest complication, and it is decisive: **the M5 generation is only
~1.2× faster at *decode* than the M4 in Apple's own measurement.** Apple ML
Research's M5-vs-M4 MacBook Pro comparison (24 GB, `mlx_lm.generate`, 4096-token
prompt / 128 generated) reports TTFT speedups of 3.33-4.06× but **generation
speedups of only 1.19-1.27×**, and attributes even that to bandwidth
(120 → 153 GB/s); BaseRT independently measures the M5 tensor units as conferring
essentially nothing on memory-bound decode (1.02-1.33× over mlx-lm, 0.98× on two
large models, and the M5 *Max* explicitly left unmeasured). M5 Max vs M4 Max
bandwidth is **614 vs 546 GB/s (+12%)** (Apple newsroom + hardware-corner's
spec table; their M5 Max LLM figures are projections, not measurements). **So
M5 Max's ~4.8× over us cannot be silicon.** The likeliest explanation is now
*verified rather than guessed* (§2.1): oMLX's M5 Max run uses **stock MLX
`quantized_matmul` for the 2-bit experts** — its fused GEMV declines 2-bit
entirely — plus fused per-stage kernels for the KDA/DSA/mHC/MLA/router machinery
(some of which are M5-only paths: oMLX's GLM PRs describe *"fused MoE gate/up
with NAX gather tiles"* and a *"tensor-unit indexer and sparse MLA"*, i.e. the
M5 Neural Accelerators under the community's "NAX" name), plus adaptive-depth
MTP, plus an affine format whose per-weight decode cost is a fraction of a
trellis codeword's. In other words the gap is (a) per-token machinery,
(b) **format instruction density**, and (c) M5-only tensor-unit paths — of which
only (a) and (b) are available to us. I would treat ~40 t/s at 2-bit as
*approximately the M5 Max answer*, and as *not evidence about M4 Max at all*.

### 3.2 Aggregate: 33.9 tok/s (B=16, same prompt), 31.5 (B=16, 4 prompts)

**No wild aggregate exists for this model on M4 Max**, and the published
aggregate numbers elsewhere are on much larger machines (M5 Ultra 141 t/s
single-stream at 4-bit; two M5 Ultras 199 t/s across 8 requests under
TensorFold; 182 t/s C4 on one DGX Spark; 1,045 t/s at c16 on 2× RTX PRO 6000). So
33.9 aggregate is **the fastest M4 Max number of any kind for this model that I
could find**, and the only aggregate. The structural reason it is the only one is
also the reason it is achievable at all: as `docs/perf.md:436-470` shows, the
shared (non-expert) portion of a step collapses from 6.1 GB/token to 3.4 GB/step
once rows share it, so batching is the only lever that exists on this hardware
short of speculation or a matrix-unit kernel.

### 3.3 The honest summary

* Single-stream 8.66: **at the top of the M4 Max field (field size: effectively
  zero), ~3x below the same-format cross-platform state of the art, ~2-3x below
  what our own chip demonstrably does for adjacent 2-bit MoEs.**
* Aggregate 33.9: **best known on this hardware for this model**; 45-48% of our
  own 70 tok/s target, with the two blockers already named in `perf.md`
  (per-weight ALU at the issue wall; per-row machinery ~23-25 ms/row).
* Anyone materially faster than us is on **different hardware** (M5 Max/Ultra,
  multi-GPU, multi-Mac) or a **different format+speculation stack** (affine int4,
  GGUF, CUDA/ROCm EXL3 with DFlash2/MTP). Nobody on M4 Max, at 2-3 bits, with
  this model, is faster.

---

## 4. Transferable to `exl3-server`, ranked

**Priority order, given the evidence in §2.5 and today's seventh pass:**
**T1** (decode once per R rows — measured, already built) → **T4** (grade the
small/expert tensor class) → **T2** (MTP in the serving path) → **T9** (cut
issued instructions per weight at rows=1) → **T3** (per-token machinery) →
**T6** → **T7** → **T8**. The two items that can plausibly close the gap to the
150+ GB/s class are T1 and T9; the rest are multipliers or enablers.

Bit-identity posture for every item: the **single-sequence path must stay
bit-identical**, and the **byte-granularity contract in `src/trellis_gemm.h:17-35,
51-64`** must not be broken (words are read as their four little-endian bytes in
that order; `aligned` may be false and the kernel must still assemble the word
from bytes). The existing gates are the batch-of-one-vs-serial logit gate, the
four-slots-vs-serial gates, and the per-tensor-class hash gate in
`tests/glm5_gemv_nt_test.cpp`. **Where an item cannot keep the byte-exact gate,
it must be gated on rows and report its deviation by name** — which is exactly
what the seventh pass already did for T1.

**T1. Decode once per R rows for every R≥8 — the staged `simdgroup_matrix` path.**
*Status: built and measured in this tree today* (`docs/perf.md:879-974`, "Seventh
pass: the staged matrix path for the batched rows"), not a proposal. *Why it is
the top item:* at 2.5 bpw the scalar QMV re-decodes the same packed words once per
row block (`mb = min(rows,4)`), so at 16 rows it pays the decode 4 times and at 26
rows 7 times, and the decode is ~10 of the ~13 issued lane-instructions per
weight (§2.5). The staged kernel decodes each 16×16 tile once into a threadgroup
fp16 tile and hands the MACs to the matrix unit. *Measured effect, per class,
rows 16 / 32:* lm_head K6 8.855 → 2.441 ms (**3.6×**, 53.7 → 194.9 GB/s), kda
qkv_proj K4 1.380 → 0.409 ms (**3.4×**), mla o_proj 0.937 → 0.268 (**3.5×**),
kda o_proj 0.479 → 0.141 (**3.4×**), dense gate/up 0.690 → 0.206 (**3.4×**),
shared experts 0.132 → 0.051 (**2.6×**), exp gate/up 0.131 → 0.048 (**2.7×**);
every class is a win at both row counts and the ratios grow with rows (up to
**4.8×** at 32).
*Bit-identity posture:* **the contract is kept where it matters and the deviation
is named where it cannot be.** `TrellisTuning.mat_min_rows` defaults to 16, so
rows 1-13 (including the entire single-sequence path, and the byte-exact table's
rows {1,2,3,13}) are unchanged *by construction* and were measured within 1% of
each other; `trellis_path_info()` (`src/trellis_gemm.h`) prints which path, row
block, padded rows and pass count a tensor takes. Above that, the matrix k-tree
cannot reproduce the QMV's reduction order
(`quad_sum(a0+a1) + quad_sum(a2+a3)` per eight stream positions + 4-lane
interleave), so the deviation is bounded and reported: **max |diff| 1.7e-5% to
8.9e-5% of max|value|** (2e-7 to 9e-7 relative) on the raw f32 output, 8 shapes ×
rows {16,32,64}, and the token-identity gates (single-slot == serial, 4
identical slots, 4 different prompts vs 4 serial runs, greedy text) all hold.
*Remaining work on T1:* bring the grouped/scale-stack forms onto it at B=16 and
re-run the end-to-end N-sweep, then re-measure the 33.9 tok/s aggregate. *Also
fold in here:* the QMV path is still 3 launches per linear (input Hadamard, GEMV,
output Hadamard); PonyExl3 `gemv_metal.py:77-107` and Kyojin
(`exl3_dec.cuh:5-11`, "the input Hadamard runs in the block prologue, the output
Hadamard in the epilogue … one launch covers what the generic path does in
three") both fuse them in-kernel — worth doing for the rows≤13 path, where it is
the only launch-count lever left, but lower priority than the row amortisation.

**T2. MTP speculation in the serving path.** `Engine::submit_batch` currently
*bails to serial when the request asks for the MTP drafter* (`docs/perf.md:381-384`),
so the drafter and the batch are mutually exclusive. *Why it matters:* it is the
only lever the wild consistently uses at 2 bits — M5 Max oQ2e 38.7 → 42.0 (+9%),
Strix Halo EXL3 26.7 → 31.6 (+18%), PonyExl3 MTP 1.70× on a dense target,
TensorFold dflash +20-50% on M3 Ultra GLM-5.3-Flash, oMLX Qwen MTP 2.22×, and our
own earlier measurement of the MTP drafter as **+30-50%** while the step is
launch-bound (`docs/perf.md:59-66`). *Effect:* expect +15-30% single-stream at
our current launch-bound step, less as T1 removes launches. *Bit-identity:* the
speculative path is already gated token-identical to plain greedy
(`docs/dspark.md`, "Verified lossless"); the contract is per-token identity, not
per-logit identity, so the single-sequence non-speculative path is untouched.
*Cost:* low-medium — the MTP head exists in the checkpoint, `glm5_mtp.cpp` and
`--draft-mtp N` exist. *Risk:* MTP acceptance on the *abliterated* 2.51 bpw pack
is unmeasured; measure acceptance before believing the gain. *Design note from
oMLX (§2.1):* its verify path needs **exact rollback** because the KDA
`ArraysCache` is untrimmable — it records the verify forward's inputs and entry
state and replays the accepted prefix through the same fused kernel. We have the
same structural problem (142 MB of KDA state per slot, `docs/perf.md:347`), so
budget for a replay-based trim rather than a memory trim.

**T3. Cut the per-token machinery: compile the hyper-connection/norm glue and
fuse the MoE router/combine.** *Why:* the wild's numbers exist because of this;
oMLX's own kernel file states it — *"Single-token decode and short verify blocks
(L <= 8) are dominated by thousands of tiny dependent dispatches, which cost both
GPU time and host encode time"* (`omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py:1-23`,
verified in this pass) — and it acts on it with one
dispatch per architectural stage (`moe_router :2135`, `moe_gate_up_swiglu :795`,
`moe_down_combine :933`, `kda_decode_step :1843`, the four DSA kernels
`:1227/:1383/:1443/:1506`, `hc_mix :1106`, `mla_head_qmv :2835`, and `multi_qmv
:2960` which fuses up to 4 same-format projections of one input). ds4 on M4 Max
spends 85% of its step on per-layer host bookkeeping (1a A1). Our own
`docs/perf.md:44-57` lists `mx::compile` and the MoE host loop (**~8 ms/token**
just in the MoE loop, `docs/perf.md:64-66`) as levers 1-2. *Effect:* the
difference between 90 and 40 ms/step is exactly this class of work. *Bit-identity:*
`mx::compile` preserves the traced elementwise order, but be careful with
reductions (the batch gates already showed MLX's fp16 reduction order moves
logits by ~0.5-2% at B>1 while tokens stay identical). *Cost:* medium; do it
site by site with the hash gate after each.

**T4. Grade every tensor class to ≥~2e9 tiles/s: the small/expert launch class is
the largest untouched lever.** *Why:* the small per-expert tensors run at
**19-50 GB/s** while the big shapes run at 289-305 GB/s (`docs/perf.md:570-634`;
independently reproduced this session: K3 4096×2048 single-expert shape =
19.4 GB/s, 2.1 MB and 1.0 MB tensors = 22.0 and 12.9 GB/s), because they are
launched once per expert with far too few threadgroups to hide latency. Those
classes are a large share of the per-token bytes — the routed experts are 2.7 GB
of the 6.1 GB step (`docs/perf.md:436-470`) — and the seventh pass shows the
staged path already lifts them (exp gate/up 2.7×, shared 2.6-2.7×), so part of
this is a T1 deployment question and part is launch geometry: the staged path
picks its own k-split aimed at ~4096 threadgroups precisely because the QMV's
split is tuned for `n_tiles * 4` warps and left the narrow-output classes at
25-35 GB/s. *The wild analogue, and the mechanism to copy:* device-side expert
indirection into one stacked trellis — PonyExl3's `tile_map`/`tile_sub` so a
single mapped launch covers all selected experts per projection group
(`exl3_moe.py:1-16`, `:139-172`, `:640-682`), Kyojin's grouped/union/dedup MoE
kernels (`exl3_gemv.cu:175`, `:225` `moe_dedup_gemv_kernel` "decodes that
expert's trellis ONCE and applies it to all of the run's rows", `:502` prefill at
16-row chunks), mlx-serve's expert-order pair execution. *Effect:* the class
moves toward the 120-195 GB/s the staged big shapes already reach; this is where
the remaining model-level factor lives (we are at 53 GB/s effective against a
~187 GB/s wall at 2.51 bpw). *Bit-identity:* the per-member k-order and the
routing-weighted combine order must be unchanged; an indirect tile map changes
which buffer a member's words come from, not the arithmetic. *Cost:* medium —
the stacked layout is built once at load, and the byte-granularity contract must
be kept through the indirect indexing (`src/trellis_gemm.h:17-35`). *Related
existing gap:* expert dedup by (row, expert) is the B>1 half of this
(`trellis_linear_group_rows(rows=R)`, `src/trellis_gemm.h:91`, still ungated
per-row per `docs/perf.md:866-870`; the shared-routing regime is worth 19.3 →
29.7 at B=8). *Note:* `moe_layer_rows` currently splits into K grouped launches
because one Metal kernel cannot take 30 members + x + out
(`docs/perf.md:355-360`) — pure expert grouping is what removes that, and the
MoE host loop is ~8 ms/token.

**T5. (absorbed by T1) The matrix path at rows ≥ 13-16 — with its remaining open
question.** The order-of-magnitude escape named by `docs/perf.md:857-860` and
`docs/research/mlxfast-bonsai2.md` §2.3 is no longer a plan: it is the seventh
pass, and it is T1. What remains genuinely open, and matters for how far T1 can
go:

* **Does `simdgroup_matrix` on M4 (Apple9) execute the MACs off the issue pipe?**
  The two in-repo readings disagree (`docs/perf.md:940-943` vs this file's §4b
  bullet), and the seventh pass's own accounting is consistent with the MACs
  being nearly free (measured 3.4× at 16 rows against a 4-pass QMV baseline). If
  the MACs *do* leave the issue pipe, then a rows=1 matrix path would
  theoretically break the 187 GB/s K2.51 wall itself; if they do not, rows≥16 is
  a hard floor. One microbenchmark settles it.
* **The wild agrees on the floor, not on the ceiling.** No implementation found
  runs a matrix unit at rows=1 for *decode*: we gate at 16, PonyExl3's v19c at
  M>8/64-row blocks, exllamav3's CUDA `mma` at m=1 only behind a 4-lane guard
  (12 of 16 rows wasted), Kyojin's WMMA GEMV at m=1 exists but its own batch-1
  dense kernel deliberately uses VALU/`v_dot2_f32_f16` instead
  (`exl3_dec.cuh:5-11`).
* **What the wild does instead at low rows:** fewer instructions per weight
  (T9), and `k`-splits/occupancy tuning (T4). That is the honest answer for the
  single-sequence path.
* *Bit-identity:* as in T1 — rows≥`mat_min_rows` only, deviation named and
  bounded, single-sequence path untouched by construction.

**T6. (implementation of T4) One mapped GEMV over a stacked expert trellis,
instead of K grouped launches.** The mechanism behind T4's grading fix: stack the
selected experts' trellises along out_tiles and index them with a device-side
`tile_map`/`tile_sub` (PonyExl3 `exl3_moe.py:1-16`, `:139-172`), or dedup and
union expert runs (`exl3_gemv.cu:175,225`), so one launch (and one x load) covers
all selected experts per projection group. Constraints to keep: the per-member
k-order, the routing-weighted combine order, and the byte-granularity contract
under indirect indexing (`src/trellis_gemm.h:17-35`); and the stacked layout must
be built once at load, not per token.

**T7. Sibling fusion — merge linears that share an input row, but respect the
crossover.** PonyExl3 makes siblings views onto one `FusedEXL3Group` that computes
all outputs in one prepare + one GEMV + one finish, *"Per decoder layer this
collapses ~9 dispatch chains into ~3"* (`exl3_fused.py:1-15`); we already merge
mlp gate|up and the KDA input projections. **Direction correction (this pass):**
their measured result is that fusion pays off when the kernels are *big* and
**regresses decode ~2×** on small/small-tensor models (`model.py:380-382`,
README:45-46) — it trades the latency overlap of independent sibling GEMVs for
throughput. So for us the candidates are the large siblings only (kda qkv, mla
q_b/o_proj per layer), and the fused-MoE classes are already covered by T1/T4.
*Bit-identity:* keep each member's GEMV order; it is a launch-count change.

**T8. Operational: raise the wired limit, warm the file cache, avoid `mx::eval`
in the loop.** Not glamorous, but load-bearing: the OrcaRouter playbook
(128 GB M4/M5 Max) documents that macOS's default Metal working-set ceiling is
**~91-96 GB** — *below* their 102 GB build and below our 121 GB working set — and
requires `sudo sysctl iogpu.wired_limit_mb=114688` (112 GB). Our
`docs/perf.md:8-13` already notes a refill from the external SSD runs at
~2.6 GB/s (400 ms/GB). And `mx::eval` at 0.11-0.2 ms per call is fatal inside a
decode loop (`docs/perf.md:64-66`, `:86-88`). *Bit-identity:* none affected.

**T9. Cut issued instructions per weight at rows=1 — the only route that lifts the
single-stream wall itself.** The wall is `74.5·K` GB/s (§2.5); at 2.51 bpw that is
~187 GB/s, which caps a *perfect* single-stream kernel at ~1.87e9 tiles/s and
~26 tok/s on our byte model, against the 70 tok/s target. Beyond T1 (rows) and T4
(class grading), the remaining move is to make each weight cost fewer instructions,
and there is now a worked example to copy from: **Kyojin's batch-1 kernel**
(`exllamav3_ext/quant/exl3_dec.cu:1-17`) — *"one lane owns one 16×16 trellis tile,
the bit windows sit at compile-time positions (fully unrolled decode, no LDS
staging, no shuffles), fp32 accumulation through `v_dot2_f32_f16`"* — at
**~4-6 lane-instructions per weight** versus our 13.75, with `pair_dot_r`
(`:288-295`) decoding each pair once for RM rows and a one-instruction window
extract (`window<>` = `v_ubfe` or `v_alignbit`, `:88-105`). The three ingredients:
(i) **one lane per tile with compile-time window positions**, which deletes our
per-group `2 loads + 64-bit funnel shift + mask` addressing (PonyExl3's `dq4`
already halved that cost, `gemv_metal.py:206-212`); (ii) **immediate/SGPR
constants** for the codebook and masks; (iii) **a 2-wide dot** — Kyojin's
`v_dot2_f32_f16` gets 2 MACs/instruction, and the open question is whether Metal
exposes an equivalent (or a `half2` FMA on packed pairs) before we budget for it.
*What is provably *not* available:* the codebook itself is only 23% of our kernel
and the no-decode ceiling is ≤27% (`docs/perf.md:600-634`), a device LUT was
measured 1.5-3.4× **slower** at rows=1, and PonyExl3 measured the same
(`gemv_metal.py:1016-1018`) — so do not spend the effort there. The bigger
in-kernel term is the **load/addressing skeleton: 0.071 of 0.168 ms = 42%** with
no memory traffic in play at all. *Effect if fully realised:* ~1.5-2× at rows=1,
i.e. the 187 GB/s wall moves toward 300+ GB/s. *Bit-identity:* the decode must
stay the same three-instruction codebook applied in the same order, and the
per-output reduction order must be reproduced (PonyExl3 v12 has *zero* hot-loop
barriers but keeps 8 register accumulators per lane for exactly this reason);
re-laying out which lane owns which tile changes who computes a weight, not the
sum order, but it is a new kernel — re-run the 25-class × rows hash table before
believing it. *Cost:* high. *Also note:* the two M4-specific rules from the
comparison — our QMV is 2.5-3.5× faster per tile than PonyExl3 v12 on the same
box/shape/format, so **copy Kyojin's structure, not PonyExl3's inner loop**; and
PonyExl3 v12 leaves 70% of the issue width idle at rows=1 while spending *fewer*
instructions per weight, which suggests a software-pipelining gap (no prefetch
ring) worth understanding before assuming our occupancy is optimal.

### 4b. Does NOT transfer — with reasons

* **Every M5 Max / M5 Ultra number in §1b-1c**, as a target for M4 Max. M5 has
  per-core Neural Accelerators reached only through the Metal 4 `mpp::tensor_ops`
  API (`-std=metal4.0`, no capability flag: BaseRT §2.1-2.2), ~614 GB/s vs 546,
  and Apple's >4× AI-compute claim. Report M5 numbers as M5 numbers, with the
  hardware named, or not at all.
* **`simdgroup_matrix` decode at rows=1.** BaseRT's own measurement says the
  tensor path does nothing for memory-bound decode and that MLX wins 2 of 15
  decode cases on M5 Pro; our `perf.md` says the matrix escape needs 13-16 rows.
  A "just use the matrix units" single-stream story is wrong on both counts.
* **M5 Ultra's 141 t/s / two-M5-Ultra 199 t/s aggregates.** Bigger chip, more
  memory, 4-bit (a *larger* quant), multi-node in the 199 case.
* **oMLX's ANE prefill knobs** (`qwen35_ane_prefill_*`): Qwen3.5-lineage only,
  prefill only. Our bottleneck is decode.
* **TurboQuant / KV quantization.** Our step reads 6.1 GB of weights; KV is not
  the binding constraint at short context (`perf.md:436-470`). No single-stream
  effect, and it costs accuracy.
* **MoE expert offload / SSD streaming / "expert cache"** (oMLX 2.7 t/s, ds4
  6.2 t/s, apus 0.085 t/s). Every mechanism in that family exists to make models
  *that do not fit* run at all. We fit. Their published numbers are a warning
  about host bookkeeping, not a speed lever.
* **REAP/REAP-pruned, OrcaSAQ 2bit-lite, oQ, Unsloth UD-IQ1_S, GGUF Q2_K/IQ3_XXS,
  NVFP4/FP8 packs.** Different quantizer or different bit-width → different
  weights → our bit-identity gate over the fixed 2.51 bpw h6 checkpoint cannot
  survive it. (Also: OrcaRouter's own 2bit-lite, the only 2-bit build that fits a
  128 GB Mac in their ladder, is +141% perplexity and 77.19% top-1 agreement —
  not a quality-equivalent substitute.)
* **`mx.quantized_matmul`-style affine int4 kernels.** Their low per-weight
  instruction count is why the same M4 Max streams affine-4bit MoEs at 2-3× our
  effective rate (§1e) — but that is a property of the *format*, and we cannot
  adopt it without re-quantizing away from EXL3.
* **DFlash2/EAGLE-3/DSpark drafters as drop-ins.** A GLM-5.3-Flash DFlash2
  drafter does exist (`incoai/GLM-5.3-Flash-DFlash2`, and an EXL3 3.00 bpw
  variant at `r0b0tlab/GLM-5.3-Flash-DFlash2-EXL3-3.00bpw`), so the *approach*
  transfers — but the draft was trained against the base model, not the
  abliterated 2.51 bpw pack, so expected acceptance on our weights is unmeasured;
  and it is a second model to load and ship. Ranked after native MTP (T2).
* **oMLX's fused GEMV kernels as a kernel to borrow.** They are verbatim copies
  of MLX's `qmv_fast`/`qmv_wide` (`decode_kernels.py:195,284`, header comment
  `:22-30`) and are affine-only, bits {3,4,5,6,8} (`_qmv_fast_ok :774`,
  `_affine_parts :785`). There is nothing in them that a trellis GEMV can take,
  and their 2-bit path is a *fallback* to stock MLX, not an optimisation. What
  transfers from oMLX is the **fusion of the non-GEMV stages** (T3) and its MTP
  controller (T2), not its arithmetic.
* **Expert offload / resident-fraction tuning as a speed lever.** oMLX's own M5
  Max numbers price it: 42.0 t/s at full residency → 22.0 at 0.85 → 9.6 at 0.25
  (1.9× and 4.4× slowdowns), mirrored by ds4's 6.2 t/s on M4 Max where GPU time
  is 15% of the step. It is a capacity mechanism. We fit; do not adopt it to
  "free memory" for KV.
* **Continuous batching above B=8 as published evidence.** oMLX's harness only
  ever measures `[2,4,8]`; our B=16 numbers have no wild counterpart and should
  not be framed as "beat someone".
* **Tensor parallelism across Macs** (TensorFold's 199 t/s across two M5 Ultras,
  the M5 Max + Strix Halo llama.cpp pair at 24 t/s, the 2× M5 Ultra GLM run).
  Multi-node plumbing for a single M4 Max; out of scope.
* **`--fast` CPU-offload "all-RAM" modes** on 24 GB cards (RTX 3090 28.6 t/s at
  3.05 bpw) — that path is a 24 GB-card workaround that decodes one stream at a
  time.
* **Anything derived from a REAP-pruned or different-expert-count model.** Ours
  is the full 288 experts, top-8, 45 layers.

---

## 5. What I could not verify, and where the community's model differs from ours

**Could not verify / open:**

1. **oMLX's `file:line` cites are transcribed, not hand-checked by me.** §2.1's
   citations come from a delegated source dig against commit `013bc3f` (repo
   cloned to `/tmp/omlx`); I did not open each of those lines myself. The
   load-bearing claims — no EXL3 loader anywhere in the tree, `oQ` is MLX
   `affine` with `mode` always `affine`, and the fused GEMV declining 2-bit — are
   the ones worth a second pair of eyes before anyone acts on them.
2. **oMLX's community numbers are one-shot and its own batching axis stops at
   B=8.** Its harness measures single-request prefill/generation at 1024…200000
   prompt tokens and continuous batching at `[2,4,8]`
   (`omlx/admin/benchmark.py:41,44`), and it deliberately busts prompt cache with
   a UUID prefix (`:543-561`). So "42.0 TG" is a single-request number with
   prompt-cache defeated, and it is *not* comparable cell-for-cell to our
   `--batch 16` aggregate; there is no published oMLX aggregate above B=8 for
   this model. Relatedly, B4's 4-bit rows (4.2-5.8 t/s) publish peaks of
   ~90-108 GB for a ~150 GB model, i.e. they are offloading regimes, not decode
   inefficiency.
3. **Two genuinely open kernel questions** (from the §2.5 source comparison, both
   answerable with one microbenchmark each): (a) does `simdgroup_matrix` on M4
   (Apple9) execute the MACs off the issue pipe, or on the ALUs? Our in-repo
   readings disagree, and the answer decides whether a rows=1 matrix path could
   ever lift the 187 GB/s wall; (b) does Metal expose a 2-wide fp16 dot
   equivalent to Kyojin's `v_dot2_f32_f16`, or can a `half2` FMA on packed pairs
   do the same job? Kyojin's per-weight advantage is partly that one instruction
   (2 MACs/instr).
4. **The seventh pass is a sibling's same-day work in flight.** `docs/perf.md:879-974`
   (the staged matrix path measurements, the `trellis_path_info` reporting, and
   the named deviation) was written *while this report was being written*, against
   a modified working tree (`src/trellis_gemm.cpp`, `src/trellis_gemm.h`,
   `tests/glm5_gemv_nt_test.cpp` were all dirty). I have read the document and
   cite it, but **I did not re-run those measurements** — they are the strongest
   in-repo evidence in §4 and the first thing to re-verify if the numbers matter.
5. **B5 (X, 35.4 t/s on M5 Max).** X blocks automated reads; only the search
   snippet is available. Its quoted "127 tok/s prompt processing" is not credible
   for a 320B model, which is itself a reason to discount the whole post.
6. **llmcheck.net's "27 tok/s".** Explicitly an estimate, and the page contradicts
   itself (header says IQ3_XXS, methodology says Q4_K_M). Not evidence. Same for
   vramcalculator's TensorFold summary table, which is a third-party digest.
7. **The two M4 Max GLM rows' relationship to our regime.** A1 is 4-bit and
   SSD-streamed; A2 is 4-bit with offload. Neither is weight-matched to our
   2.51 bpw in-memory configuration, so neither can be subtracted from our number.
8. **Whether our 8.66 tok/s single-stream included the MTP drafter.** `perf.md`'s
   numbers in the batching pass are from `--batch`/HTTP runs and the batch path
   refuses the MTP drafter; the "+30-50% from the MTP drafter" figure is from an
   earlier pass. If our 8.66 is *drafter-off* while the M5 Max 42.0 is
   *drafter-on* (B1), part of that 4.8× is speculation, not silicon. This is the
   first thing to pin down — re-measure with `--draft-mtp N` and report both.
9. **MTP acceptance on the abliterated 2.51 bpw pack.** Unmeasured anywhere.
   The Strix Halo numbers (26.7 → 31.6, +18%) and oMLX's (+9%) bracket what to
   expect, but on our weights it could be lower.
10. **No EXL3-on-Apple-Silicon GLM-5.3-Flash number exists — confirmed by
    exhaustive check, not by search failure.** PonyExl3's supported `model_type`s
    are `qwen3_5`, `qwen3_5_moe`, `llama`, `gemma4/gemma4_text`; `glm5_next` is
    absent and unmapped checkpoints raise "no architecture mapping". Its issue
    tracker has two items (an M3 Max perf report and an oracle-config request),
    neither about GLM. TensorFold's EXL3 reader is CUDA-only (4-bit `mcg` routed
    experts), it has no GLM native-CUDA EXL3 backend, and its own docs say no
    GLM-5.3-Flash EXL3 checkpoint existed when it built the mixed-expert EXL3
    kernel; its qualified Apple path for this model is MLX affine 4-bit on two M5
    Ultras. MikeRoz's `oQ3.5e`/`2.51bpw` cards carry no Apple speed figures. **So
    our engine currently holds the only EXL3-on-Apple-Silicon decode path for a
    glm5_next model that anyone has published.**
11. **M4 Max vs M5 Max for this exact model is not a clean A/B anywhere**, and
    the best available bound says the delta should be small: Apple's own M5-vs-M4
    MacBook Pro measurement (MLX `generate`, 4096 prompt/128 gen) reports
    **1.19-1.27× generation speedup** — not 4.8× — with the gain attributed to
    120 → 153 GB/s of bandwidth, and M5 Max vs M4 Max is only 614 vs 546 GB/s.
    Treat any claim that M5 Max's GLM-5.3-Flash number is "what M4 could do with
    better software" as unsupported in either direction: the mechanism
    (machinery + format + M5-only tensor-unit paths) and the silicon cannot be
    separated from outside.

**Where the community's model differs from ours:**

* **Quantizer and bpw.** Ours is EXL3 2.51 bpw, h6 codebook, from the
  *abliterated* FP8, 98.123 GiB. Wild packs: turboderp 2.05 / 3.05 / 4.05 bpw
  (plain EXL3), Yamz's 2.05+3.05 per-layer mix with tuned `su`/`sv`, oMLX `oQ2e`
  (2-bit experts, 8-bit attention/shared/head, 4-bit MTP experts, **target 2.8
  bpw, hard cap 3.0**, MLX affine + imatrix), OrcaSAQ `2bit-lite`, Unsloth
  `UD-IQ1_S` (1.56 bpw), NVFP4/FP8. Our file is *smaller* than the 3-bit packs
  and *larger* than the 2.05 bpw one.
* **No REAP / expert pruning anywhere in our comparison set.** Ours is the full
  288-expert top-8 model; the 96-139 GB "fits a 128 GB Mac" MLX builds in the
  wild are REAP25/37/50 prunes and cost +18-70% perplexity
  (`PipeNetwork/glm53-flash-mlx`). Any tok/s from a pruned build is not
  comparable to ours at all, and none of the reports in §1 are pruned.
* **Same architecture everywhere, different heads shipped.** All the GLM-5.3-Flash
  quantizers ship, drop or re-quantize the MTP head differently (MikeRoz's pack
  keeps it; PipeNetwork's mlx build *drops* the MTP layer; oMLX's `oQ2e` keeps
  and calibrates it; the EXL3 reference adds an unquantized `mtp_eh_proj.st`).
  A drafter-enabled run and a drafter-dropped run of "the same model" are not the
  same workload.
* **Format: affine vs trellis.** Ours is EXL3 trellis. Every "2-bit MLX" pack in
  §1b/1c is **MLX affine, group size 64** (`oQ`: `omlx/oq.py:4615,5667`,
  `group_size 64` at `:43`), and every GGUF is k-quant/i-quant block format.
  Affine 2-4 bit costs a fraction of the per-weight instructions a trellis
  codeword costs (one packed word holds 16 weights at 2-bit and unpacks with
  masks that MLX's `qmv_fast` — and oMLX's verbatim copy of it — already
  optimise), and that is a first-order reason the same chip streams affine MoEs
  at 2-3× our effective rate. It is also the single biggest structural reason our
  numbers and the community's are not apples-to-apples: **there is no wild
  GLM-5.3-Flash 2-3 bit report on Apple Silicon that uses our format.**
* **Prefill is where most of the community's headline speedups are.** oMLX's
  M5 Ultra row is +2.9× prefill (792 → 2,270 t/s) for a +38% decode; TensorFold's
  whole qualified list is about prefill-bound phases. Our prefill is not the
  topic here, but do not let a prefill ratio leak into a decode comparison.
* **Context.** The M5 Max 2-bit rows are at 1k-4k context with ~109-110 GB peak
  resident. Our 8.66 is at short context too, but our KDA state is 142 MB per
  slot (`perf.md:347`), so our long-context aggregate will behave differently
  from these runs as context grows.

---

### Appendix: source list

Primary: omlx.ai/benchmarks/performance (+ per-row detail pages), llamaperf.com
(model, M4 Max, M5 Max, M5 Ultra pages; reports 1864, 1880, 1813, 1655, 1653,
1648, 1744, 1874, 1875, 1866, 1367, 1368, 1357, 1535, 1439, 1028, 756, 788, 779,
1369), HuggingFace model cards (MikeRoz 2.51bpw/3.05bpw/4.05bpw, Cropduster69
oQ2e-mtp, turboderp/GLM-5.3-Flash-exl3, yamz-labs/GLM-5.3-Flash-EXL3-Yamz,
PipeNetwork/glm53-flash-mlx + its measurements table), github.com/beamivalice/PonyExl3
(cloned, read at /tmp/ponyexl3), arXiv 2607.19438 (BaseRT / M5 Neural
Accelerators), apple.com newsroom (M5 Pro/Max), ggml-org/llama.cpp#29867,
forums.developer.nvidia.com 382140, OrcaRouter "2bit-Lite MLX Playbook",
vramcalculator.com/tensorfold, unsloth.ai/docs/models/glm-5.3-flash.
In-repo: docs/perf.md (including the seventh pass, `:879-974`), docs/dspark.md,
docs/research/{omlx,mlx-serve,mlxfast,mlxfast-bonsai2,exlr8}.md,
src/trellis_gemm.{h,cpp}, src/glm5_forward.{h,cpp}, src/glm5_moe.{h,cpp}.
Cloned for this report (read-only, `/tmp`): beamivalice/PonyExl3 @8e7fa6b,
turboderp-org/exllamav3 @151539c, Yamz-Labs/kyojin @feab558, jundot/omlx
@013bc3f.
