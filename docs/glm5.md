# GLM-5-Next / GLM-5.3-Flash support plan

Target: `MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3` (EXL3 2.51 bpw average,
head 6 bits), a GLM-5-Next ("Flash" = smaller MoE member) text model.

## Architecture (from the checkpoint config.json)

| Property | Value |
|---|---|
| model_type / arch | `glm5_next_text` / `Glm5NextForConditionalGeneration` |
| Layers | 45: **34 `linear_attention` (KDA)** + **11 `deepseek_sparse_attention` (DSA)** |
| MLP | 3 dense (`intermediate_size` 12288) + 42 MoE |
| MoE | 288 routed experts, top-8, `moe_intermediate_size` 2048, 1 shared expert, sigmoid scoring, `noaux_tc`, `n_group` 1/`topk_group` 1, `routed_scaling_factor` 2.5, `norm_topk_prob`, `swiglu_limit` 10 |
| MLA (sparse layers) | `kv_lora_rank` 512, `q_lora_rank` 1536, `qk_head_dim` = `qk_nope_head_dim` 256, **`qk_rope_head_dim` 0 (`mla_use_nope`)** - no RoPE in attention, `v_head_dim` 256, 64 heads |
| DSA indexer | `index_n_heads` 32, `index_head_dim` 128, `index_topk` 2048, `index_kpool` 4 + `index_kpool_compress`, `index_kpool_always_select_tail`, `indexer_rope_interleave`, `index_share_for_mtp_iteration` |
| Hyper-connections | `mhc: true`, `hc_mult` 4, `hc_eps` 1e-6, `hc_sinkhorn_iters` 20 |
| MTP | `num_nextn_predict_layers` 1 (embedded draft block - speculation for free) |
| Other | hidden 4096, heads 64/64, `first_k_dense_replace` 3, rms_eps 1e-5, max_pos 1048576, vocab 154880, eos `[154820, 154827, 154829]`, tied embeddings false |
| Vision | `vision_config` present (VL) - **out of scope**, text only |

## References to port from

- **ds4 (antirez/DwarfStar)**: supports GLM 5.3 Flash in C + Metal.
  `docs/MODELS.md` ("recurrent KDA layers, sparse DSA attention,
  hyper-connections, and a built-in MTP block"), `docs/SPECULATIVE_DECODING.md`,
  GPU entry points `ds4_gpu_glm53_kda_decode` / `ds4_gpu_glm53_kda_prefill`
  (`ds4_gpu.h`) and `tests/test_glm53_kda.c`. This is the closest structural
  reference; its kernels are Metal, matching our backend.
- **MLX primitives already available**: `fast::gated_delta_update` (linear
  attention recurrence), `fast::scaled_dot_product_attention` (per-layer
  gathering of the selected keys will be the new part), `fast::rope` (not needed
  for MLA here).
- **HF `transformers`** `Glm5NextForConditionalGeneration` (the checkpoint was
  produced with transformers 5.x): the authoritative numerical reference for
  KDA, hyper-connections and the indexer. Check it before implementing
  hyper-connections or the sparse attention top-k selection.

## Authoritative reference (fetched, class map)

HF `transformers` GLM-5-Next implementation (Apache-2.0), used as the numerical
reference for every new mechanism - fetch from
`https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/glm5_next/`
(`modeling_glm5_next.py`, ~2200 lines; `configuration_glm5_next.py`):

| Component | Reference symbol (line in `modeling_glm5_next.py`) |
|---|---|
| KDA equations | `recurrent_kimi_delta_attention` (463), `chunk_kimi_delta_attention` (517) |
| KDA layer wiring | `Glm5NextTextLinearAttention` (620); `causal_conv1d_fn` (429) / `causal_conv1d_update` (409); `Glm5NextTextForgetGate` (339); `Glm5NextTextRMSNormGated` (374); `l2norm` (451) |
| MLA attention | `Glm5NextTextAttention` (1100), `eager_attention_forward` (1075), `repeat_kv` (1063) |
| DSA indexer | `Glm5NextTextIndexer` (772) |
| MoE | `Glm5NextTextMoE` (185), `Glm5NextTextTopkRouter` (144), `Glm5NextTextExperts` (107), `load_balancing_loss_func` (2063) |
| Hyper-connections | `Glm5NextTextHyperConnection` (218), `Glm5NextTextHyperHead` (332) |
| Layer / model wiring | `Glm5NextTextDecoderLayer` (1278), `Glm5NextTextModel` (1425), `Glm5NextForConditionalGeneration` (2133) |
| Norms | `Glm5NextTextRMSNorm` (65), `Glm5NextTextUnweightedRMSNorm` (209) |

## Tensor map (from the reference's `__init__` methods)

Per-layer parameter names to load (prefix `model.language_model.layers.{i}.`):

| Layer kind | Tensors |
|---|---|
| KDA (`linear_attn.*`) | `q_proj`, `k_proj`, `v_proj`, `conv1d` (kernel 4), `b_proj`, `g_a_proj`, `g_b_proj`, `o_norm`, `o_proj`, and `forget_gate.*` = `f_a_proj`, `f_b_proj`, `dt_bias`, `A_log` |
| MLA (`self_attn.*`) | `q_a_proj`, `q_a_layernorm`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`, `o_proj` (DeepSeek MLA shape, nope-only) |
| Indexer (`self_attn.indexer.*`) | `wq_b`, `wk`, `k_norm`, `weights_proj`, `index_kpool_compress_ape`, `index_kpool_compress_gate` |
| Hyper-connection | `input_norm`, `fn`, `base`, `scale` (the sinkhorn iterations are computed from these, not stored) |
| MoE | `experts.gate_up_proj` (fused), `experts.down_proj`, `gate.weight`, `gate.e_score_correction_bias`, `shared_experts.*` |
| Gated norm | `RMSNormGated.weight` |
| Model-level | `model.language_model.embed_tokens.weight`, `model.language_model.norm.weight`, `lm_head.{trellis,suh,svh,mul1}`, plus the `mtp.*` block (1 next-token layer) |

## Checkpoint tensor names (verified from a completed shard)

Shard `model-00005-of-00013.safetensors` (13992 tensors) confirms the on-disk
layout the loader must follow:

```
model.language_model.layers.{i}.input_layernorm.weight
model.language_model.layers.{i}.post_attention_layernorm.weight
model.language_model.layers.{i}.hc_attn_{base,fn,scale}       # hyper-connections
model.language_model.layers.{i}.hc_ffn_{base,fn,scale}
model.language_model.layers.{i}.mlp.gate.weight               # router (+ e_score_correction_bias)
model.language_model.layers.{i}.mlp.shared_experts.{gate,up,down}_proj.{trellis,suh,svh,mul1}
model.language_model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.{trellis,suh,svh,mul1}
model.language_model.layers.{i}.self_attn.A_log               # KDA forget gate (+ dt_bias, conv1d, ...)
```

Bitrates observed in that shard: **K = 2 for the routed experts** (3456 tensors)
and **K = 4 for the attention/dense tensors** (23), i.e. all integer K in 2..6 -
the existing trellis decoder and GEMV kernel cover them as-is.

## Verified per-layer tensor set (layer 15, shard 5 - a DSA/MoE layer)

Mixed quantization: EXL3 tensors carry `.trellis/.suh/.svh/.mul1` (K=2 for
experts, K=4 elsewhere), the rest are plain F16/BF16/F32. Loader must handle both.

```
hc_attn_base  F32 [24]              hc_attn_fn  F32 [24, 16384]     hc_attn_scale  F32 [3]
hc_ffn_base   F32 [24]              hc_ffn_fn   F32 [24, 16384]     hc_ffn_scale   F32 [3]
input_layernorm.weight  BF16 [4096] post_attention_layernorm.weight BF16 [4096]
mlp.gate.weight F16 [288, 4096]     mlp.gate.e_score_correction_bias F32 [288]
mlp.experts.{e}.{gate,up,down}_proj.{trellis,suh,svh,mul1}   e = 0..287   (EXL3, K=2)
mlp.shared_experts.{gate,up,down}_proj.{trellis,suh,svh,mul1}            (EXL3)
self_attn.q_a_proj.*      EXL3      q_a_layernorm.weight  BF16 [1536]
self_attn.q_b_proj.*      EXL3
self_attn.kv_a_proj_with_mqa.* EXL3 kv_a_layernorm.weight  BF16 [512]
self_attn.kv_b_proj.weight F16 [32768, 512]   # unquantized: 64 heads x (256 qk + 256 v)
self_attn.o_proj.*        EXL3
self_attn.indexer.wq_b.*  EXL3      indexer.wk.weight        F16 [128, 4096]
self_attn.indexer.k_norm.{weight,bias}      F16 [128]
self_attn.indexer.weights_proj.weight       F16 [32, 4096]
self_attn.indexer.index_kpool_compress_gate F16 [128, 4096]
self_attn.indexer.index_kpool_compress_ape  F32 [4, 128]
```

Shard 5 holds layers 15-18 (13992 tensors). KDA layers (e.g. layers 0-2, 4-6)
carry `linear_attn.*` instead of `self_attn.*` with `q_proj`, `k_proj`, `v_proj`,
`conv1d`, `b_proj`, `g_a_proj`, `g_b_proj`, `o_norm`, `o_proj` and
`forget_gate.{f_a_proj,f_b_proj,dt_bias,A_log}` (names to confirm against a
completed shard containing a KDA layer).

## Tokenizer (verified from the checkpoint's tokenizer.json)

BPE, vocab 154820 + 36 added tokens, decoder ByteLevel. Special ids:

```
154820 <|endoftext|>   154821 [MASK]      154822 [gMASK]     154823 [sMASK]
154824 <sop>           154825 <eop>       154826 <|system|>  154827 <|user|>
154828 <|assistant|>   154829 <|observation|>                154841 <think> 154842 </think>
154843 <tool_call>     154844 </tool_call> 154845 <tool_response> 154846 </tool_response>
154847 <arg_key> ...   (image/video/audio and code-fence markers in between)
```

`eos_token_id` = `[154820, 154827, 154829]` (end-of-text, user, observation) per the config.

Pre-tokenizer chain (single stage, unlike the MiniCPM5 checkpoint):

```
0  Split(Regex "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
           ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", Isolated)
1  ByteLevel(add_prefix_space=false, use_regex=false)
```

**Required engine change**: `Tokenizer::encode` currently hardcodes the two-stage
split used by MiniCPM5 (a separate `\p{N}{1,3}` stage followed by the word regex).
GLM collapses that into one alternation in a single Split stage, so the tokenizer
must interpret the `pre_tokenizer` chain from `tokenizer.json` (Split with the
given regex + ByteLevel, in order) instead of hardcoding the stage layout. The
GPT-4 alternation itself - including the `\p{N}{1,3}` branch - is already
implemented in the matcher, so this is a plumbing change, not new regex work.

## Chat template (from the checkpoint's `chat_template.jinja`)

`[gMASK]<sop>` prompt prefix, `<|system|>Reasoning Effort: {Low|High|Max}` for a
reasoning-effort control, then `<|user|>` / `<|assistant|>` turns with a
`<think>`-style reasoning block; tool definitions are rendered between the
template's tool markers.

## Implementation order

1. **Loader + config**: extend `ModelConfig` for `glm5_next_text` (nested
   `text_config`), per-layer type arrays, MoE/router/indexer/hc parameters;
   load only `model.language_model.*` + `lm_head` + the MTP block (skip
   `model.visual.*`, 987 tensors).
2. **KDA layers** (34): conv1d(kernel 4) + A_log/dt_bias + gating + recurrence
   (`fast::gated_delta_update`) + gated norm + output projection.
3. **MLA attention** (11): q_a/kv_a low-rank projections + layernorms + q_b/kv_b,
   nope-only (no RoPE), 64 heads, v 256.
4. **DSA indexer** (11): score past tokens with the 32x128 indexer, pool/pool-tail
   rules, select `topk` 2048, then run the attention over the selected set.
5. **MoE**: router (sigmoid + noaux_tc + group selection), top-8 of 288, shared
   expert, `routed_scaling_factor` 2.5, `swiglu_limit` 10 clamp, 3 dense layers.
6. **Hyper-connections** (`mhc`, hc_mult 4, 20 Sinkhorn iterations).
7. **MTP block** wired into the existing speculative loop (`Drafter` interface).
8. Tokenizer/chat template (vocab 154880, its own merges + template).

## KDA recurrence (verbatim semantics from the reference)

`recurrent_kimi_delta_attention` (modeling_glm5_next.py:463) is the per-position
form; the implementation to port is exactly this (all state math in fp32):

```
# per layer: 16 heads, k_head_dim = v_head_dim = 128, state S = zeros(16,128,128) fp32
q, k = l2norm(q), l2norm(k)            # l2norm(x) = x / sqrt(sum(x^2) + 1e-6)
q = q * (1 / sqrt(k_head_dim))         # 1/sqrt(128)
for i in range(sequence_length):
    g_i = exp(g[:, i])                 # per-head scalar decay, shape (H,1)
    b_i = beta[:, i]                   # per-head scalar, shape (H,1)
    S = S * g_i
    kv_mem = (S * k_i[..., None]).sum(-2)         # (H, v_head_dim)
    delta = (v_i - kv_mem) * b_i                  # (H, v_head_dim)
    S = S + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
    out[i] = (S * q_i.unsqueeze(-1)).sum(-2)      # (H, v_head_dim)
```

Notes that matter for a faithful port:

- `l2norm` uses `sqrt(sum(x*x) + eps)` then a plain divide (deliberately not
  `F.normalize`, to match the FLA kernels the model was trained with).
- The decay `g` is applied *before* the delta update, and the read-out uses the
  post-update state.
- `Glm5NextTextLinearAttention.forward` (620) supplies q/k/v (with the conv1d
  over the depthwise sliding window, kernel 4), `beta` (from `b_proj`) and `g`
  (from `forget_gate`: `f_a_proj`/`f_b_proj` + `dt_bias` + `A_log`, see
  `Glm5NextTextForgetGate` at 339), then applies `RMSNormGated` (`o_norm`) and
  `o_proj`.
- MLX exposes `fast::gated_delta_update`, but the reference above is the exact
  formula to validate against before trusting the primitive.

## Expected performance envelope

Rough sizing: the MoE holds ~300 B expert parameters at 2.51 bpw (~95 GB on
disk); per token only the top-8 experts are read, i.e. ~8.4 B parameters plus
attention/KDA and the quantized head:

| Item | Value |
|---|---|
| Resident weights | ~95 GB (fits a 128 GB Mac, tight) |
| Bytes read per token (estimate) | ~2.6-3.5 GB |
| Bandwidth-bound ceiling (measured 425 GB/s) | ~100-130 tok/s |
| Realistic with current kernels (180-365 GB/s) | ~40-70 tok/s |
| With the embedded MTP block | 1.5-3x on top (acceptance dependent) |

Note the quantization is *integer* K per tensor (2..6 bpw, like the other
checkpoints we support), so the existing trellis decoder and kernel apply
without fractional-bitrate work.
