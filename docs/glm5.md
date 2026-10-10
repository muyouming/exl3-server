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
