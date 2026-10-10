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
**DONE** - `Tokenizer::load` now interprets the chain: each `Split` stage's
regex is classified (a pure digit rule vs the word regex) and its digit cap is
read from `\p{N}{1,N}` (uncapped `\p{N}+` = 0), so GLM's single-Split chain
(with `\p{N}{1,3}` embedded in the word alternation) and MiniCPM5's
`\p{N}{1,3}` + word-regex pair both produce the same boundaries. Verified by
`tests/pretokenizer_test.cpp` against both models' real `tokenizer.json`:
`x1234567890y` -> `[x][123][456][789][0][y]` for both, plus encode/decode
round-trips. (Note: the *post-BPE* tokens differ between the two vocabularies -
GLM merges `456` as `45`+`6` - which is why the test asserts on pre-token
boundaries via the new `Tokenizer::pretokenize`, not on token strings.)

## Chat template (from the checkpoint's `chat_template.jinja`)

**Implemented** (`render_chat_glm` in src/tokenizer.cpp, selected automatically
when the vocabulary has `[gMASK]`/`<|user|>`/`<|assistant|>`). Verified
byte-for-byte against the real template rendered with jinja2 - see
`tools/render_chat_reference.py` and `tests/chat_template_test.cpp` (5 cases:
plain turn, effort + no generation prompt, reasoning content, tool
observations, clear_thinking).

Rendered layout (the template uses `-` whitespace control everywhere, so the
result has NO cosmetic newlines - only the special tokens delimit things):

```
[gMASK]<sop><|system|>Reasoning Effort: {Low|High|Max}
<|user|>{content}<|assistant|><think>{reasoning}</think>{content}<|assistant|><think>
```

- `reasoning_effort`: the template accepts only `low`/`high`, everything else
  (including unset) renders `Max`; exposed as render_chat's 4th argument.
- A turn whose content contains `</think>` is split at the FIRST `</think>`
  and the reasoning is emitted RAW (not stripped); only the remaining content
  is stripped.
- `clear_thinking` (= `thinking == 0`) drops reasoning from turns at or before
  the last user message.
- Tool results: `<|observation|>` opens a run of tool messages, each wrapped as
  `<tool_response>...</tool_response>` in the order given (the template's
  fallback branch; reordering by tool_call_id needs ids the engine does not
  carry).
- Out of scope (unreachable in the engine's data model): the tools preamble,
  media placeholders, tool_call argument XML.

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

## KDA layer wiring (from `Glm5NextTextLinearAttention.forward`, 660-772)

```
mixed_qkv = concat(q_proj(h), k_proj(h), v_proj(h), dim=-1)     # each qkv_dim = head_dim*heads
            .transpose(1, 2)                                     # (B, C, L) for the conv
mixed_qkv = causal_conv1d(mixed_qkv, weight=conv1d.weight.squeeze(1), bias=conv1d.bias,
                          activation=SiLU)[..., -seq_len:]       # depthwise, kernel 4, causal
q, k, v   = split(mixed_qkv, qkv_dim) -> view (B, L, heads, head_dim)

# forget gate (Glm5NextTextForgetGate, 339):
fg        = f_b_proj(f_a_proj(h))                                # (B, L, qkv_dim)
g         = (fg + dt_bias.view(1,1,-1)).view(B, L, heads, head_dim)   # dt_bias is per (head, dim)
decay     = exp(A_log.view(1,1,heads,1))                         # A_log is per head
g_out     = -decay * softplus(g)          # softplus(x) = x for x > 20 to avoid overflow
# (the reference also supports a `linear_lower_bound` "safe" variant:
#  lower_bound * sigmoid(decay * g); the published config does not set it)

beta      = sigmoid(b_proj(h))
out       = recurrent_kimi_delta_attention(q, k, v, g=g_out, beta=beta)   # formula above
out       = RMSNormGated(out)             # o_norm.weight, gated
out       = o_proj(out)                   # output gate: g_a_proj -> SiLU -> g_b_proj -> sigmoid
```

Decay granularity (verified in the reference): `g` from the forget gate is
viewed as (B, L, heads, head_dim) and `recurrent_kimi_delta_attention` does
`g[:, i][..., None].exp()`, so the decay is **per (head, k-dim)**, not per head.
`beta` on the other hand is (B, L, heads) - a per-head scalar.

The gated output norm (`Glm5NextTextRMSNormGated`, 374) is fp32 and uses
`sigmoid` (not SiLU): `y = x * rsqrt(mean(x^2,-1) + eps) * weight * sigmoid(gate)`,
with the gate coming from `g_b_proj(g_a_proj(h))` viewed as (B,L,heads,head_dim).

Status: the recurrence itself (`kda_recurrence`, `kda_l2norm`, `kda_decay`,
`kda_conv1d` in `src/glm5_kda.cpp`) is implemented and verified against the
reference formula evaluated as a scalar fp64 oracle - `tests/kda_recurrence_test.cpp`
reports max|diff| ~4e-8 with 0/56 mismatches. `kda_layer()` performs the whole layer (conv1d with
rolling state -> projections -> decay -> recurrence -> gated norm -> o_proj);
it compiles against the loader's `G5Layer`/`G5Linear` and is exercised
end-to-end once real weights land (the recurrence underneath is already
verified numerically).

Recurrent-state details for the engine: the conv keeps a 3-position rolling
window (kernel 4) and the delta rule keeps one (heads, k_head_dim, v_head_dim)
state per layer; both must be part of the KV/state cache for incremental decode.

## MLA layer (Glm5NextTextAttention, 1116-1235)

```
q_resid = q_a_layernorm(q_a_proj(h))                     # q_lora_rank 1536, fp32 RMSNorm
q       = q_b_proj(q_resid) -> (B, L, 64, 256)           # qk_head_dim = nope 256 + rope 0
kv      = kv_a_proj_with_mqa(h) -> [kv_lora_rank 512 | rope 0] -> kv_a_layernorm
kv      = kv_b_proj(kv_cached) -> (B, Ltot, 64, 256+256) -> split k_nope / v
scores  = q @ k^T * qk_head_dim**-0.5                    # scaling = 256**-0.5
mask    = causal, intersected with the DSA top-k indices (-1 = invalid)
out     = softmax(scores) @ v -> reshape(B, L, 64*256) -> o_proj
```

The compressed KV latent (512 wide) is what gets cached, not the expanded
k/v - the expansion is re-done per step (`expand_kv`), which is 32x cheaper
in memory. `build_attention_mask_from_topk` scatters the selected indices into
a per-query key mask; the indexer has already applied causality and padding,
so the mask is exactly the top-k set.

Status: `mla_layer()` implements this against the loader's `G5Layer`; the
prefill/incremental invariant is verified by `tests/mla_incremental_test.cpp`
(decode after a n-1 prefill reproduces the n-token prefill's last row to
6e-08). Numeric agreement with HF is still pending real weights.

## DSA indexer (Glm5NextTextIndexer, 795-885+) - verified core

```
q      = wq_b(q_resid).view(B, S, 32, 128)        # q_resid is SHARED with MLA:
                                                  # q_a_layernorm(q_a_proj(h))
k      = k_norm(wk(h)).view(B, S, 128)            # k_norm is a *LayerNorm* (mean+var,
                                                  # eps 1e-6, with bias), not RMSNorm
gate_s = index_kpool_compress_gate @ h            # (B, S, 128); the param is (128, hidden)
packed = cat([k, gate_s, valid_channel], -1)      # (B, S, 257) - the per-token indexer
                                                  # cache entry (k | gate | padding flag)
valid  = packed[..., -1].bool()
visible = get_visible_tokens(valid, q_len, current_len)     # causal AND padding
pool_keys, pool_indices, pool_valid = get_pooled_states(packed)  # pools of index_kpool=4
scores  = relu(matmul(q, pool_keys^T) * head_dim**-0.5)     # (B, S, 32, n_pools), relu not softmax
w       = weights_proj(h).float() * 32**-0.5                # per-head weights (B, S, 32)
index_scores = matmul(w[..., None, :], scores)              # (B, S, n_pools)
pool_end     = pool_indices[..., -1]
valid_candidates = visible.gather(pool_end) & pool_valid[:, None]
index_scores.masked_fill(~valid_candidates, min)            # then top-k over pools,
                                                            # expanded back to token ids
```

Pooling and visibility (both verified verbatim):

```
# get_pooled_states (935): pooling starts at the FIRST VALID token, not slot 0
first_key  = first valid position per row (seq_len if none)
pool_off   = arange(n_pools * 4).view(n_pools, 4)          # n_pools = ceil(seq_len / 4)
pool_idx   = first_key[:, None, None] + pool_off           # [P,P,A,B,C,D..] behaves as [A,B,C,D..]
safe       = pool_idx.clamp(0, seq_len - 1)
gv         = valid[safe] & (pool_idx < seq_len)
pool_valid = gv.all(-1)                                    # a pool needs ALL 4 slots valid
logits     = gate_scores[safe] + index_kpool_compress_ape   # ape is (4, 128): per-slot, per-dim
logits[~gv] = -inf
probs      = nan_to_num(softmax(logits, dim=2))             # over the 4 slots, per dim
pool_keys  = (probs * keys[safe]).sum(dim=2)                # per-dim weighted average
pool_idx   = pool_idx.masked_fill(~gv, -1)

# get_visible_tokens (915)
q_pos    = current_length - q_length + arange(q_length)     # absolute query positions
visible  = (kv_positions[None,None,:] <= q_pos[None,:,None]) & valid_keys[:,None,:]
```

Remaining to read before implementing: the top-k + tail-pool expansion
(`index_kpool_always_select_tail`) and the cache update `update_indexer`. The returned indices are int32 (B, S, topk)
with `-1` meaning invalid, which is exactly what `mla_layer`'s mask builder consumes.

## MoE (verified from Glm5NextTextMLP / Experts / TopkRouter)

```
# dense MLP (first_k_dense_replace layers) and experts share the same SwiGLU:
gate = gate_proj(x).clamp(max=swiglu_limit)      # swiglu_limit = 10
up   = up_proj(x).clamp(-swiglu_limit, swiglu_limit)
out  = down_proj(silu(gate) * up)

# experts: gate_up_proj is FUSED as [num_experts, 2*inter, hidden] with
# gate in the first half (chunk(2, dim=-1) -> gate, up); output is scaled by
# the token's router weight: out * top_k_weights[token, pos] and index_add'ed.

# router: weight (num_experts, hidden), e_score_correction_bias buffer,
# n_group/topk_group, norm_topk_prob, routed_scaling_factor (2.5).
```

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
