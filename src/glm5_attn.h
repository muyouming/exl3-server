#pragma once
// GLM-5-Next MLA (nope-only multi-head latent attention) + DSA selection plumbing.
// Wiring pinned in docs/glm5.md from Glm5NextTextAttention.forward (1185-1235).

#include <mlx/mlx.h>

#include "glm5.h"
#include "model.h"

namespace exl3 {
namespace mx = mlx::core;

// Standard RMSNorm (Glm5NextTextRMSNorm): fp32, x * rsqrt(mean(x^2)+eps) * weight.
mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps = 1e-6f);

// Per-layer MLA state: the *compressed* KV latent cache (kv_lora_rank wide),
// expanded per step - 32x smaller than caching expanded k/v.
struct MlaState {
  mx::array k_pass = mx::array(0.0f); // (B, L_total, kv_lora_rank) f32
  int len = 0;
  // Drops every row at or past `rows` (the compressed latent cache is a plain
  // sequence, so a partly accepted verify batch can be rewound by slicing).
  void trim(int rows);
};

// LayerNorm with bias (the indexer's k_norm is a LayerNorm, not an RMSNorm).
mx::array layer_norm(const mx::array& x, const mx::array& weight, const mx::array& bias,
                     float eps = 1e-6f);

// DSA indexer pooling + scoring (Glm5NextTextIndexer.get_pooled_states + scoring,
// 850-885). Pure function over the per-token packed indexer state so it can be
// checked against the reference independently of the projections.
//   packed (L_total, 2*head_dim + 1) = cat(key | gate_scores | valid_flag)
//   q      (L, n_heads, head_dim)
//   w      (L, n_heads)  per-head weights (weights_proj(h) * n_heads**-0.5)
//   ape    (index_kpool, head_dim)
struct IndexerPools {
  mx::array keys = mx::array(0.0f);   // (n_pools, head_dim) pooled keys
  mx::array scores = mx::array(0.0f); // (L, n_pools) index scores
  std::vector<int> indices;  // (n_pools * index_kpool) absolute token ids, -1 invalid
  std::vector<char> valid;   // (n_pools)
  int n_pools = 0;
  int kpool = 0;
};
IndexerPools indexer_pool_and_score(const mx::array& packed, const mx::array& q,
                                    const mx::array& w, const mx::array& ape);

// Selection half of the DSA indexer (880-900 + append_visible_tail at 1010):
// pick the top `select_k` pools per query row, expand them to token ids, append
// the still-incomplete tail pool (at most kpool-1 raw ids) and pad to the fixed
// output width used by the sparse attention mask. `visible` is (L, L_total)
// row-major causal+validity, `row_visible` is (L) for the padding rows of Q.
// Returns (L, output_width) int32 with -1 for absent entries.
mx::array indexer_select(const IndexerPools& pl, const std::vector<char>& visible,
                         const std::vector<char>& row_visible, int index_topk,
                         bool always_select_tail);

// q_residual shared between MLA and its indexer: q_a_layernorm(q_a_proj(h)).
mx::array mla_q_resid(const G5Layer& l, const mx::array& h);

// Per-layer indexer state: the packed per-token cache
// (key | gate_scores | valid_flag), grown by one row per step.
struct IndexerState {
  mx::array packed = mx::array(0.0f); // (1, L_total, 2*head_dim + 1) f32
  int len = 0;
  // Same as MlaState::trim: the packed per-token indexer rows are a sequence.
  void trim(int rows);
};

// DSA indexer top-k token selection for the current chunk (B = 1). `start_pos`
// is the absolute position of the chunk's first query token. Returns
// (1, L, index_topk + index_kpool - 1) int32 with -1 for absent entries -
// exactly what mla_layer's `topk` argument consumes.
mx::array indexer_topk(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                       const mx::array& q_resid, IndexerState& st, int start_pos);

// One MLA layer. `h` is (B, L, hidden); `start_pos` is the absolute position of
// the first query token. `topk` (B, L, K) holds absolute key positions selected
// by the DSA indexer (-1 = none); pass an empty array for dense causal attention.
mx::array mla_layer(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                    MlaState& state, int start_pos, const mx::array& topk = mx::array(0.0f),
                    const mx::array& q_resid_in = mx::array(0.0f));

// ---- batched decode (B sequences, one new row each) -------------------------
//
// Both per-layer caches are (B, T, ...): row b of the batch is the b-th
// sequence, a step appends every sequence's new row at the end of the array at
// once, and slot b's valid rows are the LAST len[b] rows of its own block. A
// slot that joined late therefore carries zero rows in front of its own history
// (a zero latent expands to zero k/v and a zero indexer row is invalid), and
// the caches of the different slots never alias - dim 0 separates them.
struct MlaBatchCache {
  mx::array k_pass = mx::array(0.0f); // (B, T, kv_lora_rank) f32
};
struct IndexerBatchCache {
  mx::array packed = mx::array(0.0f); // (B, T, 2*index_head_dim + 1) f32
};

// One batched MLA layer: `h` is (B, 1, hidden), the new rows are appended to
// `cache` (its width grows by one) and `len` (per slot, before the append).
// `topk` is (B, 1, W) int32 of CACHE row indices (-1 = none) for the slots' new
// query rows - exactly what indexer_topk_batch produces. Returns (B, 1, hidden).
//
// The context of every slot is expanded in one matmul (kv_b_proj over the whole
// padded cache), which is the single-sequence decoder's own cost, now shared by
// the batch; the attention itself is per (slot, head) over the slot's own rows.
mx::array mla_layer_batch(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                          MlaBatchCache& cache, const std::vector<int>& len, const mx::array& topk,
                          const mx::array& q_resid);

// The DSA indexer of a batched decode step: every row of a slot is visible to
// that slot's query, the pools are gathered from the slot's own block of the
// packed cache, and the selection is the reference's (top pools by score, then
// the incomplete tail). Appends one packed row per slot. Returns (B, 1, W)
// int32 of cache row indices, -1 for absent entries.
mx::array indexer_topk_batch(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                             const mx::array& q_resid, IndexerBatchCache& cache,
                             const std::vector<int>& len);

} // namespace exl3
