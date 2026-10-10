#pragma once
// GLM-5-Next (glm5_next_text / GLM-5.3-Flash) weight loading.
//
// Text-only. The vision tower (`model.visual.*`) is *explicitly* skipped: the
// loader prints every skipped prefix with its tensor count and fails on any
// tensor it does not recognize, so nothing is ever dropped silently.
//
// The MTP (multi-token-prediction) draft block IS loaded when the config
// declares `num_nextn_predict_layers > 0`. This checkpoint stores it as one
// extra decoder layer after the trunk (`layers.{num_hidden_layers}.*`, i.e.
// layer 45): its own MLA + indexer + MoE weights, plus `enorm`/`hnorm`/`eh_proj`
// `shared_head.norm` (the head itself is the trunk's `lm_head`). It has no
// hyper-connection parameters, so its residuals are plain.
//
// Tensors are mixed-quantization: most are EXL3 (trellis + suh/svh + mul1
// marker), while `kv_b_proj`, the indexer's small matrices, the KDA gate
// projections, the hyper-connection parameters and the router bias are plain
// F16/BF16/F32.
//
// Tensors whose file bytes can be consumed as they are (f16/f32, no transpose)
// are NOT copied: they become MLX views of a private, writable mapping of the
// safetensors shard, so the 105 GB checkpoint loads without a 105 GB heap copy
// (and without MLX having to copy each weight into a Metal buffer first, which
// is what happens for any foreign pointer that is not page aligned).
// See docs/glm5.md for the verified per-layer tensor list.

#include <mlx/mlx.h>

#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "trellis_gemm.h"

namespace exl3 {

namespace mx = mlx::core;

// A linear layer that is either an EXL3 trellis or a plain f16 (in, out) weight.
struct G5Linear {
  bool quantized = false;
  TrellisWeight tw;                  // valid when quantized
  mx::array w = mx::array(0.0f);     // f16 (in_features, out_features) otherwise
  mx::array bias = mx::array(0.0f);
  bool has_bias = false;
  int in_features = 0;
  int out_features = 0;
};

// Per-layer decode preparation for the MoE, built once from the loaded experts
// (glm5_moe.cpp) and then reused by every decode step. Everything in here is a
// pure re-layout of the checkpoint's own values - the byte-for-byte tensors the
// grouped launch needs, so the per-token path does not rebuild them:
//
//   router_w_f32 router.weight cast to f32 (what the router's matmul consumes;
//                the reference re-casts the whole (E, hidden) matrix per layer)
//   suh_*/svh_*  one (experts, in|out) f16 stack per projection class (gate|up
//                merged into one, in that member order), so the eight routed
//                scale vectors of a step are one device-side `mx::take` away
//                instead of an eight-way concatenate.
//
// `valid` is false when the layer's experts are not uniform in the way the
// grouped path needs; moe_layer then takes the per-expert path.
struct G5MoePrep {
  mx::array router_w_f32 = mx::array(0.0f);
  // The merged gate|up launch's scale stack: rows 0..E-1 are the gate experts,
  // E..2E-1 the up ones (so the merged member list and its ids line up), then
  // the down projection's own stack.
  mx::array suh_gu = mx::array(0.0f), svh_gu = mx::array(0.0f);
  mx::array suh_d = mx::array(0.0f), svh_d = mx::array(0.0f);
  bool valid = false;
};

struct G5Layer {
  ModelConfig::LayerPlan plan;
  // shared
  mx::array input_norm = mx::array(0.0f), post_norm = mx::array(0.0f); // (hidden) f16
  mx::array hc_attn_base = mx::array(0.0f), hc_attn_fn = mx::array(0.0f),
            hc_attn_scale = mx::array(0.0f);                            // f32
  mx::array hc_ffn_base = mx::array(0.0f), hc_ffn_fn = mx::array(0.0f),
            hc_ffn_scale = mx::array(0.0f);                             // f32
  // KDA (`linear_attention` layers). The checkpoint fuses q|k|v into ONE EXL3
  // tensor `self_attn.qkv_proj`, concatenated as q | k | v, each `heads *
  // head_dim` wide (the reference projects them separately and then does
  // cat([q_proj(h), k_proj(h), v_proj(h)], -1)).
  G5Linear kda_qkv, kda_b, kda_g_a, kda_g_b, kda_f_a, kda_f_b, kda_o;
  mx::array conv1d_w = mx::array(0.0f);  // (3*heads*head_dim, 1, kernel)
  mx::array conv1d_b = mx::array(0.0f);  // absent in this checkpoint (bias=False)
  bool conv1d_has_bias = false;
  mx::array o_norm = mx::array(0.0f), dt_bias = mx::array(0.0f), a_log = mx::array(0.0f);
  int kda_heads = 0;     // len(A_log)
  int kda_head_dim = 0;  // len(dt_bias) / heads
  // `linear_attn_config.gate_lower_bound`: non-zero selects fla's USE_LOWER_BOUND
  // forget gate (-bound * sigmoid(...)) instead of the softplus form. This
  // checkpoint sets -5.0.
  float kda_gate_lower_bound = 0.0f;
  // MLA + DSA indexer (sparse_attention layers)
  G5Linear q_a, q_b, kv_a, kv_b, attn_o, idx_wq_b;
  mx::array q_a_norm = mx::array(0.0f), kv_a_norm = mx::array(0.0f);
  mx::array idx_wk = mx::array(0.0f), idx_k_norm_w = mx::array(0.0f), idx_k_norm_b = mx::array(0.0f),
            idx_weights_proj = mx::array(0.0f), idx_ape = mx::array(0.0f),
            idx_gate = mx::array(0.0f);
  // MoE
  bool moe = false;
  std::vector<G5Linear> experts; // 3 per expert: gate, up, down (row-major)
  G5Linear shared_gate, shared_up, shared_down;
  mx::array router_w = mx::array(0.0f), router_bias = mx::array(0.0f);
  // Lazily built by moe_layer's first decode call (see G5MoePrep).
  mutable std::shared_ptr<G5MoePrep> moe_prep;
  // dense MLP (first_k_dense_replace layers)
  G5Linear dense_gate, dense_up, dense_down;
};

// The MTP draft block: one decoder layer fed by
//   x = eh_proj(cat[RmsNorm(Emb(t_j)), RmsNorm(h_{j-1})])
// at the token's own position, whose output goes through `head_norm` and then
// the trunk's `lm_head`. `h_{j-1}` is the trunk's final-normed hidden for the
// token before `t_j` (so row j predicts the token after `t_j`), and the layer
// attends causally over its own rows 1..j. `hc_*` fields of `layer` stay empty.
struct G5Mtp {
  bool present = false;
  int layer_index = -1;                 // cfg.num_layers (45 here)
  G5Layer layer;                        // MLA + indexer + MoE, plain residuals
  mx::array enorm = mx::array(0.0f);    // f16 (hidden)
  mx::array hnorm = mx::array(0.0f);    // f16 (hidden)
  G5Linear eh_proj;                     // (2*hidden -> hidden)
  mx::array head_norm = mx::array(0.0f); // shared_head.norm.weight, f16 (hidden)
};

struct Glm5Weights {
  mx::array embed = mx::array(0.0f);      // f16 (vocab, hidden)
  mx::array final_norm = mx::array(0.0f); // f16 (hidden)
  G5Linear lm_head;
  std::vector<G5Layer> layers;
  G5Mtp mtp;                              // present only when the config has MTP
  int64_t bytes = 0;                      // bytes of all safetensors shards (mmap'd)
  int64_t used_bytes = 0;                 // bytes of the tensors actually loaded
  int64_t used_tensors = 0;               // tensors actually loaded
  int64_t total_tensors = 0;              // tensors present in the checkpoint
  // Explicit skip sets: prefix label -> tensor count (vision tower, MTP block).
  std::vector<std::pair<std::string, int64_t>> skipped;
  size_t expert_count() const {
    size_t n = 0;
    for (auto& l : layers) n += l.experts.size() / 3;
    return n;
  }
};

// Loads the language-model weights described by `cfg` (arch must be Glm5Next).
// Missing optional tensors are reported but only required ones fail the load.
// Tensors outside the loader's knowledge (other than the documented skip sets)
// fail the load rather than being dropped silently.
bool load_glm5(const std::string& dir, const ModelConfig& cfg, Glm5Weights& out, std::string& err,
               int threads = 0, bool verbose = false);

} // namespace exl3
