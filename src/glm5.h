#pragma once
// GLM-5-Next (glm5_next_text / GLM-5.3-Flash) weight loading.
//
// Text-only: vision tensors (`model.visual.*`) and the `mtp.*` draft block are
// not loaded here yet. Linears are mixed-quantization: most are EXL3 (trellis +
// suh/svh + mul1 marker), while `kv_b_proj`, the indexer's small matrices, the
// hyper-connection parameters and the router bias are plain F16/BF16/F32.
//
// See docs/glm5.md for the verified per-layer tensor list.

#include <mlx/mlx.h>

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

struct G5Layer {
  ModelConfig::LayerPlan plan;
  // shared
  mx::array input_norm = mx::array(0.0f), post_norm = mx::array(0.0f); // (hidden) f16
  mx::array hc_attn_base = mx::array(0.0f), hc_attn_fn = mx::array(0.0f),
            hc_attn_scale = mx::array(0.0f);                            // f32
  mx::array hc_ffn_base = mx::array(0.0f), hc_ffn_fn = mx::array(0.0f),
            hc_ffn_scale = mx::array(0.0f);                             // f32
  // KDA (linear_attention layers)
  G5Linear kda_q, kda_k, kda_v, kda_b, kda_g_a, kda_g_b, kda_f_a, kda_f_b, kda_o;
  mx::array conv1d_w = mx::array(0.0f), conv1d_b = mx::array(0.0f);
  mx::array o_norm = mx::array(0.0f), dt_bias = mx::array(0.0f), a_log = mx::array(0.0f);
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
  // dense MLP (first_k_dense_replace layers)
  G5Linear dense_gate, dense_up, dense_down;
};

struct Glm5Weights {
  mx::array embed = mx::array(0.0f);      // f16 (vocab, hidden)
  mx::array final_norm = mx::array(0.0f); // f16 (hidden)
  G5Linear lm_head;
  std::vector<G5Layer> layers;
  int64_t bytes = 0;
  size_t expert_count() const {
    size_t n = 0;
    for (auto& l : layers) n += l.experts.size() / 3;
    return n;
  }
};

// Loads the language-model weights described by `cfg` (arch must be Glm5Next).
// Missing optional tensors are reported but only required ones fail the load.
bool load_glm5(const std::string& dir, const ModelConfig& cfg, Glm5Weights& out, std::string& err,
               int threads = 0, bool verbose = false);

} // namespace exl3
