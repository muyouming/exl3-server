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
};

// One MLA layer. `h` is (B, L, hidden); `start_pos` is the absolute position of
// the first query token. `topk` (B, L, K) holds absolute key positions selected
// by the DSA indexer (-1 = none); pass an empty array for dense causal attention.
mx::array mla_layer(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                    MlaState& state, int start_pos, const mx::array& topk = mx::array(0.0f));

} // namespace exl3
