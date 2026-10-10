#pragma once
// GLM-5-Next MoE: sigmoid top-k router, routed experts and the shared expert,
// plus the dense SwiGLU MLP used by the `first_k_dense_replace` layers.
// Semantics pinned in docs/glm5.md from the HF reference
// (Glm5NextTextTopkRouter.forward 157-180, Glm5NextTextExperts.forward 119-142,
//  Glm5NextTextMLP.forward 97-105, Glm5NextTextMoE.forward 199-205).

#include <mlx/mlx.h>

#include <utility>

#include "glm5.h"
#include "model.h"

namespace exl3 {
namespace mx = mlx::core;

// Group-limited top-k routing, verbatim from Glm5NextTextTopkRouter.forward:
//   scores            = sigmoid(logits)
//   scores_for_choice = scores + e_score_correction_bias
//   group_scores      = sum of the top-2 scores_for_choice inside each of the
//                       `n_group` groups (num_experts // n_group wide)
//   the `topk_group` best groups are kept, everything else masked to -inf
//   top_k experts are picked from the surviving scores_for_choice
//   weights           = *un-biased* scores of those experts, optionally
//                       normalised (norm_topk_prob) and scaled.
// Returns {weights (N, top_k) f32, indices (N, top_k) i32}.
std::pair<mx::array, mx::array> moe_topk_route(const mx::array& logits, const mx::array& bias,
                                               int n_group, int topk_group, int top_k,
                                               bool norm_topk_prob, float routed_scaling_factor);

// One sparse MoE block (Glm5NextTextMoE.forward): routed experts + shared expert.
// h is (B, L, hidden); the result is (B, L, hidden).
mx::array moe_layer(const G5Layer& l, const ModelConfig& cfg, const mx::array& h);

// Dense SwiGLU MLP (Glm5NextTextMLP.forward, `first_k_dense_replace` layers):
//   silu(clamp(gate, max=swiglu_limit)) * clamp(up, -swiglu_limit, swiglu_limit)
//   then down_proj. h is (B, L, hidden).
mx::array dense_mlp(const G5Layer& l, const ModelConfig& cfg, const mx::array& h);

} // namespace exl3
