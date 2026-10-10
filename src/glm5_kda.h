#pragma once
// GLM-5-Next KDA (Kimi-style delta attention) primitives.
// Semantics are pinned verbatim in docs/glm5.md from the HF reference
// (recurrent_kimi_delta_attention / Glm5NextTextForgetGate).

#include <mlx/mlx.h>

namespace exl3 {
namespace mx = mlx::core;

// l2norm(x) = x / sqrt(sum(x^2) + eps)   (reference l2norm, NOT F.normalize)
mx::array kda_l2norm(const mx::array& x, float eps = 1e-6f);

// forget-gate decay: -exp(A_log) * softplus(fg + dt_bias)
//   fg      (B, L, heads * head_dim)
//   dt_bias (heads * head_dim)   a_log (heads)
mx::array kda_decay(const mx::array& fg, const mx::array& dt_bias, const mx::array& a_log);

// Depthwise causal conv1d over the sequence axis, kernel `k`, SiLU activation.
//   x (B, L, C), w (C, 1, k) or (C, k), b (C) optional
mx::array kda_conv1d(const mx::array& x, const mx::array& w, const mx::array& b);

// Delta-rule recurrence (recurrent_kimi_delta_attention), all math in fp32:
//   q,k (B,L,H,kd)  v (B,L,H,vd)  g (B,L,H,1) decay per head  beta (B,L,H,1)
//   S[i] = S[i-1]*exp(g_i); kv = (S*k_i).sum(kd); delta = (v_i-kv)*beta_i;
//   S[i] += k_i (x) delta;  out[i] = (S[i]*q_i).sum(kd)
// q is l2-normalised and scaled by 1/sqrt(kd) before use.
mx::array kda_recurrence(const mx::array& q, const mx::array& k, const mx::array& v,
                         const mx::array& g, const mx::array& beta, mx::array* state = nullptr);

} // namespace exl3
