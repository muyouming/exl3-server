#pragma once
// GLM-5-Next KDA (Kimi-style delta attention) primitives.
// Semantics are pinned verbatim in docs/glm5.md from the HF reference
// (recurrent_kimi_delta_attention / Glm5NextTextForgetGate).

#include <mlx/mlx.h>

#include "glm5.h"
#include "model.h"

namespace exl3 {
namespace mx = mlx::core;

// l2norm(x) = x / sqrt(sum(x^2) + eps)   (reference l2norm, NOT F.normalize)
mx::array kda_l2norm(const mx::array& x, float eps = 1e-6f);

// Forget-gate log-decay (the value the recurrence exponentiates), fp32:
//   bound == 0: -exp(A_log) * softplus(fg + dt_bias)              (softplus arm)
//   bound != 0:  bound * sigmoid(exp(A_log) * (fg + dt_bias))     (fla USE_LOWER_BOUND)
// `linear_attn_config.gate_lower_bound` selects the arm; this checkpoint sets
// -5.0, so the bounded sigmoid replaces the softplus rather than clamping it.
//   fg      (B, L, heads, head_dim)   dt_bias (heads * head_dim)   a_log (heads)
mx::array kda_decay(const mx::array& fg, const mx::array& dt_bias, const mx::array& a_log,
                    float bound = 0.0f);

// Depthwise causal conv1d over the sequence axis, kernel `k`, SiLU activation.
//   x (B, L, C), w (C, 1, k) or (C, k), b (C) optional (ignored when it does not
//   have exactly C elements, i.e. when the checkpoint stores no conv bias)
mx::array kda_conv1d(const mx::array& x, const mx::array& w, const mx::array& b);

// Delta-rule recurrence (recurrent_kimi_delta_attention), all math in fp32:
//   q,k (B,L,H,kd)  v (B,L,H,vd)  g (B,L,H,kd) per-(head,dim) decay  beta (B,L,H,1)
//   S[i] = S[i-1]*exp(g_i); kv = (S*k_i).sum(kd); delta = (v_i-kv)*beta_i;
//   S[i] += k_i (x) delta;  out[i] = (S[i]*q_i).sum(kd)
// q is l2-normalised and scaled by 1/sqrt(kd) before use.
mx::array kda_recurrence(const mx::array& q, const mx::array& k, const mx::array& v,
                         const mx::array& g, const mx::array& beta, mx::array* state = nullptr);

// Gated RMSNorm (Glm5NextTextRMSNormGated): y = norm(x) * weight * sigmoid(gate),
// fp32 internally, norm over the last dim.
mx::array rms_norm_gated(const mx::array& x, const mx::array& weight, const mx::array& gate,
                         float eps);

// Per-layer KDA state for incremental decoding.
struct KdaState {
  mx::array conv_window = mx::array(0.0f); // (B, k-1, C) f32 conv history
  mx::array recurrent = mx::array(0.0f);   // (B, H, kd, vd) f32 delta-rule state
};

// Apply one linear (EXL3 trellis or plain f16 weight) to x.
mx::array g5_linear(const G5Linear& lin, const mx::array& x);

// One full KDA layer (Glm5NextTextLinearAttention.forward): ONE fused q|k|v
// projection (`kda_qkv`, sliced q | k | v in that order), the causal conv1d over
// that concatenation with rolling state, the forget-gate decay, the delta-rule
// recurrence, the gated output norm and the output projection. h is (B, L, hidden).
mx::array kda_layer(const G5Layer& l, const ModelConfig& cfg, const mx::array& h, KdaState& state);

} // namespace exl3
