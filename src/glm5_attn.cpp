#include "glm5_attn.h"

#include <cmath>

#include "glm5_kda.h"
#include "util.h"

namespace exl3 {

mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps) {
  mx::array xf = mx::astype(x, mx::float32);
  mx::array var = mx::mean(mx::multiply(xf, xf), std::vector<int>{-1}, true);
  mx::array y = mx::multiply(xf, mx::rsqrt(mx::add(var, mx::array(eps, mx::float32))));
  return mx::multiply(y, mx::astype(weight, mx::float32));
}

mx::array mla_layer(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                    MlaState& state, int start_pos, const mx::array& topk) {
  const int B = (int)h.shape(0), L = (int)h.shape(1);
  const int H = cfg.num_heads;
  const int kvr = (int)l.kv_b.in_features;              // kv_lora_rank (512)
  const int qk = (int)l.q_b.out_features / H;           // qk_head_dim (nope only: rope = 0)
  const int vd = (int)l.kv_b.out_features / H - qk;     // v_head_dim (256)

  // q path: q_a_proj -> q_a_layernorm -> q_b_proj
  mx::array q_resid = rms_norm(g5_linear(l.q_a, h), l.q_a_norm);
  mx::array q = mx::reshape(g5_linear(l.q_b, q_resid), {B, L, H, qk});
  // kv path: kv_a_proj_with_mqa -> (kv_lora_rank | qk_rope_head_dim=0) -> kv_a_layernorm
  mx::array kv_raw = g5_linear(l.kv_a, h); // (B, L, kvr + 0)
  mx::array k_pass = rms_norm(kv_raw, l.kv_a_norm);

  // append to the compressed cache
  mx::array k_all = k_pass;
  if (state.len > 0) k_all = mx::concatenate({state.k_pass, mx::astype(k_pass, mx::float32)}, 1);
  state.k_pass = k_all;
  state.len += L;
  const int Ltot = state.len;

  // expand: kv_b_proj -> (B, Ltot, H, qk + vd) -> k / v
  mx::array kv = mx::reshape(g5_linear(l.kv_b, k_all), {B, Ltot, H, qk + vd});
  mx::array k = mx::transpose(mx::slice(kv, {0, 0, 0, 0}, {B, Ltot, H, qk}), {0, 2, 1, 3});
  mx::array v = mx::transpose(mx::slice(kv, {0, 0, 0, qk}, {B, Ltot, H, qk + vd}), {0, 2, 1, 3});
  mx::array qh = mx::transpose(q, {0, 2, 1, 3}); // (B,H,L,qk)

  mx::array scores = mx::multiply(mx::matmul(mx::astype(qh, mx::float32),
                                             mx::transpose(mx::astype(k, mx::float32), {0, 1, 3, 2})),
                                  mx::array(1.0f / std::sqrt((float)qk), mx::float32)); // (B,H,L,Ltot)

  // mask: causal, plus the DSA top-k selection when provided
  mx::array mask = mx::zeros(mx::Shape{B, L, Ltot}, mx::bool_);
  {
    std::vector<uint8_t> mb((size_t)B * L * Ltot, 0);
    const bool sparse = topk.size() > 1;
    const int K = sparse ? (int)topk.shape(2) : 0;
    std::vector<int> tk;
    if (sparse) {
      mx::eval(topk);
      const int* tp = topk.data<int32_t>();
      tk.assign(tp, tp + (size_t)B * L * K);
    }
    for (int b = 0; b < B; ++b)
      for (int i = 0; i < L; ++i) {
        const int causal_to = start_pos + i; // inclusive
        for (int j = 0; j <= causal_to && j < Ltot; ++j) mb[((size_t)b * L + i) * Ltot + j] = true;
        if (!sparse) continue;
        for (int j = 0; j <= causal_to && j < Ltot; ++j) mb[((size_t)b * L + i) * Ltot + j] = false;
        for (int j = 0; j < K; ++j) {
          const int t = tk[((size_t)b * L + i) * K + j];
          if (t >= 0 && t < Ltot) mb[((size_t)b * L + i) * Ltot + t] = true;
        }
      }
    auto* mh = new std::vector<uint8_t>(std::move(mb));
    mask = mx::array((void*)mh->data(), mx::Shape{B, L, Ltot}, mx::bool_,
                     [mh](void*) { delete mh; });
  }
  mx::array neg_inf(-INFINITY);
  scores = mx::where(mx::reshape(mask, {B, 1, L, Ltot}), scores, neg_inf);
  mx::array probs = mx::softmax(scores, -1);
  mx::array out = mx::matmul(probs, mx::astype(v, mx::float32)); // (B,H,L,vd)
  out = mx::transpose(out, {0, 2, 1, 3});                       // (B,L,H,vd)
  out = mx::reshape(out, {B, L, H * vd});
  return g5_linear(l.attn_o, mx::astype(out, h.dtype()));
}

} // namespace exl3
