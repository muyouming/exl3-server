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

mx::array layer_norm(const mx::array& x, const mx::array& weight, const mx::array& bias, float eps) {
  mx::array xf = mx::astype(x, mx::float32);
  mx::array mean = mx::mean(xf, std::vector<int>{-1}, true);
  mx::array d = mx::subtract(xf, mean);
  mx::array var = mx::mean(mx::multiply(d, d), std::vector<int>{-1}, true);
  mx::array y = mx::multiply(d, mx::rsqrt(mx::add(var, mx::array(eps, mx::float32))));
  y = mx::multiply(y, mx::astype(weight, mx::float32));
  if (bias.size() > 1) y = mx::add(y, mx::astype(bias, mx::float32));
  return y;
}

IndexerPools indexer_pool_and_score(const mx::array& packed, const mx::array& q,
                                    const mx::array& w, const mx::array& ape) {
  const int Ltot = (int)packed.shape(0);
  const int D = (int)ape.shape(1);
  const int kpool = (int)ape.shape(0);
  const int L = (int)q.shape(0);
  const int Nh = (int)q.shape(1);
  mx::eval(packed, q, w, ape);
  const float* pk = packed.data<float>();
  auto key_at = [&](int t, int d) { return pk[(size_t)t * (2 * D + 1) + d]; };
  auto gate_at = [&](int t, int d) { return pk[(size_t)t * (2 * D + 1) + D + d]; };
  auto valid_at = [&](int t) { return pk[(size_t)t * (2 * D + 1) + 2 * D] > 0.5f; };

  // pooling starts at the first valid token (get_pooled_states)
  int first_valid = Ltot;
  for (int t = 0; t < Ltot; ++t)
    if (valid_at(t)) { first_valid = t; break; }
  IndexerPools out;
  out.kpool = kpool;
  out.n_pools = (Ltot + kpool - 1) / kpool;
  const int n = out.n_pools;
  std::vector<float> gkey((size_t)n * kpool * D, 0.0f), glog((size_t)n * kpool * D, 0.0f);
  std::vector<float> gvalid((size_t)n * kpool, 0.0f);
  out.indices.assign((size_t)n * kpool, -1);
  out.valid.assign((size_t)n, 0);
  for (int p = 0; p < n; ++p) {
    for (int s = 0; s < kpool; ++s) {
      const int t = first_valid + p * kpool + s;
      const bool in_range = t < Ltot;
      const bool ok = in_range && valid_at(t);
      if (ok) {
        out.indices[(size_t)p * kpool + s] = t;
        for (int d = 0; d < D; ++d) {
          gkey[((size_t)p * kpool + s) * D + d] = key_at(t, d);
          glog[((size_t)p * kpool + s) * D + d] = gate_at(t, d) + ape.data<float>()[(size_t)s * D + d];
        }
      } else {
        for (int d = 0; d < D; ++d) glog[((size_t)p * kpool + s) * D + d] = -INFINITY;
      }
      gvalid[(size_t)p * kpool + s] = ok ? 1.0f : 0.0f;
    }
    bool all = true;
    for (int s = 0; s < kpool; ++s) all = all && gvalid[(size_t)p * kpool + s] > 0.5f;
    out.valid[(size_t)p] = all ? 1 : 0;
  }
  auto wrap = [](std::vector<float> v, mx::Shape sh) {
    auto* h = new std::vector<float>(std::move(v));
    return mx::array((void*)h->data(), sh, mx::float32, [h](void*) { delete h; });
  };
  mx::array K = wrap(std::move(gkey), {n, kpool, D});
  mx::array Gt = wrap(std::move(glog), {n, kpool, D});
  // per-dim softmax over the pool's slots (nan_to_num for fully invalid pools)
  mx::array probs = mx::softmax(Gt, 1);
  probs = mx::where(mx::isnan(probs), mx::zeros_like(probs), probs);
  out.keys = mx::sum(mx::multiply(probs, K), std::vector<int>{1}, false); // (n, D)
  // scores: relu(q . pool_keys) * head_dim**-0.5, weighted over heads
  mx::array qf = mx::astype(q, mx::float32);
  mx::array sc = mx::matmul(qf, mx::transpose(out.keys, {1, 0})); // (L, Nh, n)
  sc = mx::multiply(mx::maximum(sc, mx::zeros_like(sc)),
                    mx::array(1.0f / std::sqrt((float)D), mx::float32));
  out.scores = mx::matmul(mx::reshape(mx::astype(w, mx::float32), {L, 1, Nh}), sc); // (L,1,n)
  out.scores = mx::reshape(out.scores, {L, n});
  return out;
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
