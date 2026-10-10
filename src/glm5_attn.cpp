#include "glm5_attn.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "glm5_kda.h"
#include "util.h"

namespace exl3 {

mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps) {
  // Glm5NextTextRMSNorm: fp32 math, then back to the input dtype (the reference
  // returns `weight * hidden_states.to(input_dtype)`), which is also what the
  // fp16-only EXL3 projections downstream expect.
  const mx::Dtype dt = x.dtype();
  mx::array xf = mx::astype(x, mx::float32);
  mx::array var = mx::mean(mx::multiply(xf, xf), std::vector<int>{-1}, true);
  mx::array y = mx::multiply(xf, mx::rsqrt(mx::add(var, mx::array(eps, mx::float32))));
  y = mx::multiply(mx::astype(y, dt), mx::astype(weight, dt));
  return y;
}

void MlaState::trim(int rows) {
  if (rows >= len) return;
  if (rows <= 0) {
    k_pass = mx::array(0.0f);
    len = 0;
    return;
  }
  k_pass = mx::slice(k_pass, {0, 0, 0}, {1, rows, (int)k_pass.shape(2)});
  len = rows;
}

void IndexerState::trim(int rows) {
  if (rows >= len) return;
  if (rows <= 0) {
    packed = mx::array(0.0f);
    len = 0;
    return;
  }
  packed = mx::slice(packed, {0, 0, 0}, {1, rows, (int)packed.shape(2)});
  len = rows;
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

mx::array indexer_select(const IndexerPools& pl, const std::vector<char>& visible,
                         const std::vector<char>& row_visible, int index_topk,
                         bool always_select_tail) {
  mx::eval(pl.scores);
  const int L = (int)pl.scores.shape(0);
  const int n = pl.n_pools;
  const int kp = pl.kpool;
  const int Ltot = (int)(visible.size() / (size_t)std::max(1, L));
  const int select_k = std::min(index_topk / std::max(1, kp), n);
  const int tail_w = always_select_tail ? kp - 1 : 0;
  const int width = index_topk + tail_w;
  const float* sc = pl.scores.data<float>();
  // pools holding a complete, visible key set
  std::vector<char> ok((size_t)L * n, 0);
  for (int l = 0; l < L; ++l)
    for (int p = 0; p < n; ++p) {
      if (l >= (int)row_visible.size() || !row_visible[(size_t)l]) continue;
      if (!pl.valid[(size_t)p]) continue;
      const int end = pl.indices[(size_t)p * kp + kp - 1];
      if (end < 0 || end >= Ltot) continue;
      ok[(size_t)l * n + p] = visible[(size_t)l * Ltot + end] ? 1 : 0;
    }
  std::vector<int32_t> out((size_t)L * width, -1);
  std::vector<std::pair<float, int>> ranked;
  for (int l = 0; l < L; ++l) {
    ranked.clear();
    for (int p = 0; p < n; ++p)
      if (ok[(size_t)l * n + p]) ranked.push_back({sc[(size_t)l * n + p], p});
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    if ((int)ranked.size() > select_k) ranked.resize((size_t)select_k);
    int w = 0;
    for (auto& [_, p] : ranked) {
      for (int s = 0; s < kp && w < index_topk; ++s) {
        const int t = pl.indices[(size_t)p * kp + s];
        if (t >= 0 && t < Ltot && visible[(size_t)l * Ltot + t]) out[(size_t)l * width + w++] = t;
      }
    }
    if (!tail_w) continue;
    // The row's current incomplete pool (reference `append_visible_tail`, 1010):
    // visible_count = the row's visible key count, tail_count = visible_count % kp,
    // tail_start = first_key + visible_count - tail_count, so query rows earlier
    // than the first full pool still select their own (short) tail.
    int visible_count = 0;
    for (int t = 0; t < Ltot; ++t)
      if (visible[(size_t)l * Ltot + t]) ++visible_count;
    if (visible_count == 0) continue;
    const int first = pl.indices.empty() ? 0 : std::max(0, pl.indices[0]);
    const int tail_count = visible_count % kp;
    const int tail_start = first + visible_count - tail_count;
    for (int j = 0; j < tail_w; ++j) {
      if (j >= tail_count || w >= width) break;
      const int t = tail_start + j;
      if (t >= 0 && t < Ltot && visible[(size_t)l * Ltot + t]) out[(size_t)l * width + w++] = t;
    }
  }
  auto* h = new std::vector<int32_t>(std::move(out));
  return mx::array((void*)h->data(), mx::Shape{L, width}, mx::int32,
                   [h](void*) { delete h; });
}

mx::array mla_q_resid(const G5Layer& l, const mx::array& h) {
  return rms_norm(g5_linear(l.q_a, h), l.q_a_norm);
}

mx::array indexer_topk(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                       const mx::array& q_resid, IndexerState& st, int start_pos) {
  const int B = (int)h.shape(0), L = (int)h.shape(1);
  if (B != 1) throw std::runtime_error("GLM indexer implemented for batch 1");
  const int Nh = cfg.index_n_heads;                                  // 32
  const int D = (int)l.idx_wk.shape(1); // index_head_dim; idx_wk is (in, out) = the loader's layout
  const int kp = cfg.index_kpool;                                    // 4
  // The loader already transposes the raw (out, in) indexer weights to (in, out).
  auto proj = [](const mx::array& w, const mx::array& x) { return mx::matmul(x, w); };
  mx::array q = mx::reshape(g5_linear(l.idx_wq_b, q_resid), {L, Nh, D});
  mx::array key = layer_norm(proj(l.idx_wk, h), l.idx_k_norm_w, l.idx_k_norm_b);
  mx::array gate = proj(l.idx_gate, h);
  mx::array valid = mx::ones(mx::Shape{B, L, 1}, mx::float32); // single sequence: no padding
  mx::array packed_new = mx::concatenate({mx::astype(key, mx::float32),
                                          mx::astype(gate, mx::float32), valid}, -1);
  mx::array all = st.len > 0 ? mx::concatenate({st.packed, packed_new}, 1) : packed_new;
  st.packed = all;
  st.len += L;
  const int Ltot = st.len;
  mx::array weights = mx::multiply(
      proj(l.idx_weights_proj, h), mx::array(1.0f / std::sqrt((float)Nh), mx::float32));
  auto pools = indexer_pool_and_score(mx::reshape(all, {Ltot, 2 * D + 1}), q, weights, l.idx_ape);
  // visibility: causal (absolute positions) and valid, per (query row, key)
  std::vector<char> visible((size_t)L * Ltot, 0), row_visible((size_t)L, 1);
  for (int i = 0; i < L; ++i)
    for (int j = 0; j <= start_pos + i && j < Ltot; ++j) visible[(size_t)i * Ltot + j] = 1;
  mx::array sel = indexer_select(pools, visible, row_visible, cfg.index_topk,
                                 cfg.index_kpool_always_select_tail);
  return mx::reshape(sel, {1, L, (int)sel.shape(1)});
}

mx::array mla_layer(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                    MlaState& state, int start_pos, const mx::array& topk,
                    const mx::array& q_resid_in) {
  const int B = (int)h.shape(0), L = (int)h.shape(1);
  const int H = cfg.num_heads;
  const int kvr = (int)l.kv_b.in_features;              // kv_lora_rank (512)
  const int qk = (int)l.q_b.out_features / H;           // qk_head_dim (nope only: rope = 0)
  const int vd = (int)l.kv_b.out_features / H - qk;     // v_head_dim (256)

  // q path: q_a_proj -> q_a_layernorm -> q_b_proj (q_resid is shared with the
  // indexer, so the caller may pass the one it already used)
  mx::array q_resid = q_resid_in.size() > 1 ? q_resid_in : mla_q_resid(l, h);
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

// ---------------------------------------------------------------------------
// Batched decode (B sequences, one new row each)
// ---------------------------------------------------------------------------

namespace {

mx::array wrap_host(std::vector<float> v, mx::Shape sh) {
  auto* h = new std::vector<float>(std::move(v));
  return mx::array((void*)h->data(), sh, mx::float32, [h](void*) { delete h; });
}

// The pooling of indexer_pool_and_score, but over slot b's own range
// [T - len[b], T) of a (B, T, 2D+1) packed cache. All slots' pools live in one
// (B, nmax, kpool, D) pair of stacks; the rows of a slot's padding pools are
// zero/-inf and are never read back (n_b bounds every consumer).
struct BatchPools {
  mx::array keys = mx::array(0.0f);   // (B, nmax, D) f32
  mx::array scores = mx::array(0.0f); // (B, nmax) f32
  std::vector<std::vector<int32_t>> indices; // per slot: n_b * kpool CACHE row ids (-1 invalid)
  std::vector<std::vector<char>> valid;      // per slot: n_b
  int kpool = 0, nmax = 0;
};

BatchPools indexer_pool_and_score_batch(const mx::array& packed, const std::vector<int>& len,
                                        const mx::array& q, const mx::array& w,
                                        const mx::array& ape) {
  const int B = (int)packed.shape(0);
  const int T = (int)packed.shape(1);
  const int D = (int)ape.shape(1);
  const int kp = (int)ape.shape(0);
  const int Nh = (int)q.shape(1);
  if ((int)len.size() != B) throw std::runtime_error("GLM indexer batch: len/B mismatch");
  mx::eval(packed, q, w, ape);
  const float* pk = packed.data<float>();
  const float* ap = ape.data<float>();
  const size_t row = (size_t)(2 * D + 1);

  BatchPools out;
  out.kpool = kp;
  std::vector<int> base((size_t)B);
  int nmax = 1;
  for (int b = 0; b < B; ++b) {
    base[(size_t)b] = T - len[(size_t)b];
    nmax = std::max(nmax, (len[(size_t)b] + kp - 1) / kp);
  }
  out.nmax = nmax;
  const size_t np = (size_t)nmax;
  std::vector<float> gkey((size_t)B * np * kp * D, 0.0f);
  std::vector<float> glog((size_t)B * np * kp * D, -INFINITY);
  std::vector<float> gvalid((size_t)B * np * kp, 0.0f);
  out.indices.assign((size_t)B, {});
  out.valid.assign((size_t)B, {});
  for (int b = 0; b < B; ++b) {
    const int Ltot = len[(size_t)b];
    const size_t bs = (size_t)base[(size_t)b];
    auto key_at = [&](int t, int d) { return pk[(bs + (size_t)t) * row + (size_t)d]; };
    auto gate_at = [&](int t, int d) { return pk[(bs + (size_t)t) * row + (size_t)D + (size_t)d]; };
    auto valid_at = [&](int t) { return pk[(bs + (size_t)t) * row + (size_t)(2 * D)] > 0.5f; };
    int first_valid = Ltot;
    for (int t = 0; t < Ltot; ++t) {
      if (valid_at(t)) {
        first_valid = t;
        break;
      }
    }
    const int nb = (Ltot + kp - 1) / kp;
    out.indices[(size_t)b].assign((size_t)nb * kp, -1);
    out.valid[(size_t)b].assign((size_t)nb, 0);
    for (int p = 0; p < nb; ++p) {
      for (int s = 0; s < kp; ++s) {
        const int t = first_valid + p * kp + s;
        const bool ok = t < Ltot && valid_at(t);
        const size_t kk = (((size_t)b * np + (size_t)p) * kp + (size_t)s) * (size_t)D;
        if (ok) {
          out.indices[(size_t)b][(size_t)p * kp + (size_t)s] = base[(size_t)b] + t;
          for (int d = 0; d < D; ++d) {
            gkey[kk + (size_t)d] = key_at(t, d);
            glog[kk + (size_t)d] = gate_at(t, d) + ap[(size_t)s * D + (size_t)d];
          }
          gvalid[((size_t)b * np + (size_t)p) * kp + (size_t)s] = 1.0f;
        }
      }
      bool all = true;
      for (int s = 0; s < kp; ++s) all = all && gvalid[((size_t)b * np + (size_t)p) * kp + (size_t)s] > 0.5f;
      out.valid[(size_t)b][(size_t)p] = all ? 1 : 0;
    }
  }
  mx::array K = wrap_host(std::move(gkey), mx::Shape{B, (int)nmax, kp, D});
  mx::array Gt = wrap_host(std::move(glog), mx::Shape{B, (int)nmax, kp, D});
  mx::array probs = mx::softmax(Gt, 2);
  probs = mx::where(mx::isnan(probs), mx::zeros_like(probs), probs);
  out.keys = mx::sum(mx::multiply(probs, K), std::vector<int>{2}, false); // (B, nmax, D)
  mx::array qf = mx::astype(q, mx::float32);                              // (B, Nh, D)
  mx::array sc = mx::matmul(qf, mx::transpose(out.keys, {0, 2, 1}));      // (B, Nh, nmax)
  sc = mx::multiply(mx::maximum(sc, mx::zeros_like(sc)),
                    mx::array(1.0f / std::sqrt((float)D), mx::float32));
  mx::array sw = mx::matmul(mx::reshape(mx::astype(w, mx::float32), {B, 1, Nh}), sc); // (B,1,nmax)
  out.scores = mx::reshape(sw, {B, (int)nmax});
  return out;
}

// One query row's DSA selection when every row of its range is visible, i.e.
// indexer_select with the visibility test folded away, expressed in the row's
// own frame plus `base` (its block's offset in the batched cache). `indices` are
// CACHE row ids. Writes `index_topk + kpool - 1` entries into `out` (-1 absent).
void select_row_dense(const float* sc, int n_pools, const std::vector<int32_t>& indices,
                      const std::vector<char>& valid, int kp, int Ltot, int base, int index_topk,
                      bool always_select_tail, int32_t* out) {
  const int select_k = std::min(index_topk / std::max(1, kp), n_pools);
  const int tail_w = always_select_tail ? kp - 1 : 0;
  const int width = index_topk + tail_w;
  for (int i = 0; i < width; ++i) out[i] = -1;
  std::vector<std::pair<float, int>> ranked;
  ranked.reserve((size_t)n_pools);
  for (int p = 0; p < n_pools; ++p) {
    if (!valid[(size_t)p]) continue;
    const int end = indices[(size_t)p * kp + kp - 1];
    if (end < 0 || end >= base + Ltot) continue;
    ranked.push_back({sc[p], p});
  }
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const auto& a, const auto& b) { return a.first > b.first; });
  if ((int)ranked.size() > select_k) ranked.resize((size_t)select_k);
  int w = 0;
  for (auto& [_, p] : ranked) {
    for (int s = 0; s < kp && w < index_topk; ++s) {
      const int32_t t = indices[(size_t)p * kp + (size_t)s];
      if (t >= base && t < base + Ltot) out[w++] = t;
    }
  }
  if (!tail_w || Ltot <= 0) return;
  const int i0 = indices.empty() ? -1 : indices[0];
  const int first = i0 >= base ? i0 - base : 0;
  const int tail_count = Ltot % kp;
  const int tail_start = first + Ltot - tail_count;
  for (int j = 0; j < tail_w; ++j) {
    if (j >= tail_count || w >= width) break;
    const int t = tail_start + j;
    if (t >= 0 && t < Ltot) out[w++] = base + t;
  }
}

} // namespace

mx::array indexer_topk_batch(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                             const mx::array& q_resid, IndexerBatchCache& cache,
                             const std::vector<int>& len) {
  const int B = (int)h.shape(0);
  if ((int)len.size() != B) throw std::runtime_error("GLM indexer batch: len/B mismatch");
  const int Nh = cfg.index_n_heads;
  const int D = (int)l.idx_wk.shape(1); // index_head_dim
  const int kp = cfg.index_kpool;
  auto proj = [](const mx::array& w, const mx::array& x) { return mx::matmul(x, w); };
  mx::array q = mx::reshape(g5_linear(l.idx_wq_b, q_resid), {B, Nh, D});
  mx::array key = layer_norm(proj(l.idx_wk, h), l.idx_k_norm_w, l.idx_k_norm_b);
  mx::array gate = proj(l.idx_gate, h);
  mx::array valid = mx::ones(mx::Shape{B, 1, 1}, mx::float32);
  mx::array packed_new =
      mx::concatenate({mx::astype(key, mx::float32), mx::astype(gate, mx::float32), valid}, -1);
  cache.packed = cache.packed.size() > 1 ? mx::concatenate({cache.packed, packed_new}, 1)
                                         : packed_new;
  const int T = (int)cache.packed.shape(1);
  mx::array weights = mx::multiply(proj(l.idx_weights_proj, h),
                                   mx::array(1.0f / std::sqrt((float)Nh), mx::float32));
  BatchPools pools = indexer_pool_and_score_batch(cache.packed, len, q, mx::reshape(weights, {B, Nh}),
                                                  l.idx_ape);
  mx::eval(pools.scores);
  const float* sc = pools.scores.data<float>();
  const int tail_w = cfg.index_kpool_always_select_tail ? kp - 1 : 0;
  const int width = cfg.index_topk + tail_w;
  std::vector<int32_t> out((size_t)B * (size_t)width, -1);
  for (int b = 0; b < B; ++b) {
    select_row_dense(sc + (size_t)b * (size_t)pools.nmax, (int)pools.valid[(size_t)b].size(),
                     pools.indices[(size_t)b], pools.valid[(size_t)b], kp, len[(size_t)b],
                     T - len[(size_t)b], cfg.index_topk, cfg.index_kpool_always_select_tail,
                     out.data() + (size_t)b * (size_t)width);
  }
  auto* hp = new std::vector<int32_t>(std::move(out));
  return mx::array((void*)hp->data(), mx::Shape{B, 1, width}, mx::int32, [hp](void*) { delete hp; });
}

mx::array mla_layer_batch(const G5Layer& l, const ModelConfig& cfg, const mx::array& h,
                          MlaBatchCache& cache, const std::vector<int>& len, const mx::array& topk,
                          const mx::array& q_resid) {
  const int B = (int)h.shape(0);
  const int H = cfg.num_heads;
  const int kvr = (int)l.kv_b.in_features;        // kv_lora_rank (512)
  const int qk = (int)l.q_b.out_features / H;     // qk_head_dim (nope only: rope = 0)
  const int vd = (int)l.kv_b.out_features / H - qk;
  if ((int)len.size() != B) throw std::runtime_error("GLM MLA batch: len/B mismatch");

  mx::array q = mx::reshape(g5_linear(l.q_b, q_resid), {B, 1, H, qk});
  mx::array kv_raw = g5_linear(l.kv_a, h);
  mx::array k_pass = rms_norm(kv_raw, l.kv_a_norm);
  mx::array k_all = cache.k_pass.size() > 1
                        ? mx::concatenate({cache.k_pass, mx::astype(k_pass, mx::float32)}, 1)
                        : mx::astype(k_pass, mx::float32);
  cache.k_pass = k_all;
  const int T = (int)k_all.shape(1);

  mx::array kv = mx::reshape(g5_linear(l.kv_b, k_all), {B, T, H, qk + vd});
  mx::array k = mx::transpose(mx::slice(kv, {0, 0, 0, 0}, {B, T, H, qk}), {0, 2, 1, 3});
  mx::array v = mx::transpose(mx::slice(kv, {0, 0, 0, qk}, {B, T, H, qk + vd}), {0, 2, 1, 3});
  mx::array qh = mx::transpose(q, {0, 2, 1, 3}); // (B,H,1,qk)
  mx::array scores =
      mx::multiply(mx::matmul(mx::astype(qh, mx::float32),
                              mx::transpose(mx::astype(k, mx::float32), {0, 1, 3, 2})),
                   mx::array(1.0f / std::sqrt((float)qk), mx::float32)); // (B,H,1,T)

  // Mask: the DSA selection when the indexer ran, the slot's whole block
  // otherwise. A decode row of slot b sees exactly the rows indexer_select
  // picked, which for a context shorter than index_topk is all of them.
  std::vector<uint8_t> mb((size_t)B * (size_t)T, 0);
  if (topk.size() > 1) {
    const int K = (int)topk.shape(2);
    mx::eval(topk);
    const int32_t* tp = topk.data<int32_t>();
    for (int b = 0; b < B; ++b) {
      for (int j = 0; j < K; ++j) {
        const int32_t t = tp[(size_t)b * (size_t)K + (size_t)j];
        if (t >= 0 && t < T) mb[(size_t)b * (size_t)T + (size_t)t] = 1;
      }
    }
  } else {
    for (int b = 0; b < B; ++b)
      for (int t = T - len[(size_t)b]; t < T; ++t) mb[(size_t)b * (size_t)T + (size_t)t] = 1;
  }
  auto* mh = new std::vector<uint8_t>(std::move(mb));
  mx::array mask((void*)mh->data(), mx::Shape{B, 1, 1, T}, mx::bool_, [mh](void*) { delete mh; });
  scores = mx::where(mask, scores, mx::array(-INFINITY));
  mx::array probs = mx::softmax(scores, -1);
  mx::array out = mx::matmul(probs, mx::astype(v, mx::float32)); // (B,H,1,vd)
  out = mx::transpose(out, {0, 2, 1, 3});                       // (B,1,H,vd)
  out = mx::reshape(out, {B, 1, H * vd});
  return g5_linear(l.attn_o, mx::astype(out, h.dtype()));
}

} // namespace exl3
