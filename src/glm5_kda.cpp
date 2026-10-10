#include "glm5_kda.h"

#include <cmath>

#include "util.h"

namespace exl3 {

mx::array kda_l2norm(const mx::array& x, float eps) {
  mx::array xf = mx::astype(x, mx::float32);
  mx::array sq = mx::sum(mx::multiply(xf, xf), std::vector<int>{-1}, true);
  mx::array inv = mx::rsqrt(mx::add(sq, mx::array(eps, mx::float32)));
  return mx::multiply(xf, inv);
}

mx::array kda_decay(const mx::array& fg, const mx::array& dt_bias, const mx::array& a_log) {
  mx::array g = mx::add(mx::astype(fg, mx::float32), mx::reshape(mx::astype(dt_bias, mx::float32),
                                                                {1, 1, -1}));
  // softplus with the reference's overflow guard: softplus(x) = x for x > 20
  mx::array sp = mx::where(mx::greater(g, mx::array(20.0f, mx::float32)),
                           g, mx::log(mx::add(mx::exp(g), mx::array(1.0f, mx::float32))));
  mx::array decay = mx::exp(mx::reshape(mx::astype(a_log, mx::float32), {1, 1, -1}));
  return mx::negative(mx::multiply(decay, sp));
}

mx::array kda_conv1d(const mx::array& x, const mx::array& w, const mx::array& b) {
  // x (B, L, C); convolve over L with a depthwise kernel (C, k)
  const int L = (int)x.shape(1), C = (int)x.shape(2);
  const int k = (int)w.shape(w.ndim() - 1);
  mx::array x32 = mx::astype(x, mx::float32);
  std::vector<mx::array> taps;
  taps.reserve((size_t)k);
  for (int t = 0; t < k; ++t) {
    // x shifted so that tap t multiplies x[i - (k-1-t)]
    const int shift = k - 1 - t;
    mx::array shifted = mx::zeros(mx::Shape{1, 1, C}, mx::float32);
    if (shift == 0) {
      shifted = x32;
    } else {
      mx::array head = mx::zeros(mx::Shape{1, shift, C}, mx::float32);
      mx::array body = mx::slice(x32, {0, 0, 0}, {(int)x.shape(0), L - shift, C});
      shifted = mx::concatenate({head, body}, 1);
    }
    mx::array wc = mx::astype(mx::reshape(w, {C, k}), mx::float32);
    mx::array wt = mx::slice(wc, {0, t}, {C, t + 1});
    taps.push_back(mx::multiply(shifted, mx::reshape(wt, {1, 1, C})));
  }
  mx::array acc = taps[0];
  for (size_t i = 1; i < taps.size(); ++i) acc = mx::add(acc, taps[i]);
  if (b.size() > 0) acc = mx::add(acc, mx::reshape(mx::astype(b, mx::float32), {1, 1, C}));
  // SiLU
  return mx::multiply(acc, mx::sigmoid(acc));
}

mx::array kda_recurrence(const mx::array& q, const mx::array& k, const mx::array& v,
                         const mx::array& g, const mx::array& beta, mx::array* state) {
  const int L = (int)q.shape(1);
  const int H = (int)q.shape(2);
  const int kd = (int)q.shape(3);
  const int vd = (int)v.shape(3);
  mx::array qf = mx::multiply(kda_l2norm(q), mx::array(1.0f / std::sqrt((float)kd), mx::float32));
  mx::array kf = kda_l2norm(k);
  mx::array vf = mx::astype(v, mx::float32);
  mx::array gf = mx::astype(g, mx::float32);
  mx::array bf = mx::astype(beta, mx::float32);

  mx::array S = state && state->size() > 1
                    ? *state
                    : mx::zeros(mx::Shape{(int)q.shape(0), H, kd, vd}, mx::float32);
  std::vector<mx::array> outs;
  outs.reserve((size_t)L);
  for (int i = 0; i < L; ++i) {
    const int B = (int)q.shape(0);
    mx::array q_i = mx::squeeze(mx::slice(qf, {0, i, 0, 0}, {B, i + 1, H, kd}), 1);
    mx::array k_i = mx::squeeze(mx::slice(kf, {0, i, 0, 0}, {B, i + 1, H, kd}), 1);
    mx::array v_i = mx::squeeze(mx::slice(vf, {0, i, 0, 0}, {B, i + 1, H, vd}), 1);
    mx::array g_i = mx::reshape(mx::squeeze(mx::slice(gf, {0, i, 0, 0}, {B, i + 1, H, 1}), 1),
                                {B, H, 1, 1});
    mx::array b_i = mx::squeeze(mx::slice(bf, {0, i, 0, 0}, {B, i + 1, H, 1}), 1);
    S = mx::multiply(S, mx::exp(g_i));
    mx::array kv = mx::sum(mx::multiply(S, mx::expand_dims(k_i, -1)), std::vector<int>{-2}, false);
    mx::array delta = mx::multiply(mx::subtract(v_i, kv), b_i);
    S = mx::add(S, mx::multiply(mx::expand_dims(k_i, -1), mx::expand_dims(delta, -2)));
    outs.push_back(mx::sum(mx::multiply(S, mx::expand_dims(q_i, -1)), std::vector<int>{-2}, false));
  }
  if (state) *state = S;
  mx::array out = outs[0];
  for (size_t i = 1; i < outs.size(); ++i) out = mx::concatenate({out, outs[i]}, 1);
  return out;
}

} // namespace exl3
