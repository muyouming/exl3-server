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
    // decay is per (head, k-dim): g has shape (B, L, H, kd)
    mx::array g_i = mx::expand_dims(mx::squeeze(mx::slice(gf, {0, i, 0, 0}, {B, i + 1, H, kd}), 1),
                                    -1);
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

mx::array rms_norm_gated(const mx::array& x, const mx::array& weight, const mx::array& gate,
                         float eps) {
  mx::array xf = mx::astype(x, mx::float32);
  mx::array var = mx::mean(mx::multiply(xf, xf), std::vector<int>{-1}, true);
  mx::array y = mx::multiply(xf, mx::rsqrt(mx::add(var, mx::array(eps, mx::float32))));
  y = mx::multiply(y, mx::astype(weight, mx::float32));
  return mx::multiply(y, mx::sigmoid(mx::astype(gate, mx::float32)));
}

mx::array g5_linear(const G5Linear& lin, const mx::array& x) {
  mx::array y = lin.quantized ? trellis_linear(lin.tw, x) : mx::matmul(x, lin.w);
  if (lin.has_bias) y = mx::add(y, mx::astype(lin.bias, y.dtype()));
  return y;
}

mx::array kda_layer(const G5Layer& l, const ModelConfig& cfg, const mx::array& h, KdaState& state) {
  const int B = (int)h.shape(0), L = (int)h.shape(1);
  const int H = cfg.layer_plan.empty() ? 0 : (l.kda_q.out_features / 128);
  const int kd = 128;
  // 1) q,k,v projections and the causal conv over their concatenation
  mx::array mixed = mx::concatenate(
      {g5_linear(l.kda_q, h), g5_linear(l.kda_k, h), g5_linear(l.kda_v, h)}, -1); // (B,L,3*H*kd)
  const int C = (int)mixed.shape(2);
  const int klen = (int)l.conv1d_w.shape(l.conv1d_w.ndim() - 1);
  mx::array conv_in = mixed;
  if (state.conv_window.size() > 1) {
    conv_in = mx::concatenate({state.conv_window, mx::astype(mixed, mx::float32)}, 1);
  }
  state.conv_window =
      mx::slice(conv_in, {0, (int)conv_in.shape(1) - (klen - 1), 0}, {B, (int)conv_in.shape(1), C});
  mx::array conved = mx::astype(kda_conv1d(conv_in, l.conv1d_w, l.conv1d_b), mixed.dtype());
  conved = mx::slice(conved, {0, (int)conved.shape(1) - L, 0}, {B, (int)conved.shape(1), C});
  const int qkv = H * kd;
  auto part = [&](int which) {
    return mx::reshape(mx::slice(conved, {0, 0, which * qkv}, {B, L, (which + 1) * qkv}),
                       {B, L, H, kd});
  };
  mx::array q = part(0), k = part(1), v = part(2);
  // 2) forget-gate decay and beta
  mx::array fg = g5_linear(l.kda_f_b, g5_linear(l.kda_f_a, h)); // (B,L,H*kd)
  mx::array g = mx::reshape(
      kda_decay(fg, l.dt_bias, l.a_log), {B, L, H, kd});         // per (head,dim) decay
  mx::array beta = mx::reshape(mx::astype(g5_linear(l.kda_b, h), mx::float32), {B, L, H, 1});
  // 3) delta-rule recurrence
  mx::array core = kda_recurrence(q, k, v, g, beta, &state.recurrent); // (B,L,H,kd)
  // 4) output gate and projection
  mx::array gate = mx::reshape(g5_linear(l.kda_g_b, g5_linear(l.kda_g_a, h)), {B, L, H, kd});
  mx::array out = rms_norm_gated(core, l.o_norm, gate, 1e-6f);
  return g5_linear(l.kda_o, mx::reshape(out, {B, L, H * kd}));
}

} // namespace exl3
