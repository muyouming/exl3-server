// GLM-5-Next mHC (hyper-connections). See glm5_hc.h for the reference line map.
#include "glm5_hc.h"

#include <stdexcept>
#include <vector>

namespace exl3 {

G5HcConfig g5_hc_config(const ModelConfig& cfg) {
  G5HcConfig hc;
  hc.mult = cfg.hc_mult;
  hc.sinkhorn_iters = cfg.hc_sinkhorn_iters;
  hc.eps = cfg.hc_eps;
  return hc;
}

// reference 321-324
mx::array g5_hc_sinkhorn(const mx::array& comb, int iters, float eps) {
  mx::array c = mx::divide(comb, mx::add(mx::sum(comb, std::vector<int>{-2}, true), mx::array(eps)));
  for (int i = 0; i < iters - 1; ++i) {
    c = mx::divide(c, mx::add(mx::sum(c, std::vector<int>{-1}, true), mx::array(eps)));
    c = mx::divide(c, mx::add(mx::sum(c, std::vector<int>{-2}, true), mx::array(eps)));
  }
  return c;
}

G5HcActivation g5_hc_prepare(const mx::array& streams, const mx::array& base,
                             const mx::array& fn, const mx::array& scale,
                             const G5HcConfig& cfg) {
  const mx::Shape shp = streams.shape();
  const int nd = (int)shp.size();
  if (nd < 2) throw std::runtime_error("g5_hc_prepare: streams must be [..., N, D]");
  const int N = cfg.mult;
  const int D = (int)shp[nd - 1];
  if (N <= 0) throw std::runtime_error("g5_hc_prepare: hc_mult must be positive");
  int P = 1;
  for (int i = 0; i < nd - 2; ++i) P *= (int)shp[i];

  // 300-301: flatten the streams and unweighted-RMSNorm over the N*D axis.
  mx::array x = mx::reshape(mx::astype(streams, mx::float32), mx::Shape{P, N * D});
  mx::array ms = mx::mean(mx::multiply(x, x), std::vector<int>{-1}, true);
  x = mx::multiply(x, mx::rsqrt(mx::add(ms, mx::array(cfg.eps))));

  // 303: F.linear(x, fn) == x @ fn.T  (fn is stored (out, in), unlike G5Linear::w)
  const int W = (2 + N) * N;
  mx::array w = mx::matmul(x, mx::transpose(mx::astype(fn, mx::float32))); // (P, W)

  // 305-310: split the projection output and the biases; comb is a matrix.
  mx::array pre_w = mx::slice(w, {0, 0}, {P, N});
  mx::array post_w = mx::slice(w, {0, N}, {P, 2 * N});
  mx::array comb_w = mx::reshape(mx::slice(w, {0, 2 * N}, {P, W}), {P, N, N});
  mx::array b32 = mx::astype(base, mx::float32);
  mx::array pre_b = mx::slice(b32, {0}, {N});
  mx::array post_b = mx::slice(b32, {N}, {2 * N});
  mx::array comb_b = mx::reshape(mx::slice(b32, {2 * N}, {W}), {N, N});
  mx::array s32 = mx::astype(scale, mx::float32);
  mx::array pre_s = mx::slice(s32, {0}, {1});
  mx::array post_s = mx::slice(s32, {1}, {2});
  mx::array comb_s = mx::slice(s32, {2}, {3});

  // 313-315
  mx::array pre = mx::add(mx::sigmoid(mx::add(mx::multiply(pre_w, pre_s), pre_b)), mx::array(cfg.eps));
  mx::array post =
      mx::multiply(mx::array(2.0f), mx::sigmoid(mx::add(mx::multiply(post_w, post_s), post_b)));
  mx::array comb = mx::add(
      mx::softmax(mx::add(mx::multiply(comb_w, comb_s), comb_b), -1), mx::array(cfg.eps));
  // 321-324
  comb = g5_hc_sinkhorn(comb, cfg.sinkhorn_iters, cfg.eps);

  // Restore the leading dims on the returned weights.
  mx::Shape lead(shp.begin(), shp.begin() + (nd - 2));
  mx::Shape post_shape = lead;
  post_shape.push_back(N);
  mx::Shape comb_shape = post_shape;
  comb_shape.push_back(N);
  mx::Shape pre4 = post_shape;
  pre4.push_back(1);

  G5HcActivation out;
  out.post = mx::reshape(post, post_shape);
  out.comb = mx::reshape(comb, comb_shape);
  // 328: (pre.unsqueeze(-1) * streams).sum(dim=-2), back to the streams dtype.
  mx::array collapsed =
      mx::sum(mx::multiply(mx::reshape(pre, pre4), mx::astype(streams, mx::float32)),
              std::vector<int>{nd - 2}, false);
  out.collapsed = mx::astype(collapsed, streams.dtype());
  return out;
}

mx::array g5_hc_mix(const mx::array& block_out, const mx::array& post,
                    const mx::array& comb, const mx::array& residual) {
  const mx::Shape shp = residual.shape();
  const int nd = (int)shp.size();
  if (nd < 2) throw std::runtime_error("g5_hc_mix: residual must be [..., N, D]");
  const mx::Dtype dt = block_out.dtype();
  mx::Shape post_shape(shp.begin(), shp.begin() + (nd - 1)); // lead + N
  // reference 1335-1337: post.unsqueeze(-1) * out.unsqueeze(-2) + comb^T @ residual
  mx::array p = mx::expand_dims(mx::reshape(mx::astype(post, dt), post_shape), -1);
  mx::array b = mx::expand_dims(mx::astype(block_out, dt), -2);
  mx::array c = mx::astype(comb, dt);
  return mx::add(mx::multiply(p, b),
                 mx::matmul(mx::swapaxes(c, -1, -2), mx::astype(residual, dt)));
}

mx::array g5_hc_expand(const mx::array& x, int mult) {
  const mx::Shape shp = x.shape();
  if (shp.empty()) throw std::runtime_error("g5_hc_expand: x must be [..., D]");
  mx::Shape target;
  for (int i = 0; i + 1 < (int)shp.size(); ++i) target.push_back(shp[i]);
  target.push_back(mult); // ..., mult, D
  target.push_back(shp[shp.size() - 1]);
  // materialised, like the reference's `.expand(...).contiguous()` (1493)
  return mx::contiguous(mx::broadcast_to(mx::expand_dims(x, -2), target));
}

mx::array g5_hc_head(const mx::array& streams) {
  const int nd = streams.ndim();
  if (nd < 3) throw std::runtime_error("g5_hc_head: streams must be [..., N, D]");
  return mx::mean(streams, std::vector<int>{nd - 2}, false);
}

} // namespace exl3
