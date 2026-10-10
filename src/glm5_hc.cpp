// GLM-5-Next mHC (hyper-connections). See glm5_hc.h for the reference line map.
#include "glm5_hc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace exl3 {

namespace {

// ---------------------------------------------------------------------------
// Fused hyper-connection kernel.
//
// The reference (and the MLX expression of it) is ~110 tiny operators per site:
// an RMSNorm, a skinny matmul, a dozen slices, sigmoid/softmax and a 20-iteration
// Sinkhorn on a tiny (N,N) matrix. On this machine every one of those operators
// costs several microseconds of dispatch, so 2 sites x 45 layers cost tens of
// milliseconds per token (measured: ~0.45-0.95 ms per site, of which ~65% is the
// Sinkhorn loop alone).
//
// This kernel computes the whole Glm5NextTextHyperConnection.forward (300-328) in
// ONE threadgroup per position: pass 1 reduces sum(x^2) over the N*D axis, pass 2
// gives every warp one of the (2+N)*N projection outputs, the scalar part does the
// pre/post/comb activations, softmax and the Sinkhorn with the reference's exact
// loop order, and pass 3 collapses the streams with `pre`.
//
// Numerics: the same formulas, the same order of operations and the same f32
// precision as the MLX expression (an f32 RMSNorm of the fp16 streams, f32
// accumulation, exp/rsqrt from metal_stdlib - MLX's sigmoid/softmax/rsqrt use the
// same intrinsics). Only the summation ORDER of the reductions differs, which is
// the fp32 rounding of ~16k additions.
// ---------------------------------------------------------------------------

const char* kHcSource = R"METAL(
  uint p = threadgroup_position_in_grid.x;
  uint tid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup;
  uint warp = tid / 32u;

  const uint nd = uint(ND);
  const uint off = p * nd;

  threadgroup float red[NWARPS];
  threadgroup float sm_w[NW];
  threadgroup float sm_pre[HC_N];
  threadgroup float sm_post[HC_N];

  // --- pass 1: RMSNorm denominator ----------------------------------------
  float ss = 0.0f;
  for (uint i = tid; i < nd; i += uint(TGSIZE)) {
    float v = float(streams[off + i]);
    ss = fma(v, v, ss);
  }
  ss += simd_shuffle_xor(ss, ushort(16));
  ss += simd_shuffle_xor(ss, ushort(8));
  ss += simd_shuffle_xor(ss, ushort(4));
  ss += simd_shuffle_xor(ss, ushort(2));
  ss += simd_shuffle_xor(ss, ushort(1));
  if (lane == 0u) red[warp] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint w = 0u; w < uint(NWARPS); ++w) total += red[w];
  float rms = rsqrt(total / float(nd) + float(EPS));

  // --- pass 2: one projection output per warp ------------------------------
  if (warp < uint(NW)) {
    const device float* fj = fn + warp * nd;
    float d = 0.0f;
    for (uint i = lane; i < nd; i += 32u) d = fma(float(streams[off + i]) * rms, fj[i], d);
    d += simd_shuffle_xor(d, ushort(16));
    d += simd_shuffle_xor(d, ushort(8));
    d += simd_shuffle_xor(d, ushort(4));
    d += simd_shuffle_xor(d, ushort(2));
    d += simd_shuffle_xor(d, ushort(1));
    if (lane == 0u) sm_w[warp] = d;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- scalar part: pre/post/comb, softmax, Sinkhorn (single thread) -------
  if (tid == 0u) {
    float ww[NW];
    for (uint j = 0u; j < uint(NW); ++j) ww[j] = sm_w[j];
    for (uint n = 0u; n < uint(HC_N); ++n) {
      sm_pre[n] = 1.0f / (1.0f + exp(-(ww[n] * scale[0] + base[n]))) + float(EPS);
      sm_post[n] = 2.0f * (1.0f / (1.0f + exp(-(ww[uint(HC_N) + n] * scale[1] + base[uint(HC_N) + n]))));
      post[p * uint(HC_N) + n] = sm_post[n];
    }
    float c[HC_N * HC_N];
    for (uint i = 0u; i < uint(HC_N); ++i) {
      float m = -INFINITY;
      for (uint j = 0u; j < uint(HC_N); ++j) {
        float z = ww[2u * uint(HC_N) + i * uint(HC_N) + j] * scale[2] +
                  base[2u * uint(HC_N) + i * uint(HC_N) + j];
        c[i * uint(HC_N) + j] = z;
        m = max(m, z);
      }
      float s = 0.0f;
      for (uint j = 0u; j < uint(HC_N); ++j) {
        c[i * uint(HC_N) + j] = exp(c[i * uint(HC_N) + j] - m);
        s += c[i * uint(HC_N) + j];
      }
      for (uint j = 0u; j < uint(HC_N); ++j) c[i * uint(HC_N) + j] = c[i * uint(HC_N) + j] / s + float(EPS);
    }
    // Sinkhorn: column normalise, then (ITERS-1) x (row, column) normalises.
    for (uint j = 0u; j < uint(HC_N); ++j) {
      float cs = 0.0f;
      for (uint i = 0u; i < uint(HC_N); ++i) cs += c[i * uint(HC_N) + j];
      for (uint i = 0u; i < uint(HC_N); ++i) c[i * uint(HC_N) + j] = c[i * uint(HC_N) + j] / (cs + float(EPS));
    }
    for (uint it = 0u; it < uint(ITERS - 1); ++it) {
      for (uint i = 0u; i < uint(HC_N); ++i) {
        float rs = 0.0f;
        for (uint j = 0u; j < uint(HC_N); ++j) rs += c[i * uint(HC_N) + j];
        for (uint j = 0u; j < uint(HC_N); ++j) c[i * uint(HC_N) + j] = c[i * uint(HC_N) + j] / (rs + float(EPS));
      }
      for (uint j = 0u; j < uint(HC_N); ++j) {
        float cs = 0.0f;
        for (uint i = 0u; i < uint(HC_N); ++i) cs += c[i * uint(HC_N) + j];
        for (uint i = 0u; i < uint(HC_N); ++i) c[i * uint(HC_N) + j] = c[i * uint(HC_N) + j] / (cs + float(EPS));
      }
    }
    for (uint k = 0u; k < uint(HC_N * HC_N); ++k) comb[p * uint(HC_N * HC_N) + k] = c[k];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // --- pass 3: collapsed = sum_n pre[n] * streams[n] -----------------------
  for (uint dd = tid; dd < uint(DIM); dd += uint(TGSIZE)) {
    float a = 0.0f;
    for (uint n = 0u; n < uint(HC_N); ++n) {
      a = fma(sm_pre[n], float(streams[off + n * uint(DIM) + dd]), a);
    }
    collapsed[p * uint(DIM) + dd] = XTOD(a);
  }
)METAL";

struct HcKernelKey {
  int n, dim, iters;
  float eps;
  bool half_streams;
  bool operator==(const HcKernelKey& o) const {
    return n == o.n && dim == o.dim && iters == o.iters && eps == o.eps &&
           half_streams == o.half_streams;
  }
};

// The key doubles as the Metal kernel name, so it must be a valid identifier
// (eps goes in as its raw bit pattern).
std::string hc_key(const HcKernelKey& k) {
  uint32_t bits = 0;
  std::memcpy(&bits, &k.eps, sizeof(bits));
  char buf[96];
  std::snprintf(buf, sizeof(buf), "hc_%d_%d_%d_%u_%d", k.n, k.dim, k.iters, bits,
                (int)k.half_streams);
  return buf;
}

std::unordered_map<std::string, mx::fast::CustomKernelFunction> g_hc_kernels;
std::mutex g_hc_mutex;

const mx::fast::CustomKernelFunction& hc_kernel_for(const HcKernelKey& k) {
  const std::string key = hc_key(k);
  auto it = g_hc_kernels.find(key);
  if (it != g_hc_kernels.end()) return it->second;
  const int W = (2 + k.n) * k.n;
  char hdr[256];
  std::snprintf(hdr, sizeof(hdr),
                "#include <metal_stdlib>\nusing namespace metal;\n"
                "#define HC_N %d\n#define NW %d\n#define NWARPS %d\n#define TGSIZE %d\n"
                "#define DIM %d\n#define ND %d\n#define ITERS %d\n#define EPS %af\n"
                "#define XTOD(v) %s\n",
                k.n, W, W, W * 32, k.dim, k.n * k.dim, k.iters, (double)k.eps,
                k.half_streams ? "half(v)" : "(v)");
  std::string header = hdr;
  char name[64];
  std::snprintf(name, sizeof(name), "exl3_%s", key.c_str());
  auto fn = mx::fast::metal_kernel(name, {"streams", "fn", "base", "scale"},
                                   {"collapsed", "post", "comb"}, kHcSource, header, true, false,
                                   mx::CompileOptions{});
  return g_hc_kernels.emplace(key, std::move(fn)).first->second;
}

bool hc_fused_enabled() {
  static const bool on = std::getenv("EXL3_HC_MLX") == nullptr;
  return on;
}

} // namespace

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
  const int W = (2 + N) * N;

  // The fused path needs one warp per projection output, fp16 streams and the
  // f32 parameters exactly as the loader stores them.
  const bool half_st = streams.dtype() == mx::float16;
  if (hc_fused_enabled() && (half_st || streams.dtype() == mx::float32) && N <= 4 &&
      W * 32 <= 1024 &&
      (int64_t)fn.size() == (int64_t)W * N * D && fn.dtype() == mx::float32 &&
      base.dtype() == mx::float32 && (int64_t)base.size() == W && scale.dtype() == mx::float32 &&
      (int64_t)scale.size() == 3) {
    mx::Shape lead(shp.begin(), shp.begin() + (nd - 2));
    mx::Shape post_shape = lead;
    post_shape.push_back(N);
    mx::Shape comb_shape = post_shape;
    comb_shape.push_back(N);
    mx::Shape coll_shape = lead;
    coll_shape.push_back(D);

    const HcKernelKey key{N, D, cfg.sinkhorn_iters, cfg.eps, half_st};
    std::vector<mx::array> outs;
    {
      std::lock_guard<std::mutex> lk(g_hc_mutex); // kernel map only, never held while launching
      const auto& kernel = hc_kernel_for(key);
      const int tgs = W * 32;
      outs = kernel({streams, fn, base, scale}, {coll_shape, post_shape, comb_shape},
                    {streams.dtype(), mx::float32, mx::float32}, {P * tgs, 1, 1}, {tgs, 1, 1},
                    std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                    false, mx::default_device());
    }
    G5HcActivation out;
    out.collapsed = outs[0];
    out.post = outs[1];
    out.comb = outs[2];
    return out;
  }

  // 300-301: flatten the streams and unweighted-RMSNorm over the N*D axis.
  mx::array x = mx::reshape(mx::astype(streams, mx::float32), mx::Shape{P, N * D});
  mx::array ms = mx::mean(mx::multiply(x, x), std::vector<int>{-1}, true);
  x = mx::multiply(x, mx::rsqrt(mx::add(ms, mx::array(cfg.eps))));

  // 303: F.linear(x, fn) == x @ fn.T  (fn is stored (out, in), unlike G5Linear::w)
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
