// GLM-5-Next mHC (hyper-connections) verification against a scalar CPU oracle
// transcribed line-by-line from the HF reference
// (modeling_glm5_next.py: Glm5NextTextHyperConnection.forward 300-328, the
// (row,col) Sinkhorn loop 321-324, the pre-collapse 328; Glm5NextTextDecoderLayer
// mix 1335-1337; HyperHead mean 335-336; the model-level stream expansion 1493).
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/hc_test \
//     tests/hc_test.cpp src/glm5_hc.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib
//
// Checks:
//   (a) Sinkhorn makes a positive (N,N) matrix doubly stochastic after EXACTLY
//       hc_sinkhorn_iters (20) iterations: |row_sum-1| and |col_sum-1| <= 1e-5.
//       Iteration count / loop order / eps placement are load-bearing: 1 iteration
//       leaves the row sums far from 1, 3 vs 4 iterations differ, and the result
//       is compared against a double-precision scalar Sinkhorn transcription at
//       3 iterations (also with a large eps). A different count therefore fails.
//   (b) post/comb/collapsed + the stream mix agree with an independent scalar
//       oracle on a hand-computed case: hc_mult 4, hidden 4, 2 positions, run
//       with the checkpoint eps (1e-6) and a large eps (1e-3) so that every eps
//       in the reference is pinned.
//   (c) expand/head/mix shapes and the expand->mix identity roundtrip.
//
// Expected (verified, M1-class Mac): "hc (hyper-connections): PASS" with
//   (a) |rowsum-1|/|colsum-1| ~1e-06 at 20 iters, ~0.3 at 1 iter, 3v4 ~6e-4,
//       3-iter vs scalar ~4e-08;
//   (b) max|diff| < 4e-07 for post/comb/collapsed/mixed (both eps);
//   (c) expand copies exact, head/mix max|diff| <= 1.2e-07.
// Mutation-tested: swapping the row/column order, ignoring the iteration count,
// dropping eps from any denominator, skipping the input RMSNorm, using mean
// instead of the pre-weighted sum, skipping Sinkhorn, softmax on the wrong axis,
// no comb^T, post without x2, and expanding on the wrong axis all make it FAIL.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "glm5_hc.h"

namespace mx = mlx::core;

// MLX arrays over heap buffers (the deleter keeps them alive; never a read-only
// mmap view).
static mx::array wrap(const std::vector<float>& v, mx::Shape sh) {
  auto* h = new std::vector<float>(v);
  return mx::array((void*)h->data(), std::move(sh), mx::float32, [h](void*) { delete h; });
}
static std::vector<float> host(const mx::array& a) {
  mx::array c = mx::contiguous(mx::astype(a, mx::float32)); // broadcast views are strided
  mx::eval(c);
  const float* p = c.data<float>();
  return std::vector<float>(p, p + c.size());
}
static std::vector<int> ishape(const mx::array& a) {
  std::vector<int> s;
  const int nd = (int)a.ndim();
  for (int i = 0; i < nd; ++i) s.push_back(a.shape(i));
  return s;
}

// 321-324 in double precision: column-normalise, then (iters-1) x (row, column),
// each division by (sum + eps). Used by both the Sinkhorn and the oracle checks.
static void scalar_sinkhorn(std::vector<double>& m, int N, int iters, double eps) {
  std::vector<double> cs((size_t)N);
  for (int j = 0; j < N; ++j) cs[(size_t)j] = 0.0;
  for (int r = 0; r < N; ++r)
    for (int j = 0; j < N; ++j) cs[(size_t)j] += m[(size_t)r * N + j];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) m[(size_t)i * N + j] /= cs[(size_t)j] + eps;
  for (int it = 0; it < iters - 1; ++it) {
    for (int i = 0; i < N; ++i) {
      double rs = 0;
      for (int j = 0; j < N; ++j) rs += m[(size_t)i * N + j];
      for (int j = 0; j < N; ++j) m[(size_t)i * N + j] /= rs + eps;
    }
    for (int j = 0; j < N; ++j) cs[(size_t)j] = 0.0;
    for (int r = 0; r < N; ++r)
      for (int j = 0; j < N; ++j) cs[(size_t)j] += m[(size_t)r * N + j];
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) m[(size_t)i * N + j] /= cs[(size_t)j] + eps;
  }
}

// ---------------------------------------------------------------- (a) Sinkhorn
static bool test_sinkhorn() {
  const int N = 4;         // hc_mult
  const int ITERS = 20;    // hc_sinkhorn_iters
  const int L = 2;         // two independent matrices
  const float EPS = 1e-6f; // hc_eps
  std::mt19937 rng(20261010);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::uniform_real_distribution<float> lg(-3.0f, 3.0f);
  bool all = true;
  // kind 0: raw random positive entries; kind 1: the matrix the layer actually
  // feeds in, softmax(w, dim=-1) + hc_eps (ref 315).
  for (int kind = 0; kind < 2; ++kind) {
    std::vector<float> comb((size_t)L * N * N);
    for (int l = 0; l < L; ++l)
      for (int i = 0; i < N; ++i) {
        float row[4], m = -1e30f;
        for (int j = 0; j < N; ++j) { row[j] = lg(rng); m = std::fmax(m, row[j]); }
        double s = 0;
        for (int j = 0; j < N; ++j) s += std::exp((double)row[j] - m);
        for (int j = 0; j < N; ++j) {
          size_t k = ((size_t)l * N + i) * N + j;
          comb[k] = (kind == 0) ? 0.05f + u(rng) : (float)(std::exp((double)row[j] - m) / s) + EPS;
        }
      }
    mx::array C = wrap(comb, {L, N, N});
    mx::array S20 = exl3::g5_hc_sinkhorn(C, ITERS, EPS);
    mx::array S19 = exl3::g5_hc_sinkhorn(C, ITERS - 1, EPS);
    mx::array S4 = exl3::g5_hc_sinkhorn(C, 4, EPS);
    mx::array S3 = exl3::g5_hc_sinkhorn(C, 3, EPS);
    mx::array S1 = exl3::g5_hc_sinkhorn(C, 1, EPS);
    mx::eval(S20, S19, S4, S3, S1);
    std::vector<float> h20 = host(S20), h19 = host(S19), h4 = host(S4), h3 = host(S3),
                       h1 = host(S1);
    double maxrow = 0, maxcol = 0, maxrow1 = 0, maxd = 0, max3 = 0, max34 = 0;
    for (int l = 0; l < L; ++l) {
      for (int i = 0; i < N; ++i) {
        double rs = 0;
        for (int j = 0; j < N; ++j) rs += h20[((size_t)l * N + i) * N + j];
        maxrow = std::fmax(maxrow, std::fabs(rs - 1.0));
        double rs1 = 0;
        for (int j = 0; j < N; ++j) rs1 += h1[((size_t)l * N + i) * N + j];
        maxrow1 = std::fmax(maxrow1, std::fabs(rs1 - 1.0));
      }
      for (int j = 0; j < N; ++j) {
        double cs = 0;
        for (int i = 0; i < N; ++i) cs += h20[((size_t)l * N + i) * N + j];
        maxcol = std::fmax(maxcol, std::fabs(cs - 1.0));
      }
      // The iteration count must enter the computation: at a low count the
      // matrices are far from the fixed point, so 3 vs 4 iterations differ a lot
      // (19 vs 20 only differ by the not-yet-converged residual, printed for
      // information - once converged the last pass is a no-op in fp32).
      for (int k = 0; k < N * N; ++k) {
        size_t idx = (size_t)l * N * N + k;
        maxd = std::fmax(maxd, std::fabs((double)h20[idx] - h19[idx]));
        max34 = std::fmax(max34, std::fabs((double)h4[idx] - h3[idx]));
      }
      // 3 iterations vs the scalar transcription: pins the loop body/order, which
      // the doubly-stochastic property alone cannot (all orders reach the same limit)
      std::vector<double> ref3(comb.begin() + (size_t)l * N * N, comb.begin() + (size_t)(l + 1) * N * N);
      scalar_sinkhorn(ref3, N, 3, EPS);
      for (int k = 0; k < N * N; ++k)
        max3 = std::fmax(max3, std::fabs((double)h3[(size_t)l * N * N + k] - ref3[(size_t)k]));
    }
    std::printf("(a) sinkhorn[%s] N=%d iters=%d: max|rowsum-1|=%.3g max|colsum-1|=%.3g "
                "(1 iter: max|rowsum-1|=%.3g) 3 vs 4 iters=%.3g (20-19=%.3g) "
                "3-iter vs scalar=%.3g\n",
                kind ? "softmax+eps" : "random>0", N, ITERS, maxrow, maxcol, maxrow1, max34, maxd,
                max3);
    if (kind == 0)
      std::printf("    comb[0] = %.6f %.6f %.6f %.6f (row sum=%.7f)\n", h20[0], h20[1], h20[2],
                  h20[3], (double)h20[0] + h20[1] + h20[2] + h20[3]);
    all &= maxrow <= 1e-5 && maxcol <= 1e-5 && maxrow1 > 1e-3 && max34 > 1e-4 && max3 < 1e-6;
  }
  // Large eps: pins that hc_eps really is added to EVERY normalisation
  // denominator (the reference adds it to all of them, 321-324).
  {
    const float E2 = 0.05f;
    std::vector<float> c2((size_t)N * N);
    for (auto& v : c2) v = 0.05f + u(rng);
    mx::array C2 = wrap(c2, {N, N});
    mx::array S3 = exl3::g5_hc_sinkhorn(C2, 3, E2);
    mx::eval(S3);
    std::vector<float> h3 = host(S3);
    std::vector<double> ref(c2.begin(), c2.end());
    scalar_sinkhorn(ref, N, 3, E2);
    double d = 0;
    for (int k = 0; k < N * N; ++k) d = std::fmax(d, std::fabs((double)h3[k] - ref[(size_t)k]));
    std::printf("(a) sinkhorn 3 iters with eps=%.2g vs scalar: max|diff|=%.3g\n", E2, d);
    all &= d < 1e-6;
  }
  std::printf("(a) doubly stochastic after exactly %d iters; loop order, iteration count and eps "
              "handling are load-bearing: %s\n",
              ITERS, all ? "PASS" : "FAIL");
  return all;
}

// ------------------------------------------------------------------- (b) oracle
// Scalar transcription of Glm5NextTextHyperConnection.forward (285-329) plus the
// decoder-layer mix (1335-1337). Double precision, independent of glm5_hc.cpp.
struct Oracle {
  std::vector<double> post, comb, collapsed, mixed;
};
static Oracle oracle_hc(const std::vector<double>& streams, const std::vector<double>& fn,
                        const std::vector<double>& base, const std::vector<double>& scale,
                        const std::vector<double>& bo, int N, int D, double eps, int iters) {
  const int F = N * D;
  Oracle o;
  o.post.assign(N, 0.0);
  o.comb.assign((size_t)N * N, 0.0);
  o.collapsed.assign(D, 0.0);
  o.mixed.assign((size_t)N * D, 0.0);
  // 300: flattened = streams.view(-1).float()
  std::vector<double> flat(F);
  for (int k = 0; k < F; ++k) flat[k] = streams[k];
  // 301 + 209-214: unweighted RMSNorm over the flattened stream axis
  double ms = 0;
  for (int k = 0; k < F; ++k) ms += flat[k] * flat[k];
  ms /= F;
  for (int k = 0; k < F; ++k) flat[k] = flat[k] / std::sqrt(ms + eps);
  // 303: F.linear(flat, fn) = flat @ fn.T
  std::vector<double> w(24, 0.0);
  for (int j = 0; j < 24; ++j)
    for (int k = 0; k < F; ++k) w[j] += flat[k] * fn[(size_t)j * F + k];
  // 305-310: split; comb is (N,N) row-major over the last N*N weights
  std::vector<double> pre_w(N), post_w(N), pre(N);
  std::vector<double> cw((size_t)N * N);
  for (int j = 0; j < N; ++j) { pre_w[j] = w[j]; }
  for (int j = 0; j < N; ++j) { post_w[j] = w[N + j]; }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) cw[(size_t)i * N + j] = w[2 * N + i * N + j];
  // 313-315
  for (int j = 0; j < N; ++j) {
    pre[j] = 1.0 / (1.0 + std::exp(-(pre_w[j] * scale[0] + base[j]))) + eps;
    o.post[j] = 2.0 / (1.0 + std::exp(-(post_w[j] * scale[1] + base[N + j])));
  }
  for (int i = 0; i < N; ++i) { // softmax over the last dim
    double m = -1e30, s = 0;
    for (int j = 0; j < N; ++j) m = std::fmax(m, cw[(size_t)i * N + j] * scale[2] + base[2 * N + i * N + j]);
    for (int j = 0; j < N; ++j) s += std::exp(cw[(size_t)i * N + j] * scale[2] + base[2 * N + i * N + j] - m);
    for (int j = 0; j < N; ++j)
      o.comb[(size_t)i * N + j] =
          std::exp(cw[(size_t)i * N + j] * scale[2] + base[2 * N + i * N + j] - m) / s + eps;
  }
  // 321-324 (shared double-precision transcription)
  scalar_sinkhorn(o.comb, N, iters, eps);
  // 328: collapsed = (pre[:,None] * streams).sum(stream axis)
  for (int d = 0; d < D; ++d) {
    double s = 0;
    for (int i = 0; i < N; ++i) s += pre[i] * streams[(size_t)i * D + d];
    o.collapsed[d] = s;
  }
  // 1335-1337: out[i] = post[i]*bo + sum_j comb[j][i]*res[j]
  for (int i = 0; i < N; ++i)
    for (int d = 0; d < D; ++d) {
      double s = o.post[i] * bo[d];
      for (int j = 0; j < N; ++j) s += o.comb[(size_t)j * N + i] * streams[(size_t)j * D + d];
      o.mixed[(size_t)i * D + d] = s;
    }
  return o;
}

// One full oracle comparison at a given hc_eps.
static bool oracle_case(float EPS, const char* tag) {
  const int N = 4, D = 4, L = 2; // hc_mult 4, hidden 4, 2 positions
  const int ITERS = 20;
  std::mt19937 rng(4242);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::normal_distribution<float> nf(0.0f, 0.03f);
  std::vector<float> streams((size_t)L * N * D);
  for (auto& v : streams) v = nd(rng);
  std::vector<float> fn((size_t)24 * N * D);
  for (auto& v : fn) v = nf(rng);
  std::vector<float> base(24);
  for (int i = 0; i < 24; ++i) base[i] = 0.1f * std::sin(0.9f * (float)i);
  std::vector<float> scale{1.3f, 0.7f, 1.1f};
  std::vector<float> bo((size_t)L * D);
  for (int l = 0; l < L; ++l)
    for (int d = 0; d < D; ++d) bo[(size_t)l * D + d] = 0.25f * (float)(d + 1) - 0.1f * (float)l;

  exl3::G5HcConfig hc;
  hc.mult = N;
  hc.sinkhorn_iters = ITERS;
  hc.eps = EPS;
  mx::array S = wrap(streams, {1, L, N, D});
  mx::array F = wrap(fn, {24, N * D});
  mx::array B = wrap(base, {24});
  mx::array SC = wrap(scale, {3});
  exl3::G5HcActivation a = exl3::g5_hc_prepare(S, B, F, SC, hc);
  mx::array BO = wrap(bo, {1, L, D});
  mx::array mixed = exl3::g5_hc_mix(BO, a.post, a.comb, S);
  mx::eval(a.post, a.comb, a.collapsed, mixed);

  std::vector<float> hpost = host(a.post), hcomb = host(a.comb), hcoll = host(a.collapsed),
                     hmix = host(mixed);
  double mp = 0, mc = 0, mcol = 0, mmix = 0;
  for (int l = 0; l < L; ++l) {
    std::vector<double> st((size_t)N * D), bd(D);
    for (int k = 0; k < N * D; ++k) st[k] = streams[(size_t)l * N * D + k];
    for (int d = 0; d < D; ++d) bd[d] = bo[(size_t)l * D + d];
    std::vector<double> fnd(fn.begin(), fn.end()), bd24(base.begin(), base.end()),
        scl(scale.begin(), scale.end());
    Oracle o = oracle_hc(st, fnd, bd24, scl, bd, N, D, EPS, ITERS);
    for (int i = 0; i < N; ++i) {
      mp = std::fmax(mp, std::fabs((double)hpost[(size_t)l * N + i] - o.post[i]));
      for (int j = 0; j < N; ++j)
        mc = std::fmax(mc, std::fabs((double)hcomb[((size_t)l * N + i) * N + j] - o.comb[(size_t)i * N + j]));
    }
    for (int d = 0; d < D; ++d) {
      mcol = std::fmax(mcol, std::fabs((double)hcoll[(size_t)l * D + d] - o.collapsed[d]));
      for (int i = 0; i < N; ++i)
        mmix = std::fmax(mmix, std::fabs((double)hmix[((size_t)l * N + i) * D + d] - o.mixed[(size_t)i * D + d]));
    }
  }
  std::printf("(b) oracle[%s] eps=%.0e N=%d D=%d L=%d: max|diff| post=%.3g comb=%.3g "
              "collapsed=%.3g mixed=%.3g\n",
              tag, (double)EPS, N, D, L, mp, mc, mcol, mmix);
  std::printf("    collapsed[0] = %.7f (oracle) | post = %.6f %.6f %.6f %.6f | comb[0] sum = %.7f\n",
              (double)hcoll[0], hpost[0], hpost[1], hpost[2], hpost[3],
              (double)hcomb[0] + hcomb[1] + hcomb[2] + hcomb[3]);
  bool ok = mp < 1e-5 && mc < 1e-5 && mcol < 1e-5 && mmix < 1e-5;
  std::printf("(b) weights + collapse + mix match the scalar oracle: %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

static bool test_oracle() {
  // 1e-6 is the checkpoint value; 1e-3 makes every hc_eps addition (rms_norm,
  // pre, comb, each Sinkhorn denominator) observable, so a missing one fails.
  return oracle_case(1e-6f, "checkpoint") && oracle_case(1e-3f, "large-eps");
}

// ----------------------------------------------------- (c) shapes / roundtrip
static bool test_shapes_roundtrip() {
  const int N = 4, D = 4, L = 3, B = 2;
  std::vector<float> x((size_t)B * L * D);
  for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * (float)i - 0.3f;
  mx::array X = wrap(x, {B, L, D});
  mx::array E = exl3::g5_hc_expand(X, N);
  mx::array H = exl3::g5_hc_head(E);
  mx::eval(E, H);
  std::vector<int> es = ishape(E), hs = ishape(H);
  // every stream must be an exact copy of x
  std::vector<float> he = host(E), hh = host(H);
  double maxexp = 0, maxhead = 0;
  for (int b = 0; b < B; ++b)
    for (int l = 0; l < L; ++l)
      for (int d = 0; d < D; ++d) {
        float v = x[((size_t)b * L + l) * D + d];
        for (int i = 0; i < N; ++i)
          maxexp = std::fmax(maxexp, std::fabs((double)he[(((size_t)b * L + l) * N + i) * D + d] - v));
        maxhead = std::fmax(maxhead, std::fabs((double)hh[((size_t)b * L + l) * D + d] - v));
      }
  // expand -> mix identity: post = 1, comb = I  =>  out[i] = bo + streams[i]
  std::vector<float> ones((size_t)B * L * N, 1.0f);
  std::vector<float> ident((size_t)B * L * N * N, 0.0f);
  for (int b = 0; b < B; ++b)
    for (int l = 0; l < L; ++l)
      for (int i = 0; i < N; ++i) ident[(((size_t)b * L + l) * N + i) * N + i] = 1.0f;
  std::vector<float> bo((size_t)B * L * D);
  for (size_t i = 0; i < bo.size(); ++i) bo[i] = 0.05f * (float)i + 1.0f;
  mx::array P = wrap(ones, {B, L, N}), CI = wrap(ident, {B, L, N, N}), BO = wrap(bo, {B, L, D});
  mx::array M = exl3::g5_hc_mix(BO, P, CI, E);
  mx::eval(M);
  std::vector<int> ms = ishape(M);
  std::vector<float> hm = host(M);
  double maxid = 0;
  for (int b = 0; b < B; ++b)
    for (int l = 0; l < L; ++l)
      for (int d = 0; d < D; ++d) {
        double want = (double)bo[((size_t)b * L + l) * D + d] + (double)x[((size_t)b * L + l) * D + d];
        for (int i = 0; i < N; ++i)
          maxid = std::fmax(maxid, std::fabs((double)hm[(((size_t)b * L + l) * N + i) * D + d] - want));
      }
  // prepare() shapes on the expanded streams (1, L, N, D)
  mx::array S1 = exl3::g5_hc_expand(wrap(x, {1, L, D}), N);
  std::mt19937 rng(99);
  std::normal_distribution<float> nf(0.0f, 0.05f);
  std::vector<float> fn((size_t)24 * N * D);
  for (auto& v : fn) v = nf(rng);
  std::vector<float> base(24, 0.0f), scale{1.0f, 1.0f, 1.0f};
  exl3::G5HcConfig hc;
  hc.mult = N;
  hc.sinkhorn_iters = 20;
  exl3::G5HcActivation a = exl3::g5_hc_prepare(S1, wrap(base, {24}), wrap(fn, {24, N * D}),
                                               wrap(scale, {3}), hc);
  std::vector<int> ps = ishape(a.post), cs = ishape(a.comb), os = ishape(a.collapsed);
  auto fmt = [](const std::vector<int>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? "," : "") + std::to_string(s[i]);
    return r + "]";
  };
  std::printf("(c) expand %s -> %s (copies max|diff|=%.3g), head -> %s (max|diff|=%.3g), "
              "mix -> %s (identity max|diff|=%.3g)\n",
              fmt({B, L, D}).c_str(), fmt(es).c_str(), maxexp, fmt(hs).c_str(), maxhead,
              fmt(ms).c_str(), maxid);
  std::printf("    prepare: post %s comb %s collapsed %s\n", fmt(ps).c_str(), fmt(cs).c_str(),
              fmt(os).c_str());
  bool ok = es == std::vector<int>({B, L, N, D}) && hs == std::vector<int>({B, L, D}) &&
            ms == std::vector<int>({B, L, N, D}) && ps == std::vector<int>({1, L, N}) &&
            cs == std::vector<int>({1, L, N, N}) && os == std::vector<int>({1, L, D}) &&
            maxexp == 0.0 && maxhead < 1e-6 && maxid < 1e-6;
  std::printf("(c) shapes + expand->mix roundtrip: %s\n", ok ? "PASS" : "FAIL");
  return ok;
}

int main() {
  bool a = test_sinkhorn();
  bool b = test_oracle();
  bool c = test_shapes_roundtrip();
  bool ok = a && b && c;
  std::printf("hc (hyper-connections): %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
