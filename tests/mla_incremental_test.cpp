// MLA layer invariant: incremental decode (prefill n-1 then one token) MUST
// reproduce the last row of a single n-token prefill - catches cache, offset
// and causal-mask bugs. Weights are synthetic.
// Build (from the repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/mla_test \
//     tests/mla_incremental_test.cpp src/glm5_attn.cpp src/glm5_kda.cpp src/glm5.cpp \
//     src/exl3.cpp src/safetensors.cpp src/trellis_gemm.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib
// Expected: "max|diff| = ~6e-08, mismatches = 0 / 8, non-finite = 0"
// MLA layer smoke test: the incremental path (prefill n + decode 1) must agree
// with a single prefill of n+1 tokens - catches cache/offset/mask bugs.
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <mlx/mlx.h>
#include "glm5_attn.h"
#include "glm5_kda.h"
namespace mx = mlx::core;

static exl3::G5Linear lin(int in, int out, std::mt19937& rng) {
  std::normal_distribution<float> nd(0, 0.3f);
  auto* w = new std::vector<float>((size_t)in * out);
  for (auto& x : *w) x = nd(rng);
  // plain weights are consumed as (in, out) f16
  auto* h = new std::vector<float>((size_t)in * out);
  for (size_t i = 0; i < h->size(); ++i) (*h)[i] = (*w)[i];
  delete w;
  exl3::G5Linear l;
  l.quantized = false;
  l.in_features = in; l.out_features = out;
  l.w = mx::array((void*)h->data(), mx::Shape{in, out}, mx::float32,
                  [h](void*) { delete h; });
  return l;
}
static mx::array normw(int n, std::mt19937& rng) {
  std::normal_distribution<float> nd(0, 0.2f);
  auto* h = new std::vector<float>((size_t)n);
  for (auto& x : *h) x = 1.0f + nd(rng);
  return mx::array((void*)h->data(), mx::Shape{n}, mx::float32, [h](void*) { delete h; });
}
int main() {
  const int Hd = 8, Nh = 2, QK = 4, VD = 4, KVR = 5, QLR = 6, L = 4;
  std::mt19937 rng(7); std::normal_distribution<float> nd(0, 1);
  exl3::G5Layer l;
  l.q_a = lin(Hd, QLR, rng); l.q_b = lin(QLR, Nh * QK, rng);
  l.kv_a = lin(Hd, KVR, rng); l.kv_b = lin(KVR, Nh * (QK + VD), rng);
  l.attn_o = lin(Nh * VD, Hd, rng);
  l.q_a_norm = normw(QLR, rng); l.kv_a_norm = normw(KVR, rng);
  exl3::ModelConfig cfg; cfg.num_heads = Nh;
  auto* xh = new std::vector<float>((size_t)L * Hd);
  for (auto& x : *xh) x = nd(rng);
  mx::array x((void*)xh->data(), mx::Shape{1, L, Hd}, mx::float32, [xh](void*) { delete xh; });

  // full prefill
  exl3::MlaState s1;
  mx::array full = exl3::mla_layer(l, cfg, x, s1, 0);
  mx::eval(full);
  // incremental: prefill L-1 then decode the last token
  exl3::MlaState s2;
  mx::array pre = exl3::mla_layer(l, cfg, mx::slice(x, {0, 0, 0}, {1, L - 1, Hd}), s2, 0);
  mx::array last = exl3::mla_layer(l, cfg, mx::slice(x, {0, L - 1, 0}, {1, L, Hd}), s2, L - 1);
  mx::eval(pre); mx::eval(last);
  const float* f = full.data<float>();
  const float* d = last.data<float>();
  double maxdiff = 0; int bad = 0, nan = 0;
  for (int i = 0; i < Hd; ++i) {
    // the decode row is the LAST row of the full prefill
    double a = (double)f[(L - 1) * Hd + i], b = (double)d[i];
    if (!std::isfinite(a) || !std::isfinite(b)) nan++;
    maxdiff = std::max(maxdiff, std::fabs(a - b));
    if (std::fabs(a - b) > 2e-3) bad++;
  }
  {
    const float* pr = pre.data<float>();
    double md0 = 0, md2 = 0;
    for (int j = 0; j < 2 * Hd; ++j) md0 = std::max(md0, std::fabs((double)f[j] - (double)pr[j]));
    for (int j = 0; j < 2 * Hd; ++j) md2 = std::max(md2, std::fabs((double)f[2 * Hd + j] - (double)pr[2 * Hd + j]));
  std::printf("MLA prefill-vs-incremental: max|diff| = %.3g, mismatches = %d / %d, non-finite = %d\n",
              maxdiff, bad, Hd, nan);
  return bad == 0 && nan == 0 ? 0 : 1;
}
