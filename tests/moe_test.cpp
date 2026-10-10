// GLM-5-Next MoE + dense MLP numeric check.
//
// Synthetic PLAIN weights and a scalar CPU oracle recomputed from the HF
// reference (/tmp/modeling_glm5_next.py):
//   Glm5NextTextTopkRouter.forward  :157-180
//   Glm5NextTextExperts.forward     :119-142
//   Glm5NextTextMLP.forward         :97-105
//   Glm5NextTextMoE.forward         :199-205
//
// Build (from the repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/moe_test \
//     tests/moe_test.cpp src/glm5_moe.cpp src/glm5_kda.cpp src/glm5_attn.cpp src/glm5.cpp \
//     src/model.cpp src/exl3.cpp src/safetensors.cpp src/trellis_gemm.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <mlx/mlx.h>

#include "glm5_kda.h"
#include "glm5_moe.h"

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static std::vector<float> rndv(int n, std::mt19937& rng, float s = 0.3f) {
  std::normal_distribution<float> nd(0, s);
  std::vector<float> v((size_t)n);
  for (auto& x : v) x = nd(rng);
  return v;
}
static exl3::G5Linear mk(const std::vector<float>& W, int in, int out) {
  auto* h = new std::vector<float>(W);
  exl3::G5Linear l;
  l.quantized = false;
  l.in_features = in;
  l.out_features = out;
  l.w = mx::array((void*)h->data(), mx::Shape{in, out}, mx::float32, [h](void*) { delete h; });
  return l;
}
static exl3::G5Linear mk16(const std::vector<float>& W, int in, int out) {
  auto* h = new std::vector<float>(W);
  mx::array a = mx::array((void*)h->data(), mx::Shape{in, out}, mx::float32, [h](void*) { delete h; });
  exl3::G5Linear l;
  l.quantized = false;
  l.in_features = in;
  l.out_features = out;
  l.w = mx::astype(a, mx::float16);
  mx::eval(l.w); // a owns the heap buffer; materialise before it is destroyed
  return l;
}
static mx::array mk1(const std::vector<float>& v) {
  auto* h = new std::vector<float>(v);
  return mx::array((void*)h->data(), mx::Shape{(int)v.size()}, mx::float32,
                   [h](void*) { delete h; });
}
static mx::array mk2(const std::vector<float>& v, int r, int c) {
  auto* h = new std::vector<float>(v);
  return mx::array((void*)h->data(), mx::Shape{r, c}, mx::float32, [h](void*) { delete h; });
}

// ---------------------------------------------------------------------------
// scalar CPU oracle (double), transcribed from the reference formula
// ---------------------------------------------------------------------------
static double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double silu(double x) { return x * sigm(x); }

struct Exp {
  std::vector<float> g, u, d; // g,u: (H,I) row-major; d: (I,H) row-major
};

struct Router {
  std::vector<int> idx;      // top_k
  std::vector<double> w;     // top_k
};

static Router oracle_route(const std::vector<double>& logits, const std::vector<double>& bias,
                           int n_group, int topk_group, int top_k, bool norm, double scale) {
  const int E = (int)logits.size();
  std::vector<double> sc((size_t)E), sfc((size_t)E);
  for (int e = 0; e < E; ++e) {
    sc[(size_t)e] = sigm(logits[(size_t)e]);
    sfc[(size_t)e] = sc[(size_t)e] + bias[(size_t)e];
  }
  const int gsize = E / n_group;
  std::vector<double> gs((size_t)n_group);
  for (int g = 0; g < n_group; ++g) {
    std::vector<double> inner(sfc.begin() + (size_t)g * gsize,
                              sfc.begin() + (size_t)(g + 1) * gsize);
    std::sort(inner.begin(), inner.end(), std::greater<double>());
    gs[(size_t)g] = inner[0] + (gsize > 1 ? inner[1] : 0.0);
  }
  std::vector<int> gorder((size_t)n_group);
  std::iota(gorder.begin(), gorder.end(), 0);
  std::stable_sort(gorder.begin(), gorder.end(),
                   [&](int a, int b) { return gs[(size_t)a] > gs[(size_t)b]; });
  std::vector<char> allowed((size_t)E, 0);
  for (int i = 0; i < topk_group; ++i)
    for (int j = 0; j < gsize; ++j) allowed[(size_t)gorder[(size_t)i] * gsize + j] = 1;

  std::vector<int> order((size_t)E);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    double ca = allowed[(size_t)a] ? sfc[(size_t)a] : -std::numeric_limits<double>::infinity();
    double cb = allowed[(size_t)b] ? sfc[(size_t)b] : -std::numeric_limits<double>::infinity();
    return ca > cb;
  });
  std::vector<int> idx(order.begin(), order.begin() + top_k);
  std::vector<double> w((size_t)top_k);
  for (int k = 0; k < top_k; ++k) w[(size_t)k] = sc[(size_t)idx[(size_t)k]];
  if (norm) {
    double denom = 0;
    for (double x : w) denom += x;
    denom += 1e-20;
    for (double& x : w) x /= denom;
  }
  for (double& x : w) x *= scale;
  return {idx, w};
}

static std::vector<double> oracle_swiglu(const std::vector<double>& x, const Exp& e, int H, int I,
                                         double limit, double weight) {
  std::vector<double> gate((size_t)I), up((size_t)I), cur((size_t)I);
  for (int i = 0; i < I; ++i) {
    double g = 0, u = 0;
    for (int h = 0; h < H; ++h) {
      g += x[(size_t)h] * e.g[(size_t)h * I + i];
      u += x[(size_t)h] * e.u[(size_t)h * I + i];
    }
    gate[(size_t)i] = std::min(g, limit);                        // clamp(max=limit)
    up[(size_t)i] = std::min(std::max(u, -limit), limit);        // clamp(-limit, limit)
    cur[(size_t)i] = silu(gate[(size_t)i]) * up[(size_t)i];
  }
  std::vector<double> out((size_t)H, 0.0);
  for (int h = 0; h < H; ++h)
    for (int i = 0; i < I; ++i) out[(size_t)h] += weight * cur[(size_t)i] * e.d[(size_t)i * H + h];
  return out;
}

// Returns {max|diff|, mismatches}; tolerance is relative-ish.
static bool g_ok = true;
static void check(const char* name, const mx::array& got, const std::vector<double>& exp,
                  double atol) {
  mx::array g = mx::astype(got, mx::float32);
  mx::eval(g);
  const float* p = g.data<float>();
  double maxd = 0;
  int bad = 0;
  for (size_t i = 0; i < exp.size(); ++i) {
    double d = std::fabs((double)p[i] - exp[i]);
    maxd = std::max(maxd, d);
    if (d > atol * (1.0 + std::fabs(exp[i]))) ++bad;
  }
  std::printf("  %-28s max|diff| = %.3g  mismatches = %d / %zu\n", name, maxd, bad, exp.size());
  if (bad) {
    std::printf("    !! FAILED\n");
    g_ok = false;
  }
}

static void eq(const char* what, bool ok) {
  std::printf("  %-28s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok) g_ok = false;
}

int main() {
  const int H = 2, I = 3, E = 5, K = 2;
  const float lim = 10.0f;
  std::mt19937 rng(11);

  // Expert weights + shared expert.
  std::vector<Exp> ew((size_t)E);
  exl3::G5Layer l;
  l.moe = true;
  l.experts.resize((size_t)E * 3);
  for (int e = 0; e < E; ++e) {
    ew[(size_t)e].g = rndv(H * I, rng);
    ew[(size_t)e].u = rndv(H * I, rng);
    ew[(size_t)e].d = rndv(I * H, rng);
    l.experts[(size_t)(e * 3 + 0)] = mk(ew[(size_t)e].g, H, I);
    l.experts[(size_t)(e * 3 + 1)] = mk(ew[(size_t)e].u, H, I);
    l.experts[(size_t)(e * 3 + 2)] = mk(ew[(size_t)e].d, I, H);
  }
  Exp sh;
  sh.g = rndv(H * I, rng);
  sh.u = rndv(H * I, rng);
  sh.d = rndv(I * H, rng);

  // Router: hand-crafted logits (row 0 = logits when h = [1, 0]) + a bias that
  // flips the top-2 away from the two highest *un-biased* scores.
  std::vector<float> rw(E * H), rb((size_t)E, 0.0f);
  const double raw[E] = {2.0, 1.0, 0.5, -2.0, -3.0};
  const double raw2[E] = {0.4, 0.9, 3.0, 0.1, -0.5};
  const double bias[E] = {0.0, 0.0, 0.0, 0.8, 0.75};
  for (int e = 0; e < E; ++e) {
    rw[(size_t)e * H + 0] = (float)raw[e];
    rw[(size_t)e * H + 1] = (float)raw2[e];
    rb[(size_t)e] = (float)bias[e];
  }
  l.router_w = mk2(rw, E, H);
  l.router_bias = mk1(rb);
  l.shared_gate = mk(sh.g, H, I);
  l.shared_up = mk(sh.u, H, I);
  l.shared_down = mk(sh.d, I, H);

  exl3::ModelConfig cfg;
  cfg.hidden_size = H;
  cfg.n_routed_experts = E;
  cfg.num_experts_per_tok = K;
  cfg.moe_intermediate_size = I;
  cfg.n_shared_experts = 1;
  cfg.routed_scaling_factor = 2.5f;
  cfg.swiglu_limit = lim;
  cfg.norm_topk_prob = true;

  std::printf("== (a) router: bias flips selection, weights stay un-biased ==\n");
  {
    // h = [1, 0] -> logits = raw
    std::vector<float> rf;
    for (int e = 0; e < E; ++e) rf.push_back((float)raw[e]);
    mx::array la = mk1(rf); // (E,) but route wants (N,E): reshape below
    la = mx::reshape(la, {1, E});
    auto [w, idx] = exl3::moe_topk_route(la, mk1(rb), 1, 1, K, true, 2.5f);
    mx::eval(w);
    mx::eval(idx);
    const int32_t* ip = idx.data<int32_t>();
    const float* wp = w.data<float>();
    std::vector<int> sel = {ip[0], ip[1]};
    std::sort(sel.begin(), sel.end());
    eq("selected == {0,3} (bias flips)", sel == std::vector<int>({0, 3}));
    // weights per selected expert (sorted): un-biased scores, normalised, *2.5
    std::vector<std::pair<int, double>> got{{sel[0], 0}, {sel[1], 0}};
    for (int k = 0; k < K; ++k)
      for (auto& g : got)
        if (g.first == ip[k]) g.second = wp[k];
    eq("w[e0] == 2.20199", std::fabs(got[0].second - 2.201993) < 1e-4);
    eq("w[e3] == 0.29801 (un-biased)", std::fabs(got[1].second - 0.298008) < 1e-4);
    // the two highest un-biased scores are 0 and 1 (bias is what changed it)
    std::vector<double> sc(E);
    for (int e = 0; e < E; ++e) sc[(size_t)e] = sigm(raw[e]);
    std::vector<int> ord(E);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int c) { return sc[(size_t)a] > sc[(size_t)c]; });
    eq("un-biased top2 == {0,1}", (std::vector<int>{ord[0], ord[1]} == std::vector<int>({0, 1})));
  }

  std::printf("== (b) group limit: n_group=2, topk_group=1 ==\n");
  {
    // group 0 = {0,1}, group 1 = {2,3}; global best is expert 2 but group 0 wins
    // the (sum of top-2) group score, so experts 0 and 1 are selected.
    std::vector<float> lg = {1.0f, 0.9f, 2.0f, -1.0f};
    std::vector<float> zb(4, 0.0f);
    std::vector<double> lgd = {1.0, 0.9, 2.0, -1.0}, zd = {0, 0, 0, 0};
    Router ref = oracle_route(lgd, zd, 2, 1, 2, false, 1.0);
    auto [w, idx] = exl3::moe_topk_route(mx::reshape(mk1(lg), {1, 4}), mk1(zb), 2, 1, 2, false, 1.0f);
    mx::eval(w);
    mx::eval(idx);
    const int32_t* ip = idx.data<int32_t>();
    std::vector<int> sel = {ip[0], ip[1]};
    std::sort(sel.begin(), sel.end());
    eq("selected == {0,1} (group)", sel == std::vector<int>({0, 1}));
    std::vector<int> refsel = ref.idx;
    std::sort(refsel.begin(), refsel.end());
    eq("oracle agrees", sel == refsel);
    eq("global best (expert 2) excluded", std::find(sel.begin(), sel.end(), 2) == sel.end());
  }

  std::printf("== (c) multi-token routing + weighted expert sum ==\n");
  {
    const int L = 3;
    std::vector<float> xs = {1, 0, 0, 1, 1, 1}; // (1,3,2)
    mx::array h = mk2(xs, 1 * L, H);
    h = mx::reshape(h, {1, L, H});
    mx::array out = exl3::moe_layer(l, cfg, h);
    mx::eval(out);
    const float* o = out.data<float>();

    std::vector<double> exp((size_t)L * H, 0.0);
    for (int t = 0; t < L; ++t) {
      std::vector<double> x = {(double)xs[(size_t)t * H], (double)xs[(size_t)t * H + 1]};
      std::vector<double> logits((size_t)E), b((size_t)E);
      for (int e = 0; e < E; ++e) {
        logits[(size_t)e] = x[0] * rw[(size_t)e * H] + x[1] * rw[(size_t)e * H + 1];
        b[(size_t)e] = bias[e];
      }
      Router r = oracle_route(logits, b, 1, 1, K, true, 2.5);
      std::vector<double> acc((size_t)H, 0.0);
      for (int k = 0; k < K; ++k) {
        std::vector<double> c = oracle_swiglu(x, ew[(size_t)r.idx[(size_t)k]], H, I, lim,
                                              r.w[(size_t)k]);
        for (int hh = 0; hh < H; ++hh) acc[(size_t)hh] += c[(size_t)hh];
      }
      std::vector<double> s = oracle_swiglu(x, sh, H, I, lim, 1.0);
      for (int hh = 0; hh < H; ++hh) exp[(size_t)t * H + hh] = acc[(size_t)hh] + s[(size_t)hh];
    }
    std::printf("  token experts: ");
    std::vector<std::vector<int>> per_tok((size_t)L);
    for (int t = 0; t < L; ++t) {
      std::vector<double> x = {(double)xs[(size_t)t * H], (double)xs[(size_t)t * H + 1]};
      std::vector<double> logits((size_t)E), b((size_t)E);
      for (int e = 0; e < E; ++e) {
        logits[(size_t)e] = x[0] * rw[(size_t)e * H] + x[1] * rw[(size_t)e * H + 1];
        b[(size_t)e] = bias[e];
      }
      Router r = oracle_route(logits, b, 1, 1, K, true, 2.5);
      std::sort(r.idx.begin(), r.idx.end());
      per_tok[(size_t)t] = r.idx;
      std::printf("[%d,%d]", r.idx[0], r.idx[1]);
    }
    std::printf("\n");
    eq("tokens route differently", per_tok[0] != per_tok[1] || per_tok[1] != per_tok[2]);
    check("moe_layer vs oracle", out, exp, 3e-3);
    (void)o;
  }

  std::printf("== (d) dense MLP with swiglu_limit clamp ==\n");
  {
    // gate/up = 12 (> limit 10), so both clamps are active.
    std::vector<float> Wg(2 * 4, 1.0f), Wu(2 * 4, 1.0f);
    std::vector<float> Wd = {1, 0, 0, 1, 1, 0, 0, 1}; // (4,2)
    l.dense_gate = mk(Wg, 2, 4);
    l.dense_up = mk(Wu, 2, 4);
    l.dense_down = mk(Wd, 4, 2);
    std::vector<float> xs = {6, 6};
    mx::array h = mx::reshape(mk1(xs), {1, 1, 2});
    mx::array out = exl3::dense_mlp(l, cfg, h);
    mx::eval(out);
    Exp de{Wg, Wu, Wd};
    std::vector<double> exp = oracle_swiglu({6, 6}, de, H, 4, lim, 1.0);
    std::vector<double> unclamped = oracle_swiglu({6, 6}, de, H, 4, 1e9, 1.0);
    std::printf("  clamped out: %.6f %.6f | unclamped: %.6f %.6f\n", exp[0], exp[1], unclamped[0],
                unclamped[1]);
    eq("clamp changes the result", std::fabs(exp[0] - unclamped[0]) > 1e-3);
    check("dense_mlp vs oracle", out, exp, 2e-3);
  }

  std::printf("== (e) f16 weights + f16 activations (engine dtype) ==\n");
  {
    exl3::G5Layer l16;
    l16.moe = true;
    l16.experts.resize((size_t)E * 3);
    for (int e = 0; e < E; ++e) {
      l16.experts[(size_t)(e * 3 + 0)] = mk16(ew[(size_t)e].g, H, I);
      l16.experts[(size_t)(e * 3 + 1)] = mk16(ew[(size_t)e].u, H, I);
      l16.experts[(size_t)(e * 3 + 2)] = mk16(ew[(size_t)e].d, I, H);
    }
    l16.shared_gate = mk16(sh.g, H, I);
    l16.shared_up = mk16(sh.u, H, I);
    l16.shared_down = mk16(sh.d, I, H);
    l16.router_w = mx::astype(mk2(rw, E, H), mx::float16);
    l16.router_bias = mk1(rb);

    const int L = 3;
    std::vector<float> xs = {1, 0, 0, 1, 1, 1};
    mx::array h16 = mx::astype(mx::reshape(mk2(xs, L, H), {1, L, H}), mx::float16);
    mx::eval(h16); // materialise before the mk2 owner dies
    mx::array out16 = exl3::moe_layer(l16, cfg, h16);
    mx::eval(out16);
    // oracle from the same (un-rounded) weights, loose tolerance for f16
    std::vector<double> exp((size_t)L * H, 0.0);
    for (int t = 0; t < L; ++t) {
      std::vector<double> x = {(double)xs[(size_t)t * H], (double)xs[(size_t)t * H + 1]};
      std::vector<double> logits((size_t)E), b((size_t)E);
      for (int e = 0; e < E; ++e) {
        logits[(size_t)e] = x[0] * rw[(size_t)e * H] + x[1] * rw[(size_t)e * H + 1];
        b[(size_t)e] = bias[e];
      }
      Router r = oracle_route(logits, b, 1, 1, K, true, 2.5);
      std::vector<double> acc((size_t)H, 0.0);
      for (int k = 0; k < K; ++k) {
        std::vector<double> c = oracle_swiglu(x, ew[(size_t)r.idx[(size_t)k]], H, I, lim, r.w[(size_t)k]);
        for (int hh = 0; hh < H; ++hh) acc[(size_t)hh] += c[(size_t)hh];
      }
      std::vector<double> s = oracle_swiglu(x, sh, H, I, lim, 1.0);
      for (int hh = 0; hh < H; ++hh) exp[(size_t)t * H + hh] = acc[(size_t)hh] + s[(size_t)hh];
    }
    check("moe_layer(f16) vs oracle", out16, exp, 2e-2);

    l16.dense_gate = mk16(std::vector<float>(8, 1.0f), 2, 4);
    l16.dense_up = mk16(std::vector<float>(8, 1.0f), 2, 4);
    l16.dense_down = mk16(std::vector<float>{1, 0, 0, 1, 1, 0, 0, 1}, 4, 2);
    mx::array dh = mx::astype(mx::reshape(mk1(std::vector<float>{6, 6}), {1, 1, 2}), mx::float16);
    mx::eval(dh);
    mx::array dout = exl3::dense_mlp(l16, cfg, dh);
    std::vector<double> dexp = oracle_swiglu({6, 6}, Exp{std::vector<float>(8, 1.0f),
                                                         std::vector<float>(8, 1.0f),
                                                         std::vector<float>{1, 0, 0, 1, 1, 0, 0, 1}},
                                             H, 4, lim, 1.0);
    check("dense_mlp(f16) vs oracle", dout, dexp, 3e-2);
  }

  std::printf("%s\n", g_ok ? "ALL OK" : "FAILURES");
  return g_ok ? 0 : 1;
}
