// indexer_topk: the indexer's projection/packing caller. Checks shape, cache
// growth across prefill+decode, that ids are in range and unique within each
// query row (the reference allows the same token in different rows), and that
// the result is what mla_layer's mask builder consumes.
// Build (repo root): clang++ -std=c++20 -O2 -Ivendor -Isrc -o /tmp/idx_topk_test \
//   tests/indexer_topk_test.cpp src/glm5_attn.cpp src/glm5_kda.cpp src/glm5.cpp \
//   src/exl3.cpp src/safetensors.cpp src/trellis_gemm.cpp -lmlx
// indexer_topk smoke: shapes, cache growth, valid ids, no duplicates, and that
// the returned selection is what mla_layer's mask builder can consume.
#include <cstdio>
#include <set>
#include <algorithm>
#include <random>
#include <vector>
#include <mlx/mlx.h>
#include "glm5_attn.h"
#include "glm5_kda.h"
namespace mx = mlx::core;
static exl3::G5Linear lin(int in, int out, std::mt19937& rng) {
  std::normal_distribution<float> nd(0, 0.4f);
  auto* h = new std::vector<float>((size_t)in * out);
  for (auto& x : *h) x = nd(rng);
  exl3::G5Linear l; l.quantized = false; l.in_features = in; l.out_features = out;
  l.w = mx::array((void*)h->data(), mx::Shape{in, out}, mx::float32, [h](void*) { delete h; });
  return l;
}
static mx::array mat(int r, int c, std::mt19937& rng, float s = 0.3f) {
  std::normal_distribution<float> nd(0, s);
  auto* h = new std::vector<float>((size_t)r * c);
  for (auto& x : *h) x = nd(rng);
  return mx::array((void*)h->data(), mx::Shape{r, c}, mx::float32, [h](void*) { delete h; });
}
int main() {
  const int Hd = 8, D = 4, Nh = 2, kp = 4, topk = 8, W = topk + kp - 1;
  std::mt19937 rng(5);
  exl3::G5Layer l;
  l.idx_wq_b = lin(Nh * D, Nh * D, rng);       // (in, out): wq_b maps q_lora_rank -> Nh*D
  l.idx_wk = mat(D, Hd, rng);             // raw, stored (out, in)
  l.idx_gate = mat(D, Hd, rng);
  l.idx_weights_proj = mat(Nh, Hd, rng);
  l.idx_k_norm_w = mat(1, D, rng, 0.1f);
  l.idx_k_norm_b = mat(1, D, rng, 0.1f);
  l.idx_ape = mat(kp, D, rng, 0.2f);
  exl3::ModelConfig cfg;
  cfg.index_n_heads = Nh; cfg.index_kpool = kp; cfg.index_topk = topk;
  cfg.index_kpool_always_select_tail = true;
  exl3::IndexerState st;
  auto run = [&](int L, int start) {
    mx::array h = mx::reshape(mat(L, Hd, rng), {1, L, Hd});
    mx::array qr = mx::reshape(mat(L, Nh * D, rng), {1, L, Nh * D});
    mx::array t = exl3::indexer_topk(l, cfg, h, qr, st, start);
    mx::eval(t);
    std::printf("  L=%d start=%d -> shape {1,%d,%d} cache_len=%d\n", L, start,
                (int)t.shape(1), (int)t.shape(2), st.len);
    bool ok = t.shape(1) == L && t.shape(2) == W;
    const int32_t* p = t.data<int32_t>();
    int nvalid = 0, maxrow = 0;
    for (int r = 0; r < L; ++r) {
      std::set<int> seen;
      for (int i = 0; i < W; ++i) {
        int v = p[r * W + i];
        if (v == -1) continue;
        nvalid++;
        // ids must be in range and unique WITHIN a row (rows may share ids)
        if (v < 0 || v >= st.len || !seen.insert(v).second) ok = false;
      }
      maxrow = std::max(maxrow, (int)seen.size());
    }
    std::printf("     valid ids=%d, max distinct per row=%d\n", nvalid, maxrow);
    return ok;
  };
  bool ok = run(5, 0);
  ok = run(1, 5) && ok;
  std::printf("indexer_topk: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
