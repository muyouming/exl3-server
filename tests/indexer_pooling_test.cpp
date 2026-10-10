// DSA indexer pooling + scoring against a scalar CPU oracle. Exercises the
// correctness-critical details: pooling anchors at the FIRST VALID token (the
// first two tokens here are invalid), a pool is valid only if all index_kpool
// slots are, and the pooled key is a per-dim softmax(gate_scores + ape)
// weighted average of the slot keys.
// Build (repo root): clang++ -std=c++20 -O2 -Ivendor -Isrc -o /tmp/idx_test \
//   tests/indexer_pooling_test.cpp src/glm5_attn.cpp src/glm5_kda.cpp src/glm5.cpp \
//   src/exl3.cpp src/safetensors.cpp src/trellis_gemm.cpp -lmlx
// Expected: pooled keys max|diff| ~5e-08, scores ~1e-08, PASS
// Indexer pooling + scoring vs a scalar CPU oracle (reference semantics:
// first-valid anchoring, per-dim softmax(gate+ape) weighting, relu scoring).
#include <cstdio>
#include <cmath>
#include <vector>
#include <mlx/mlx.h>
#include "glm5_attn.h"
namespace mx = mlx::core;
int main() {
  const int Ltot = 10, D = 4, KP = 4, Nh = 2, L = 3;
  // tokens 0 and 1 invalid -> pooling must start at 2
  std::vector<float> packed((size_t)Ltot * (2 * D + 1), 0.0f);
  for (int t = 0; t < Ltot; ++t) {
    for (int d = 0; d < D; ++d) {
      packed[(size_t)t * (2 * D + 1) + d] = 0.1f * (t + 1) + 0.05f * d;
      packed[(size_t)t * (2 * D + 1) + D + d] = 0.2f * ((t * 7 + d * 3) % 5) - 0.4f;
    }
    packed[(size_t)t * (2 * D + 1) + 2 * D] = (t >= 2) ? 1.0f : 0.0f;
  }
  std::vector<float> q((size_t)L * Nh * D), w((size_t)L * Nh), ape((size_t)KP * D);
  for (size_t i = 0; i < q.size(); ++i) q[i] = 0.3f * std::sin((float)i * 1.7f);
  for (size_t i = 0; i < w.size(); ++i) w[i] = 0.5f + 0.1f * (float)i;
  for (size_t i = 0; i < ape.size(); ++i) ape[i] = 0.25f * std::cos((float)i);
  auto wrap = [](std::vector<float>& v, mx::Shape sh) {
    auto* h = new std::vector<float>(v);
    return mx::array((void*)h->data(), sh, mx::float32, [h](void*) { delete h; });
  };
  mx::array P = wrap(packed, {Ltot, 2 * D + 1});
  mx::array Q = wrap(q, {L, Nh, D}), W = wrap(w, {L, Nh}), A = wrap(ape, {KP, D});
  auto r = exl3::indexer_pool_and_score(P, Q, W, A);
  mx::eval(r.keys, r.scores);
  // oracle
  int first = 2;
  int n = (Ltot + KP - 1) / KP;
  double maxk = 0, maxs = 0;
  int badk = 0, bads = 0;
  const float* K = r.keys.data<float>();
  const float* S = r.scores.data<float>();
  for (int p = 0; p < n; ++p) {
    std::vector<double> kk((size_t)D, 0.0);
    std::vector<int> idx((size_t)KP, -1);
    bool all = true;
    std::vector<std::vector<double>> logits((size_t)KP, std::vector<double>(D, 0.0));
    for (int s = 0; s < KP; ++s) {
      int t = first + p * KP + s;
      bool ok = t < Ltot;  // tokens >= 2 are valid by construction
      idx[(size_t)s] = ok ? t : -1;
      all = all && ok;
      for (int d = 0; d < D; ++d) {
        if (ok) logits[(size_t)s][(size_t)d] = packed[(size_t)t * (2 * D + 1) + D + d] + ape[(size_t)s * D + d];
        else logits[(size_t)s][(size_t)d] = -INFINITY;
      }
      if (!ok) continue;
    }
    // softmax per dim over slots, then weighted key sum
    for (int d = 0; d < D; ++d) {
      double mx_ = -INFINITY, sum = 0;
      for (int s = 0; s < KP; ++s) mx_ = std::max(mx_, logits[(size_t)s][(size_t)d]);
      for (int s = 0; s < KP; ++s) sum += std::exp(logits[(size_t)s][(size_t)d] - mx_);
      for (int s = 0; s < KP; ++s) {
        int t = idx[(size_t)s];
        if (t < 0) continue;
        double pr = std::exp(logits[(size_t)s][(size_t)d] - mx_) / sum;
        kk[(size_t)d] += pr * packed[(size_t)t * (2 * D + 1) + d];
      }
    }
    for (int d = 0; d < D; ++d) {
      double diff = std::fabs(kk[(size_t)d] - (double)K[(size_t)p * D + d]);
      maxk = std::max(maxk, diff);
      if (diff > 1e-4) badk++;
    }
    if (r.valid[(size_t)p] != (all ? 1 : 0)) { std::printf("pool %d validity mismatch\n", p); badk++; }
    if (r.indices[(size_t)p * KP] != idx[0]) { std::printf("pool %d first index mismatch\n", p); badk++; }
    // scores for row 0 only (reference masks by visibility, not applied here)
    for (int l = 0; l < L; ++l) {
      double acc = 0;
      for (int h = 0; h < Nh; ++h) {
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += q[((size_t)l * Nh + h) * D + d] * kk[(size_t)d];
        double relu = dot > 0 ? dot : 0.0;
        acc += w[(size_t)l * Nh + h] * relu * (1.0 / std::sqrt((double)D));
      }
      double diff = std::fabs(acc - (double)S[(size_t)l * n + p]);
      maxs = std::max(maxs, diff);
      if (diff > 1e-4) bads++;
    }
  }
  std::printf("indexer pools=%d (kpool=%d), pooled keys: max|diff|=%.3g bad=%d, scores: max|diff|=%.3g bad=%d\n",
              n, KP, maxk, badk, maxs, bads);
  bool ok = badk == 0 && bads == 0;
  std::printf("indexer pooling/scoring: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
