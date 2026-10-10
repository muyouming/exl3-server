// KDA delta-rule recurrence: MLX implementation vs the reference formula
// (recurrent_kimi_delta_attention) evaluated as a scalar oracle in fp64.
//
// Build (from the repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include \
//     -o /tmp/kda_test tests/kda_recurrence_test.cpp src/glm5_kda.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib
// Expected: "max|diff| = ~4e-08, mismatches(>1e-4) = 0 / 56"
// KDA recurrence: MLX implementation vs a scalar CPU oracle (the reference formula).
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <mlx/mlx.h>
#include "glm5_kda.h"
namespace mx = mlx::core;
int main() {
  const int B = 1, L = 7, H = 2, KD = 4, VD = 4;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> q((size_t)B*L*H*KD), k(q.size()), v((size_t)B*L*H*VD), g((size_t)B*L*H), beta((size_t)B*L*H);
  for (auto& x : q) x = nd(rng);
  for (auto& x : k) x = nd(rng);
  for (auto& x : v) x = nd(rng);
  for (auto& x : g) x = -0.1f - 0.3f * std::fabs(nd(rng));   // negative decay, as the model produces
  for (auto& x : beta) x = 0.2f + 0.6f * std::fabs(nd(rng)) / 3.0f;
  auto A = [&](std::vector<float>& src, mx::Shape sh) {
    auto* h = new std::vector<float>(src);
    return mx::array((void*)h->data(), sh, mx::float32, [h](void*) { delete h; });
  };
  mx::array Q = A(q, {B, L, H, KD}), K = A(k, {B, L, H, KD}), V = A(v, {B, L, H, VD});
  mx::array G = A(g, {B, L, H, 1}), BE = A(beta, {B, L, H, 1});
  mx::array out = exl3::kda_recurrence(Q, K, V, G, BE);
  mx::eval(out);
  const float* o = out.data<float>();
  // scalar oracle
  double maxdiff = 0; int bad = 0;
  auto l2 = [&](const std::vector<float>& x, int b, int i, int h) {
    double s = 0; for (int d = 0; d < KD; ++d) { double t = x[((size_t)b*L + i)*H*KD + (size_t)h*KD + d]; s += t*t; }
    return std::sqrt(s + 1e-6);
  };
  for (int b = 0; b < B; ++b) {
    std::vector<double> S((size_t)H*KD*VD, 0.0);
    for (int i = 0; i < L; ++i) {
      for (int h = 0; h < H; ++h) {
        const double nq = l2(q, b, i, h), nk = l2(k, b, i, h);
        const double gi = g[((size_t)b*L + i)*H + h], bi = beta[((size_t)b*L + i)*H + h];
        for (int d = 0; d < KD; ++d)
          for (int e = 0; e < VD; ++e) S[(size_t)(h*KD + d)*VD + e] *= std::exp(gi);
        std::vector<double> kv((size_t)VD, 0.0);
        for (int d = 0; d < KD; ++d) {
          const double ki = k[((size_t)b*L + i)*H*KD + (size_t)h*KD + d] / nk;
          for (int e = 0; e < VD; ++e) kv[(size_t)e] += S[(size_t)(h*KD + d)*VD + e] * ki;
        }
        for (int e = 0; e < VD; ++e) {
          const double vi = v[((size_t)b*L + i)*H*VD + (size_t)h*VD + e];
          kv[(size_t)e] = (vi - kv[(size_t)e]) * bi;
        }
        for (int d = 0; d < KD; ++d) {
          const double ki = k[((size_t)b*L + i)*H*KD + (size_t)h*KD + d] / nk;
          for (int e = 0; e < VD; ++e) S[(size_t)(h*KD + d)*VD + e] += ki * kv[(size_t)e];
        }
        for (int e = 0; e < VD; ++e) {
          double acc = 0;
          for (int d = 0; d < KD; ++d) {
            const double qi = q[((size_t)b*L + i)*H*KD + (size_t)h*KD + d] / nq / std::sqrt((double)KD);
            acc += S[(size_t)(h*KD + d)*VD + e] * qi;
          }
          double got = o[(((size_t)b*L + i)*H + h)*VD + e];
          double d_ = std::fabs(got - acc);
          maxdiff = std::max(maxdiff, d_);
          if (d_ > 1e-4) bad++;
        }
      }
    }
  }
  std::printf("KDA recurrence vs scalar oracle: max|diff| = %.3g, mismatches(>1e-4) = %d / %d\n",
              maxdiff, bad, B*L*H*VD);
  std::printf("  first 6 mlx : %.5f %.5f %.5f %.5f %.5f %.5f\n", o[0], o[1], o[2], o[3], o[4], o[5]);
  {
    // oracle values for the same first entries (recompute crudely for printing)
    std::vector<double> S2((size_t)H*KD*VD, 0.0), ref;
    for (int i = 0; i < L; ++i)
      for (int h = 0; h < H; ++h) {
        const double nq = l2(q, 0, i, h), nk = l2(k, 0, i, h);
        const double gi = g[(size_t)i*H + h], bi = beta[(size_t)i*H + h];
        for (int d = 0; d < KD; ++d)
          for (int e = 0; e < VD; ++e) S2[(size_t)(h*KD + d)*VD + e] *= std::exp(gi);
        std::vector<double> kv((size_t)VD, 0.0);
        for (int d = 0; d < KD; ++d) {
          const double ki = k[((size_t)i)*H*KD + (size_t)h*KD + d] / nk;
          for (int e = 0; e < VD; ++e) kv[(size_t)e] += S2[(size_t)(h*KD+d)*VD+e] * ki;
        }
        for (int e = 0; e < VD; ++e) kv[(size_t)e] = (v[((size_t)i)*H*VD + (size_t)h*VD + e] - kv[(size_t)e]) * bi;
        for (int d = 0; d < KD; ++d) {
          const double ki = k[((size_t)i)*H*KD + (size_t)h*KD + d] / nk;
          for (int e = 0; e < VD; ++e) S2[(size_t)(h*KD+d)*VD+e] += ki * kv[(size_t)e];
        }
        for (int e = 0; e < VD; ++e) {
          double acc = 0;
          for (int d = 0; d < KD; ++d) {
            const double qi = q[((size_t)i)*H*KD + (size_t)h*KD + d] / nq / std::sqrt((double)KD);
            acc += S2[(size_t)(h*KD+d)*VD+e] * qi;
          }
          if (i == 0) ref.push_back(acc);
        }
      }
    std::printf("  oracle(H*VD=8) : ");
    for (double x : ref) std::printf("%.5f ", x);
    std::printf("\n");
  }
  return bad == 0 ? 0 : 1;
}
