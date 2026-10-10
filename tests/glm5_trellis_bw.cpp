// Effective weight-read bandwidth of the trellis GEMV, per tensor shape and row
// count - the measurement that says whether the trellis path is bandwidth-bound
// or launch/latency-bound. The eval() round-trip (~0.2 ms) is excluded by
// building a batch of independent iterations into one graph and evaluating once;
// an f16 matmul of the same output shape is included as the "what MLX's own
// GEMV achieves" reference.
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/glm5_trellis_bw \
//     tests/glm5_trellis_bw.cpp src/glm5_forward.cpp src/glm5.cpp src/glm5_kda.cpp \
//     src/glm5_attn.cpp src/glm5_hc.cpp src/glm5_moe.cpp src/glm5_mtp.cpp src/trellis_gemm.cpp \
//     src/safetensors.cpp src/model.cpp src/exl3.cpp src/tokenizer.cpp src/engine.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib \
//     -framework Foundation -framework Metal -framework Accelerate
//
//   usage: /tmp/glm5_trellis_bw [iterations_per_graph]
//
// The iteration count matters: at 8 iterations per graph the host-side dispatch
// of the 3 kernels per iteration is what gets measured (~123 GB/s for the
// 4096x24576 tensor at rows=1). From ~64 iterations on, the number is the
// kernel's. Measured on an M4 Max (128 GB), GB/s of packed weights read, raw
// QMV (no Hadamards), 64 iterations per graph, before -> after the aligned-word
// / row-blocking pass in trellis_gemm.cpp:
//
//   shape                        rows=1        rows=2       rows=4      rows=8
//   qkv_proj  4096x24576 K4   254 -> 305    139 -> 244   71 -> 138   26 -> 72
//   o_proj    8192x4096  K4   199 -> 225    115 -> 176  112 -> 124   32 -> 67
//   expert    4096x2048  K3   110 -> 134     64 -> 101   74 ->  77   24 -> 44
//   expert    2048x4096  K3   105 -> 123     60 ->  96   55 ->  58   22 -> 30
//   f_a       4096x256   K4    51 ->  45     29 ->  29   26 ->  28    9 -> 16
//   f16 matmul 4096x24576(ref)       420 GB/s at rows=1, 200 GB/s at rows 2-8
//
// so the trellis GEMV is 1.35x off a plain f16 GEMV in *time* at one row (it
// reads 4x fewer bytes), the rows collapse was accumulator spills rather than
// weight re-reads, and the small per-layer tensors (f_a) are launch-bound:
// batching them, not a better kernel, is what they need.
// Effective weight-read bandwidth of the trellis GEMV per shape and row count,
// with the mx::eval overhead excluded (a batch of iterations is built into one
// graph and evaluated once; a no-op baseline shows what dispatch alone costs).
#include <mlx/mlx.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "glm5.h"
#include "glm5_kda.h"
#include "trellis_gemm.h"

namespace mx = mlx::core;

static double now_ms2() {
  using c = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(c::now().time_since_epoch()).count();
}

// The packed words are built through the engine's own trellis_pack, so the
// benchmark measures the buffer layout the model actually runs on (aligned,
// engine-owned; a zero-copy shard view would be the only alternative).
static exl3::G5Linear make_trellis(int in, int out, int K, unsigned seed) {
  exl3::G5Linear l;
  l.quantized = true;
  l.in_features = in;
  l.out_features = out;
  l.tw.K = K;
  l.tw.codebook = 2;
  l.tw.in_features = in;
  l.tw.out_features = out;
  const int64_t nwords = (int64_t)(in / 16) * (out / 16) * (8 * K);
  std::vector<uint16_t> src((size_t)nwords);
  std::mt19937 rng(seed);
  for (auto& v : src) v = (uint16_t)rng();
  l.tw.packed = exl3::trellis_pack(src.data(), in / 16, out / 16, K);
  l.tw.suh = mx::ones({in}, mx::float16);
  l.tw.svh = mx::ones({out}, mx::float16);
  mx::eval(l.tw.packed, l.tw.suh, l.tw.svh);
  return l;
}

// Build `reps` independent evaluations of `build()` into ONE graph, then
// evaluate once - so the ~0.2 ms eval round-trip is amortised instead of being
// charged to every sample.
static double batch_ms(int reps, int rounds, auto&& build) {
  mx::array acc = mx::zeros({1}, mx::float32);
  mx::eval(acc);
  double best = 1e9;
  for (int rr = 0; rr < rounds; ++rr) {
    const double t0 = now_ms2();
    for (int i = 0; i < reps; ++i) acc = mx::add(acc, build(i));
    mx::eval(acc);
    best = std::min(best, (now_ms2() - t0) / reps);
  }
  return best;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 64;
  struct Shape {
    const char* name;
    int in, out, K;
  } shapes[] = {
      {"qkv_proj     4096x24576 K4", 4096, 24576, 4},
      {"o_proj       8192x4096  K4", 8192, 4096, 4},
      {"expert gate  4096x2048  K3", 4096, 2048, 3},
      {"expert down  2048x4096  K3", 2048, 4096, 3},
      {"kda f_a      4096x256   K4", 4096, 256, 4},
  };
  std::printf("=== trellis GEMV effective weight-read bandwidth (eval excluded, %d iters/graph) ===\n",
              reps);
  for (auto& sh : shapes) {
    exl3::G5Linear t = make_trellis(sh.in, sh.out, sh.K, 7u * sh.in + sh.out);
    const double bytes = (double)sh.in * sh.out * sh.K / 8.0;
    for (int rows : {1, 2, 4, 8}) {
      std::vector<float> xv((size_t)rows * sh.in, 0.05f);
      mx::array x = mx::astype(mx::array((void*)xv.data(), {rows, sh.in}, mx::float32, [](void*) {}),
                               mx::float16);
      mx::eval(x);
      const double ms = batch_ms(reps, 3, [&](int) { return exl3::trellis_linear(t.tw, x); });
      std::printf("  %-30s rows %d: %7.3f ms -> %5.1f GB/s\n", sh.name, rows, ms,
                  bytes / (ms * 1e-3) / 1e9);
    }
  }
  // reference: a plain f16 matmul reading the same number of bytes
  {
    const int in = 4096, out = 24576;
    const double bytes = (double)in * out * 2;
    std::vector<float> wv((size_t)in * out, 0.01f);
    mx::array w = mx::astype(mx::array((void*)wv.data(), {in, out}, mx::float32, [](void*) {}),
                             mx::float16);
    mx::eval(w);
    for (int rows : {1, 2, 4, 8}) {
      std::vector<float> xv((size_t)rows * in, 1.0f);
      mx::array x = mx::astype(mx::array((void*)xv.data(), {rows, in}, mx::float32, [](void*) {}),
                               mx::float16);
      mx::eval(x);
      const double ms = batch_ms(reps, 3, [&](int) { return mx::matmul(x, w); });
      std::printf("  %-30s rows %d: %7.3f ms -> %5.1f GB/s\n", "f16 matmul 4096x24576 (ref)", rows,
                  ms, bytes / (ms * 1e-3) / 1e9);
    }
  }
  return 0;
}
