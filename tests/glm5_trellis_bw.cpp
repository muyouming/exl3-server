// Effective weight-read bandwidth of the trellis GEMV, per REAL tensor shape of
// the checkpoint and row count - the measurement that says whether the trellis
// path is bandwidth-bound or issue/latency-bound. The eval() round-trip
// (~0.1-0.2 ms) is excluded by building a batch of independent iterations into
// one graph and evaluating once; an f16 matmul of the same output shape is
// included as the "what MLX's own GEMV achieves" reference.
//
// Build (repo root):
//   MLX=$(brew --prefix mlx); SRC=$(ls src/*.cpp | grep -v '^src/main.cpp$')
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$MLX/include -o /tmp/glm5_trellis_bw \
//     tests/glm5_trellis_bw.cpp $SRC -L$MLX/lib -lmlx -Wl,-rpath,$MLX/lib \
//     -framework Foundation -framework Metal -framework Accelerate
//
//   usage: /tmp/glm5_trellis_bw [iterations_per_graph] [shapes|all] [rounds]
//
// Three rows per (shape, rows) cell, all over the same packed buffer:
//   qmv     yhat = xhat @ W_hat                      - the custom Metal kernel
//   lin     trellis_linear = qmv + both hadamards     - what a decode step calls
// `qmv` is the kernel's own rate (the number that says whether the kernel is
// fine); `lin` is what the model pays, so the difference is pure dispatch. The
// `xN` column is how many times the row blocking streams the weights (rows are
// padded to mb = min(rows,4) per threadgroup block), i.e. a multi-row cell's
// GB/s is not a DRAM rate unless it is multiplied by N.
//
// The iteration count matters: at 8 iterations per graph the host-side dispatch
// of the 3 kernels per iteration is what gets measured (~123 GB/s for the
// 4096x24576 tensor at rows=1). From ~64 iterations on, the number is the
// kernel's. The reported time is the median of `rounds` (default 5): the box
// drifts 10-30% between runs of the same binary, so only interleaved
// same-session runs of two builds are comparable at all. Measured on an M4 Max
// (128 GB), raw QMV, GB/s of packed weights read, after the fourth pass's
// geometry rule (docs/perf.md):
//
//   shape (real checkpoint geometry)   rows=1  rows=2  rows=4   rows=13  rows=26
//   lm_head     4096x154880  K6         450     362     211(DRAM 201)    (x7 = 206)
//   kda qkv     4096x24576   K4         308     252     138      (x4 139)   (x7 141)
//   mla o_proj  16384x4096   K4         325     242     129      (x4 133)   (x7 136)
//   exp8 gate|up 4096x2048   K3         202     (rows 2 149)
//   exp8 gate|up 4096x2048   K2         134
//   kda f_a     4096x256     K4          55
//   f16 matmul 4096x24576 (ref)         327 GB/s at rows=1, 202 at rows 2-8
//
// The tile rate behind those numbers is ~2.4e9 (16x16 output tiles)/s for every
// K - the machine's SIMD issue width - so the GB/s a shape shows is its bytes
// per tile (K/2 bytes per weight * 256) times that rate, and the small tensors
// (f_a, kv_a, single-tensor expert shapes) are latency/dispatch-bound.
#include <mlx/mlx.h>

#include <algorithm>
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
  // trellis_pack reads the raw i16 stream: 16*K u16 per 16x16 tile.
  const int64_t nwords = (int64_t)(in / 16) * (out / 16) * (16 * K);
  std::vector<uint16_t> src((size_t)nwords);
  std::mt19937 rng(seed);
  for (auto& v : src) v = (uint16_t)rng();
  l.tw.packed = exl3::trellis_pack(src.data(), in / 16, out / 16, K);
  l.tw.suh = mx::ones({in}, mx::float16);
  l.tw.svh = mx::ones({out}, mx::float16);
  mx::eval(l.tw.packed, l.tw.suh, l.tw.svh);
  return l;
}

static mx::array f16_rows(int rows, int in, float fill) {
  std::vector<float> xv((size_t)rows * in, fill);
  mx::array x =
      mx::astype(mx::array((void*)xv.data(), {rows, in}, mx::float32, [](void*) {}), mx::float16);
  mx::eval(x);
  return x;
}

// Build `reps` independent evaluations of `build()` into ONE graph, then
// evaluate once - so the ~0.2 ms eval round-trip is amortised instead of being
// charged to every sample. Every intermediate stays alive in `acc`.
//
// The reported number is the MEDIAN of `rounds` repeats: the box drifts by
// 10-30% between runs of the same binary (clocks and page cache), the min is
// biased by whichever repeat got lucky first, and only interleaved same-session
// runs of the two builds are comparable at all.
static double batch_ms(int reps, int rounds, auto&& build) {
  mx::array acc = mx::zeros({1}, mx::float32);
  mx::eval(acc);
  std::vector<double> t((size_t)rounds);
  for (int rr = 0; rr < rounds; ++rr) {
    const double t0 = now_ms2();
    for (int i = 0; i < reps; ++i) acc = mx::add(acc, build(i));
    mx::eval(acc);
    t[(size_t)rr] = (now_ms2() - t0) / reps;
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

// The real checkpoint's EXL3 shape classes (from the shard headers): in, out, K.
struct Shape {
  const char* name;
  int in, out, K, count; // count = tensors of this class in the checkpoint
} kShapes[] = {
    {"lm_head          4096x154880 K6", 4096, 154880, 6, 1},
    {"kda qkv_proj     4096x24576  K4", 4096, 24576, 4, 34},
    {"mla o_proj      16384x4096   K4", 16384, 4096, 4, 11},
    {"kda o_proj       8192x4096   K4", 8192, 4096, 4, 34},
    {"exp gate K2      4096x2048   K2", 4096, 2048, 2, 22 * 288},
    {"exp gate K3      4096x2048   K3", 4096, 2048, 3, 20 * 288},
    {"exp down K2      2048x4096   K2", 2048, 4096, 2, 22 * 288},
    {"exp down K3      2048x4096   K3", 2048, 4096, 3, 20 * 288},
    {"shared gate      4096x2048   K4", 4096, 2048, 4, 43},
    {"shared down      2048x4096   K4", 2048, 4096, 4, 43},
    {"dense gate/up    4096x12288  K3", 4096, 12288, 3, 3 * 2},
    {"dense down      12288x4096   K3", 12288, 4096, 3, 3},
    {"mla q_b          1536x16384  K4", 1536, 16384, 4, 11},
    {"mla kv_a         4096x512    K4", 4096, 512, 4, 11},
    {"kda f_a          4096x256    K4", 4096, 256, 4, 34},
};

// The kernel reads a weight tile once per row block: rows are padded to mb
// (= min(rows,4)) rows per threadgroup block, and each block streams the whole
// tensor. So a cell's DRAM traffic is bytes * passes(), and a GB/s figure that
// does not say so understates the memory rate for the multi-row cells.
static int weight_passes(int rows) {
  const int mb = rows >= 4 ? 4 : rows;
  int pad = ((rows + mb - 1) / mb) * mb;
  if (pad >= 64) pad = ((pad + 31) / 32) * 32;
  return pad / mb;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 64;
  const char* sel = argc > 2 ? argv[2] : "all";
  const int rounds = argc > 3 ? std::atoi(argv[3]) : 5;
  std::printf("=== trellis GEMV effective weight-read bandwidth (eval excluded, %d iters/graph, median of %d) ===\n",
              reps, rounds);
  for (auto& sh : kShapes) {
    if (std::string(sel) != "all" && std::string(sh.name).find(sel) == std::string::npos) continue;
    exl3::G5Linear t = make_trellis(sh.in, sh.out, sh.K, 7u * sh.in + sh.out);
    const double bytes = (double)sh.in * sh.out * sh.K / 8.0;
    for (int rows : {1, 2, 3, 4, 13, 26}) {
      mx::array x = f16_rows(rows, sh.in, 0.05f);
      mx::array xhat = f16_rows(rows, sh.in, 0.05f);
      const double t_raw = batch_ms(reps, rounds, [&](int) { return exl3::trellis_matmul_raw(t.tw, xhat); });
      const double t_lin = batch_ms(reps, rounds, [&](int) { return exl3::trellis_linear(t.tw, x); });
      const int passes = weight_passes(rows);
      std::printf("  %-32s rows %2d: qmv %8.4f ms %6.1f GB/s x%d (%6.1f DRAM) | lin %8.4f ms "
                  "%6.1f GB/s\n",
                  sh.name, rows, t_raw, bytes / (t_raw * 1e-3) / 1e9, passes,
                  bytes * passes / (t_raw * 1e-3) / 1e9, t_lin, bytes / (t_lin * 1e-3) / 1e9);
    }
    std::fflush(stdout);
  }
  // the routed experts of one decode step: 8 members, one grouped launch each
  // for gate|up and one for the down (the shapes the model's MoE pays for)
  for (int K : {2, 3}) {
    std::vector<exl3::TrellisWeight> gu, dn;
    for (int m = 0; m < 8; ++m) {
      gu.push_back(make_trellis(4096, 2048, K, 100u + m).tw);
      dn.push_back(make_trellis(2048, 4096, K, 200u + m).tw);
    }
    const double bytes = 2.0 * 8.0 * 4096.0 * 2048.0 * K / 8.0; // gate|up and down
    for (int rows : {1, 2, 4}) {
      mx::array x = f16_rows(rows, 4096, 0.05f);
      mx::array xd = f16_rows(rows, 2048, 0.05f);
      const double t_gu = batch_ms(reps, rounds, [&](int) { return exl3::trellis_linear_group(gu, x); });
      const double t_dn = batch_ms(reps, rounds, [&](int) { return exl3::trellis_linear_group(dn, xd); });
      std::printf("  exp8 gate|up+down K%d           rows %2d: gu %8.4f dn %8.4f ms %6.1f GB/s\n", K,
                  rows, t_gu, t_dn, bytes / ((t_gu + t_dn) * 1e-3) / 1e9);
    }
    std::fflush(stdout);
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
      mx::array x = f16_rows(rows, in, 1.0f);
      const double ms = batch_ms(reps, rounds, [&](int) { return mx::matmul(x, w); });
      std::printf("  %-32s rows %2d: %8.4f ms %6.1f GB/s\n", "f16 matmul 4096x24576 (ref)", rows,
                  ms, bytes / (ms * 1e-3) / 1e9);
    }
  }
  return 0;
}
