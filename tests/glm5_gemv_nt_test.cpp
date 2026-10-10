// Bit-identity of the trellis GEMV under its *scheduling* knobs.
//
// The output-tiles-per-threadgroup (nt) and rows-per-threadgroup (mb) knobs
// only decide which threadgroup covers which output tile / row block, and in
// what order the threadgroup's simdgroups walk the k range. They change no
// accumulation order: per output element the k-tiles are visited in the same
// order and the two reductions (the 4 simdgroup partials, then the k-split
// partials) are unchanged. So every (nt, mb) pair must produce the *same bytes*
// as any other - which is what lets the auto-tuning rule in trellis_gemm.cpp be
// changed without touching the model's output.
//
// This is the local gate for that claim; the checkpoint-level gates
// (tests/glm5_exl3_decode_test.cpp against the reference decoder,
// tests/moe_test.cpp's grouped == per-expert == general hashes and the end-to-end
// greedy text) sit on top of it. Note that the k-split *count* is deliberately
// NOT in this set: it is a function of the tensor's own geometry only, because
// the split partials are summed by mx::sum in the caller and a different count
// is a different f32 association.
//
// Build (repo root):
//   MLX=$(brew --prefix mlx); SRC=$(ls src/*.cpp | grep -v '^src/main.cpp$')
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$MLX/include -o /tmp/glm5_gemv_nt_test \
//     tests/glm5_gemv_nt_test.cpp $SRC -L$MLX/lib -lmlx -Wl,-rpath,$MLX/lib \
//     -framework Foundation -framework Metal -framework Accelerate
//
//   usage: /tmp/glm5_gemv_nt_test        (no checkpoint, a few hundred MB resident)
#include <mlx/mlx.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "glm5.h"
#include "trellis_gemm.h"

namespace mx = mlx::core;

static exl3::TrellisWeight make_trellis(int in, int out, int K, unsigned seed) {
  exl3::TrellisWeight tw;
  tw.K = K;
  tw.codebook = 2;
  tw.in_features = in;
  tw.out_features = out;
  const int64_t nwords = (int64_t)(in / 16) * (out / 16) * (16 * K);
  std::vector<uint16_t> src((size_t)nwords);
  std::mt19937 rng(seed);
  for (auto& v : src) v = (uint16_t)rng();
  tw.packed = exl3::trellis_pack(src.data(), in / 16, out / 16, K);
  // Non-unit scales: a scheduling change that moved an operation across the
  // scale multiply would show up as a byte difference.
  std::vector<float> su((size_t)in), sv((size_t)out);
  std::mt19937 r2(seed ^ 0x5bd1u);
  std::uniform_real_distribution<float> d(0.5f, 1.5f);
  for (auto& v : su) v = d(r2);
  for (auto& v : sv) v = d(r2);
  tw.suh = mx::astype(mx::array(su.data(), {in}, mx::float32), mx::float16);
  tw.svh = mx::astype(mx::array(sv.data(), {out}, mx::float32), mx::float16);
  mx::eval(tw.packed, tw.suh, tw.svh);
  return tw;
}

static mx::array rand_f16(int rows, int in, unsigned seed) {
  std::vector<float> xv((size_t)rows * in);
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, 1.0f);
  for (auto& v : xv) v = d(rng);
  mx::array x = mx::astype(mx::array(xv.data(), {rows, in}, mx::float32), mx::float16);
  mx::eval(x);
  return x;
}

// The evaluated bytes of an array, as a std::vector<char>.
static std::vector<char> bytes_of(const mx::array& a) {
  mx::array c = mx::contiguous(a);
  mx::eval(c);
  const char* p = (const char*)c.data<void>();
  return std::vector<char>(p, p + c.nbytes());
}

static int g_fail = 0;

static void expect_same(const char* what, const std::vector<char>& want,
                        const std::vector<char>& got) {
  if (want == got) {
    std::printf("  ok   %s\n", what);
    return;
  }
  ++g_fail;
  size_t first = 0;
  while (first < want.size() && first < got.size() && want[first] == got[first]) ++first;
  std::printf("  FAIL %s: %zu bytes differ, first at %zu\n", what, want.size() - first, first);
}

struct Shape {
  const char* name;
  int in, out, K;
} kShapes[] = {
    {"kda f_a     4096x256    K4", 4096, 256, 4},
    {"exp gate    4096x2048   K2", 4096, 2048, 2},
    {"exp gate    4096x2048   K3", 4096, 2048, 3},
    {"exp down    2048x4096   K3", 2048, 4096, 3},
    {"kda o_proj  8192x4096   K4", 8192, 4096, 4},
    {"mla q_b     1536x16384  K4", 1536, 16384, 4},
    {"small K6    4096x512    K6", 4096, 512, 6},
    {"kda qkv     4096x24576  K4", 4096, 24576, 4},
};

// Every EXL3 tensor class the checkpoint's shard headers declare: 25 distinct
// (in, out, K) triples over the text tower, the visual tower excluded (`h6` in
// the model name is the last layer's K=6 set). The count column is how many
// tensors of the checkpoint carry that class.
struct RealShape {
  const char* name;
  int in, out, K, count;
} kRealShapes[] = {
    {"lm_head              4096x154880 K6", 4096, 154880, 6, 1},
    {"kda qkv_proj          4096x24576  K4", 4096, 24576, 4, 34},
    {"mla o_proj           16384x4096   K6", 16384, 4096, 6, 1},
    {"mla o_proj           16384x4096   K4", 16384, 4096, 4, 11},
    {"mtp eh_proj           8192x4096   K5", 8192, 4096, 5, 1},
    {"dense down           12288x4096   K3", 12288, 4096, 3, 3},
    {"dense gate/up         4096x12288  K3", 4096, 12288, 3, 6},
    {"mla q_b               1536x16384  K6", 1536, 16384, 6, 1},
    {"kda o_proj            8192x4096   K4", 8192, 4096, 4, 34},
    {"mla q_b               1536x16384  K4", 1536, 16384, 4, 11},
    {"shared down           2048x4096   K6", 2048, 4096, 6, 1},
    {"shared gate/up        4096x2048   K6", 4096, 2048, 6, 2},
    {"indexer wq_b          1536x4096   K6", 1536, 4096, 6, 1},
    {"indexer wq_b          4096x1536   K6", 4096, 1536, 6, 1},
    {"shared down           2048x4096   K4", 2048, 4096, 4, 330},
    {"shared gate/up        4096x2048   K4", 4096, 2048, 4, 660},
    {"exp down              2048x4096   K3", 2048, 4096, 3, 5760},
    {"exp gate/up           4096x2048   K3", 4096, 2048, 3, 11520},
    {"indexer wq_b          1536x4096   K4", 1536, 4096, 4, 11},
    {"indexer wq_b          4096x1536   K4", 4096, 1536, 4, 11},
    {"exp down              2048x4096   K2", 2048, 4096, 2, 6336},
    {"exp gate/up           4096x2048   K2", 4096, 2048, 2, 12672},
    {"mla kv_a              4096x512    K6", 4096, 512, 6, 1},
    {"mla kv_a              4096x512    K5", 4096, 512, 5, 2},
    {"mla kv_a              4096x512    K4", 4096, 512, 4, 9},
};

static uint64_t fnv1a(const std::vector<char>& b) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : b) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

// The byte-identity table: for every real tensor class and row count, the hash
// of the raw QMV output and of the whole trellis_linear. Two builds of this
// file must print the same table; that is the "no output byte moved" gate for
// the geometry rule (the per-config sweep above is the same claim at a finer
// grain, and moe_test.cpp's grouped == per-expert == general is the gate for
// the grouped launch).
static void hash_table() {
  std::printf("=== kernel output hashes over the real tensor classes x rows ===\n");
  for (auto& sh : kRealShapes) {
    exl3::TrellisWeight tw = make_trellis(sh.in, sh.out, sh.K, 13u * sh.in + sh.out);
    for (int rows : {1, 2, 3, 13, 26}) {
      mx::array x = rand_f16(rows, sh.in, 3u * rows + sh.out);
      mx::array xhat = rand_f16(rows, sh.in, 5u * rows + sh.out);
      exl3::trellis_set_tuning(exl3::TrellisTuning{0, 0, 0});
      const uint64_t h_raw = fnv1a(bytes_of(exl3::trellis_matmul_raw(tw, xhat)));
      const uint64_t h_lin = fnv1a(bytes_of(exl3::trellis_linear(tw, x)));
      std::printf("hash %-32s rows %2d x%5d raw %016llx lin %016llx\n", sh.name, rows, sh.count,
                  (unsigned long long)h_raw, (unsigned long long)h_lin);
      std::fflush(stdout);
    }
  }
}

int main() {
  std::printf("=== trellis GEMV: nt/mb scheduling must not move a byte ===\n");
  for (auto& sh : kShapes) {
    exl3::TrellisWeight tw = make_trellis(sh.in, sh.out, sh.K, 11u * sh.in + sh.out);
    for (int rows : {1, 2, 3, 4, 13, 26}) {
      mx::array x = rand_f16(rows, sh.in, 7u * rows + sh.in);
      // The reference: whatever the auto rule picks (nt=0, mb=0).
      exl3::trellis_set_tuning(exl3::TrellisTuning{0, 0, 0});
      std::vector<char> want = bytes_of(exl3::trellis_linear(tw, x));
      for (int nt : {1, 2, 4, 8}) {
        for (int mb : {1, 2, 4}) {
          char what[128];
          std::snprintf(what, sizeof(what), "%s rows %2d nt %d mb %d", sh.name, rows, nt, mb);
          exl3::trellis_set_tuning(exl3::TrellisTuning{nt, 0, mb});
          expect_same(what, want, bytes_of(exl3::trellis_linear(tw, x)));
        }
      }
      if (g_fail) {
        std::printf("trellis GEMV nt/mb invariance: FAIL\n");
        return 1;
      }
    }
    std::fflush(stdout);
  }
  // The same for the grouped launch (members differ only in which weight buffer
  // and which scale vector they read) and for its two scale-stack forms.
  {
    const int in = 4096, out = 2048, K = 3;
    std::vector<exl3::TrellisWeight> ws;
    for (int m = 0; m < 3; ++m) ws.push_back(make_trellis(in, out, K, 31u + m));
    for (int rows : {1, 2, 4}) {
      mx::array x = rand_f16(rows, in, 5u + rows);
      exl3::trellis_set_tuning(exl3::TrellisTuning{0, 0, 0});
      std::vector<char> want = bytes_of(exl3::trellis_linear_group(ws, x));
      std::vector<char> per;
      {
        std::vector<mx::array> ys;
        for (auto& w : ws) ys.push_back(exl3::trellis_linear(w, x));
        per = bytes_of(mx::stack(ys, 0));
      }
      char what[128];
      std::snprintf(what, sizeof(what), "grouped 3 members rows %d == per-expert", rows);
      expect_same(what, want, per);
      std::snprintf(what, sizeof(what), "grouped 3 members rows %d == scale-stack form", rows);
      expect_same(what, want, bytes_of(exl3::trellis_linear_group_scaled(
                                   ws, x, mx::concatenate({ws[0].suh, ws[1].suh, ws[2].suh}, 0),
                                   mx::concatenate({ws[0].svh, ws[1].svh, ws[2].svh}, 0))));
      for (int nt : {1, 2, 4}) {
        for (int mb : {1, 2, 4}) {
          std::snprintf(what, sizeof(what), "grouped 3 members rows %d nt %d mb %d", rows, nt, mb);
          exl3::trellis_set_tuning(exl3::TrellisTuning{nt, 0, mb});
          expect_same(what, want, bytes_of(exl3::trellis_linear_group(ws, x)));
        }
      }
      if (g_fail) {
        std::printf("trellis GEMV nt/mb invariance: FAIL\n");
        return 1;
      }
    }
  }
  std::printf("trellis GEMV nt/mb invariance: %s\n", g_fail ? "FAIL" : "PASS");
  hash_table();
  return g_fail ? 1 : 0;
}
