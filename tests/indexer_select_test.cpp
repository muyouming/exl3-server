// DSA indexer selection: top select_k pools expanded to token ids, the
// incomplete tail pool appended (append_visible_tail), fixed output width and
// padding rows masked to -1. Crafted pools so the ranking and tail are both
// observable.
// Build (repo root): clang++ -std=c++20 -O2 -Ivendor -Isrc -o /tmp/isel_test \
//   tests/indexer_select_test.cpp src/glm5_attn.cpp src/glm5_kda.cpp src/glm5.cpp \
//   src/exl3.cpp src/safetensors.cpp src/trellis_gemm.cpp -lmlx
// Expected: all checks ok, PASS
// Indexer selection: top select_k pools expanded to token ids, incomplete tail
// appended, padding rows masked to -1.
#include <cstdio>
#include <set>
#include <vector>
#include <mlx/mlx.h>
#include "glm5_attn.h"
namespace mx = mlx::core;
static int fails = 0;
static void chk(bool c, const char* what) { if (c) std::printf("  ok   %s\n", what); else { fails++; std::printf("  FAIL %s\n", what); } }
int main() {
  const int kp = 4, index_topk = 8, Ltot = 11;  // pools: 0..7 (2 full pools) + tail 8,9,10
  const int L = 2, n = (Ltot + kp - 1) / kp;    // n = 3
  exl3::IndexerPools pl;
  pl.kpool = kp; pl.n_pools = n;
  pl.valid.assign((size_t)n, 1);
  pl.valid[2] = 0;  // the incomplete pool is never selectable
  pl.indices.assign((size_t)n * kp, -1);
  for (int t = 0; t < 8; ++t) pl.indices[(size_t)t] = t;
  for (int s = 0; s < 3; ++s) pl.indices[(size_t)2 * kp + s] = 8 + s;  // incomplete last pool
  // scores: row 0 prefers pool 2 then 0; row 1 is fully masked (padding row)
  std::vector<float> sc((size_t)L * n, -INFINITY);
  sc[0 * n + 2] = 5.0f; sc[0 * n + 0] = 3.0f;
  auto wrap = [](std::vector<float>& v, mx::Shape sh) {
    auto* h = new std::vector<float>(v);
    return mx::array((void*)h->data(), sh, mx::float32, [h](void*) { delete h; });
  };
  pl.scores = wrap(sc, {L, n});
  std::vector<char> vis((size_t)L * Ltot, 0);
  for (int t = 0; t < Ltot; ++t) vis[t] = 1;          // row 0 sees everything
  std::vector<char> rowvis = {1, 0};                   // row 1 is a padding row
  mx::array out = exl3::indexer_select(pl, vis, rowvis, index_topk, true);
  mx::eval(out);
  chk(out.shape(0) == L && out.shape(1) == index_topk + kp - 1, "output width = topk + kpool - 1");
  const int32_t* o = out.data<int32_t>();
  const int W = index_topk + kp - 1;
  std::printf("  row0:");
  for (int i = 0; i < W; ++i) std::printf(" %d", o[0 * W + i]);
  std::printf("\n  row1:");
  for (int i = 0; i < W; ++i) std::printf(" %d", o[1 * W + i]);
  std::printf("\n");
  // row 0: the preferred pool (2) comes first with its 3 valid ids, then pool 0's 4
  // pool 0 scored 3.0 and pool 1 -inf (pool 2 is unscorable: incomplete), so the
  // ranking is pool 0 then pool 1
  chk(o[0] == 0 && o[1] == 1 && o[2] == 2 && o[3] == 3, "best-scoring pool first, ids in order");
  chk(o[4] == 4 && o[5] == 5 && o[6] == 6 && o[7] == 7, "second pool expanded in order");
  chk(o[8] == 8 && o[9] == 9 && o[10] == 10, "incomplete tail pool appended after the pools");
  std::set<int> seen;
  bool alldistinct = true, inrange = true;
  for (int i = 0; i < W; ++i) {
    int v = o[i];
    if (v == -1) continue;
    if (v < 0 || v >= Ltot) inrange = false;
    if (!seen.insert(v).second) alldistinct = false;
  }
  chk(alldistinct, "no duplicate token ids");
  chk(inrange, "all ids within the cache");
  bool row1_all_minus1 = true;
  for (int i = 0; i < W; ++i) if (o[1 * W + i] != -1) row1_all_minus1 = false;
  chk(row1_all_minus1, "padding row is all -1");
  std::printf("indexer selection: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
