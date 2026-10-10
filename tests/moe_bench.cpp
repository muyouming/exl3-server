// Micro-benchmark of ONE decode-step MoE block (the per-layer machinery that
// dominates the step). Synthetic weights with the real checkpoint's shapes
// (hidden 4096, moe_intermediate 2048, 288 routed experts top-8, trellis K=3,
// shared expert K=4), so it needs no 100 GB checkpoint.
//
// It reports two things:
//   1. sub-stage times, each an (build graph -> mx::eval) cycle, so every
//      number includes one CPU/GPU sync (~0.13 ms on this machine - subtract
//      the "null" row to get the stage's own cost), and
//   2. the whole block with the ids handed over by the host (no sync inside),
//      which is what device-side routing would look like.
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/moe_bench \
//     tests/moe_bench.cpp src/glm5_moe.cpp src/glm5_kda.cpp src/glm5_attn.cpp src/glm5.cpp \
//     src/glm5_forward.cpp src/glm5_hc.cpp src/glm5_mtp.cpp src/model.cpp src/exl3.cpp \
//     src/safetensors.cpp src/trellis_gemm.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib
//   usage: /tmp/moe_bench [reps] [rows] [model_dir]
// Without a model_dir the layer is synthetic (the same shapes, 8 distinct
// buffers); with one, layer 4's real weights are used and the run also prints
// the bit-identity line (prepared vs old-grouped vs general).
//
// Weight-storage variants (the questions the numbers in docs/perf.md answer):
//   MB_MALLOC_PACK=1  trellis_pack's malloc'd buffer instead of an mmap'd one
//   MB_BYTE_PATH=1    force the four-byte spelling of a tile word
//   MB_FILE_BACKED=1  a real file mapping with the tensor 2 bytes into a page
//                     (the loader's zero-copy case: unaligned, file-backed)
#include <mlx/mlx.h>

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "engine.h"
#include "glm5_forward.h"
#include "glm5_attn.h"
#include "glm5_hc.h"
#include "glm5_kda.h"
#include "glm5_moe.h"
#include "trellis_gemm.h"

namespace mx = mlx::core;

static double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// One packed trellis weight of the requested shape, filled with a deterministic
// bit pattern (the values do not matter for timing).
// The real loader keeps the packed words in a page-aligned mapping of the shard
// file (a zero-copy Metal buffer); trellis_pack instead hands MLX a malloc'd
// pointer that must be copied into a Metal buffer on first use. Both are
// measured because they are not the same speed.
static int g_mmap_backed = 1;
static int g_force_byte_path = 0; // 0 = aligned word loads, 1 = the four-byte spelling
static int g_file_backed = 0;     // 1 = a real file mapping, data 2 bytes into a page

static exl3::G5Linear mkq(int in, int out, int K, unsigned seed) {
  exl3::G5Linear lin;
  lin.quantized = true;
  const int kt = in / 16, nt = out / 16;
  std::vector<uint16_t> words((size_t)kt * nt * 16 * K);
  std::mt19937 rng(seed);
  for (auto& w : words) w = (uint16_t)(rng() & 0xffffu);
  if (g_file_backed) {
    // The real loader's case: a page-aligned mapping of a file whose tensor
    // starts at a byte offset that is not a multiple of 4.
    static int fd = -1;
    if (fd < 0) {
      char path[] = "/tmp/moe_bench_pack_XXXXXX";
      fd = mkstemp(path);
      if (fd < 0) { std::printf("mkstemp failed\n"); std::exit(1); }
      unlink(path);
    }
    const size_t nb = words.size() * 2;
    if (ftruncate(fd, (off_t)(2 + nb)) != 0) { std::printf("ftruncate failed\n"); std::exit(1); }
    if (pwrite(fd, words.data(), nb, 2) != (ssize_t)nb) { std::printf("pwrite failed\n"); std::exit(1); }
    const int64_t page = 16384;
    void* base = ::mmap(nullptr, (size_t)(nb + page), PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { std::printf("mmap failed\n"); std::exit(1); }
    mx::array raw(base, mx::Shape{(int)(nb + page)}, mx::uint8, [](void*) {});
    mx::array bytes = mx::slice(raw, {2}, {(int)(2 + nb)});
    lin.tw.packed = mx::reshape(bytes, {kt, nt, (int)(nb / (size_t)(kt * nt))});
  } else if (g_mmap_backed) {
    const size_t nb = words.size() * 2;
    void* base = ::mmap(nullptr, nb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (base == MAP_FAILED) { std::printf("mmap failed\n"); std::exit(1); }
    std::memcpy(base, words.data(), nb);
    lin.tw.packed = mx::array(base, mx::Shape{kt, nt, (int)nb / (kt * nt)}, mx::uint8,
                              [](void*) {});
  } else {
    lin.tw.packed = exl3::trellis_pack(words.data(), kt, nt, K);
  }
  lin.tw.aligned = !g_force_byte_path && !g_file_backed;
  lin.tw.in_features = in;
  lin.tw.out_features = out;
  lin.tw.K = K;
  lin.tw.codebook = 2;
  std::vector<uint16_t> su((size_t)in), sv((size_t)out);
  for (size_t i = 0; i < su.size(); ++i) su[i] = (uint16_t)(0x3800u + (i % 23));
  for (size_t i = 0; i < sv.size(); ++i) sv[i] = (uint16_t)(0x3400u + (i % 17));
  lin.tw.suh = mx::array(su.data(), mx::Shape{in}, mx::float16);
  lin.tw.svh = mx::array(sv.data(), mx::Shape{out}, mx::float16);
  mx::eval(lin.tw.suh, lin.tw.svh);
  lin.in_features = in;
  lin.out_features = out;
  return lin;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
  if (const char* e = std::getenv("MB_MALLOC_PACK")) g_mmap_backed = std::atoi(e) ? 0 : 1;
  if (const char* e = std::getenv("MB_BYTE_PATH")) g_force_byte_path = std::atoi(e);
  if (const char* e = std::getenv("MB_FILE_BACKED")) g_file_backed = std::atoi(e);
  const int rows = argc > 2 ? std::atoi(argv[2]) : 1;
  const int H = 4096, I = 2048, E = 288, K = 8;
  exl3::ModelConfig cfg;
  cfg.hidden_size = H;
  cfg.moe_intermediate_size = I;
  cfg.n_routed_experts = E;
  cfg.num_experts_per_tok = K;
  cfg.n_group = 1;
  cfg.topk_group = 1;
  cfg.norm_topk_prob = true;
  cfg.routed_scaling_factor = 2.5f;
  cfg.swiglu_limit = 10.0f;

  exl3::G5Layer l;
  l.moe = true;
  const int NEXP = 8; // distinct buffers; the 288 slots alias them (timing only)
  std::vector<exl3::G5Linear> pool;
  for (int e = 0; e < NEXP; ++e) {
    const unsigned s = 100u + (unsigned)e;
    pool.push_back(mkq(H, I, 3, s * 3 + 0));
    pool.push_back(mkq(H, I, 3, s * 3 + 1));
    pool.push_back(mkq(I, H, 3, s * 3 + 2));
  }
  l.experts.resize((size_t)E * 3);
  for (int e = 0; e < E; ++e)
    for (int j = 0; j < 3; ++j) l.experts[(size_t)(e * 3 + j)] = pool[(size_t)((e % NEXP) * 3 + j)];
  std::vector<uint16_t> rw16((size_t)H * E);
  for (size_t i = 0; i < rw16.size(); ++i) rw16[i] = (uint16_t)(0x2800u + (i % 41));
  std::vector<float> rb((size_t)E, 0.01f);
  l.router_w = mx::array(rw16.data(), mx::Shape{E, H}, mx::float16); // the checkpoint's dtype
  l.router_bias = mx::array(rb.data(), mx::Shape{E}, mx::float32);
  l.shared_gate = mkq(H, I, 4, 901);
  l.shared_up = mkq(H, I, 4, 902);
  l.shared_down = mkq(I, H, 4, 903);
  l.dense_gate = l.shared_gate;
  l.dense_up = l.shared_up;
  l.dense_down = l.shared_down;
  mx::eval(l.router_w, l.router_bias);

  std::vector<uint16_t> hh((size_t)rows * H);
  for (size_t i = 0; i < hh.size(); ++i) hh[i] = (uint16_t)(0x3000u + (i % 29));
  mx::array h(hh.data(), mx::Shape{1, rows, H}, mx::float16);
  mx::eval(h);

  // Real-model mode: use layer 4's actual weights (and the checkpoint's own
  // config) instead of the synthetic ones.
  std::unique_ptr<exl3::Engine> engine;
  const exl3::Glm5Weights* rw = nullptr;
  if (argc > 3) {
    engine = std::make_unique<exl3::Engine>();
    std::string err;
    if (!engine->load(argv[3], err, 0, false)) {
      std::fprintf(stderr, "load: %s\n", err.c_str());
      return 1;
    }
    exl3::Glm5Model* g5 = engine->model().glm5();
    if (!g5) { std::fprintf(stderr, "not a GLM-5 checkpoint\n"); return 1; }
    cfg = engine->config();
    rw = &g5->weights();
    l = rw->layers[4];
    l.dense_gate = l.shared_gate; // the shared-expert row times the same weights
    l.dense_up = l.shared_up;
    l.dense_down = l.shared_down;
    mx::eval(exl3::moe_layer(l, cfg, h));
    std::printf("real layer 4: %zu experts, router (%d,%d)\n", l.experts.size() / 3,
                (int)l.router_w.shape(0), (int)l.router_w.shape(1));
  }

  for (int r = 0; r < 3; ++r) mx::eval(exl3::moe_layer(l, cfg, h)); // warm up

  // ---- sub-stages ----------------------------------------------------------
  struct Row {
    std::string name;
    std::function<mx::array()> run;
    std::vector<double> t;
  };
  std::vector<Row> rows_t;
  auto add = [&](const std::string& n, std::function<mx::array()> f) {
    rows_t.push_back(Row{n, std::move(f), {}});
  };

  const int N = (int)h.size() / H;
  add("null (one elementwise op + sync)", [&] { return mx::add(h, mx::array(0.0f, mx::float16)); });
  add("router: logits matmul", [&] {
    mx::array x = mx::reshape(mx::astype(h, mx::float32), {N, H});
    return mx::matmul(x, mx::transpose(mx::astype(l.router_w, mx::float32)));
  });
  add("router: + topk route + ids eval", [&] {
    mx::array x = mx::reshape(mx::astype(h, mx::float32), {N, H});
    mx::array logits = mx::matmul(x, mx::transpose(mx::astype(l.router_w, mx::float32)));
    auto w_i = exl3::moe_topk_route(logits, l.router_bias, cfg.n_group, cfg.topk_group, K,
                                    cfg.norm_topk_prob, cfg.routed_scaling_factor);
    return w_i.second;
  });
  add("reshape+tile activation to (K,rows,in)", [&] {
    mx::array xr = mx::reshape(h, {1, N, H});
    return mx::reshape(mx::tile(xr, {K, 1, 1}), {K, N, H});
  });
  add("1 grouped linear (8 members, K=3)", [&] {
    std::vector<exl3::TrellisWeight> wv;
    for (int k = 0; k < K; ++k) wv.push_back(l.experts[(size_t)(k * 3)].tw);
    return exl3::trellis_linear_group(wv, mx::reshape(h, {N, H}));
  });
  add("3 grouped linears (gate/up/down)", [&] {
    std::vector<exl3::TrellisWeight> wg, wu, wd;
    for (int k = 0; k < K; ++k) {
      wg.push_back(l.experts[(size_t)(k * 3 + 0)].tw);
      wu.push_back(l.experts[(size_t)(k * 3 + 1)].tw);
      wd.push_back(l.experts[(size_t)(k * 3 + 2)].tw);
    }
    mx::array flat = mx::reshape(h, {N, H});
    mx::array gate = exl3::trellis_linear_group(wg, flat);
    mx::array up = exl3::trellis_linear_group(wu, flat);
    gate = mx::clip(gate, std::nullopt, mx::array(cfg.swiglu_limit, gate.dtype()));
    up = mx::clip(up, mx::array(-cfg.swiglu_limit, up.dtype()), mx::array(cfg.swiglu_limit, up.dtype()));
    mx::array act = mx::multiply(gate, mx::sigmoid(gate));
    mx::array actup = mx::multiply(act, up);
    mx::array down = exl3::trellis_linear_group_rows(wd, actup, (int)(actup.size() / ((int64_t)K * I)));
    return down;
  });
  // The pieces of one grouped linear: the raw QMV with pre-built inputs (no
  // tile, no concatenate) vs the full grouped call.
  std::vector<exl3::TrellisWeight> wv;
  std::vector<mx::array> svh_v, suh_v;
  for (int k = 0; k < K; ++k) {
    wv.push_back(l.experts[(size_t)(k * 3)].tw);
    svh_v.push_back(wv.back().svh);
    suh_v.push_back(wv.back().suh);
  }
  mx::array sstack_v = mx::concatenate(svh_v, 0);
  mx::array sstack_u = mx::concatenate(suh_v, 0);
  mx::array xr = mx::reshape(mx::tile(mx::reshape(h, {1, N, H}), {K, 1, 1}), {K, N, H});
  mx::array xhat = exl3::trellis_hadamard(xr, sstack_u, H / 128, /*pre=*/true, N, /*stacked=*/true);
  mx::eval(xhat);
  // The real decode step reads a *different* 88 MB of expert weights every
  // layer, so nothing stays in the cache. This row launches the same QMV over
  // 8 disjoint 8-expert sets out of a 128-expert pool (400 MB) in one eval:
  // (total - null) / 8 is the cold per-launch cost.
  static std::vector<exl3::G5Linear> big_pool;
  for (int e = 0; e < 128; ++e) {
    const unsigned s2 = 5000u + (unsigned)e;
    big_pool.push_back(mkq(H, I, 3, s2));
  }
  add("8x grouped QMV over 8 different layers (real bytes)", [&] {
    mx::array acc = mx::array(0.0f);
    for (int m = 0; m < 8; ++m) {
      const exl3::G5Layer& lm = rw ? rw->layers[(size_t)(4 + m)] : l;
      std::vector<exl3::TrellisWeight> wc;
      for (int k = 0; k < K; ++k) wc.push_back(lm.experts[(size_t)(k * 3)].tw);
      mx::array xh = exl3::trellis_hadamard(xr, sstack_u, H / 128, /*pre=*/true, N, /*stacked=*/true);
      mx::array y = exl3::trellis_matmul_group(wc, xh, N);
      acc = (m == 0) ? y : mx::add(acc, y);
    }
    return acc;
  });
  add("8x raw QMV over a 400MB pool (cold data)", [&] {
    mx::array acc = mx::array(0.0f);
    for (int m = 0; m < 8; ++m) {
      std::vector<exl3::TrellisWeight> wc;
      for (int k = 0; k < 8; ++k) wc.push_back(big_pool[(size_t)(m * 8 + k)].tw);
      mx::array xh = exl3::trellis_hadamard(xr, sstack_u, H / 128, /*pre=*/true, N, /*stacked=*/true);
      mx::array y = exl3::trellis_matmul_group(wc, xh, N);
      acc = (m == 0) ? y : mx::add(acc, y);
    }
    return acc;
  });
  add("raw QMV only (prebuilt xhat, one launch)", [&] {
    return exl3::trellis_matmul_group(wv, xhat, N);
  });
  add("raw QMV + out hadamard (no concat/tile)", [&] {
    mx::array y = exl3::trellis_matmul_group(wv, xhat, N);
    return exl3::trellis_hadamard(mx::reshape(y, {K * N, 2048}), sstack_v, 2048 / 128, /*pre=*/false, N, /*stacked=*/true);
  });
  add("2x raw QMV in one eval (independent)", [&] {
    mx::array a1 = exl3::trellis_matmul_group(wv, xhat, N);
    mx::array a2 = exl3::trellis_matmul_group(wv, xhat, N);
    return mx::add(a1, a2);
  });
  add("8x raw QMV in one eval (independent)", [&] {
    mx::array acc = exl3::trellis_matmul_group(wv, xhat, N);
    for (int i = 1; i < 8; ++i) acc = mx::add(acc, exl3::trellis_matmul_group(wv, xhat, N));
    return acc;
  });
  add("one grouped linear w/o eval (encode only)", [&] {
    std::vector<exl3::TrellisWeight> wg;
    for (int k = 0; k < K; ++k) wg.push_back(l.experts[(size_t)(k * 3)].tw);
    mx::array y = exl3::trellis_linear_group(wg, mx::reshape(h, {N, H}));
    mx::eval(mx::array(0.0f)); // the null sync; the grouped graph above is never evaluated
    return y;
  });
  add("shared expert (dense_mlp, K=4)", [&] { return exl3::dense_mlp(l, cfg, h); });
  add("full moe_layer (shipped path, +ids eval)", [&] { return exl3::moe_layer(l, cfg, h); });
  exl3::G5HcConfig hc;
  mx::array streams = mx::array(0.0f), coll = mx::array(0.0f);
  exl3::G5HcActivation A;
  if (rw) {
    hc = exl3::g5_hc_config(cfg);
    streams = exl3::g5_hc_expand(h, hc.mult);
    mx::eval(streams);
    A = exl3::g5_hc_prepare(streams, l.hc_attn_base, l.hc_attn_fn, l.hc_attn_scale, hc);
    mx::eval(A.collapsed, A.post, A.comb);
    coll = exl3::rms_norm(A.collapsed, l.input_norm, cfg.rms_norm_eps);
    mx::eval(coll);
    add("hc_prepare (real fn 24x16384 f32)", [&] {
      return exl3::g5_hc_prepare(streams, l.hc_attn_base, l.hc_attn_fn, l.hc_attn_scale, hc)
          .collapsed;
    });
    add("hc_mix", [&] { return exl3::g5_hc_mix(coll, A.post, A.comb, streams); });
    add("rms_norm (1,1,4096)", [&] {
      return exl3::rms_norm(A.collapsed, l.input_norm, cfg.rms_norm_eps);
    });
    add("lm_head (trellis, real vocab)", [&] {
      return exl3::trellis_linear(rw->lm_head.tw, mx::astype(h, mx::float16));
    });
    add("2nd MoE layer (layer 12, its own K)", [&] {
      return exl3::moe_layer(rw->layers[12], cfg, h);
    });
    // The rest of the decode step, uninstrumented (the profile in glm5_bench
    // charges one eval barrier to every stage; these are the same stages with
    // the barrier subtracted).
    static exl3::KdaState kst;
    static exl3::MlaState mst;
    static exl3::IndexerState ist;
    static int mpos = 0;
    const exl3::G5Layer& lk = rw->layers[0];  // linear_attention (KDA)
    const exl3::G5Layer& lm = rw->layers[3];  // sparse_attention (MLA + indexer)
    // Bit-identity on the real shapes and weights: the prepared decode path,
    // the path that concatenates the members' own scale vectors, and the
    // general gather/scatter path must produce the same bytes.
    {
      auto hash = [](const mx::array& y) {
        mx::eval(y);
        const uint8_t* p = (const uint8_t*)y.data<uint16_t>();
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < (size_t)y.size() * 2; ++i) h = (h ^ p[i]) * 1099511628211ull;
        return h;
      };
      // Is hoisting the router's f16->f32 cast bit-safe? A lazy cast can be
      // fused into the matmul kernel; materialising it first cannot.
      {
        mx::array xf = mx::reshape(mx::astype(h, mx::float32), {1, H});
        mx::array lg_lazy = mx::matmul(xf, mx::transpose(mx::astype(l.router_w, mx::float32)));
        mx::array w32 = mx::astype(l.router_w, mx::float32);
        mx::eval(w32);
        mx::array lg_mat = mx::matmul(xf, mx::transpose(w32));
        mx::eval(lg_lazy, lg_mat);
        const float* pa = lg_lazy.data<float>();
        const float* pb = lg_mat.data<float>();
        float md = 0.0f;
        for (int i = 0; i < E; ++i) md = std::max(md, std::fabs(pa[i] - pb[i]));
        auto ra = exl3::moe_topk_route(lg_lazy, l.router_bias, cfg.n_group, cfg.topk_group, K,
                                       cfg.norm_topk_prob, cfg.routed_scaling_factor);
        auto rb = exl3::moe_topk_route(lg_mat, l.router_bias, cfg.n_group, cfg.topk_group, K,
                                       cfg.norm_topk_prob, cfg.routed_scaling_factor);
        mx::eval(ra.second, rb.second);
        const int32_t* ia = ra.second.data<int32_t>();
        const int32_t* ib = rb.second.data<int32_t>();
        bool same_ids = true;
        for (int i = 0; i < K; ++i) same_ids = same_ids && ia[i] == ib[i];
        std::printf("  [router] lazy-cast vs materialised-cast logits max|diff|=%.3g, ids equal: %d\n",
                    (double)md, (int)same_ids);
      }
      unsetenv("EXL3_MOE_OLDPREP");
      const uint64_t hn = hash(exl3::moe_layer(l, cfg, h));
      setenv("EXL3_MOE_OLDPREP", "1", 1);
      const uint64_t ho = hash(exl3::moe_layer(l, cfg, h));
      unsetenv("EXL3_MOE_OLDPREP");
      setenv("EXL3_MOE_GENERAL", "1", 1);
      const uint64_t hg = hash(exl3::moe_layer(l, cfg, h));
      unsetenv("EXL3_MOE_GENERAL");
      std::printf("  [identity] prepared %016llx  old-grouped %016llx  general %016llx -> %s\n",
                  (unsigned long long)hn, (unsigned long long)ho, (unsigned long long)hg,
                  (hn == ho && hn == hg) ? "BIT-IDENTICAL" : "MISMATCH");
    }
    add("kda_layer (real, 34/token)", [&] { return exl3::kda_layer(lk, cfg, h, kst); });
    add("mla_q_resid + indexer_topk (real)", [&] {
      mx::array qr = exl3::mla_q_resid(lm, h);
      return exl3::indexer_topk(lm, cfg, h, qr, ist, 1000 + mpos++);
    });
    add("mla_layer (+indexer, real)", [&] {
      mx::array qr = exl3::mla_q_resid(lm, h);
      mx::array tk = exl3::indexer_topk(lm, cfg, h, qr, ist, 1000 + mpos);
      mx::array y = exl3::mla_layer(lm, cfg, h, mst, 1000 + mpos++, tk, qr);
      return y;
    });
  }

  // the whole routed path with the ids already on the host: no sync inside
  std::vector<exl3::TrellisWeight> wg, wu, wd;
  mx::array routed_out = mx::array(0.0f);
  mx::array routed_w = mx::array(0.0f);
  {
    mx::array x = mx::reshape(mx::astype(h, mx::float32), {N, H});
    mx::array logits = mx::matmul(x, mx::transpose(mx::astype(l.router_w, mx::float32)));
    auto w_i = exl3::moe_topk_route(logits, l.router_bias, cfg.n_group, cfg.topk_group, K,
                                    cfg.norm_topk_prob, cfg.routed_scaling_factor);
    mx::eval(w_i.second);
    const int32_t* idx = w_i.second.data<int32_t>();
    for (int k = 0; k < K; ++k) {
      const int e = (int)idx[k];
      wg.push_back(l.experts[(size_t)(e * 3 + 0)].tw);
      wu.push_back(l.experts[(size_t)(e * 3 + 1)].tw);
      wd.push_back(l.experts[(size_t)(e * 3 + 2)].tw);
    }
    routed_w = w_i.first;
  }
  add("routed path, host ids (no sync)", [&] {
    mx::array flat = mx::reshape(h, {N, H});
    mx::array gate = exl3::trellis_linear_group(wg, flat);
    mx::array up = exl3::trellis_linear_group(wu, flat);
    gate = mx::clip(gate, std::nullopt, mx::array(cfg.swiglu_limit, gate.dtype()));
    up = mx::clip(up, mx::array(-cfg.swiglu_limit, up.dtype()), mx::array(cfg.swiglu_limit, up.dtype()));
    mx::array act = mx::multiply(gate, mx::sigmoid(gate));
    mx::array actup = mx::multiply(act, up);
    mx::array down = exl3::trellis_linear_group_rows(wd, actup, (int)(actup.size() / ((int64_t)K * I)));
    mx::array st = mx::reshape(down, {K, N, 1, H});
    mx::array wt = mx::reshape(routed_w, {K, 1, 1, 1});
    return mx::reshape(mx::sum(mx::astype(mx::multiply(st, wt), h.dtype()), 0), {1, N, H});
  });

  for (int r = 0; r < reps; ++r)
    for (auto& row : rows_t) {
      if (r == 0) std::fprintf(stderr, "[stage] %s\n", row.name.c_str());
      const double t0 = now_s();
      mx::array y = row.run();
      mx::eval(y);
      row.t.push_back((now_s() - t0) * 1000.0);
    }

  std::printf("rows=%d reps=%d (each row = build graph + one mx::eval)\n", rows, reps);
  double null_ms = 0;
  for (auto& row : rows_t) {
    std::sort(row.t.begin(), row.t.end());
    const double best = row.t.front(), med = row.t[row.t.size() / 2];
    const double of = rows_t[0].t.front();
    if (row.name.rfind("null", 0) == 0) null_ms = best;
    std::printf("  %-42s best %7.3f  median %7.3f  (-null %7.3f)\n", row.name.c_str(), best, med,
                best - null_ms);
  }
  return 0;
}
