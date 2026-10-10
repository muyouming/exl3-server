// Verifies the in-kernel EXL3 trellis decode (the path the engine actually runs)
// against the independent reference decoder in src/exl3.cpp, element by element,
// on tensors of the real checkpoint.
//
// This is the regression test for the packed-word alignment: the trellis words
// are addressed as raw little-endian BYTES (TrellisWeight::packed) because a
// safetensors tensor does not have to start on a 4-byte boundary - a shard's
// header length is arbitrary - and MLX truncates a buffer offset when an array
// is viewed as a wider dtype. Every trellis tensor in this checkpoint whose
// shard header length is not a multiple of 4 used to decode from a byte-shifted
// word stream, which silently produced a "plausible" but wrong weight matrix for
// the whole model (including the lm_head). Comparing the two decoders element
// by element catches that immediately: with a shifted stream, essentially every
// weight differs.
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include \
//     -o /tmp/glm5_exl3_decode_test tests/glm5_exl3_decode_test.cpp src/glm5_forward.cpp \
//     src/glm5.cpp src/glm5_kda.cpp src/glm5_attn.cpp src/glm5_hc.cpp src/glm5_moe.cpp \
//     src/trellis_gemm.cpp src/safetensors.cpp src/model.cpp src/exl3.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib \
//     -framework Foundation -framework Metal -framework Accelerate
//
//   usage: /tmp/glm5_exl3_decode_test <model_dir>
//
// Expected on MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3 (all four
// tensors exercise a different K and a different byte alignment; the head is
// checked in full, the others over their first 2 k-tiles):
//
//   lm_head                       K=6 shard_off%4=2  mismatches 0 / 634388480
//   layers.0.self_attn.qkv_proj   K=4                mismatches 0 / 786432
//   layers.3.self_attn.q_b_proj   K=4                mismatches 0 / 524288
//   layers.3.mlp.experts.0.gate   K=3                mismatches 0 / 65536
//   -> "trellis decode: PASS" (about 12 s: the loader maps the shards, the
//      decode comparison itself is a few seconds)
//
#include <mlx/mlx.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "exl3.h"
#include "fp16.h"
#include "glm5.h"
#include "model.h"
#include "safetensors.h"
#include "util.h"

namespace mx = mlx::core;
using namespace exl3;

// perm table (copied from exl3.cpp: the tile's 256 stream positions -> elements)
static const int32_t* perm_table() {
  static int32_t pm[256];
  static bool once = [] {
    for (int lane = 0; lane < 32; ++lane) {
      const int r0 = (lane % 4) * 2;
      const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
      const int c0 = lane / 4;
      for (int i = 0; i < 4; ++i) {
        pm[lane * 8 + i] = rows[i] * 16 + c0;
        pm[lane * 8 + 4 + i] = rows[i] * 16 + c0 + 8;
      }
    }
    return true;
  }();
  (void)once;
  return pm;
}

static std::string shard_for(const std::string& dir, const std::string& key) {
  for (int i = 1; i <= 64; ++i) {
    char num[16];
    std::snprintf(num, sizeof num, "%05d", i);
    const std::string cand = dir + "/model-" + num + "-of-00013.safetensors";
    SafetensorsFile f;
    std::string e;
    if (!SafetensorsFile::load(cand, f, e)) continue;
    const bool has = f.has(key + ".trellis");
    f.unmap();
    if (has) return cand;
  }
  return "";
}

// One tensor, element-wise: the kernel's decode (via a one-hot raw matmul) vs
// the reference decoder's tile values.
static int check(const std::string& dir, const std::string& key, const TrellisWeight& tw,
                 int k_tiles_checked) {
  const std::string sh = shard_for(dir, key);
  std::string err;
  SafetensorsFile f;
  if (sh.empty() || !SafetensorsFile::load(sh, f, err)) {
    std::printf("  %-46s SKIP (shard not found)\n", key.c_str());
    return 1;
  }
  const STTensor& info = f.tensor(key + ".trellis");
  const int64_t off = f.header_len + info.offset;
  const int K = (int)(info.shape[2] / 16);
  const int in = (int)info.shape[0] * 16, out = (int)info.shape[1] * 16;

  Exl3Tensor t;
  t.trellis = (const uint16_t*)f.i16_data(key + ".trellis", err);
  t.K = K;
  t.codebook = f.has(key + ".mul1") ? Codebook::Mul1
                                    : (f.has(key + ".mcg") ? Codebook::McG : Codebook::k3Inst);
  t.in_features = in;
  t.out_features = out;
  t.suh = f.f16_data(key + ".suh", err);
  t.svh = f.f16_data(key + ".svh", err);
  std::vector<uint16_t> inner((size_t)in * out);
  decode_inner(t, inner.data(), 0);

  // A sub-tensor covering `k_tiles_checked` k-tiles of the same data, addressed
  // exactly like the engine's (byte-granular) TrellisWeight.
  TrellisWeight sub = tw;
  sub.in_features = 16 * k_tiles_checked;
  sub.out_features = out;
  sub.packed = mx::slice(tw.packed, {0, 0, 0},
                         {k_tiles_checked, (int)tw.packed.shape(1), (int)tw.packed.shape(2)});

  int64_t bad = 0, total = 0;
  for (int r = 0; r < sub.in_features; ++r) {
    std::vector<uint16_t> xb((size_t)sub.in_features, 0);
    xb[(size_t)r] = mx::array(1.0f, mx::float16).item<uint16_t>();
    mx::array xhat((void*)xb.data(), mx::Shape{1, sub.in_features}, mx::float16, [](void*) {});
    mx::array y = trellis_matmul_raw(sub, xhat); // (1, out) f32 = W_hat row r
    mx::eval(y);
    const float* got = y.data<float>();
    const uint16_t* ref = inner.data() + (size_t)r * out;
    for (int n = 0; n < out; ++n) {
      ++total;
      if (got[n] != (float)fp16::from_bits(ref[n]).to_double()) ++bad;
    }
  }
  std::printf("  %-46s K=%d shard_off%%4=%lld  mismatches %lld / %lld  %s\n", key.c_str(), K,
              (long long)(off % 4), (long long)bad, (long long)total, bad == 0 ? "OK" : "FAIL");
  f.unmap();
  return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model_dir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  std::string err;
  ModelConfig cfg;
  if (!ModelConfig::load(dir, cfg, err)) {
    std::fprintf(stderr, "config: %s\n", err.c_str());
    return 1;
  }
  Glm5Weights w;
  if (!load_glm5(dir, cfg, w, err, 0, false)) {
    std::fprintf(stderr, "load_glm5: %s\n", err.c_str());
    return 1;
  }
  int fails = 0;
  // The head is the tensor that exposed the alignment bug (its shard's header
  // length is 2 mod 4); the others cover K=4 and K=3 and an aligned shard.
  fails += check(dir, "lm_head", w.lm_head.tw, 256);
  fails += check(dir, "model.language_model.layers.0.self_attn.qkv_proj", w.layers[0].kda_qkv.tw,
                 2);
  fails += check(dir, "model.language_model.layers.3.self_attn.q_b_proj", w.layers[3].q_b.tw, 2);
  fails += check(dir, "model.language_model.layers.3.mlp.experts.0.gate_proj",
                 w.layers[3].experts[0].tw, 2);
  std::printf("%s\n", fails == 0 ? "trellis decode: PASS" : "trellis decode: FAIL");
  return fails == 0 ? 0 : 1;
}
