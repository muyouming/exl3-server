// KDA rewind: the state a partly accepted verify chunk is rewound to must equal
// the state serial decoding would hold after that many rows.
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include \
//     -o /tmp/kda_rewind_test tests/kda_rewind_test.cpp src/glm5_kda.cpp \
//     src/trellis_gemm.cpp -L$(brew --prefix mlx)/lib -lmlx \
//     -Wl,-rpath,$(brew --prefix mlx)/lib -framework Metal -framework Accelerate
// Expected: "kda rewind: OK"
//
// Checks (random weights, fp32, no checkpoint needed):
//   1. kda_recurrence's row_states agree with running the recurrence from the
//      same initial state over just the first r+1 rows.
//   2. kda_layer + kda_rewind to row m reproduces both the delta-rule state and
//      the conv window of a layer that only ever saw those m rows.
#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "glm5_kda.h"

namespace mx = mlx::core;

namespace {

double max_diff(const mx::array& a_in, const mx::array& b_in) {
  mx::array a = mx::astype(a_in, mx::float32);
  mx::array b = mx::astype(b_in, mx::float32);
  mx::eval(a, b);
  if (a.size() != b.size()) return 1e9;
  const float* pa = a.data<float>();
  const float* pb = b.data<float>();
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)pa[i] - (double)pb[i]));
  return m;
}

mx::array rand_f32(const mx::Shape& sh, std::mt19937& rng, double scale = 0.2) {
  std::normal_distribution<float> nd(0.0f, (float)scale);
  int64_t n = 1;
  for (auto d : sh) n *= d;
  std::vector<float> f((size_t)n);
  for (int64_t i = 0; i < n; ++i) f[(size_t)i] = nd(rng);
  auto* holder = new std::vector<float>(std::move(f));
  return mx::array((void*)holder->data(), sh, mx::float32, [holder](void*) { delete holder; });
}

mx::array rand_f16(const mx::Shape& sh, std::mt19937& rng, double scale = 0.2) {
  return mx::astype(rand_f32(sh, rng, scale), mx::float16);
}

} // namespace

int main() {
  const int L = 5, H = 3, KD = 4, VD = 4, HID = 8, KPOOL = 4;
  (void)KPOOL;
  std::mt19937 rng(7);
  int bad = 0;

  // ---- 1) kda_recurrence prefix states -------------------------------------
  {
    mx::array q = rand_f32({1, L, H, KD}, rng);
    mx::array k = rand_f32({1, L, H, KD}, rng);
    mx::array v = rand_f32({1, L, H, VD}, rng);
    mx::array g = rand_f32({1, L, H, KD}, rng, 0.1);
    mx::array beta = rand_f32({1, L, H, 1}, rng, 0.5);
    mx::array state = rand_f32({1, H, KD, VD}, rng, 0.5);
    mx::array s0 = state;
    std::vector<mx::array> rows;
    mx::array full = exl3::kda_recurrence(q, k, v, g, beta, &state, &rows);
    mx::eval(full, state);
    if ((int)rows.size() != L - 1) {
      std::printf("FAIL: %d prefix states for L=%d\n", (int)rows.size(), L);
      bad++;
    }
    for (int r = 0; r + 1 < L; ++r) {
      mx::array m = s0;
      auto slice = [&](const mx::array& a) {
        return mx::slice(a, {0, 0, 0, 0}, {1, r + 1, H, (int)a.shape(3)});
      };
      exl3::kda_recurrence(slice(q), slice(k), slice(v), slice(g), slice(beta), &m);
      const double d = max_diff(rows[(size_t)r], m);
      if (d != 0.0) {
        std::printf("FAIL: prefix state %d differs by %g\n", r, d);
        bad++;
      }
    }
    std::printf("recurrence prefix states: exact\n");
  }

  // ---- 2) kda_layer + kda_rewind -------------------------------------------
  {
    exl3::ModelConfig cfg;
    cfg.hidden_size = HID;
    cfg.rms_norm_eps = 1e-5f;
    const int qkv = H * KD;
    exl3::G5Layer l;
    auto lin = [&](int in, int out) {
      exl3::G5Linear g_;
      g_.w = rand_f16({in, out}, rng);
      g_.in_features = in;
      g_.out_features = out;
      return g_;
    };
    l.kda_qkv = lin(HID, 3 * qkv);
    l.kda_b = lin(HID, H);
    l.kda_g_a = lin(HID, KD);
    l.kda_g_b = lin(KD, qkv);
    l.kda_f_a = lin(HID, KD);
    l.kda_f_b = lin(KD, qkv);
    l.kda_o = lin(qkv, HID);
    l.conv1d_w = rand_f16({3 * qkv, 1, 4}, rng, 0.5);
    l.o_norm = rand_f16({KD}, rng, 0.5);
    l.dt_bias = rand_f32({qkv}, rng, 0.5);
    l.a_log = rand_f32({H}, rng, 0.5);
    l.kda_heads = H;
    l.kda_head_dim = KD;
    l.kda_gate_lower_bound = -5.0f;

    mx::array h = rand_f16({1, L, HID}, rng, 1.0);
    exl3::KdaState st; // fresh state, as after a first prefill token
    exl3::KdaRewind tape;
    exl3::kda_layer(l, cfg, h, st, &tape);
    mx::eval(st.recurrent, st.conv_window);
    if (!tape.captured || (int)tape.prefix_states.size() != L - 1) {
      std::printf("FAIL: rewind tape not captured (%d states)\n", (int)tape.prefix_states.size());
      bad++;
    }
    for (int m = 1; m <= L; ++m) {
      exl3::KdaState ref;
      mx::array hm = mx::slice(h, {0, 0, 0}, {1, m, HID});
      exl3::kda_layer(l, cfg, hm, ref);
      mx::eval(ref.recurrent, ref.conv_window);
      exl3::KdaState got = st; // the state after the whole chunk, as the layer left it
      if (m < L) exl3::kda_rewind(got, tape, m);
      const double dr = max_diff(got.recurrent, ref.recurrent);
      const double dw = max_diff(got.conv_window, ref.conv_window);
      if (dr != 0.0 || dw != 0.0) {
        std::printf("FAIL: rewind to %d rows: state diff %g, window diff %g\n", m, dr, dw);
        bad++;
      }
    }
    std::printf("layer rewind from a fresh chunk: exact\n");

    // A layer that already holds a conv window (the usual case in a long run):
    // one chunk without a tape, then a second one with it, rewound mid-chunk.
    exl3::KdaState base;
    mx::array h1 = rand_f16({1, 3, HID}, rng, 1.0);
    exl3::kda_layer(l, cfg, h1, base);
    mx::eval(base.recurrent, base.conv_window);
    exl3::KdaState st2 = base;
    exl3::KdaRewind tape2;
    mx::array h2 = rand_f16({1, L, HID}, rng, 1.0);
    exl3::kda_layer(l, cfg, h2, st2, &tape2);
    for (int m = 1; m <= L; ++m) {
      exl3::KdaState ref = base;
      mx::array hm = mx::slice(h2, {0, 0, 0}, {1, m, HID});
      exl3::kda_layer(l, cfg, hm, ref);
      exl3::KdaState got = st2;
      if (m < L) exl3::kda_rewind(got, tape2, m);
      const double dr = max_diff(got.recurrent, ref.recurrent);
      const double dw = max_diff(got.conv_window, ref.conv_window);
      if (dr != 0.0 || dw != 0.0) {
        std::printf("FAIL: mid-run rewind to %d rows: state diff %g, window diff %g\n", m, dr, dw);
        bad++;
      }
    }
    std::printf("layer rewind with a pre-existing window: exact\n");
  }
  std::printf("%s\n", bad == 0 ? "kda rewind: OK" : "kda rewind: FAILURES");
  return bad == 0 ? 0 : 1;
}
