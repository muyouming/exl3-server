// Loads the real GLM-5-Next checkpoint and runs the two attention layer kinds on
// real weights: a KDA layer (linear_attention) and an MLA + DSA indexer layer
// (deepseek_sparse_attention). Reports the loaded inventory, every observed
// shape/dtype and the output statistics.
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/glm5_load_test \
//     tests/glm5_load_test.cpp src/glm5.cpp src/glm5_kda.cpp src/glm5_attn.cpp \
//     src/trellis_gemm.cpp src/safetensors.cpp src/model.cpp src/exl3.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib \
//     -framework Foundation -framework Metal -framework Accelerate
//
//   usage: /tmp/glm5_load_test <model_dir> [--mla N] [--quiet]
//
// Expected on MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3: the load reports
// 147039/151554 tensors (100.9 GB, ~10-25 s) with the two documented skip sets,
// then every smoke line ends in "OK" (no NaN, no zeros) and the exit status is 0.
#include <mlx/mlx.h>
#include <mach/mach.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "glm5.h"
#include "glm5_attn.h"
#include "glm5_kda.h"
#include "model.h"
#include "util.h"

namespace mx = mlx::core;

static size_t rss_mb() {
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
    return 0;
  return (size_t)(info.resident_size / (1024 * 1024));
}

static const char* dtype_name(mx::Dtype d) {
  if (d == mx::float16) return "f16";
  if (d == mx::bfloat16) return "bf16";
  if (d == mx::float32) return "f32";
  if (d == mx::uint32) return "u32";
  if (d == mx::int32) return "i32";
  if (d == mx::uint8) return "u8";
  return "?";
}

static std::string shape_str(const mx::array& a) {
  std::string s = "[";
  for (int i = 0; i < (int)a.ndim(); ++i) s += (i ? "," : "") + std::to_string(a.shape(i));
  return s + "]";
}

static void dump_lin(const char* name, const exl3::G5Linear& l) {
  std::printf("    %-24s %-4s in=%5d out=%6d", name, l.quantized ? "EXL3" : "f16", l.in_features,
              l.out_features);
  if (l.quantized) {
    std::printf("  K=%d codebook=%d words=%s %s suh=%s svh=%s", l.tw.K, l.tw.codebook,
                shape_str(l.tw.packed).c_str(), dtype_name(l.tw.packed.dtype()),
                shape_str(l.tw.suh).c_str(), shape_str(l.tw.svh).c_str());
  } else {
    std::printf("  w=%s %s", shape_str(l.w).c_str(), dtype_name(l.w.dtype()));
  }
  if (l.has_bias) std::printf(" +bias");
  std::printf("\n");
}

static void dump_vec(const char* name, const mx::array& a) {
  std::printf("    %-22s %s %s\n", name, shape_str(a).c_str(), dtype_name(a.dtype()));
}

struct Stats {
  float lo = 0, hi = 0, mean = 0, absmax = 0;
  int64_t nan = 0, zero = 0, n = 0;
  bool ok = true;
};

static Stats stats_of(const mx::array& y, const char* label, int64_t expect_rows,
                      int64_t expect_cols) {
  mx::array yf = mx::astype(y, mx::float32);
  mx::eval(yf);
  Stats s;
  if (y.ndim() != 3) {
    std::printf("  !! %s: expected a rank-3 output, got %s\n", label, shape_str(y).c_str());
    s.ok = false;
    return s;
  }
  const int64_t L = y.shape(0) * y.shape(1) * y.shape(2);
  if (y.shape(0) != expect_rows || y.shape(1) != expect_cols) {
    std::printf("  !! %s: expected (%lld,%lld,*), got %s\n", label, (long long)expect_rows,
                (long long)expect_cols, shape_str(y).c_str());
    s.ok = false;
  }
  const float* p = yf.data<float>();
  s.n = L;
  s.lo = s.hi = p[0];
  double sum = 0;
  for (int64_t i = 0; i < L; ++i) {
    const float v = p[i];
    if (!std::isfinite(v)) ++s.nan;
    if (v == 0.0f) ++s.zero;
    s.lo = std::min(s.lo, v);
    s.hi = std::max(s.hi, v);
    s.absmax = std::max(s.absmax, std::fabs(v));
    sum += v;
  }
  s.mean = (float)(sum / (double)L);
  s.ok = s.ok && s.nan == 0 && s.absmax > 0.0f;
  return s;
}

static void report(const char* label, const mx::array& y, int64_t rows, int64_t cols) {
  const Stats s = stats_of(y, label, rows, cols);
  std::printf("  %-28s %s %s  min=%.6g max=%.6g mean=%.6g absmax=%.6g  zeros=%lld nan=%lld  %s\n",
              label, shape_str(y).c_str(), dtype_name(y.dtype()), s.lo, s.hi, s.mean, s.absmax,
              (long long)s.zero, (long long)s.nan, s.ok ? "OK" : "FAIL");
}

static mx::array hidden(int64_t L, int hidden_size, uint64_t seed) {
  mx::array h = mx::random::normal(mx::Shape{(int)L, (int)hidden_size}, mx::float16,
                                   mx::random::key(seed));
  return mx::reshape(h, {1, (int)L, (int)hidden_size});
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model_dir> [--mla N] [--quiet]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  int mla_layer_idx = -1;
  bool verbose = true;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mla") && i + 1 < argc) mla_layer_idx = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--quiet")) verbose = false;
  }

  std::string err;
  exl3::ModelConfig cfg;
  if (!exl3::ModelConfig::load(dir, cfg, err)) {
    std::fprintf(stderr, "config: %s\n", err.c_str());
    return 1;
  }
  std::printf("config: %d layers, hidden %d, %d heads x %d (qk), v %d, q_lora %d kv_lora %d,\n"
              "        %d experts top-%d inter %d, first_k_dense %d, indexer %dx%d topk %d kpool %d\n",
              cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.qk_head_dim, cfg.v_head_dim,
              cfg.q_lora_rank, cfg.kv_lora_rank, cfg.n_routed_experts, cfg.num_experts_per_tok,
              cfg.moe_intermediate_size, cfg.first_k_dense_replace, cfg.index_n_heads,
              cfg.index_head_dim, cfg.index_topk, cfg.index_kpool);
  int kda_n = 0, dsa_n = 0, moe_n = 0;
  for (auto& p : cfg.layer_plan) {
    kda_n += p.linear_attention;
    dsa_n += p.sparse_attention;
    moe_n += p.moe;
  }
  std::printf("        %d KDA + %d DSA attention layers, %d dense + %d MoE MLPs\n", kda_n, dsa_n,
              cfg.num_layers - moe_n, moe_n);

  const int64_t t0 = exl3::now_ms();
  exl3::Glm5Weights w;
  if (!exl3::load_glm5(dir, cfg, w, err, 0, verbose)) {
    std::fprintf(stderr, "load_glm5: %s\n", err.c_str());
    return 1;
  }
  const double secs = (exl3::now_ms() - t0) / 1000.0;
  std::printf("\nLOAD OK in %.1f s: %lld/%lld tensors, %.2f GB used of %.2f GB mapped, "
              "%zu experts\n",
              secs, (long long)w.used_tensors, (long long)w.total_tensors, w.used_bytes / 1e9,
              w.bytes / 1e9, w.expert_count());
  for (auto& [label, count] : w.skipped)
    std::printf("  skipped %lld: %s\n", (long long)count, label.c_str());
  std::printf("  mlx active %.2f GB, peak %.2f GB, process RSS %zu MB (zero-copy check)\n",
              mx::get_active_memory() / 1e9, mx::get_peak_memory() / 1e9, rss_mb());

  // ---------------------------------------------------------------- layer 0 (KDA)
  const exl3::G5Layer& l0 = w.layers[0];
  std::printf("\nlayer 0 (%s%s): heads=%d head_dim=%d gate_lower_bound=%.3g conv_bias=%d\n",
              l0.plan.linear_attention ? "KDA" : "?", l0.moe ? " + MoE MLP" : " + dense MLP",
              l0.kda_heads, l0.kda_head_dim, l0.kda_gate_lower_bound, (int)l0.conv1d_has_bias);
  dump_lin("self_attn.qkv_proj", l0.kda_qkv);
  dump_lin("self_attn.b_proj", l0.kda_b);
  dump_lin("self_attn.f_a_proj", l0.kda_f_a);
  dump_lin("self_attn.f_b_proj", l0.kda_f_b);
  dump_lin("self_attn.g_a_proj", l0.kda_g_a);
  dump_lin("self_attn.g_b_proj", l0.kda_g_b);
  dump_lin("self_attn.o_proj", l0.kda_o);
  dump_vec("self_attn.conv1d.weight", l0.conv1d_w);
  dump_vec("self_attn.o_norm.weight", l0.o_norm);
  dump_vec("self_attn.dt_bias", l0.dt_bias);
  dump_vec("self_attn.A_log", l0.a_log);
  dump_vec("input_layernorm.weight", l0.input_norm);
  dump_lin("mlp.gate_proj (dense)", l0.dense_gate);
  dump_lin("mlp.down_proj (dense)", l0.dense_down);

  exl3::KdaState ks;
  mx::array h1 = hidden(1, cfg.hidden_size, 7);
  mx::array y0 = mx::array(0.0f);
  try {
    y0 = exl3::kda_layer(l0, cfg, h1, ks);
    mx::eval(y0);
  } catch (const std::exception& e) {
    std::printf("!! kda_layer threw: %s\n", e.what());
    std::fflush(stdout);
    return 1;
  }
  std::printf("layer 0 KDA on (1,1,%d):\n", cfg.hidden_size);
  report("kda_layer", y0, 1, 1);

  // Longer sequence + one decode step: the rolling conv window must reproduce a
  // single 5-token prefill's last row.
  mx::array h5 = hidden(5, cfg.hidden_size, 11);
  exl3::KdaState ks_pre;
  mx::array y_pre = exl3::kda_layer(l0, cfg, h5, ks_pre);
  mx::eval(y_pre);
  exl3::KdaState ks_split;
  mx::array y4 = exl3::kda_layer(l0, cfg, mx::slice(h5, {0, 0, 0}, {1, 4, cfg.hidden_size}), ks_split);
  mx::eval(y4);
  mx::array y_dec = exl3::kda_layer(l0, cfg, mx::slice(h5, {0, 4, 0}, {1, 5, cfg.hidden_size}), ks_split);
  mx::eval(y_dec);
  {
    mx::array a = mx::astype(mx::slice(y_pre, {0, 4, 0}, {1, 5, cfg.hidden_size}), mx::float32);
    mx::array b = mx::astype(y_dec, mx::float32);
    mx::array d = mx::abs(mx::subtract(a, b));
    mx::eval(d);
    std::printf("  prefill-5 last row vs prefill-4+decode: max|diff|=%.3g absmax(prefill)=%.3g\n",
                mx::max(d).item<float>(), mx::max(mx::abs(a)).item<float>());
  }

  // ------------------------------------------------------- first sparse layer (MLA)
  if (mla_layer_idx < 0) {
    for (int i = 0; i < cfg.num_layers; ++i)
      if (cfg.layer_plan[(size_t)i].sparse_attention) {
        mla_layer_idx = i;
        break;
      }
  }
  if (mla_layer_idx >= 0) {
    const exl3::G5Layer& l = w.layers[(size_t)mla_layer_idx];
    std::printf("\nlayer %d (MLA + DSA indexer%s):\n", mla_layer_idx, l.moe ? " + MoE MLP" : "");
    dump_lin("self_attn.q_a_proj", l.q_a);
    dump_lin("self_attn.q_b_proj", l.q_b);
    dump_lin("self_attn.kv_a_proj_with_mqa", l.kv_a);
    dump_lin("self_attn.kv_b_proj", l.kv_b);
    dump_lin("self_attn.o_proj", l.attn_o);
    dump_vec("self_attn.q_a_layernorm.weight", l.q_a_norm);
    dump_vec("self_attn.kv_a_layernorm.weight", l.kv_a_norm);
    dump_lin("indexer.wq_b", l.idx_wq_b);
    dump_vec("indexer.wk.weight", l.idx_wk);
    dump_vec("indexer.k_norm.weight", l.idx_k_norm_w);
    dump_vec("indexer.k_norm.bias", l.idx_k_norm_b);
    dump_vec("indexer.weights_proj.weight", l.idx_weights_proj);
    dump_vec("indexer.index_kpool_compress_gate", l.idx_gate);
    dump_vec("indexer.index_kpool_compress_ape", l.idx_ape);

    for (int64_t L : {(int64_t)1, (int64_t)5}) {
      mx::array h = hidden(L, cfg.hidden_size, 23);
      mx::array q_resid = exl3::mla_q_resid(l, h);
      exl3::IndexerState ist;
      exl3::MlaState mst;
      mx::array topk = exl3::indexer_topk(l, cfg, h, q_resid, ist, 0);
      mx::eval(topk);
      std::printf("layer %d indexer_topk(%lld tokens): %s %s (first row:", mla_layer_idx,
                  (long long)L, shape_str(topk).c_str(), dtype_name(topk.dtype()));
      {
        mx::array t = mx::astype(topk, mx::int32);
        mx::eval(t);
        const int32_t* p = t.data<int32_t>();
        const int32_t wide = (int32_t)t.shape(t.ndim() - 1);
        for (int32_t j = 0; j < std::min<int32_t>(wide, 6) && j < 6; ++j) std::printf(" %d", p[j]);
        std::printf(" ...)\n");
      }
      mx::array y = exl3::mla_layer(l, cfg, h, mst, 0, topk, q_resid);
      mx::eval(y);
      report("mla_layer (DSA topk)", y, 1, (int64_t)L);
      // dense causal attention on the same hidden state, for comparison
      exl3::MlaState mst2;
      mx::array yd = exl3::mla_layer(l, cfg, h, mst2, 0, mx::array(0.0f), q_resid);
      mx::eval(yd);
      report("mla_layer (dense causal)", yd, 1, (int64_t)L);
      {
        mx::array d = mx::abs(mx::subtract(mx::astype(y, mx::float32), mx::astype(yd, mx::float32)));
        mx::eval(d);
        std::printf("  DSA vs dense max|diff|=%.3g\n", mx::max(d).item<float>());
      }
    }
  }

  std::printf("\nfinal: mlx active %.2f GB, peak %.2f GB, RSS %zu MB\n",
              mx::get_active_memory() / 1e9, mx::get_peak_memory() / 1e9, rss_mb());
  return 0;
}
