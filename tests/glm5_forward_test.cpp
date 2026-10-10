// GLM-5-Next end-to-end forward check on the REAL checkpoint.
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/glm5_forward_test \
//     tests/glm5_forward_test.cpp src/glm5_forward.cpp src/glm5.cpp src/glm5_kda.cpp \
//     src/glm5_attn.cpp src/glm5_hc.cpp src/glm5_moe.cpp src/trellis_gemm.cpp \
//     src/safetensors.cpp src/model.cpp src/exl3.cpp src/tokenizer.cpp src/engine.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib \
//     -framework Foundation -framework Metal -framework Accelerate
//
//   usage: /tmp/glm5_forward_test <model_dir> [prompt] [n_tokens]
//
// Checks (all on the real weights):
//   1. prefill-vs-incremental: the logits of a fixed short token sequence from
//      one prefill must equal prefill(n-1) + decode(1) within f16 tolerance.
//   2. greedy generation of `n_tokens` from a chat-templated prompt, reporting
//      the decoded text, tokens/s and the peak memory.
#include <mach/mach.h>
#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine.h"
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

static void mem_line(const char* tag) {
  std::printf("  [%s] mlx active %.2f GB, mlx peak %.2f GB, process RSS %zu MB\n", tag,
              mx::get_active_memory() / 1e9, mx::get_peak_memory() / 1e9, rss_mb());
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model_dir> [prompt] [n_tokens]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const std::string user_prompt = argc > 2 ? argv[2] : "In one sentence, what is a trellis?";
  const int n_tokens = argc > 3 ? std::atoi(argv[3]) : 32;

  exl3::Engine engine;
  std::string err;
  const int64_t t0 = exl3::now_ms();
  if (!engine.load(dir, err, /*threads=*/0, /*fp16_weights=*/false)) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  const double load_s = (exl3::now_ms() - t0) / 1000.0;
  const auto& cfg = engine.config();
  std::printf("\nLOADED in %.1fs: arch=%s, %d layers, %d KDA + %d DSA, vocab %d\n", load_s,
              cfg.arch == exl3::Arch::Glm5Next ? "glm5_next" : "llama", cfg.num_layers,
              [&] {
                int n = 0;
                for (auto& p : cfg.layer_plan) n += p.linear_attention;
                return n;
              }(),
              [&] {
                int n = 0;
                for (auto& p : cfg.layer_plan) n += p.sparse_attention;
                return n;
              }(),
              cfg.vocab_size);
  std::printf("  hc_mult %d eps %g sinkhorn %d | n_group %d topk_group %d gate_lower_bound %g | "
              "swiglu_limit %g routed_scaling %g norm_topk_prob %d\n",
              cfg.hc_mult, cfg.hc_eps, cfg.hc_sinkhorn_iters, cfg.n_group, cfg.topk_group,
              cfg.linear_gate_lower_bound, cfg.swiglu_limit, cfg.routed_scaling_factor,
              (int)cfg.norm_topk_prob);
  std::printf("  weights %.2f GB, %lld/%lld tensors\n",
              (double)engine.model().weight_bytes() / 1e9,
              (long long)engine.model().used_tensors(), (long long)engine.model().total_tensors());
  mem_line("after load");

  // ---------------------------------------------------------------- 1. consistency
  const exl3::Tokenizer& tok = engine.tokenizer();
  std::vector<exl3::ChatMessage> msgs = {{"user", user_prompt}};
  const std::string rendered = tok.render_chat(msgs, /*add_generation_prompt=*/true, /*thinking=*/0);
  std::vector<int> ids = tok.encode(rendered, /*add_bos=*/false);
  std::printf("\nprompt: %zu chars -> %zu tokens\n", rendered.size(), ids.size());
  std::printf("  rendered: %s\n", rendered.c_str());

  const int n = std::min<int>((int)ids.size(), 24);
  std::vector<int> seq(ids.begin(), ids.begin() + n);
  std::printf("consistency on the first %d prompt tokens:", n);
  for (int i = 0; i < std::min(8, n); ++i) std::printf(" %d", seq[(size_t)i]);
  std::printf(" ...\n");

  exl3::Model& model = engine.model();
  exl3::KVCache c_full, c_inc;
  c_full.init(cfg.num_layers);
  c_inc.init(cfg.num_layers);
  const int64_t a0 = exl3::now_ms();
  std::vector<float> full = model.forward_multi(seq, c_full, false).logits_last;
  const double full_s = (exl3::now_ms() - a0) / 1000.0;
  std::vector<int> prefix(seq.begin(), seq.end() - 1);
  const int64_t b0 = exl3::now_ms();
  model.forward_multi(prefix, c_inc, false);
  std::vector<float> inc = model.forward_multi({seq.back()}, c_inc, false).logits_last;
  const double inc_s = (exl3::now_ms() - b0) / 1000.0;

  double max_abs = 0, max_rel = 0, logit_scale = 0;
  size_t argmax_a = 0, argmax_b = 0;
  for (size_t i = 0; i < full.size(); ++i) {
    const double d = std::fabs((double)full[i] - (double)inc[i]);
    max_abs = std::max(max_abs, d);
    logit_scale = std::max(logit_scale, std::fabs((double)full[i]));
  }
  for (size_t i = 0; i < full.size(); ++i) {
    max_rel = std::max(max_rel, std::fabs((double)full[i] - (double)inc[i]) / logit_scale);
    if (full[i] > full[argmax_a]) argmax_a = i;
    if (inc[i] > inc[argmax_b]) argmax_b = i;
  }
  std::printf("  prefill(%d) %.2fs | prefill(%d)+decode(1) %.2fs\n", n, full_s, n - 1, inc_s);
  std::printf("  max|diff| = %.3g  (max|logit| = %.3g, relative %.3g)  argmax %zu vs %zu %s\n",
              max_abs, logit_scale, max_rel, argmax_a, argmax_b,
              argmax_a == argmax_b ? "SAME" : "DIFFERENT");
  // top-5 overlap as a second, more robust signal
  auto top5 = [](const std::vector<float>& v) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < v.size(); ++i) idx[i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return v[(size_t)a] > v[(size_t)b]; });
    idx.resize(5);
    return idx;
  };
  std::vector<int> ta = top5(full), tb = top5(inc);
  int overlap = 0;
  for (int a : ta)
    for (int b : tb)
      if (a == b) ++overlap;
  std::printf("  top-5: prefill");
  for (int t : ta) std::printf(" %d(%s)", t, tok.text_of(t).c_str());
  std::printf("\n         incr   ");
  for (int t : tb) std::printf(" %d(%s)", t, tok.text_of(t).c_str());
  std::printf("\n  top-5 overlap %d/5 (%s)\n", overlap, overlap == 5 ? "OK" : "MISMATCH");
  mem_line("after consistency");

  // ---------------------------------------------------------------- 2. generation
  model.reset_cache(c_full);
  model.reset_cache(c_inc);
  std::printf("\n--- greedy generation (%d tokens, chat-templated prompt) ---\n", n_tokens);
  exl3::GenerateOptions opt;
  opt.sample.temperature = 0.0f; // greedy
  opt.max_tokens = n_tokens;
  opt.split_think = false; // keep the <think> markers in the text
  exl3::GenerateResult r = engine.generate(ids, opt);
  std::printf("finish=%s prompt=%d completion=%d prefill=%.2fs decode=%.2fs (%.2f tok/s)\n",
              r.finish_reason.c_str(), r.prompt_tokens, r.completion_tokens, r.prefill_seconds,
              r.decode_seconds,
              r.decode_seconds > 0 ? r.completion_tokens / r.decode_seconds : 0.0);
  std::printf("tokens:");
  for (int t : r.tokens) std::printf(" %d", t);
  std::printf("\nTEXT: <<<%s>>>\n", r.text.c_str());
  std::printf("decoded whole sequence: <<<%s>>>\n",
              tok.decode([&] {
                std::vector<int> all = ids;
                all.insert(all.end(), r.tokens.begin(), r.tokens.end());
                return all;
              }()).c_str());
  std::printf("mlx peak %.2f GB, RSS %zu MB\n", mx::get_peak_memory() / 1e9, rss_mb());
  return 0;
}
