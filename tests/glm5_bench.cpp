// GLM-5.3-Flash single-stream decode benchmark: the acceptance instrument for
// the launch-overhead work. It prints
//
//   1. a per-component breakdown of one decode step (every stage boundary is an
//      mx::eval(), so the numbers are upper bounds that include one sync), and
//   2. the end-to-end ms/step of the uninstrumented path (three repeats, so the
//      run-to-run noise is visible).
//
// Measurement protocol: the first passes over this checkpoint are dominated by
// page faults on the mmap'd weight file (105 GB vs a 128 GB machine), which
// makes them ~2.5x slower than the steady state. The bench therefore warms up
// with `warmup` untimed decode steps before it measures anything, and reports
// every repeat plus the minimum.
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/glm5_bench \
//     tests/glm5_bench.cpp src/glm5_forward.cpp src/glm5.cpp src/glm5_kda.cpp \
//     src/glm5_attn.cpp src/glm5_hc.cpp src/glm5_moe.cpp src/glm5_mtp.cpp src/trellis_gemm.cpp \
//     src/safetensors.cpp src/model.cpp src/exl3.cpp src/tokenizer.cpp src/engine.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib \
//     -framework Foundation -framework Metal -framework Accelerate
//
//   usage: /tmp/glm5_bench <model_dir> [n_tokens] [warmup] [--no-profile]
#include <mlx/mlx.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "engine.h"
#include "glm5_forward.h"
#include "model.h"
#include "util.h"

namespace mx = mlx::core;

static double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model_dir> [n_tokens] [warmup] [--no-profile]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  int n_tokens = 16, warmup = 24;
  bool do_profile = true;
  int pos_args = 0;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-profile") == 0) {
      do_profile = false;
      continue;
    }
    if (pos_args == 0) n_tokens = std::atoi(argv[i]);
    else if (pos_args == 1) warmup = std::atoi(argv[i]);
    ++pos_args;
  }

  exl3::Engine engine;
  std::string err;
  const int64_t t0 = exl3::now_ms();
  if (!engine.load(dir, err, /*threads=*/0, /*fp16_weights=*/false)) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  const double load_s = (exl3::now_ms() - t0) / 1000.0;
  exl3::Model& model = engine.model();
  exl3::Glm5Model* g5 = model.glm5();
  if (!g5) {
    std::fprintf(stderr, "not a GLM-5-Next checkpoint\n");
    return 1;
  }
  const exl3::ModelConfig& cfg = engine.config();
  std::printf("=== GLM-5.3-Flash decode bench ===\n");
  std::printf("loaded in %.1fs: %d layers, weights %.2f GB, mlx active %.2f GB\n", load_s,
              cfg.num_layers, (double)model.weight_bytes() / 1e9, mx::get_active_memory() / 1e9);

  const exl3::Tokenizer& tok = engine.tokenizer();
  std::vector<exl3::ChatMessage> msgs = {
      {"user", "In one sentence, what is a trellis? Answer briefly."}};
  const std::string rendered = tok.render_chat(msgs, /*add_generation_prompt=*/true, /*thinking=*/0);
  std::vector<int> ids = tok.encode(rendered, /*add_bos=*/false);
  std::printf("prompt: %zu tokens, %d warm-up steps, %d measured steps per repeat\n", ids.size(),
              warmup, n_tokens);

  const int hs = cfg.hidden_size;
  // One decode step: the exact geometry of the real decode path (a 1-token
  // chunk), with the lm_head applied to the last hidden row by hand so the
  // bench owns the timing.
  auto step = [&](int token) {
    mx::array h = g5->forward_hidden({token}, g5->position());
    return mx::slice(h, {0, (int)h.shape(1) - 1, 0}, {1, (int)h.shape(1), hs});
  };
  auto logits_of = [&](const mx::array& row) {
    mx::array logits = g5->apply_lm_head(row); // (1, vocab) f32, eval'd inside
    const float* p = logits.data<float>();
    return std::vector<float>(p, p + g5->vocab_size());
  };
  auto argmax = [](const std::vector<float>& v) {
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
  };

  exl3::KVCache cache;
  cache.init(cfg.num_layers);
  const double p0 = now_s();
  exl3::Model::ForwardResult fr = model.forward_multi(ids, cache, false);
  const double prefill_s = now_s() - p0;
  int next = argmax(fr.logits_last);
  std::printf("prefill %.2fs (%.0f ms/token), first token %d (%s), pos %d\n", prefill_s,
              prefill_s * 1000.0 / (double)ids.size(), next, tok.text_of(next).c_str(),
              g5->position());

  std::vector<int> gen;
  gen.push_back(next);
  auto one_step = [&] {
    std::vector<float> logits = logits_of(step(gen.back()));
    gen.push_back(argmax(logits));
  };

  // ---- warm-up -------------------------------------------------------------
  const double w0 = now_s();
  for (int i = 0; i < warmup; ++i) one_step();
  const double warmup_ms = (now_s() - w0) * 1000.0 / std::max(1, warmup);
  std::printf("warm-up %d steps: %.1f ms/step\n", warmup, warmup_ms);

  // ---- per-component breakdown (instrumented) ------------------------------
  exl3::Glm5Profile prof;
  double bd_total_ms = 0, bd_lm_head_ms = 0;
  if (do_profile) {
    g5->set_profile(&prof);
    const double s0 = now_s();
    mx::array last = step(gen.back());
    bd_total_ms = (now_s() - s0) * 1000.0; // wall time of the instrumented forward
    g5->set_profile(nullptr);
    const double l0 = now_s();
    std::vector<float> lg = logits_of(last);
    bd_lm_head_ms = (now_s() - l0) * 1000.0;
    gen.push_back(argmax(lg));
  }

  // ---- end-to-end (uninstrumented) ----------------------------------------
  double rep[3] = {0, 0, 0};
  for (int r = 0; r < 3; ++r) {
    const double d0 = now_s();
    for (int i = 0; i < n_tokens; ++i) one_step();
    rep[r] = (now_s() - d0) * 1000.0 / n_tokens;
  }
  const double best = std::min({rep[0], rep[1], rep[2]});

  if (do_profile && prof.steps > 0) {
    const double st = (double)prof.steps;
    struct Row {
      const char* name;
      double ms;
    };
    Row rows[] = {
        {"hc_prepare (2/layer: norm + proj + softmax + Sinkhorn)", prof.hc_prepare_ms / st},
        {"rms_norm (2/layer)", prof.rms_norm_ms / st},
        {"KDA (34 layers)", prof.kda_ms / st},
        {"indexer (11 layers)", prof.indexer_ms / st},
        {"MLA (11 layers)", prof.mla_ms / st},
        {"MoE (42 layers)", prof.moe_ms / st},
        {"dense MLP (3 layers)", prof.dense_ms / st},
        {"hc_mix (2/layer)", prof.hc_mix_ms / st},
        {"hyper-head + final norm", prof.head_ms / st},
        {"lm_head (bench-timed)", bd_lm_head_ms},
    };
    std::printf("\n--- per-component breakdown (ms/step, eval barrier per stage) ---\n");
    double sum = 0;
    for (auto& r : rows) {
      std::printf("  %-54s %9.2f\n", r.name, r.ms);
      sum += r.ms;
    }
    std::printf("  %-54s %9.2f\n", "sum of components (eval barrier per stage)", sum);
    std::printf("  %-54s %9.2f\n", "TOTAL forward_hidden (instrumented wall)", bd_total_ms);
  }

  std::printf("\n--- end-to-end (uninstrumented, %d tokens/take) ---\n", n_tokens);
  for (int r = 0; r < 3; ++r)
    std::printf("  repeat %d: %7.2f ms/step -> %5.2f tok/s\n", r + 1, rep[r], 1000.0 / rep[r]);
  std::printf("  BEST:     %7.2f ms/step -> %5.2f tok/s\n", best, 1000.0 / best);
  std::printf("  text: %s\n", tok.decode(gen).substr(0, 200).c_str());

  // ---- speculative decoding (the MTP draft layer), if the model has one -----
  if (auto drafter = model.make_mtp_drafter()) {
    engine.set_drafter(drafter);
    exl3::GenerateOptions opt;
    opt.sample.temperature = 0.0f; // speculation is greedy-only
    opt.max_tokens = n_tokens * 3;
    opt.draft_tokens = 4;          // 3 proposals + the token the target already fed
    const double s0 = now_s();
    exl3::GenerateResult r = engine.generate(ids, opt);
    const double dt = now_s() - s0;
    std::printf("\n--- speculative (--draft-mtp 4, %d tokens) ---\n", opt.max_tokens);
    std::printf("  %d tokens in %.2fs -> %.2f ms/step, %.2f tok/s effective (%.2f serial-equivalent)\n",
                r.completion_tokens, dt, r.decode_seconds * 1000.0 / std::max(1, r.completion_tokens),
                r.completion_tokens / std::max(1e-9, r.decode_seconds),
                r.completion_tokens / std::max(1e-9, dt));
    std::printf("  text: %s\n", r.text.substr(0, 160).c_str());
  }
  std::printf("mlx peak %.2f GB\n", mx::get_peak_memory() / 1e9);
  return 0;
}
