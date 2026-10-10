// GLM-5-Next MTP drafter on the REAL checkpoint: token identity against serial
// greedy decoding, plus the acceptance rate and the throughput it buys.
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/glm5_mtp_test \
//     tests/glm5_mtp_test.cpp src/glm5_forward.cpp src/glm5_mtp.cpp src/glm5.cpp \
//     src/glm5_kda.cpp src/glm5_attn.cpp src/glm5_hc.cpp src/glm5_moe.cpp \
//     src/trellis_gemm.cpp src/safetensors.cpp src/model.cpp src/exl3.cpp \
//     src/tokenizer.cpp src/engine.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib \
//     -framework Foundation -framework Metal -framework Accelerate
//
//   usage: /tmp/glm5_mtp_test <model_dir> [depth] [tokens] [reps]
//          depth 0 sweeps 2..min(4, model cap) instead of one depth
//
// For every prompt and every configuration it decodes greedily from the same
// state - once per repetition, interleaved - and requires the token sequence to
// match the serial one: the speculative loop's invariant is that the target's
// own argmax verifies every proposal, so nothing may change. EOS is ignored so
// both runs always emit the full budget and the comparison covers the whole
// sequence instead of stopping wherever the text happens to end.
#include <mlx/mlx.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine.h"
#include "model.h"
#include "util.h"

namespace mx = mlx::core;

namespace {

std::string brief(const std::vector<int>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size(); ++i) {
    s += std::to_string(ids[i]);
    if (i + 1 < ids.size()) s += " ";
  }
  return s;
}

std::string escaped(const std::string& s, size_t max_len) {
  std::string out;
  for (char c : s) {
    if (out.size() >= max_len) { out += "..."; break; }
    if (c == '\n') out += "\\n";
    else out += c;
  }
  return out;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model_dir> [depth] [tokens] [reps]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const int want_depth = argc > 2 ? std::atoi(argv[2]) : 2;
  const int ntok = argc > 3 ? std::atoi(argv[3]) : 32;
  const int reps = argc > 4 ? std::max(1, std::atoi(argv[4])) : 1;

  exl3::Engine engine;
  std::string err;
  if (!engine.load(dir, err, /*threads=*/0, /*fp16_weights=*/false)) {
    std::fprintf(stderr, "load failed: %s\n", err.c_str());
    return 1;
  }
  const int cap = engine.model().max_speculative_rows();
  std::shared_ptr<exl3::Drafter> drafter = engine.model().make_mtp_drafter();
  if (!drafter) {
    std::fprintf(stderr, "the checkpoint has no MTP draft block\n");
    return 1;
  }
  std::vector<int> depths;
  if (want_depth <= 0) {
    for (int d = 2; d <= (cap > 0 ? cap : 4); ++d) depths.push_back(d);
  } else {
    depths.push_back(want_depth);
  }
  const exl3::Tokenizer& tok = engine.tokenizer();

  const char* prompts[] = {
      "The history of the printing press begins in",
      "- Write a short Python function that returns the sum of a list.\n\n def sum_list(",
  };

  int failures = 0;
  for (const char* prompt : prompts) {
    const std::vector<int> ids = tok.encode(prompt, /*add_bos=*/true);
    exl3::GenerateOptions g;
    g.max_tokens = ntok;
    g.ignore_eos = true;
    g.split_think = false;

    // config 0 is serial, then one per depth.
    std::vector<std::vector<int>> tokens(depths.size() + 1);
    std::vector<std::vector<double>> tps(depths.size() + 1);
    std::vector<int64_t> rounds(depths.size() + 1, 0), prop(depths.size() + 1, 0),
        acc(depths.size() + 1, 0);
    std::string text;
    for (int rep = 0; rep < reps; ++rep) {
      for (size_t c = 0; c < depths.size() + 1; ++c) {
        engine.reset();
        if (c == 0) {
          engine.set_drafter(nullptr);
          g.draft_tokens = 0;
        } else {
          engine.set_drafter(drafter);
          g.draft_tokens = depths[c - 1];
        }
        exl3::GenerateResult r = engine.generate(ids, g);
        const double t = r.decode_seconds > 0 ? r.completion_tokens / r.decode_seconds : 0.0;
        if (rep == 0) {
          tokens[c] = r.tokens;
          text = r.text;
        } else if (tokens[c] != r.tokens) {
          failures++;
          std::printf("  !! repeat %d of config %zu changed the token sequence\n", rep + 1, c);
        }
        tps[c].push_back(t);
        rounds[c] += r.spec_rounds;
        prop[c] += r.spec_proposals;
        acc[c] += r.spec_accepted;
      }
    }

    std::printf("\n=== prompt: %s\n", escaped(prompt, 60).c_str());
    std::printf("  text    %s\n", escaped(text, 150).c_str());
    std::printf("  %-8s %10s %10s %8s %8s  %s\n", "config", "min", "median", "rounds", "tok/round",
                "identity");
    for (size_t c = 0; c < depths.size() + 1; ++c) {
      const bool same = tokens[c] == tokens[0];
      if (!same) failures++;
      char label[32];
      if (c == 0) std::snprintf(label, sizeof(label), "serial");
      else std::snprintf(label, sizeof(label), "mtp-%d", depths[c - 1]);
      const double mn = *std::min_element(tps[c].begin(), tps[c].end());
      std::printf("  %-8s %10.2f %10.2f %8lld %8.2f  %s", label, mn, median(tps[c]),
                  (long long)rounds[c],
                  rounds[c] > 0 ? 1.0 + (double)acc[c] / (double)rounds[c] : 1.0,
                  c == 0 ? "-" : (same ? "IDENTICAL" : "DIFFERENT"));
      if (c > 0) {
        std::printf("  (acc %lld/%lld = %.1f%%)", (long long)acc[c], (long long)prop[c],
                    prop[c] > 0 ? 100.0 * (double)acc[c] / (double)prop[c] : 0.0);
      }
      std::printf("\n");
      if (!same) std::printf("    serial: %s\n    got   : %s\n", brief(tokens[0]).c_str(),
                             brief(tokens[c]).c_str());
    }
  }
  std::printf("\n%s (%zu prompts, %d tokens each, %d rep(s))\n",
              failures == 0 ? "all configurations token-identical to serial greedy"
                            : "TOKEN MISMATCH",
              sizeof(prompts) / sizeof(prompts[0]), ntok, reps);
  return failures == 0 ? 0 : 1;
}
