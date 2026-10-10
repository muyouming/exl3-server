// GLM-5-Next: is a speculative verify batch equivalent to serial decoding?
//
// Build (repo root):
//   clang++ -std=c++20 -O2 -Ivendor -Isrc -I$(brew --prefix mlx)/include -o /tmp/glm5_rewind_test \
//     tests/glm5_rewind_test.cpp src/glm5_forward.cpp src/glm5_mtp.cpp src/glm5.cpp \
//     src/glm5_kda.cpp src/glm5_attn.cpp src/glm5_hc.cpp src/glm5_moe.cpp \
//     src/trellis_gemm.cpp src/safetensors.cpp src/model.cpp src/exl3.cpp \
//     src/tokenizer.cpp src/engine.cpp \
//     -L$(brew --prefix mlx)/lib -lmlx -Wl,-rpath,$(brew --prefix mlx)/lib \
//     -framework Foundation -framework Metal -framework Accelerate
//
//   usage: /tmp/glm5_rewind_test <model_dir> [prompt]
//
// Three probes on the real checkpoint, all from the same prompt.
//   A. batching: the logits a multi-row chunk computes for its FIRST row versus
//      the logits a one-row forward computes for the same token. Different GEMM
//      shapes may round differently, so this measures by how much (and whether
//      the argmax survives) - the speculative loop always verifies >= 1 row.
//   B. rewind round trip: a chunk whose first rows are kept and whose deeper
//      rows are dropped (what a partly accepted verify batch does) must leave
//      the state serial decoding would reach. Same shapes on both sides, so this
//      isolates the rewind itself: the recurrent KDA state is restored from the
//      states the chunk recorded, the latent/indexer caches are sliced.
//   C. the draft-row invariant: Glm5Model::mtp_rows() == position() - 1 after
//      every operation, plus how often the draft row's argmax is already the
//      token the trunk picks (the acceptance the speculative loop can expect).
#include <mlx/mlx.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.h"
#include "glm5_forward.h"
#include "model.h"
#include "util.h"

namespace mx = mlx::core;

namespace {

// Row `r` of a (rows, vocab) logits array as a compact (1, vocab) array.
mx::array row_of(const mx::array& logits, int r) {
  return mx::contiguous(mx::slice(logits, {r, 0}, {r + 1, (int)logits.shape(1)}));
}

int argmax_of(const mx::array& logits) {
  mx::array a = mx::astype(logits, mx::float32);
  mx::eval(a);
  const float* p = a.data<float>();
  int best = 0;
  for (int64_t i = 1; i < a.size(); ++i) {
    if (p[i] > p[best]) best = (int)i;
  }
  return best;
}

struct Diff {
  double max_abs = 0.0;
  double over_proj_scale = 0.0;
  double ref_max = 0.0;
  double margin = 0.0; // reference top-1 minus top-2: a flip needs |d| > margin
  int argmax_ref = -1, argmax_got = -1;
};

Diff compare(const mx::array& ref_in, const mx::array& got_in) {
  mx::array a = mx::astype(ref_in, mx::float32);
  mx::array b = mx::astype(got_in, mx::float32);
  mx::eval(a, b);
  Diff d;
  if (a.size() != b.size()) {
    d.max_abs = -1;
    return d;
  }
  const float* pa = a.data<float>();
  const float* pb = b.data<float>();
  double second = -1e30;
  // Project the absolute difference onto the logits' own scale: that is the
  // quantity that matters for an argmax flip.
  for (size_t i = 0; i < a.size(); ++i) {
    d.max_abs = std::max(d.max_abs, std::fabs((double)pa[i] - (double)pb[i]));
    d.ref_max = std::max(d.ref_max, std::fabs((double)pa[i]));
    if (d.argmax_ref < 0 || pa[i] > pa[d.argmax_ref]) {
      second = (d.argmax_ref < 0) ? second : (double)pa[d.argmax_ref];
      d.argmax_ref = (int)i;
    } else if ((double)pa[i] > second) {
      second = (double)pa[i];
    }
    if (d.argmax_got < 0 || pb[i] > pb[d.argmax_got]) d.argmax_got = (int)i;
  }
  d.margin = (double)pa[d.argmax_ref] - second;
  // The fp16 resolution at the logits' magnitude (1 ulp of the mantissa).
  const double ulp = std::ldexp(1.0, (int)std::floor(std::log2(std::max(d.ref_max, 1e-9))) - 10);
  d.over_proj_scale = ulp > 0 ? d.max_abs / ulp : 0.0;
  return d;
}

void report(const char* what, const Diff& d) {
  std::printf("  %-26s max|d| %.3e (%.1f ulp)  top-1 margin %.3f  argmax %s\n", what, d.max_abs,
              d.over_proj_scale, d.margin, (d.argmax_ref == d.argmax_got) ? "same" : "FLIPS");
}

// Row `r` of a (1, rows, hidden) hidden-state chunk as a compact (1, 1, hidden).
mx::array hrow_of(const mx::array& h, int r) {
  return mx::contiguous(mx::slice(h, {0, r, 0}, {1, r + 1, (int)h.shape(2)}));
}

int argmax_of_vec(const std::vector<float>& v) {
  int best = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[(size_t)best]) best = (int)i;
  }
  return best;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model_dir> [prompt]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const std::string prompt = argc > 2 ? argv[2] : "The history of the printing press begins in";

  exl3::Engine engine;
  std::string err;
  if (!engine.load(dir, err, 0, false)) {
    std::fprintf(stderr, "load failed: %s\n", err.c_str());
    return 1;
  }
  exl3::Model& model = engine.model();
  exl3::Glm5Model* g = model.glm5();
  if (!g) {
    std::fprintf(stderr, "not a GLM-5-Next model\n");
    return 1;
  }
  if (!model.make_mtp_drafter()) {
    std::fprintf(stderr, "no MTP block\n");
    return 1;
  }
  const int vocab = g->vocab_size();
  const std::vector<int> ids = engine.tokenizer().encode(prompt, /*add_bos=*/true);
  const int P = (int)ids.size();
  int bad = 0;

  // The prompt's hidden states and the first three greedy tokens, each from a
  // fresh prefill.
  auto prefill = [&]() { return g->forward_hidden(ids, 0, /*drafting=*/true); };
  mx::array h = prefill();
  const int t1 = argmax_of(g->apply_lm_head(mx::slice(h, {0, P - 1, 0}, {1, P, (int)g->config().hidden_size})));

  // ---- the tokens the reference run feeds ----------------------------------
  g->reset();
  prefill();
  mx::array l1 = g->forward_tokens({t1}, P);            // logits at P+1
  const int t2 = argmax_of(l1);
  mx::array l2s = g->forward_tokens({t2}, P + 1);       // serial: logits at P+2
  const int t3 = argmax_of(l2s);
  mx::array l3s = g->forward_tokens({t3}, P + 2);       // serial: logits at P+3
  const int t4 = argmax_of(l3s);
  std::printf("prompt %d tokens; greedy chain %d %d %d %d\n", P, t1, t2, t3, t4);

  // ---- A: does a multi-row chunk round the same as a one-row forward? -----
  std::printf("\nA. multi-row vs one-row logits (row 0 of the chunk)\n");
  {
    g->reset();
    prefill();
    mx::array single = g->forward_tokens({t1}, P); // (1, vocab)
    g->reset();
    prefill();
    mx::array batch2 = g->forward_tokens({t1, t2}, P); // (2, vocab)
    Diff d2 = compare(row_of(single, 0), row_of(batch2, 0));
    report("2 rows vs 1 row", d2);
    report("2 rows, row 1 vs serial", compare(l2s, row_of(batch2, 1)));

    g->reset();
    prefill();
    mx::array batch3 = g->forward_tokens({t1, t2, t3}, P); // (3, vocab)
    Diff d3 = compare(row_of(single, 0), row_of(batch3, 0));
    report("3 rows vs 1 row", d3);
    Diff d3b = compare(mx::astype(l3s, mx::float32), row_of(batch3, 2));
    report("3 rows, row 2 vs serial", d3b);
    // Both sides multi-row: if only the one-row kernel differs, these are equal.
    report("2 rows vs 3 rows (row 0)", compare(row_of(batch2, 0), row_of(batch3, 0)));
    report("2 rows vs 3 rows (row 1)", compare(row_of(batch2, 1), row_of(batch3, 1)));
    if (d2.max_abs != 0.0 || d3.max_abs != 0.0) {
      std::printf("  -> a multi-row verify batch is NOT bit-identical to serial decoding\n");
    } else {
      std::printf("  -> multi-row batching is bit-identical to serial here\n");
    }
  }

  // ---- A2: where does the difference enter? --------------------------------
  // H1: the head alone, fed the SAME hidden rows: a 2-row head call's last row
  //     versus a 1-row head call on that same row.
  // H2: the trunk hidden itself: row 0 of a 2-row chunk versus the 1-row chunk.
  std::printf("\nA2. head vs trunk hidden\n");
  {
    g->reset();
    mx::array h1 = prefill();
    mx::array hs_last = mx::contiguous(mx::slice(h1, {0, P - 1, 0}, {1, P, (int)g->config().hidden_size}));
    mx::array hs_two = mx::concatenate(
        {mx::contiguous(mx::slice(h1, {0, P - 2, 0}, {1, P - 1, (int)g->config().hidden_size})),
         hs_last},
        1);
    mx::array head_single = g->apply_lm_head(hs_last);      // (1, vocab)
    mx::array head_double = g->apply_lm_head(hs_two);       // (2, vocab)
    report("head: 2 rows row 1 vs 1 row", compare(head_single, row_of(head_double, 1)));

    g->reset();
    prefill();
    mx::array chunk1 = g->forward_hidden({t1}, P, true);          // (1,1,hidden)
    g->reset();
    prefill();
    mx::array chunk2 = g->forward_hidden({t1, t2}, P, true);      // (1,2,hidden)
    Diff dh = compare(hrow_of(chunk1, 0), hrow_of(chunk2, 0));
    std::printf("  %-26s max|d| %.3e  (hidden max %.3f, %.4f%% of it)\n",
                "hidden: 2 rows row 0 vs 1 row", dh.max_abs, dh.ref_max,
                100.0 * dh.max_abs / std::max(dh.ref_max, 1e-9));
  }

  // ---- B: rewind round trip (same shapes on both sides) --------------------
  std::printf("\nB. rewind after a partly accepted chunk\n");
  {
    g->reset();
    prefill();
    g->forward_hidden({t1, t2, t3}, P, /*drafting=*/true);
    if (g->position() != P + 3 || g->mtp_rows() != P + 2) {
      std::printf("  chunk state wrong: position %d (want %d), mtp rows %d (want %d)\n",
                  g->position(), P + 3, g->mtp_rows(), P + 2);
      bad++;
    }
    g->rewind(P + 1); // only t1 is committed
    if (g->position() != P + 1 || g->mtp_rows() != P) {
      std::printf("  rewind state wrong: position %d (want %d), mtp rows %d (want %d)\n",
                  g->position(), P + 1, g->mtp_rows(), P);
      bad++;
    }
    mx::array got2 = g->forward_hidden({t2}, P + 1, true);
    Diff d2 = compare(mx::astype(l2s, mx::float32), g->apply_lm_head(got2));
    report("rewind to P+1, re-feed t2", d2);
    mx::array got3 = g->forward_hidden({t3}, P + 2, true);
    Diff d3 = compare(mx::astype(l3s, mx::float32), g->apply_lm_head(got3));
    report("then t3", d3);
    // The rewind itself is exact (tests/kda_rewind_test.cpp). What is left here
    // is the chunk-shape effect of A: the state the rewind restores is the one
    // the chunk's own (3-row) projections produced, not a 1-row decode's.
    if (d2.max_abs != 0.0 || d3.max_abs != 0.0) {
      std::printf("  -> residual = the chunk-shape effect of A, not a rewind error\n");
    } else {
      std::printf("  -> the rewind reproduces serial decoding exactly\n");
    }
  }

  // ---- C: draft rows and how often the draft already agrees ----------------
  std::printf("\nC. draft rows\n");
  {
    g->reset();
    prefill();
    exl3::Glm5MtpState scratch = g->mtp_state();
    int agree = 0, total = 0;
    std::vector<int> chain = {t1, t2, t3};
    std::vector<int> truth = {t2, t3, t4};
    mx::array prev = g->last_hidden();
    for (size_t i = 0; i < chain.size(); ++i) {
      mx::array row = g->mtp_row(chain[i], prev, P + (int)i, scratch);
      const std::vector<float> lg = g->mtp_logits(row);
      const int draft = argmax_of_vec(lg);
      total++;
      agree += (draft == truth[i]) ? 1 : 0;
      std::printf("  row %d (t_%zu): draft %d, trunk %d %s\n", P + (int)i, i + 1, draft, truth[i],
                  draft == truth[i] ? "agree" : "differ");
      prev = row;
    }
    std::printf("  %d/%d draft rows agree with the trunk's own choice\n", agree, total);
    if (g->position() != P || g->mtp_rows() != P - 1) {
      std::printf("  the scratch chain disturbed the committed state\n");
      bad++;
    }
  }
  std::printf("\n%s\n", bad == 0 ? "rewind probe: OK" : "rewind probe: PROBLEMS");
  return bad == 0 ? 0 : 1;
}
