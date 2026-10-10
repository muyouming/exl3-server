// GLM-5-Next request-level batching: correctness gates and the aggregate
// throughput of N concurrent sequences, on the REAL checkpoint.
//
// The batched decode advances every live slot by one token in one fused forward
// pass (rows = sequences), so the single-sequence path must stay untouched and
// the batched rows must produce exactly what serial decoding of the same tokens
// produces. Gates, in order:
//
//   1. one slot in the batch (with its own state) == the serial decode, logits
//      compared row by row over several steps;
//   2. four slots fed the SAME prompt and the SAME tokens == the serial decode
//      (the batch adds no cross-slot coupling);
//   3. four slots with four DIFFERENT prompts, decoded greedily for N steps ==
//      four independent serial runs, token by token;
//   4. the engine's scheduler: concurrent requests submitted from other threads
//      through the serving path == their serial runs, and a scheduler stopped
//      with a request in flight never leaves its submitter waiting (the MLX
//      stream is bound to the thread that created it, so the serving thread
//      contract is a correctness gate - see docs/perf.md);
//   5. the aggregate decode rate for B = 1/2/4/8 over 1/2/4/8 distinct prompts
//      (same session, warmed), the expert-set size printed beside each cell.
//
// usage: glm5_batch_test <model_dir> [steps] [--bench]
#include <mlx/mlx.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "engine.h"
#include "glm5_forward.h"
#include "glm5_moe.h" // MoeRowsDiag: the routed-expert term's bytes per step
#include "model.h"
#include "trellis_gemm.h" // TrellisPathInfo: which kernel the step's classes take
#include "util.h"

namespace mx = mlx::core;

static int g_fail = 0;
static void eq(const char* what, bool ok) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++g_fail;
}

static double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct Row {
  std::vector<float> logits;
  int argmax = -1;
};

static int run(int argc, char** argv);

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 1;
  }
}

static int run(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model_dir> [steps] [--bench]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  int steps = 6;
  bool bench = false;
  int only = 0; // 0 = every section (primarily for the iteration loop)
  int pos_args = 0;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bench") == 0) {
      bench = true;
      continue;
    }
    if (std::strcmp(argv[i], "--only") == 0) {
      only = std::atoi(argv[++i]);
      continue;
    }
    if (pos_args == 0) steps = std::atoi(argv[i]);
    ++pos_args;
  }
  const bool sec1 = only == 0 || only == 1, sec2 = only == 0 || only == 2;
  const bool sec3 = only == 0 || only == 3;
  const bool sec4 = only == 0 || only == 4;

  exl3::Engine engine;
  std::string err;
  const int64_t t0 = exl3::now_ms();
  if (!engine.load(dir, err, 0, false)) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  std::printf("loaded in %.1fs\n", (exl3::now_ms() - t0) / 1000.0);
  exl3::Model& model = engine.model();
  exl3::Glm5Model* g5 = model.glm5();
  if (!g5) {
    std::fprintf(stderr, "not a GLM-5-Next checkpoint\n");
    return 1;
  }
  const exl3::Tokenizer& tok = engine.tokenizer();
  const int V = g5->vocab_size();

  auto logits_of = [&](const mx::array& h) {
    mx::array row = mx::slice(h, {0, (int)h.shape(1) - 1, 0}, {1, (int)h.shape(1), (int)h.shape(2)});
    mx::array lg = g5->apply_lm_head(row); // (1, V) f32
    const float* p = lg.data<float>();
    Row r;
    r.logits.assign(p, p + V);
    r.argmax = (int)(std::max_element(r.logits.begin(), r.logits.end()) - r.logits.begin());
    return r;
  };
  auto row_of = [&](const mx::array& logits, int b) {
    const float* p = logits.data<float>();
    Row r;
    r.logits.assign(p + (size_t)b * (size_t)V, p + (size_t)(b + 1) * (size_t)V);
    r.argmax = (int)(std::max_element(r.logits.begin(), r.logits.end()) - r.logits.begin());
    return r;
  };
  auto cmp = [&](const Row& a, const Row& b) {
    double maxd = 0;
    for (size_t i = 0; i < a.logits.size(); ++i)
      maxd = std::max(maxd, std::fabs((double)a.logits[i] - (double)b.logits[i]));
    return maxd;
  };
  // max|logit| of a row: the scale a *relative* tolerance is expressed against.
  // An absolute one would be meaningless here - a 45-layer fp16 stack whose
  // kernels change shape (rows=1 -> rows=B) accumulates in a different order, so
  // the batch's rows can differ from the serial row in the last bits, and the
  // difference compounds across the layers. Token identity is the hard gate.
  auto lscale = [&](const Row& a) {
    double m = 0;
    for (float v : a.logits) m = std::max(m, (double)std::fabs(v));
    return m;
  };

  // The first four drive the correctness gates; the whole set is used by the
  // bench's `prom` axis (rows that share a prompt share their routing, so the
  // number of distinct prompts is what the expert-set size follows).
  std::vector<std::string> prompts = {
      "In one sentence, what is a trellis? Answer briefly.",
      "Name the largest planet in the solar system.",
      "Write the first eight digits of pi, separated by spaces.",
      "What colour is a ripe banana?",
      "Translate 'good morning' into Japanese and Korean.",
      "List four prime numbers above twenty.",
      "Explain what a hash table is in one sentence.",
      "What is the capital of Australia?",
  };
  std::vector<std::vector<int>> ids;
  for (auto& p : prompts) {
    std::vector<exl3::ChatMessage> msgs = {{"user", p}};
    const std::string rendered = tok.render_chat(msgs, true, 0);
    ids.push_back(tok.encode(rendered, false));
  }
  std::printf("prompt tokens:");
  for (auto& v : ids) std::printf(" %zu", v.size());
  std::printf("\n");

  // ---- serial reference: `steps` greedy steps per prompt --------------------
  struct SerialRun {
    std::vector<Row> rows;          // logits after each fed token (rows[0] = after the prompt)
    std::vector<int> tokens;        // the token fed at each step (tokens[0] = sampled from rows[0])
  };
  std::vector<SerialRun> ref(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    SerialRun& r = ref[i];
    std::vector<float> lg = g5->forward_last_logits(ids[i], 0);
    Row first;
    first.logits = lg;
    first.argmax = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    r.rows.push_back(first);
    r.tokens.push_back(first.argmax);
    for (int s = 1; s < steps; ++s) {
      mx::array h = g5->forward_hidden({r.tokens.back()}, g5->position());
      Row row = logits_of(h);
      r.rows.push_back(row);
      r.tokens.push_back(row.argmax);
    }
  }
  std::printf("\nserial reference (prompt 0), %d tokens:", steps);
  for (int t : ref[0].tokens) std::printf(" %d", t);
  std::printf("\n  text: <<<%s>>>\n", tok.decode(ref[0].tokens).c_str());

  // ---- 1. one slot in the batch == the serial decode -----------------------
  if (sec1) {
    std::printf("\n== 1. batch with one slot vs serial (prompt 0) ==\n");
    // The join's own row IS the serial prefill's row, and each step's row is the
    // row the serial run reached after the same token - rows[s] is the row token
    // s was sampled from, so feeding tokens[s] must reproduce rows[s+1].
    std::vector<float> lg0;
    const int slot = g5->batch_join(ids[0], &lg0);
    eq("join returns slot 0", slot == 0);
    Row joined;
    joined.logits = lg0;
    joined.argmax = (int)(std::max_element(lg0.begin(), lg0.end()) - lg0.begin());
    double worst = cmp(joined, ref[0].rows[0]);
    eq("join logits' argmax == the serial prefill's", joined.argmax == ref[0].rows[0].argmax);
    for (int s = 0; s + 1 < steps; ++s) {
      mx::array lg = g5->batch_step({ref[0].tokens[(size_t)s]});
      Row got = row_of(lg, 0);
      worst = std::max(worst, cmp(got, ref[0].rows[(size_t)(s + 1)]));
      if (s == 0) eq("after feeding token 1, argmax == token 2", got.argmax == ref[0].tokens[1]);
    }
    std::printf("  max|logit diff| over the prefill + %d steps = %.6g\n", steps - 1, worst);
    eq("single-slot batch logits == serial logits (<= 1e-3)", worst <= 1e-3);
    g5->batch_leave(0);
    eq("batch empty after leave", g5->batch_size() == 0);
  }

  // ---- 2. four slots, same prompt, same tokens == serial -------------------
  if (sec2) {
    std::printf("\n== 2. four slots on the same prompt/tokens ==\n");
    std::vector<int> slots;
    for (int b = 0; b < 4; ++b) slots.push_back(g5->batch_join(ids[0]));
    eq("four joins give slots 0..3", slots == std::vector<int>({0, 1, 2, 3}));
    double worst = 0, worst_rel = 0, rowd = 0;
    bool argmax_ok = true;
    for (int s = 0; s + 1 < steps; ++s) {
      std::vector<int> feed(4, ref[0].tokens[(size_t)s]);
      mx::array lg = g5->batch_step(feed);
      std::vector<Row> got(4);
      for (int b = 0; b < 4; ++b) got[(size_t)b] = row_of(lg, b);
      // Four slots fed identical tokens must produce identical rows, bit for
      // bit: any difference here is cross-slot coupling, not noise.
      for (int b = 1; b < 4; ++b) rowd = std::max(rowd, cmp(got[0], got[(size_t)b]));
      for (int b = 0; b < 4; ++b) {
        const Row& want = ref[0].rows[(size_t)(s + 1)];
        worst = std::max(worst, cmp(got[(size_t)b], want));
        worst_rel = std::max(worst_rel, cmp(got[(size_t)b], want) / lscale(want));
        argmax_ok = argmax_ok && got[(size_t)b].argmax == ref[0].tokens[(size_t)(s + 1)];
      }
    }
    eq("all four rows' argmax track the serial run", argmax_ok);
    eq("the four rows are bit-identical to each other", rowd == 0);
    std::printf("  max|logit diff| = %.4g (%.3g%% of max|logit|), row-vs-row = %g\n", worst,
                worst_rel * 100, rowd);
    eq("four identical slots == serial within 2%", worst_rel <= 0.02);
    for (int b = 3; b >= 0; --b) g5->batch_leave(b);
    eq("batch empty", g5->batch_size() == 0);
  }

  // ---- 3. four different prompts == four serial runs ----------------------
  if (sec3) {
    std::printf("\n== 3. four different prompts, greedy, vs four serial runs ==\n");
    for (int b = 0; b < 4; ++b) g5->batch_join(ids[(size_t)b]);
    std::vector<int> feed(4);
    for (int b = 0; b < 4; ++b) feed[(size_t)b] = ref[(size_t)b].tokens[0];
    std::vector<std::vector<int>> got(4);
    for (int b = 0; b < 4; ++b) got[(size_t)b].push_back(feed[(size_t)b]);
    double worst = 0, worst_rel = 0;
    bool argmax_ok = true;
    for (int s = 0; s + 1 < steps; ++s) {
      mx::array lg = g5->batch_step(feed);
      for (int b = 0; b < 4; ++b) {
        Row r = row_of(lg, b);
        const Row& want = ref[(size_t)b].rows[(size_t)(s + 1)];
        worst = std::max(worst, cmp(r, want));
        worst_rel = std::max(worst_rel, cmp(r, want) / lscale(want));
        argmax_ok = argmax_ok && r.argmax == ref[(size_t)b].tokens[(size_t)(s + 1)];
        feed[(size_t)b] = r.argmax;
        got[(size_t)b].push_back(feed[(size_t)b]);
      }
    }
    eq("every step's argmax == the serial run's next token", argmax_ok);
    for (int b = 0; b < 4; ++b) {
      char msg[160];
      std::snprintf(msg, sizeof(msg), "prompt %d batched tokens == serial", b);
      eq(msg, got[(size_t)b] == ref[(size_t)b].tokens);
      std::printf("     batched <<<%s>>>\n", tok.decode(got[(size_t)b]).c_str());
    }
    std::printf("  max|logit diff| over the 4 runs = %.4g (%.3g%% of max|logit|)\n", worst,
                worst_rel * 100);
    // The hard gate is the token identity above (and the argmax per step); the
    // logits themselves move by ~1-3% because a batched matmul accumulates in a
    // different order than a rows=1 one (the trellis GEMVs are bit-exact per row
    // - tests/moe_test.cpp (g) - so this is MLX's batched f16 reduction, not the
    // expert path).
    eq("different prompts: logits within 5% of the serial run", worst_rel <= 0.05);
    g5->batch_reset();
  }

  // ---- 4. the serving thread (the engine's scheduler) ----------------------
  //
  // The HTTP server loads the checkpoint on main and serves requests on other
  // threads. MLX binds a stream to the thread that created it and refuses to
  // evaluate a primitive on a stream that thread does not have, so two things
  // must hold, and this gate is the regression test for both:
  //   (a) nothing the loader leaves in the model may be an unresolved graph
  //       (a lazy `slice/view/reshape` chain over a zero-copy shard view carries
  //       the LOADING thread's stream). That was the "There is no Stream(gpu, 0)
  //       in current thread." failure: every served request threw it;
  //   (b) every request must be built and evaluated on the same thread - the
  //       engine's scheduler thread - not on whichever HTTP worker picked the
  //       request up.
  // And a request must never be left waiting: the scheduler answers everything
  // it holds when it stops, so a submitter cannot block on a thread that is gone
  // (the old failure left the next submitter in cv.wait() forever, which is why
  // tools/batch_server_bench.sh printed nothing for 25 minutes).
  if (sec4) {
    std::printf("\n== 4. serving thread: the engine's scheduler ==\n");
    exl3::GenerateOptions o;
    o.max_tokens = steps;
    o.sample.temperature = 0.0f; // greedy, as the batch test's reference runs
    o.ignore_eos = true;         // so the token count is exactly `steps`
    o.split_think = false;
    for (int mode : {0, 2}) {
      engine.start_batch(mode);
      eq("scheduler running", engine.batch_running());
      std::vector<exl3::GenerateResult> got(2);
      std::vector<std::thread> ts;
      for (int i = 0; i < 2; ++i)
        ts.emplace_back([&engine, &got, &o, &ids, i] {
          got[(size_t)i] = engine.submit_batch(ids[(size_t)i], o);
        });
      for (auto& t : ts) t.join();
      engine.stop_batch();
      for (int i = 0; i < 2; ++i) {
        std::vector<int> want(ref[(size_t)i].tokens.begin(),
                              ref[(size_t)i].tokens.begin() + steps);
        char msg[160];
        std::snprintf(msg, sizeof(msg), "--batch %d: concurrent request %d == its serial run",
                      mode, i);
        eq(msg, got[(size_t)i].tokens == want);
      }
    }
    // Stop the scheduler while a request is mid-generation: the submitter must
    // come back (either with what it generated, or as "cancelled"), not hang.
    engine.start_batch(4);
    std::thread slow([&engine, &ids, &o] {
      exl3::GenerateOptions lo = o;
      lo.max_tokens = 512; // long enough that the request is still in flight
      (void)engine.submit_batch(ids[0], lo);
    });
    std::this_thread::sleep_for(std::chrono::seconds(3));
    const double t0 = now_s();
    engine.stop_batch();
    slow.join();
    const double stopped = now_s() - t0;
    std::printf("  stop_batch() with a request in flight returned in %.2fs\n", stopped);
    eq("stop_batch() releases the in-flight submitter (no hang)", stopped < 30.0);
  }

  // ---- 5. aggregate throughput --------------------------------------------
  if (bench) {
    std::printf("\n== 5. aggregate decode rate (same session, warmed) ==\n");
    const int warm = 12, meas = 8;
    // Which kernel the step's shared-weight classes take: a batched forward's
    // rows are the row count of every fused linear (kda qkv_proj/o_proj, the
    // shared expert, the dense MLP, mla q_b/o_proj), so one query per row count
    // says it for all of them (src/trellis_gemm.h, TrellisPathInfo). Printed
    // because a throughput cell without the path that produced it is not
    // comparable with a cell from a different build.
    {
      const exl3::TrellisTuning tn = exl3::trellis_get_tuning();
      std::printf("  tuning: mat_min_rows=%d mat_max_rows=%d nt=%d sg=%d mb=%d\n", tn.mat_min_rows,
                  tn.mat_max_rows, tn.nt, tn.sg, tn.mb);
      const exl3::G5Layer& l0 = g5->weights().layers[0];
      auto probe = [&](const char* what, const exl3::G5Linear& lin) {
        if (!lin.quantized || lin.tw.packed.size() <= 1) return false;
        std::printf("  %s (%dx%d K%d):", what, lin.in_features, lin.out_features, lin.tw.K);
        for (int rows : {1, 4, 8, 16, 32}) {
          const exl3::TrellisPathInfo pi = exl3::trellis_path_info(lin.tw, rows);
          std::printf(" %d:%s(mb%d x%d)%s", rows, pi.staged ? "staged" : "QMV", pi.rows_per_block,
                      pi.passes, rows == 32 ? "\n" : "");
        }
        return true;
      };
      if (!probe("kda qkv_proj", l0.kda_qkv)) (void)probe("shared gate", l0.shared_gate);
      std::fflush(stdout);
    }
    // serial
    {
      std::vector<int> t = ref[0].tokens;
      for (int i = 0; i < warm; ++i) {
        mx::array h = g5->forward_hidden({t.back()}, g5->position());
        t.push_back(logits_of(h).argmax);
      }
      double best = 1e30;
      for (int r = 0; r < 3; ++r) {
        const double s0 = now_s();
        for (int i = 0; i < meas; ++i) {
          mx::array h = g5->forward_hidden({t.back()}, g5->position());
          t.push_back(logits_of(h).argmax);
        }
        best = std::min(best, (now_s() - s0) / meas);
      }
      std::printf("  B=1 serial: %.2f ms/step -> %.2f tok/s (aggregate %.2f)\n", best * 1000,
                  1.0 / best, 1.0 / best);
    }
    // Interleaved rounds: the box drifts ~30% between runs and the page cache's
    // state is part of the measurement (a step whose experts are not resident
    // runs at the SSD's rate), so every configuration is measured next to every
    // other one and the reported number is the best round of each.
    //
    // `prom` is how many DISTINCT prompts the B rows are drawn from: the rows
    // that share one prompt share its routing (and so its experts), which is the
    // only thing that makes a batch's expert reads collapse (1 prompt: 8
    // distinct experts; 4 prompts: ~28; 8: ~57.6, expected = E*(1-(1-K/E)^prom)
    // for independent routings - printed so the gradient can be read against the
    // expert-set size rather than against the row count).
    struct Case {
      int B;
      int prom;
      // The three arms of the interleaved measurement, each best-of-3-rounds:
      //   ms       the shipped geometry (the default tuning: a run of rows >= the
      //            threshold takes the staged matrix path)
      //   ms_off   grouping WITHOUT staging: every run blocked to rows = r on the
      //            byte-exact QMV (EXL3_MOE_QMVROWS + mat_min_rows = 0) - the
      //            ablation that says the QMV row blocking does not pay
      //   ms_legacy one member per (row, expert) pair (EXL3_MOE_ROWS1) - the
      //            pre-grouping tree this change replaces, and identical to the
      //            shipped geometry below the threshold
      std::vector<double> ms;
      std::vector<double> ms_off;
      std::vector<double> ms_legacy;
      // The routed-expert term the arm streamed, and the step's distinct
      // experts - weight bytes per step (MoeRowsDiag), one entry per arm in the
      // order above, plus the measured runs (grouped members) and distinct
      // experts per layer per step. The gap between streamed and shared is the
      // whole effect of the run geometry.
      double streamed[3] = {0, 0, 0};
      double shared[3] = {0, 0, 0};
      double runs[3] = {0, 0, 0};
      double staged_runs[3] = {0, 0, 0};
      double distinct[3] = {0, 0, 0};
      double layers[3] = {0, 0, 0}; // layers per step that took the rows path
    };
    const exl3::TrellisTuning dflt_tuning = exl3::trellis_get_tuning();
    std::vector<Case> cases;
    for (int B : {1, 2, 4, 8}) cases.push_back({B, 4, {}});
    for (int B : {4, 8}) cases.push_back({B, 1, {}});
    // Appended last so the six cases above keep their position in the round
    // order (the box drifts; a reordered round is not comparable).
    cases.push_back({8, 2, {}});
    cases.push_back({8, 8, {}});
    // 13-16 rows is the width a simdgroup_matrix path would need (matrix ops
    // zero-pad to 13 or 16 rows), so the batch must also be measured there - it
    // is the row count the next phase's kernel would want, and the pair layout
    // above is deliberately free of any 2*B launch bound.
    cases.push_back({16, 1, {}});
    cases.push_back({16, 4, {}});
    const int E = engine.config().n_routed_experts, K = engine.config().num_experts_per_tok;
    const int nprom = (int)ids.size();
    int moe_layers = 0;
    for (const auto& l : g5->weights().layers)
      if (l.moe) moe_layers += 1;
    // One measured round of one case: join `prom` slots, warm `warm` steps, time
    // `meas`. The slots' state is per-batch, so dropping it here is what a
    // finished request does (batch_leave); reset is the bulk form.
    auto run_case = [&](Case& c, std::vector<double>& ms, int arm) {
      std::vector<int> feed((size_t)c.B);
      exl3::moe_rows_diag_reset(); // the expert term of THIS run only
      bool ran = false;
      try {
        for (int b = 0; b < c.B; ++b) {
          const size_t pi = (size_t)(b % std::min(c.prom, nprom));
          (void)g5->batch_join(ids[pi]);
          feed[(size_t)b] = ref[pi].tokens[0];
        }
        auto one = [&]() {
          mx::array lg = g5->batch_step(feed);
          const float* p = lg.data<float>();
          for (int b = 0; b < c.B; ++b) {
            const float* row = p + (size_t)b * (size_t)V;
            feed[(size_t)b] = (int)(std::max_element(row, row + V) - row);
          }
        };
        for (int i = 0; i < warm; ++i) one();
        const double s0 = now_s();
        for (int i = 0; i < meas; ++i) one();
        ms.push_back((now_s() - s0) / meas);
        ran = true;
      } catch (const std::exception& e) {
        // A width this box cannot hold (the batch state is ~150 MB per slot
        // plus the caches) is reported, not fatal: the numbers above it stay
        // valid.
        std::printf("  B=%d %d prompts: SKIPPED (%s)\n", c.B, c.prom, e.what());
      }
      if (ran) {
        const exl3::MoeRowsDiag d = exl3::moe_rows_diag();
        const double steps_all = (double)(warm + meas);
        c.streamed[arm] = (double)d.streamed_bytes / steps_all;
        c.shared[arm] = (double)d.shared_bytes / steps_all;
        // Per moe_layer_rows call == per layer per step.
        c.runs[arm] = d.calls ? (double)d.runs / (double)d.calls : 0.0;
        c.staged_runs[arm] = d.calls ? (double)d.staged_runs / (double)d.calls : 0.0;
        c.distinct[arm] = d.calls ? (double)d.distinct_experts / (double)d.calls : 0.0;
        c.layers[arm] = (double)d.calls / steps_all;
      }
      g5->batch_reset();
    };
    // Interleaved rounds: each round measures every case through all three arms,
    // and the arm order rotates by round so none is always first (the box drifts
    // monotonically inside a round - the three arms must not be ordered by
    // position). The box drifts far more than any lever (this pass saw the B=8
    // two-prompt cell at 465 and 289 ms for the SAME code, and the B=1 serial
    // cell at 135/171/177 ms across three processes), so only arms measured back
    // to back in one process are comparable. The printed "ms/step" line below
    // stays the best-of-3 of the default arm.
    exl3::TrellisTuning off_tuning = dflt_tuning;
    off_tuning.mat_min_rows = 0; // the byte-exact QMV (with the runs still blocked)
    auto set_arm = [&](int arm) {
      if (arm == 2) {
        unsetenv("EXL3_MOE_QMVROWS");
        exl3::trellis_set_tuning(dflt_tuning);
        setenv("EXL3_MOE_ROWS1", "1", 1);
      } else if (arm == 1) {
        unsetenv("EXL3_MOE_ROWS1");
        setenv("EXL3_MOE_QMVROWS", "1", 1);
        exl3::trellis_set_tuning(off_tuning);
      } else {
        unsetenv("EXL3_MOE_ROWS1");
        unsetenv("EXL3_MOE_QMVROWS");
        exl3::trellis_set_tuning(dflt_tuning);
      }
    };
    for (int r = 0; r < 3; ++r) {
      for (auto& c : cases) {
        for (int step = 0; step < 3; ++step) {
          const int arm = (r + step) % 3;
          set_arm(arm);
          run_case(c, arm == 0 ? c.ms : (arm == 1 ? c.ms_off : c.ms_legacy), arm);
        }
      }
    }
    unsetenv("EXL3_MOE_ROWS1");
    unsetenv("EXL3_MOE_QMVROWS");
    exl3::trellis_set_tuning(dflt_tuning);
    for (auto& c : cases) {
      if (c.ms.empty()) continue;
      const double best = *std::min_element(c.ms.begin(), c.ms.end());
      const int distinct = (int)(E * (1.0 - std::pow(1.0 - (double)K / E, std::min(c.prom, nprom))));
      std::printf("  B=%d %2d prompt%s (~%2d distinct experts): %7.2f ms/step -> %5.2f tok/s "
                  "per seq, aggregate %6.2f tok/s\n",
                  c.B, std::min(c.prom, nprom), c.prom == 1 ? " " : "s", distinct, best * 1000,
                  1.0 / best, (double)c.B / best);
      std::printf("      expert term (%d of %d MoE layers on the rows path): shipped %5.2f GB "
                  "streamed / %5.2f GB shared per step (%4.1f members, %4.1f staged, %4.1f distinct "
                  "experts per layer) | one per pair %5.2f / %5.2f GB (%4.1f members)\n",
                  (int)(c.layers[0] + 0.5), moe_layers, c.streamed[0] / 1e9, c.shared[0] / 1e9,
                  c.runs[0], c.staged_runs[0], c.distinct[0], c.streamed[2] / 1e9,
                  c.shared[2] / 1e9, c.runs[2]);
    }

    // The QMV row blocking's own contribution, measured with the staged path off
    // in BOTH arms (so the pair isolates the run geometry): the shipped geometry
    // keeps the rows=1 members below the threshold, so this arm is the ablation.
    // The box drifts far more than any lever (this pass measured the B=1 serial
    // cell, identical code in both arms, at 135 vs 177 ms across two back-to-back
    // processes - a 24% spread), so only a back-to-back pair in one process is a
    // measurement.
    std::printf("  --- paired (shipped / every run row-blocked, staged off), interleaved rounds ---\n");
    {
      for (auto& c : cases) {
        if (c.ms.empty() || c.ms_off.empty()) continue;
        const double on = *std::min_element(c.ms.begin(), c.ms.end());
        const double of = *std::min_element(c.ms_off.begin(), c.ms_off.end());
        std::printf("  pair B=%d %2d prompt%s: shipped %7.2f ms (%5.2f tok/s) | row-blocked QMV "
                    "%7.2f ms (%5.2f tok/s) -> %.2fx\n",
                    c.B, std::min(c.prom, nprom), c.prom == 1 ? " " : "s", on * 1000,
                    (double)c.B / on, of * 1000, (double)c.B / of, of / on);
      }
    }

    // The before/after that matters: the shipped geometry against one rows=1
    // member per (row, expert) pair - the pre-grouping tree. Both arms are in
    // this process, and both carry the expert term's bytes so the numbers can be
    // read against the mechanism rather than only against ms. Below the
    // threshold the two arms ARE the same geometry, so those cells read 1.00x by
    // construction (an internal consistency check, not a measurement).
    std::printf("  --- paired (shipped / one member per pair), interleaved rounds ---\n");
    for (auto& c : cases) {
      if (c.ms.empty() || c.ms_legacy.empty()) continue;
      const double grp = *std::min_element(c.ms.begin(), c.ms.end());
      const double leg = *std::min_element(c.ms_legacy.begin(), c.ms_legacy.end());
      std::printf("  pair B=%d %2d prompt%s: shipped %7.2f ms (%5.2f tok/s) | one per pair %7.2f ms "
                  "(%5.2f tok/s) -> %.2fx | expert term %5.2f -> %5.2f GB/step\n",
                  c.B, std::min(c.prom, nprom), c.prom == 1 ? " " : "s", grp * 1000,
                  (double)c.B / grp, leg * 1000, (double)c.B / leg, leg / grp,
                  c.streamed[2] / 1e9, c.streamed[0] / 1e9);
    }
  }

  std::printf("\n%s\n", g_fail == 0 ? "ALL OK" : "FAILURES");
  return g_fail == 0 ? 0 : 1;
}
