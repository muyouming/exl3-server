// GLM-5-Next MoE / dense MLP. See glm5_moe.h for the pinned reference lines.
#include "glm5_moe.h"

#include "glm5_kda.h" // g5_linear

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <vector>

namespace exl3 {

std::pair<mx::array, mx::array> moe_topk_route(const mx::array& logits, const mx::array& bias,
                                               int n_group, int topk_group, int top_k,
                                               bool norm_topk_prob, float routed_scaling_factor) {
  const int N = (int)logits.shape(0);
  const int E = (int)logits.shape(1);
  // n_group/topk_group come from the config (1/1 for this checkpoint, which the
  // `topk_group == n_group` branch below degenerates to "keep every expert").
  const int ng = n_group > 1 ? n_group : 1;
  const int tg = topk_group > 0 ? (topk_group < ng ? topk_group : ng) : 1;
  const int gsize = E / ng;

  mx::array scores = mx::sigmoid(logits);
  mx::array sfc = mx::add(scores, mx::astype(bias, mx::float32)); // scores_for_choice

  // group_scores = sum of the top-2 scores_for_choice in each group.
  mx::array r = mx::reshape(sfc, {N, ng, gsize});
  const int gtop = gsize >= 2 ? 2 : 1;
  mx::array gperm = mx::argpartition(mx::negative(r), gtop - 1, 2); // (N, ng, gsize)
  mx::array gidx = mx::slice(gperm, {0, 0, 0}, {N, ng, gtop});
  mx::array gtop2 = mx::take_along_axis(r, gidx, 2);          // (N, ng, gtop)
  mx::array group_scores = mx::sum(gtop2, 2);                 // (N, ng)

  // Keep the `topk_group` best groups (topk_group == n_group keeps everything -
  // this is the path taken by the shipped checkpoint).
  mx::array gperm2 = mx::argpartition(mx::negative(group_scores), tg - 1, 1);
  mx::array gsel = mx::slice(gperm2, {0, 0}, {N, tg}); // (N, tg)
  mx::array gmask = mx::scatter_add_axis(mx::zeros({N, ng}, mx::float32), gsel,
                                         mx::ones({N, tg}, mx::float32), 1);
  mx::array score_mask =
      mx::reshape(mx::broadcast_to(mx::expand_dims(gmask, -1), {N, ng, gsize}), {N, E});
  mx::array masked = mx::where(mx::greater(score_mask, mx::array(0.0f)), sfc,
                               mx::array(-std::numeric_limits<float>::infinity(), mx::float32));

  // top_k of the masked scores_for_choice; weights come from the un-biased scores.
  mx::array perm = mx::argpartition(mx::negative(masked), top_k - 1, 1); // (N, E)
  mx::array indices = mx::astype(mx::slice(perm, {0, 0}, {N, top_k}), mx::int32);
  mx::array weights = mx::take_along_axis(scores, indices, 1); // (N, top_k) f32

  if (norm_topk_prob) {
    mx::array denom = mx::add(mx::sum(weights, 1, /*keepdims=*/true), mx::array(1e-20f));
    weights = mx::divide(weights, denom);
  }
  weights = mx::multiply(weights, mx::array(routed_scaling_factor, mx::float32));
  return {weights, indices};
}

namespace {

// silu(gate_clamped) * up_clamped -> down_proj (shared by the experts and the
// dense MLP). `silu(x) = x * sigmoid(x)` - MLX C++ has no silu op.
mx::array swiglu(const G5Linear& g, const G5Linear& u, const G5Linear& d, const mx::array& x,
                 float limit) {
  mx::array gate = g5_linear(g, x);
  mx::array up = g5_linear(u, x);
  gate = mx::clip(gate, std::nullopt, mx::array(limit, gate.dtype()));
  up = mx::clip(up, mx::array(-limit, up.dtype()), mx::array(limit, up.dtype()));
  mx::array act = mx::multiply(gate, mx::sigmoid(gate));
  return g5_linear(d, mx::multiply(act, up));
}

} // namespace

mx::array dense_mlp(const G5Layer& l, const ModelConfig& cfg, const mx::array& h) {
  return swiglu(l.dense_gate, l.dense_up, l.dense_down, h, cfg.swiglu_limit);
}

namespace {

// Defined below, with the rest of the layer's decode preparation (G5MoePrep):
// declared here because moe_layer_rows needs it before the definition.
const G5MoePrep* moe_prep_of(const G5Layer& l, const ModelConfig& cfg);

// A grouped launch's member list must share this, and the kernel is limited to
// 31 buffers: x + n member weights + the output, so n <= 29. Measured: a launch
// with 30 members fails to build ("buffer attribute parameter is out of
// bounds: must be between 0 and 30").
constexpr int kMaxGroupMembers = 29;

// The routed-expert byte accounting (MoeRowsDiag): one process-global accumulator
// that moe_layer_rows fills and tests/glm5_batch_test.cpp reads per step.
MoeRowsDiag& moe_rows_diag_state() {
  static MoeRowsDiag d;
  return d;
}

// Everything a grouped launch requires of one member. The *cross-member*
// uniformity (same shape, K, codebook, alignment, no bias, for every expert of a
// projection class) is established once per layer by moe_prep_of(), which
// returns nullptr unless every one of the 288 experts of each class matches its
// class' first one - so a chunk may mix any experts into one launch.
bool groupable(const G5Linear& g) {
  return g.quantized && !g.has_bias && g.tw.aligned && g.tw.packed.size() > 1;
}

} // namespace

mx::array moe_layer_rows(const G5Layer& l, const ModelConfig& cfg, const mx::array& h) {
  const int B = (int)h.shape(0), L = (int)h.shape(1), H = (int)h.shape(2);
  const int E = cfg.n_routed_experts;
  const int K = cfg.num_experts_per_tok;
  const G5MoePrep* prep = moe_prep_of(l, cfg);
  // A one-row call IS the serial decode path (its fast path is the same
  // arithmetic and one launch per class).
  if (B <= 1 || L != 1 || prep == nullptr || E <= 0 || K <= 0) {
    return moe_layer(l, cfg, h);
  }

  // The shared expert does not depend on the router: dispatch it first so the
  // GPU has it queued while the host waits for the ids (as moe_layer does).
  mx::array shared = swiglu(l.shared_gate, l.shared_up, l.shared_down, h, cfg.swiglu_limit);

  mx::array x = mx::reshape(mx::astype(h, mx::float32), {B, H});
  mx::array logits = mx::matmul(x, mx::transpose(prep->router_w_f32));
  auto [weights, indices] = moe_topk_route(logits, l.router_bias, cfg.n_group, cfg.topk_group, K,
                                           cfg.norm_topk_prob, cfg.routed_scaling_factor);
  mx::eval(indices); // the step's one host round trip
  const int32_t* idx = indices.data<int32_t>();

  // ---- the step's work items, grouped by expert -----------------------------
  //
  // The step has B*K (row, expert) pairs. Routing order visits them row by row,
  // so the same expert's weight tile is read once per row that selected it - and
  // on this checkpoint the rows of a real batch barely overlap when the prompts
  // differ (57.6 distinct experts of 64 selections at B=8, top-8 of 288).
  // Executing the pairs in EXPERT order makes every pair of one expert run back
  // to back, so its tile is read from cache instead of DRAM for all but the
  // first of them (the mechanism oMLX's `moe_verify_gather` ports;
  // docs/research/omlx.md §B).
  //
  // Ordering alone cannot remove the *duplication*: with rows=1 members the
  // expert's tile is still decoded once per selecting row. The RUN - the rows of
  // the batch that selected one expert - is therefore the launch's unit: a run of
  // r rows whose r reaches the staged threshold is routed through the
  // single-tensor entry point with its (r, 1, hidden) activation, which IS
  // decode-shaped, so its r rows share one staged tile and one weight pass
  // instead of r row-at-a-time QMV members (the mechanism, and why the
  // sub-threshold runs deliberately keep the rows=1 geometry, are in the
  // header). The per-run results are reordered back with a `take` (a pure move)
  // before the weighted sum, which sees the (K, B, 1, H) layout and the
  // reduction order it always had. tests/moe_test.cpp (g)/(h) hashes the rows
  // path against N single-row calls and against the general gather/scatter form,
  // bit for bit at every width it can be reached with.
  const int P = B * K;
  struct Pair {
    int expert, row, rank;
  };
  std::vector<Pair> pairs;
  pairs.reserve((size_t)P);
  for (int b = 0; b < B; ++b) {
    for (int k = 0; k < K; ++k) {
      const int e = (int)idx[(size_t)b * K + k];
      if (e < 0 || e >= E || (int)l.experts.size() != E * 3) return moe_layer(l, cfg, h);
      const G5Linear& g = l.experts[(size_t)(e * 3 + 0)];
      const G5Linear& u = l.experts[(size_t)(e * 3 + 1)];
      const G5Linear& d = l.experts[(size_t)(e * 3 + 2)];
      if (!groupable(g) || !groupable(u) || !groupable(d)) {
        // The layer goes to the general gather/scatter path. On this checkpoint
        // that is what happens to the two thirds of the MoE layers whose packed
        // words are not 4-byte aligned (TrellisWeight::aligned: the shard's
        // header length, which is arbitrary, fixes the parity), and it is a
        // real condition rather than a rare fallback - so it is traced by name.
        if (std::getenv("EXL3_MOE_RUNS_DEBUG")) {
          std::fprintf(stderr,
                       "MOE_ROWS_BAIL e=%d quantized=%d%d%d bias=%d%d%d aligned=%d%d%d "
                       "packed=%zu/%zu/%zu\n",
                       e, (int)g.quantized, (int)u.quantized, (int)d.quantized, (int)g.has_bias,
                       (int)u.has_bias, (int)d.has_bias, (int)g.tw.aligned, (int)u.tw.aligned,
                       (int)d.tw.aligned, g.tw.packed.size(), u.tw.packed.size(),
                       d.tw.packed.size());
        }
        return moe_layer(l, cfg, h);
      }
      pairs.push_back(Pair{e, b, k});
    }
  }
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const Pair& a, const Pair& b) { return a.expert < b.expert; });
  // The slot each sorted item belongs to: the rank-major layout (rank * B + row)
  // the weighted sum at the end of this function has always consumed, and the
  // inverse map that puts the sorted results back into it.
  std::vector<int32_t> perm((size_t)P); // perm[rank * B + row] = sorted position
  for (int p = 0; p < P; ++p) {
    const Pair& q = pairs[(size_t)p];
    perm[(size_t)(q.rank * B + q.row)] = p;
  }

  // The step's members, one per launch slot. A RUN - the rows of the batch that
  // selected one expert - is the unit the staged path takes: its activation is
  // (r, 1, hidden), so routing it through the single-tensor entry point puts its
  // r rows on one staged tile instead of r row-at-a-time QMV members, and that
  // is the only part of the row grouping that paid.
  //
  // Below the threshold the step keeps the pre-grouping geometry - one rows=1
  // member per (row, expert) pair - deliberately. Row-blocking the QMV is real
  // in bytes (mb = min(rows, 4) rows share one decode pass, so a run of r >= 4
  // rows costs ceil(r/4) passes instead of r: 4x at r = 16) and was measured NOT
  // to move the time: mb = 4 inside a threadgroup costs the QMV its nt (mb * nt
  // <= 4 is the accumulator budget, and nt = 4 is what amortises the
  // per-iteration activation path), which cancels the pass saving - the
  // ablation measured +4..5% on the same-prompt cells and -5..9% on B=16
  // four-prompts with the grouping alone, and the same again with it plus every
  // r >= 16 run staged. So the QMV keeps its rows=1 geometry. Two switches keep
  // the ablation reproducible: EXL3_MOE_QMVROWS groups every run (the
  // measured-worse variant) and EXL3_MOE_ROWS1 refuses to group at all (the
  // tree this change replaces).
  const bool per_pair = std::getenv("EXL3_MOE_ROWS1") != nullptr;
  const bool qmv_rows = std::getenv("EXL3_MOE_QMVROWS") != nullptr;
  struct Run {
    int expert, off, r;
  };
  std::vector<Run> runs;
  // Which runs the staged fp16 + simdgroup_matrix path takes. Its scope rule is a
  // *shape* rule - a decode-shaped activation (>= 3-D, sequence axis 1) at rows
  // >= the tuning threshold - and a run's activation IS (r, 1, hidden), so
  // routing the run through the single-tensor entry point is exactly what puts
  // it on the staged path. The three projections of an expert are one shape
  // class, so their answers agree; the gate itself lives in trellis_path_info.
  std::vector<char> staged;
  runs.reserve((size_t)P);
  staged.reserve((size_t)P);
  for (int p = 0; p < P;) {
    int q = p;
    while (q < P && pairs[(size_t)q].expert == pairs[(size_t)p].expert) ++q;
    const int e = pairs[(size_t)p].expert, r = q - p;
    const bool st =
        !per_pair && trellis_path_info(l.experts[(size_t)(e * 3 + 0)].tw, r).staged &&
        trellis_path_info(l.experts[(size_t)(e * 3 + 1)].tw, r).staged &&
        trellis_path_info(l.experts[(size_t)(e * 3 + 2)].tw, r).staged;
    const bool grouped = !per_pair && (qmv_rows || st);
    if (grouped) {
      runs.push_back(Run{e, p, r});
      staged.push_back(st ? 1 : 0);
    } else {
      for (int i = p; i < q; ++i) {
        runs.push_back(Run{pairs[(size_t)i].expert, i, 1});
        staged.push_back(0);
      }
    }
    p = q;
  }

  const float limit = cfg.swiglu_limit;
  mx::array hf = mx::reshape(h, {B, H});
  auto run_rows = [&](const Run& rn) {
    std::vector<int32_t> rows((size_t)rn.r);
    for (int t = 0; t < rn.r; ++t) rows[(size_t)t] = pairs[(size_t)(rn.off + t)].row;
    return rows;
  };

  // The expert term's byte accounting (MoeRowsDiag): the distinct experts' bytes
  // and what these launches will actually read.
  {
    MoeRowsDiag& st = moe_rows_diag_state();
    st.calls += 1;
    st.pairs += P;
    st.runs += (int64_t)runs.size();
    auto proj_bytes = [&](int e, int w) { return l.experts[(size_t)(e * 3 + w)].tw.bytes(); };
    for (int p = 0; p < P; ++p) {
      const int e = pairs[(size_t)p].expert;
      if (p == 0 || e != pairs[(size_t)(p - 1)].expert) {
        st.distinct_experts += 1;
        st.shared_bytes += proj_bytes(e, 0) + proj_bytes(e, 1) + proj_bytes(e, 2);
      }
    }
    for (size_t i = 0; i < runs.size(); ++i) {
      if (staged[i]) st.staged_runs += 1;
      for (int w = 0; w < 3; ++w) {
        const TrellisWeight& tw = l.experts[(size_t)(runs[i].expert * 3 + w)].tw;
        st.streamed_bytes += tw.bytes() * trellis_path_info(tw, runs[i].r).passes;
      }
    }
  }
  if (std::getenv("EXL3_MOE_RUNS_DEBUG")) {
    std::map<int, int> hist;
    for (const Run& rn : runs) hist[rn.r] += 1;
    std::fprintf(stderr, "MOE_ROWS P=%d runs=%zu staged=%zu rhist=", P, runs.size(),
                 (size_t)std::count(staged.begin(), staged.end(), (char)1));
    for (const auto& hh : hist) std::fprintf(stderr, "%dx%d ", hh.second, hh.first);
    std::fprintf(stderr, "\n");
  }

  // The per-run output (r, 1, H), filled by whichever path serves the run and
  // concatenated in run order at the end; run order IS sorted-pair order, which
  // is the order `perm` undoes.
  std::vector<mx::array> run_out;
  run_out.reserve(runs.size());
  for (size_t i = 0; i < runs.size(); ++i) run_out.emplace_back(mx::array(0.0f));

  // (a) the staged runs: one expert, r rows, three decode-shaped single-tensor
  // calls. The gather is the grouped path's own row gather; the (r, 1, H) shape
  // is what makes the call decode-shaped, and the r rows then share one staged
  // tile instead of r rows=1 members.
  for (size_t i = 0; i < runs.size(); ++i) {
    if (!staged[i]) continue;
    const Run& rn = runs[i];
    std::vector<int32_t> rows = run_rows(rn);
    mx::array xr = mx::reshape(
        mx::take(hf, mx::array(rows.data(), {rn.r}, mx::int32), 0), {rn.r, 1, H});
    run_out[i] = swiglu(l.experts[(size_t)(rn.expert * 3 + 0)],
                        l.experts[(size_t)(rn.expert * 3 + 1)],
                        l.experts[(size_t)(rn.expert * 3 + 2)], xr, limit);
  }

  // (b) the runs the QMV serves, as members of the grouped launches, bucketed by
  // row count: the grouped activation is one (n, rows, in) block, so its members
  // share `rows` - and with it the geometry mb = min(rows, 4) that decides how
  // many rows share one decode pass. Every run of a bucket is one member pair:
  // the merged gate|up launch holds the bucket's gate members first and its up
  // members second (moe_prep_of's scale-stack order, and the gu4 split below), so
  // a chunk is at most kMaxGroupMembers/2 runs; the down launch holds one member
  // per run and is chunked separately (kMaxGroupMembers), exactly as the
  // per-pair form was. A bucket of rows=1 runs with one run per pair is exactly
  // that per-pair form, which is what EXL3_MOE_ROWS1 selects.
  std::map<int, std::vector<size_t>> buckets;
  for (size_t i = 0; i < runs.size(); ++i)
    if (!staged[i]) buckets[runs[i].r].push_back(i);
  std::vector<mx::array> act_run;
  act_run.reserve(runs.size());
  for (size_t i = 0; i < runs.size(); ++i) act_run.emplace_back(mx::array(0.0f));
  const int gu_chunk = kMaxGroupMembers / 2;
  for (const auto& bucket : buckets) {
    const int r = bucket.first;
    const std::vector<size_t>& ks = bucket.second;
    const int c = (int)ks.size();
    for (int s = 0; s < c; s += gu_chunk) {
      const int cc = std::min(gu_chunk, c - s);
      std::vector<TrellisWeight> wgu((size_t)2 * cc);
      std::vector<int32_t> gu_ids((size_t)2 * cc), grow((size_t)2 * cc * r);
      for (int i = 0; i < cc; ++i) {
        const Run& rn = runs[ks[(size_t)(s + i)]];
        const int e = rn.expert;
        std::vector<int32_t> rows = run_rows(rn);
        wgu[(size_t)i] = l.experts[(size_t)(e * 3 + 0)].tw;
        wgu[(size_t)(cc + i)] = l.experts[(size_t)(e * 3 + 1)].tw;
        gu_ids[(size_t)i] = e;
        gu_ids[(size_t)(cc + i)] = E + e;
        for (int t = 0; t < r; ++t) {
          grow[(size_t)(i * r + t)] = rows[(size_t)t];
          grow[(size_t)((cc + i) * r + t)] = rows[(size_t)t];
        }
      }
      // The activation block is `r` rows per member: member i<cc is run i's rows
      // for its gate expert, member cc+i the same rows for its up expert.
      mx::array xgu = mx::reshape(
          mx::take(hf, mx::array(grow.data(), {2 * cc * r}, mx::int32), 0), {2 * cc, r, H});
      mx::array gu = trellis_linear_group_rows_scaled(
          wgu, xgu, r,
          mx::take(prep->suh_gu, mx::array(gu_ids.data(), {2 * cc}, mx::int32), 0),
          mx::take(prep->svh_gu, mx::array(gu_ids.data(), {2 * cc}, mx::int32), 0));
      const int out_i = wgu[0].out_features;
      mx::array gu4 = mx::reshape(gu, {2, cc, r, out_i});
      mx::array gate = mx::reshape(mx::slice(gu4, {0, 0, 0, 0}, {1, cc, r, out_i}), {cc, r, out_i});
      mx::array up = mx::reshape(mx::slice(gu4, {1, 0, 0, 0}, {2, cc, r, out_i}), {cc, r, out_i});
      gate = mx::clip(gate, std::nullopt, mx::array(limit, gate.dtype()));
      up = mx::clip(up, mx::array(-limit, up.dtype()), mx::array(limit, up.dtype()));
      mx::array act = mx::multiply(mx::multiply(gate, mx::sigmoid(gate)), up); // (cc, r, inn)
      const int inn = (int)act.shape(2);
      for (int i = 0; i < cc; ++i)
        act_run[ks[(size_t)(s + i)]] =
            mx::reshape(mx::slice(act, {i, 0, 0}, {i + 1, r, inn}), {r, 1, inn});
    }
  }
  for (const auto& bucket : buckets) {
    const int r = bucket.first;
    const std::vector<size_t>& ks = bucket.second;
    const int c = (int)ks.size();
    for (int s = 0; s < c; s += kMaxGroupMembers) {
      const int cc = std::min(kMaxGroupMembers, c - s);
      std::vector<TrellisWeight> wd((size_t)cc);
      std::vector<int32_t> d_ids((size_t)cc);
      std::vector<mx::array> parts;
      parts.reserve((size_t)cc);
      for (int i = 0; i < cc; ++i) {
        const Run& rn = runs[ks[(size_t)(s + i)]];
        wd[(size_t)i] = l.experts[(size_t)(rn.expert * 3 + 2)].tw;
        d_ids[(size_t)i] = rn.expert;
        parts.push_back(act_run[ks[(size_t)(s + i)]]);
      }
      mx::array xd = parts.size() == 1 ? parts[0] : mx::concatenate(parts, 0);
      const int inn = (int)xd.shape(xd.ndim() - 1);
      xd = mx::reshape(xd, {cc, r, inn});
      mx::array down = trellis_linear_group_rows_scaled(
          wd, xd, r,
          mx::take(prep->suh_d, mx::array(d_ids.data(), {cc}, mx::int32), 0),
          mx::take(prep->svh_d, mx::array(d_ids.data(), {cc}, mx::int32), 0));
      for (int i = 0; i < cc; ++i)
        run_out[ks[(size_t)(s + i)]] =
            mx::reshape(mx::slice(down, {i, 0, 0}, {i + 1, r, H}), {r, 1, H});
    }
  }
  mx::array down_all = run_out.size() == 1 ? run_out[0] : mx::concatenate(run_out, 0);

  // Back to the rank-major (K, B, 1, H) stack this function has always built,
  // then the general path's weighted sum, term by term in rank order: (K, B, 1,
  // H) times the per-(row, rank) weight, rounded to h's dtype, summed over K.
  mx::array stk =
      mx::reshape(mx::take(down_all, mx::array(perm.data(), {P}, mx::int32), 0), {K, B, 1, H});
  mx::array wt = mx::reshape(mx::transpose(weights), {K, B, 1, 1}); // (B,K) f32 -> (K,B,1,1)
  mx::array routed = mx::reshape(mx::sum(mx::astype(mx::multiply(stk, wt), h.dtype()), 0),
                                 {B, L, H});
  return mx::add(routed, shared);
}

MoeRowsDiag moe_rows_diag() { return moe_rows_diag_state(); }
void moe_rows_diag_reset() { moe_rows_diag_state() = MoeRowsDiag{}; }

namespace {

// The layer's decode preparation (G5MoePrep in glm5.h), built on first use:
// the router weight in the layout its matmul consumes, and one stacked scale
// vector per projection class. Returns nullptr when the experts are not uniform
// in the way the grouped forms need, in which case moe_layer takes the
// per-expert path (the arithmetic is the same either way). The build is a pure
// re-layout of the layer's own tensors; nothing here is per token.
const G5MoePrep* moe_prep_of(const G5Layer& l, const ModelConfig& cfg) {
  // EXL3_MOE_OLDPREP forces the pre-prepared path (the grouped launches built
  // from the members' own scale vectors) so the two can be compared.
  if (std::getenv("EXL3_MOE_OLDPREP")) return nullptr;
  // Lazily built, without a lock: the decode path is single-threaded
  // (Engine::generate holds the engine's mutex), and a raced build would only
  // waste work, never produce a wrong result - the cache is written once, last.
  if (l.moe_prep && l.moe_prep->valid) return l.moe_prep.get();
  if (!l.moe || l.router_w.size() == 0) return nullptr;
  const int E = cfg.n_routed_experts;
  if (E <= 0 || (int)l.experts.size() != (size_t)E * 3) return nullptr;
  // Every expert of a class must share the shape/quantization its neighbours
  // have, otherwise a grouped launch cannot be geometrically uniform.
  auto same = [](const G5Linear& a, const G5Linear& b) {
    return a.quantized && b.quantized && !a.has_bias && !b.has_bias && a.tw.K == b.tw.K &&
           a.tw.codebook == b.tw.codebook && a.tw.in_features == b.tw.in_features &&
           a.tw.out_features == b.tw.out_features && a.tw.aligned == b.tw.aligned;
  };
  for (int e = 0; e < E; ++e) {
    if (!same(l.experts[0], l.experts[(size_t)(e * 3 + 0)]) ||
        !same(l.experts[1], l.experts[(size_t)(e * 3 + 1)]) ||
        !same(l.experts[2], l.experts[(size_t)(e * 3 + 2)])) {
      return nullptr;
    }
  }
  auto prep = std::make_shared<G5MoePrep>();
  // One (E*classes, width) stack per side: `which` lists the projection classes
  // in member order (the merged launch's gate rows then up rows).
  auto stack = [&](const std::vector<int>& which, bool in_side, mx::array& out) {
    std::vector<mx::array> v;
    v.reserve((size_t)E * which.size());
    for (int w : which) {
      for (int e = 0; e < E; ++e) {
        const G5Linear& lin = l.experts[(size_t)(e * 3 + w)];
        v.push_back(in_side ? lin.tw.suh : lin.tw.svh);
      }
    }
    const G5Linear& ref = l.experts[(size_t)which[0]];
    out = mx::reshape(mx::concatenate(v, 0),
                      {(int)which.size() * E,
                       in_side ? ref.tw.in_features : ref.tw.out_features});
  };
  // Only the cast is hoisted; the matmul keeps the operand graph it had (a
  // transposed f32 view). Hoisting the cast alone is bit-safe - the prepared
  // router and the per-token expression agree to 0.0 and select the same
  // experts (tests/moe_bench.cpp prints that check) - and it removes the
  // (E, hidden) f16->f32 cast, 14 MB of traffic, from every layer's step.
  prep->router_w_f32 = mx::astype(l.router_w, mx::float32);
  stack({0, 1}, true, prep->suh_gu);   // gate rows, then up rows
  stack({0, 1}, false, prep->svh_gu);
  stack({2}, true, prep->suh_d);
  stack({2}, false, prep->svh_d);
  mx::eval(prep->router_w_f32, prep->suh_gu, prep->svh_gu, prep->suh_d, prep->svh_d);
  prep->valid = true;
  l.moe_prep = prep;
  return prep.get();
}

} // namespace

mx::array moe_layer(const G5Layer& l, const ModelConfig& cfg, const mx::array& h) {
  const int B = (int)h.shape(0), L = (int)h.shape(1), H = (int)h.shape(2);
  const int N = B * L;
  const int E = cfg.n_routed_experts;
  const int K = cfg.num_experts_per_tok;

  // The shared expert takes the same input and does not depend on the router,
  // so it is built (and dispatched) before the routed path: the MoE's one host
  // round trip - reading the selected ids back - then waits on a GPU that
  // already has this work queued instead of on an idle one.
  mx::array shared = swiglu(l.shared_gate, l.shared_up, l.shared_down, h, cfg.swiglu_limit);

  // Router logits in fp32 exactly as the reference; the cast the expression
  // below spells out per token is the prepared one when the layer has a prep.
  const G5MoePrep* prep = moe_prep_of(l, cfg);
  mx::array x = mx::reshape(mx::astype(h, mx::float32), {N, H});
  mx::array logits = prep ? mx::matmul(x, mx::transpose(prep->router_w_f32))
                          : mx::matmul(x, mx::transpose(mx::astype(l.router_w, mx::float32)));
  auto [weights, indices] = moe_topk_route(logits, l.router_bias, cfg.n_group, cfg.topk_group, K,
                                          cfg.norm_topk_prob, cfg.routed_scaling_factor);

  mx::array routed = mx::array(0.0f);
  // Both A/B switches (moe_test drives all three implementations) must keep
  // their meaning: EXL3_MOE_GENERAL forces the general gather/scatter path and
  // EXL3_MOE_NONGROUPED the per-expert one.
  if (N == 1 && prep != nullptr && std::getenv("EXL3_MOE_NONGROUPED") == nullptr &&
      std::getenv("EXL3_MOE_GENERAL") == nullptr) {
    // Decode fast path with the layer's prepared scale stacks: every work item
    // reads the SAME row, so the per-expert gather/scatter bookkeeping collapses
    // into three grouped launches (gate|up merged into one, then down) plus the
    // weighted sum. The arithmetic is the general path's: each expert output is
    // scaled in f32, rounded to the streams' dtype, and the K terms are summed
    // k = 0, 1, ... in the same order the general path accumulates them.
    mx::eval(indices);
    const int32_t* idx = indices.data<int32_t>();
    // The merged launch's member order is the scale stacks' row order: every
    // routed gate expert first, then every routed up expert (see moe_prep_of).
    std::vector<TrellisWeight> wgu, wd;
    wgu.reserve((size_t)(2 * K));
    wd.reserve((size_t)K);
    for (int k = 0; k < K; ++k) wgu.push_back(l.experts[(size_t)((int)idx[k] * 3 + 0)].tw);
    for (int k = 0; k < K; ++k) wgu.push_back(l.experts[(size_t)((int)idx[k] * 3 + 1)].tw);
    for (int k = 0; k < K; ++k) wd.push_back(l.experts[(size_t)((int)idx[k] * 3 + 2)].tw);
    mx::array flat = mx::reshape(h, {N, H});
    // The scale stacks hold every gate expert's rows first (0..E-1), then every
    // up expert's (E..2E-1): the up half of the launch indexes the stack at
    // E + id.
    mx::array ids_up = mx::add(indices, mx::array((int)E, mx::int32));
    mx::array ids2 = mx::concatenate({indices, ids_up}, 0); // gate ids, then up ids
    const float limit = cfg.swiglu_limit;
    mx::array gu = trellis_linear_group_scaled(wgu, flat, mx::take(prep->suh_gu, ids2, 0),
                                               mx::take(prep->svh_gu, ids2, 0));
    const int inn = (int)wd[0].in_features;
    mx::array gate = mx::slice(gu, {0, 0, 0}, {K, N, (int)wd[0].out_features});
    mx::array up = mx::slice(gu, {K, 0, 0}, {2 * K, N, (int)wd[0].out_features});
    gate = mx::clip(gate, std::nullopt, mx::array(limit, gate.dtype()));
    up = mx::clip(up, mx::array(-limit, up.dtype()), mx::array(limit, up.dtype()));
    mx::array act = mx::multiply(gate, mx::sigmoid(gate));
    mx::array actup = mx::multiply(act, up);
    const int rows_per = (int)(actup.size() / ((int64_t)K * inn));
    mx::array down = trellis_linear_group_rows_scaled(wd, actup, rows_per,
                                                      mx::take(prep->suh_d, indices, 0),
                                                      mx::take(prep->svh_d, indices, 0));
    // (K, rows, H) -> (K, rows, 1, H) and the general path's weighted sum.
    mx::array st = mx::reshape(down, {K, N, 1, H});
    mx::array wt = mx::reshape(weights, {K, 1, 1, 1});
    routed = mx::reshape(mx::sum(mx::astype(mx::multiply(st, wt), h.dtype()), 0), {B, L, H});
    return mx::add(routed, shared);
  }

  mx::eval(indices);

  // Group the (token, top-k) work items by expert, mirroring the reference's
  // per-expert loop (each expert's weights are applied in one matmul).
  const int32_t* idx = indices.data<int32_t>();
  // The single-row (decode) fast path is numerically the general path's; this
  // switch forces the general one so the two can be compared (moe_test).
  const bool moe_general = std::getenv("EXL3_MOE_GENERAL") != nullptr;
  if (N == 1 && !moe_general) {
    // Decode fallback (no prepared stacks, or EXL3_MOE_NONGROUPED): the 8
    // routed experts are applied as three multi-buffer launches - gate, up and
    // down - instead of 3*K single-tensor linears, each launch keeping every
    // member's own scale vectors, split geometry and accumulation order (see
    // trellis_linear_group). The elementwise swiglu then runs on the stacked
    // (K, 1, out) arrays, which is the same arithmetic on the same values.
    bool grouped = std::getenv("EXL3_MOE_NONGROUPED") == nullptr;
    std::vector<G5Linear const*> gsel;
    if (grouped) {
      for (int k = 0; k < K; ++k) {
        const int e = (int)idx[k];
        const G5Linear& g0 = l.experts[(size_t)(e * 3 + 0)];
        const G5Linear& u0 = l.experts[(size_t)(e * 3 + 1)];
        const G5Linear& d0 = l.experts[(size_t)(e * 3 + 2)];
        if (!g0.quantized || !u0.quantized || !d0.quantized || g0.has_bias || u0.has_bias ||
            d0.has_bias || g0.tw.in_features != u0.tw.in_features ||
            g0.tw.out_features != u0.tw.out_features || d0.tw.in_features != u0.tw.out_features ||
            d0.tw.out_features != g0.tw.in_features ||
            d0.tw.codebook != g0.tw.codebook || g0.tw.codebook != u0.tw.codebook ||
            d0.tw.K != g0.tw.K || g0.tw.K != u0.tw.K) {
          grouped = false; // fall back to the per-expert path below
          break;
        }
        gsel.push_back(&g0);
        gsel.push_back(&u0);
        gsel.push_back(&d0);
      }
    }
    if (grouped) {
      std::vector<TrellisWeight> ws_gate, ws_up, ws_down;
      for (int k = 0; k < K; ++k) {
        ws_gate.push_back(gsel[(size_t)(k * 3 + 0)]->tw);
        ws_up.push_back(gsel[(size_t)(k * 3 + 1)]->tw);
        ws_down.push_back(gsel[(size_t)(k * 3 + 2)]->tw);
      }
      mx::array flat = mx::reshape(h, {N, H});
      const float limit = cfg.swiglu_limit;
      // gate / up: (K, rows, out), one launch each, then swiglu on the stacks.
      mx::array gate = trellis_linear_group(ws_gate, flat);
      mx::array up = trellis_linear_group(ws_up, flat);
      gate = mx::clip(gate, std::nullopt, mx::array(limit, gate.dtype()));
      up = mx::clip(up, mx::array(-limit, up.dtype()), mx::array(limit, up.dtype()));
      mx::array act = mx::multiply(gate, mx::sigmoid(gate));
      mx::array actup = mx::multiply(act, up);
      const int inn = ws_down[0].in_features;
      const int rows_per = (int)(actup.size() / ((int64_t)K * inn));
      mx::array down = trellis_linear_group_rows(ws_down, actup, rows_per);
      // (K, rows, H) -> (K, rows, 1, H) and the general path's weighted sum.
      mx::array st = mx::reshape(down, {K, N, 1, H});
      mx::array wt = mx::reshape(weights, {K, 1, 1, 1});
      routed = mx::reshape(mx::sum(mx::astype(mx::multiply(st, wt), h.dtype()), 0), {B, L, H});
    } else {
      std::vector<mx::array> outs;
      outs.reserve((size_t)K);
      for (int k = 0; k < K; ++k) {
        const int e = (int)idx[k];
        outs.push_back(swiglu(l.experts[(size_t)(e * 3 + 0)], l.experts[(size_t)(e * 3 + 1)],
                              l.experts[(size_t)(e * 3 + 2)], h, cfg.swiglu_limit));
      }
      mx::array st = mx::stack(outs, 0);                       // (K, 1, 1, H)
      mx::array wt = mx::reshape(weights, {K, 1, 1, 1});       // (K, 1, 1, 1), f32
      routed = mx::reshape(mx::sum(mx::astype(mx::multiply(st, wt), h.dtype()), 0), {B, L, H});
    }
  } else {
    std::vector<std::vector<int>> work(E);
    for (int t = 0; t < N; ++t)
      for (int k = 0; k < K; ++k) work[(size_t)idx[(size_t)t * K + k]].push_back(t * K + k);

    mx::array hflat = mx::reshape(h, {N, H});
    mx::array wflat = mx::reshape(weights, {N * K});
    mx::array acc = mx::zeros({N * K, H}, h.dtype());
    for (int e = 0; e < E; ++e) {
      const std::vector<int>& ids = work[(size_t)e];
      if (ids.empty()) continue;
      const int n = (int)ids.size();
      std::vector<int32_t> toks((size_t)n);
      for (int i = 0; i < n; ++i) toks[(size_t)i] = ids[(size_t)i] / K;
      mx::array wid = mx::array(ids.begin(), mx::Shape{n}, mx::int32);
      mx::array tid = mx::array(toks.begin(), mx::Shape{n}, mx::int32);

      mx::array rows = mx::take(hflat, tid, 0); // (n, H)
      mx::array cur = swiglu(l.experts[(size_t)(e * 3 + 0)], l.experts[(size_t)(e * 3 + 1)],
                             l.experts[(size_t)(e * 3 + 2)], rows, cfg.swiglu_limit);
      mx::array wt = mx::take(wflat, wid, 0); // (n,)
      cur = mx::astype(mx::multiply(cur, mx::reshape(wt, {n, 1})), acc.dtype());
      mx::array scatter_idx = mx::broadcast_to(mx::reshape(wid, {n, 1}), {n, H});
      acc = mx::scatter_add_axis(acc, scatter_idx, cur, 0);
    }
    routed = mx::reshape(mx::sum(mx::reshape(acc, {N, K, H}), 1), {B, L, H});
  }
  return mx::add(routed, shared);
}

} // namespace exl3
