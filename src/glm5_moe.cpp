// GLM-5-Next MoE / dense MLP. See glm5_moe.h for the pinned reference lines.
#include "glm5_moe.h"

#include "glm5_kda.h" // g5_linear

#include <cstdint>
#include <cstdlib>
#include <limits>
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
