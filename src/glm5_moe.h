#pragma once
// GLM-5-Next MoE: sigmoid top-k router, routed experts and the shared expert,
// plus the dense SwiGLU MLP used by the `first_k_dense_replace` layers.
// Semantics pinned in docs/glm5.md from the HF reference
// (Glm5NextTextTopkRouter.forward 157-180, Glm5NextTextExperts.forward 119-142,
//  Glm5NextTextMLP.forward 97-105, Glm5NextTextMoE.forward 199-205).

#include <mlx/mlx.h>

#include <cstdint>
#include <utility>

#include "glm5.h"
#include "model.h"

namespace exl3 {
namespace mx = mlx::core;

// Group-limited top-k routing, verbatim from Glm5NextTextTopkRouter.forward:
//   scores            = sigmoid(logits)
//   scores_for_choice = scores + e_score_correction_bias
//   group_scores      = sum of the top-2 scores_for_choice inside each of the
//                       `n_group` groups (num_experts // n_group wide)
//   the `topk_group` best groups are kept, everything else masked to -inf
//   top_k experts are picked from the surviving scores_for_choice
//   weights           = *un-biased* scores of those experts, optionally
//                       normalised (norm_topk_prob) and scaled.
// Returns {weights (N, top_k) f32, indices (N, top_k) i32}.
std::pair<mx::array, mx::array> moe_topk_route(const mx::array& logits, const mx::array& bias,
                                               int n_group, int topk_group, int top_k,
                                               bool norm_topk_prob, float routed_scaling_factor);

// One sparse MoE block (Glm5NextTextMoE.forward): routed experts + shared expert.
// h is (B, L, hidden); the result is (B, L, hidden).
mx::array moe_layer(const G5Layer& l, const ModelConfig& cfg, const mx::array& h);

// The routed-expert block of a DECODE BATCH: `h` is (B, 1, hidden), one row per
// sequence, and every row routes on its own. The arithmetic is moe_layer's
// general path per row - each row's K experts see that row's own activation -
// and the step's B*K (row, expert) pairs are executed in EXPERT ORDER and
// GROUPED BY EXPERT, which is what makes the launch win and the bandwidth win
// the same operation.
//
// The unit is a RUN: the rows of the batch that selected one expert. The step is
// B*K pairs but at most E runs, and a run's row count r is the sharing that
// exists (identical prompt rows route identically, so a same-prompt batch of 16
// is 8 runs of 16 rows, not 128 pairs).
//
// A run of r rows reaches the staged fp16 + simdgroup_matrix path exactly when
// the tensor's rows >= the tuning threshold: the run's activation IS decode
// shaped - (r, 1, hidden), the shape the request-level batch itself passes - and
// the run is routed through the single-tensor entry point (trellis_linear), which
// owns that gate. That is the geometry this function ships, and the only one that
// paid: it is worth +18% on the B=16 same-prompt aggregate (the expert term's
// weight passes fall 16x, 17.0 -> 1.1 GB/step, and the r rows share one staged
// tile instead of r row-at-a-time QMV members).
//
// Below the threshold the step keeps the pre-grouping geometry - one rows=1
// member per (row, expert) pair - deliberately. Row-blocking the QMV is real in
// bytes (mb = min(rows, 4) shares one decode pass, so a run of r >= 4 costs
// ceil(r/4) passes instead of r) and was measured NOT to move the time: mb = 4
// inside a threadgroup costs the QMV its nt (mb * nt <= 4 is the accumulator
// budget, and nt = 4 amortises the per-iteration activation path), which cancels
// the saving - +4..5% on the same-prompt cells and -5..9% on B=16 four-prompts.
// The ablation stays reproducible: EXL3_MOE_QMVROWS groups every run (the
// measured-worse variant) and EXL3_MOE_ROWS1 refuses to group at all. The
// grouped forms' sequence axis is flattened, so their scope cannot be made and
// they are never guessed at (src/trellis_gemm.cpp, matmul_group_impl).
//
// Exactness: the grouped QMV form with r rows per member is bit-identical per
// row to the rows=1 form (mb/nt are scheduling; glm5_gemv_nt_test.cpp proves the
// invariance), and rows < 16 never leave the QMV, so the single-sequence path and
// every prefill are untouched. The staged runs carry the staged path's two named
// f32-association deviations and nothing else. A `take` (a pure move) puts the
// run results back into the rank-major (K, B, 1, H) stack the weighted sum has
// always consumed. tests/moe_test.cpp (g)/(h) hashes the result against N
// single-row calls and against the general path, bit for bit.
//
// Scope of the whole path, stated because it is not "every layer": moe_layer_rows
// needs the layer's experts to pass groupable() - and on this checkpoint only 14
// of the 42 MoE layers do, the other 28 having packed words that are not 4-byte
// aligned (the shard's arbitrary header length fixes the parity; trellis_pack-ing
// 96 GB of weights to fix it was measured a 2x loss, load_linear). Those 28
// layers take moe_layer's general gather/scatter path, which already groups the
// step's tokens per expert (mb = min(n, 4) rows per decode pass) and is what the
// prefill uses. So the run geometry above is the 14 aligned layers' win, and the
// staged runs are not yet available to the other 28 - the named next lever.
//
// Chunking: a Metal kernel takes at most 31 buffers (x, the member weights, the
// output), so a launch holds at most 29 members - 14 runs for the merged gate|up
// launch (two members per run) and 29 for down. A bucket of runs of one row count
// is chunked, so only runs that straddle a chunk boundary can re-read their
// tile; a run of a row count reached by other runs shares the launch, i.e. the
// chunking lives inside a bucket. A batch of any width is handled; the layers
// whose experts are not uniform in the way the grouped forms need fall back to
// moe_layer (see moe_prep_of).
mx::array moe_layer_rows(const G5Layer& l, const ModelConfig& cfg, const mx::array& h);

// The routed-expert term's weight bytes, accumulated by moe_layer_rows. The
// decoder is single-threaded (Engine::generate holds the engine's mutex), so the
// counters are process-global; `streamed_bytes` is what the launches actually
// read (each run's expert bytes x the weight passes its row blocking needs),
// `shared_bytes` is the step's distinct experts' bytes (one pass over each), and
// `distinct_experts` is how many experts the step's pairs selected - the expert
// term is linear in that and in the passes, not in the pair count. The gap
// between streamed and shared is the whole effect of the run geometry, and the
// per-projection passes are the ones TrellisPathInfo reports. `calls` is how many
// layers took this path (the rest fall back to moe_layer's general form - see
// moe_layer_rows' scope note), so a per-step figure divided by `calls` is per
// layer on this path, not per layer of the model. Used by
// tests/glm5_batch_test.cpp's bench to print the expert term per step.
struct MoeRowsDiag {
  int64_t calls = 0;           // moe_layer_rows calls seen
  int64_t pairs = 0;           // (row, expert) pairs per call: B * top_k
  int64_t runs = 0;            // grouped members (runs of one expert)
  int64_t distinct_experts = 0; // experts the step's pairs selected
  int64_t staged_runs = 0;     // runs that took the staged matrix path
  int64_t streamed_bytes = 0;
  int64_t shared_bytes = 0;
};
MoeRowsDiag moe_rows_diag();
void moe_rows_diag_reset();

// Dense SwiGLU MLP (Glm5NextTextMLP.forward, `first_k_dense_replace` layers):
//   silu(clamp(gate, max=swiglu_limit)) * clamp(up, -swiglu_limit, swiglu_limit)
//   then down_proj. h is (B, L, hidden).
mx::array dense_mlp(const G5Layer& l, const ModelConfig& cfg, const mx::array& h);

} // namespace exl3
