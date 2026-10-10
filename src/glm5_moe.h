#pragma once
// GLM-5-Next MoE: sigmoid top-k router, routed experts and the shared expert,
// plus the dense SwiGLU MLP used by the `first_k_dense_replace` layers.
// Semantics pinned in docs/glm5.md from the HF reference
// (Glm5NextTextTopkRouter.forward 157-180, Glm5NextTextExperts.forward 119-142,
//  Glm5NextTextMLP.forward 97-105, Glm5NextTextMoE.forward 199-205).

#include <mlx/mlx.h>

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
// and the step's B*K (row, expert) pairs are executed in EXPERT ORDER, which is
// what makes the launch win and the bandwidth win the same operation.
//
// Why expert order: on this checkpoint a real batch's rows barely share their
// routing (57.6 distinct experts of 64 selections at B=8, top-8 of 288), so
// reading each pair's weight tile in routing order streams the same expert's
// tile once per row that picked it. Sorting the pairs by expert id makes every
// pair of one expert run back to back, so its tile is read from cache for all
// but the first of them (the mechanism oMLX's `moe_verify_gather` ports;
// docs/research/omlx.md §B). It is a transcription, not an approximation: each
// pair is still one rows=1 member of a grouped launch with its own weight
// buffer and its own slice of the prepared scale stacks, so the kernel's tile
// geometry, k-order and f32 split sum are untouched, and a `take` (a pure move)
// puts the per-pair results back into the rank-major (K, B, 1, H) stack that the
// weighted sum consumes exactly as before. tests/moe_test.cpp (g) hashes the
// result against N single-row calls and against the general path, bit for bit.
//
// Chunking: a Metal kernel takes at most 31 buffers (x, the member weights, the
// output), so a launch holds at most 29 members - 14 pairs for the merged
// gate|up launch (two members per pair) and 29 for down. Each chunk is
// contiguous in the expert-sorted order, so only experts that straddle a chunk
// boundary can re-read their tile. A batch of any width is handled; the layers
// whose experts are not uniform in the way the grouped forms need fall back to
// moe_layer (see moe_prep_of).
mx::array moe_layer_rows(const G5Layer& l, const ModelConfig& cfg, const mx::array& h);

// Dense SwiGLU MLP (Glm5NextTextMLP.forward, `first_k_dense_replace` layers):
//   silu(clamp(gate, max=swiglu_limit)) * clamp(up, -swiglu_limit, swiglu_limit)
//   then down_proj. h is (B, L, hidden).
mx::array dense_mlp(const G5Layer& l, const ModelConfig& cfg, const mx::array& h);

} // namespace exl3
