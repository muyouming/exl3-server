#pragma once
// GLM-5-Next MTP (multi-token-prediction) draft block: the row forward and the
// speculative Drafter built on it.
//
// The block is the checkpoint's extra layer `model.language_model.layers.{45}.*`
// (`num_nextn_predict_layers` 1): its own MLA + DSA indexer + MoE weights, plus
// `enorm`, `hnorm`, `eh_proj` and `shared_head.norm`. One row of it is
//
//   x_j   = eh_proj(cat[ RMSNorm(Emb(t_j), enorm), RMSNorm(h_{j-1}, hnorm) ])
//   y_j   = MLP(MLA(x_j))                  one plain decoder layer
//   row_j = RMSNorm(y_j, shared_head.norm)
//   logits_j = lm_head(row_j)              the *trunk's* head (untied, on disk)
//
// where `h_{j-1}` is the trunk's final-normed hidden at position j-1 and j the
// row's own absolute position: row j predicts the token after t_j, and the draft
// layer attends causally over its own rows 1..j. The draft block has no
// hyper-connection parameters, so its residuals are plain (the mHC mix
// `post * block + comb @ streams` degenerates to a residual add).
//
// Sources: the checkpoint's tensor inventory (3456 expert + 52 non-expert
// tensors under layers.45.*) pins the structure; the wiring above follows the
// two public implementations of this architecture, which agree with each other:
// vLLM's `HYV4MultiTokenPredictorLayer` (cat order embedding-first, draft input
// `previous_hidden_states` = the target's returned - final-normed - hidden, a
// block built without iHC, the shared head over `shared_head.norm`) and SGLang's
// `Glm4MoeModelNextN` (same cat order, same `shared_head.norm`, draft input
// `spec_info.hidden_states`). The local HF port
// (`/tmp/modeling_glm5_next.py`) has NO MTP class - it lists `layers\.45\.` and
// `layers\.\d+\.shared_head\.` in `_keys_to_ignore_on_load_unexpected` - so no
// reference row forward exists to diff against, and drafts are validated by
// their acceptance rate instead. Two details stay convention-dependent and are
// NOT verifiable here: the row-0 slot (dropped here; vLLM zeroes its embedding
// instead) and the draft layer's indexer pool anchoring below index_topk
// contexts (its grid starts at position 1, so pooling is offset by one row from
// the trunk's; every context under 2048 tokens selects densely either way).
//
// Chain semantics inside one speculative round (depth >= 2): the next row's
// `prev` is the previous row's own `row_j` - the value the shared head consumes -
// which is what both implementations feed back.

#include <memory>

#include <mlx/mlx.h>

#include <vector>

#include "glm5.h"
#include "glm5_attn.h"
#include "model.h"

namespace exl3 {

namespace mx = mlx::core;

class Drafter;
class Glm5Model;

// The draft layer's own state: one row per committed token position (rows
// 1..rows), each holding the layer's compressed MLA latent and its packed DSA
// indexer row.
struct Glm5MtpState {
  MlaState mla;
  IndexerState idx;
  int rows = 0;
  // Drops every row at or past `n` (the speculative loop rewinds a verify batch
  // the trunk only partly accepted).
  void trim(int n);
};

// Runs the draft layer over R rows: `tokens` (1,R) int32, `prev` (1,R,hidden)
// the predecessor hidden of each row (the trunk's final-normed hidden for the
// first row of a round, the previous row's `row_j` inside a chain), `start_pos`
// the absolute position of the first row. Appends the R rows to `st` and returns
// `row_j` (1,R,hidden) - what the shared head consumes.
//
// `st` must already hold the rows before `start_pos` (row j belongs to the token
// at position j); passing a scratch copy keeps the caller free to drop the rows
// again.
mx::array mtp_block_rows(const G5Mtp& m, const ModelConfig& cfg, const mx::array& embed,
                   const mx::array& tokens, const mx::array& prev, Glm5MtpState& st,
                   int start_pos);

// The MTP drafter for `model`, or nullptr when the checkpoint carries no MTP
// block. The trunk keeps the draft rows in step with its own positions while
// drafting is on (Glm5Model::forward_hidden), so a round only runs the draft
// chain itself: row `position()` for the token the engine just emitted, then one
// row per deeper draft, all in a scratch copy of the committed state.
std::shared_ptr<Drafter> make_mtp_drafter(Glm5Model& model);

} // namespace exl3
