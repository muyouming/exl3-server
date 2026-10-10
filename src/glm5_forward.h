#pragma once
// GLM-5-Next full text model forward: embedding -> hyper-connection stream
// expand -> 45 decoder layers -> hyper-head -> final norm -> EXL3 lm_head.
//
// Wiring pinned to the HF reference (modeling_glm5_next.py):
//
//   Glm5NextTextModel.forward                                   (1461-1512)
//     hidden = inputs_embeds.unsqueeze(2).expand(-1,-1,hc_mult,-1)  (1493)
//     for layer in layers: hidden, topk = layer(hidden, ...)        (1495-1506)
//     hidden = self.norm(self.hc_head(hidden))                      (1508)
//   Glm5NextTextDecoderLayer.forward                            (1299-1348)
//     residual = h; post, comb, h = attn_hc(h)                      (1312-1313)
//     h = input_layernorm(h); h = self_attn(h)                    (1315-1334)
//     h = post.unsqueeze(-1)*h.unsqueeze(-2) + comb^T @ residual  (1335-1337)
//     ... same with ffn_hc / post_attention_layernorm / mlp       (1339-1346)
//   Glm5NextTextAttention.forward                               (1176-1236)
//     q_resid = q_a_layernorm(q_a_proj(h)) is shared with the indexer
//     topk_indices = self.indexer(h, q_resid, ...)  (every layer here: the
//     config leaves `indexer_types` at "full" for all layers, see
//     configuration_glm5_next.py:175-184 with index_topk_freq=1)
//   Glm5NextTextLinearAttention.forward                          (660-772)
//   Glm5NextForConditionalGeneration.forward: logits = lm_head(hidden), the head
//   is NOT tied (config tie_word_embeddings=false, lm_head.* is on disk).
//
// State: one KdaState per KDA layer (34: conv window + delta-rule matrix) and
// one {MlaState, IndexerState} per sparse layer (11: compressed KV latent cache
// + the packed per-token indexer state), so decode can continue a prefill one
// token at a time. Nothing here is batchable: the GLM path is single-sequence.

#include <mlx/mlx.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glm5.h"
#include "glm5_attn.h"
#include "glm5_hc.h"
#include "glm5_kda.h"
#include "glm5_mtp.h"
#include "model.h"

namespace exl3 {

namespace mx = mlx::core;

// Per-stage decode profile (benchmark instrumentation only). Each stage is
// `mx::eval`d at its boundary, so the reported milliseconds include the stage's
// lazy graph evaluation, its kernel launches and one CPU/GPU sync. Enabling it
// therefore perturbs the schedule; the totals are an upper bound on each
// component and the *uninstrumented* run is the end-to-end instrument.
struct Glm5Profile {
  double hc_prepare_ms = 0.0; // 2 per layer: RMSNorm + projection + softmax + Sinkhorn
  double rms_norm_ms = 0.0;   // 2 per layer: the norm before attention and before the MLP
  double kda_ms = 0.0;        // KDA (linear-attention) layers only
  double indexer_ms = 0.0;    // DSA indexer (pooling/scoring + host top-k selection)
  double mla_ms = 0.0;        // MLA (sparse-attention) layers only
  double moe_ms = 0.0;        // routed + shared experts
  double dense_ms = 0.0;      // the first_k_dense_replace layers
  double hc_mix_ms = 0.0;     // 2 per layer: post/comb + residual mixing
  double head_ms = 0.0;       // hyper-head + final norm + lm_head
  double total_ms = 0.0;      // whole forward_hidden (profiled)
  int64_t steps = 0;
  void reset() { *this = Glm5Profile{}; }
};

class Glm5Model {
 public:
  Glm5Model() = default;

  // Loads the checkpoint in `dir` (its config must already be parsed into `cfg`,
  // arch == Arch::Glm5Next) and builds the per-layer state. `verbose` prints the
  // loader's per-layer inventory.
  bool load(const std::string& dir, const ModelConfig& cfg, std::string& err, int threads = 0,
            bool verbose = false);

  // Drops every per-layer state and rewinds to position 0.
  void reset();

  // Benchmark instrumentation: when non-null, forward_hidden() evaluates (and
  // accumulates the time of) every named stage. Null by default, so the
  // production path keeps its single lazy graph.
  void set_profile(Glm5Profile* p) { profile_ = p; }

  // Absolute position of the next token fed to forward_tokens().
  int position() const { return pos_; }
  const ModelConfig& config() const { return cfg_; }
  int vocab_size() const { return vocab_; }
  int64_t weight_bytes() const { return w_.used_bytes; }
  int64_t mapped_bytes() const { return w_.bytes; }
  int64_t used_tensors() const { return w_.used_tensors; }
  int64_t total_tensors() const { return w_.total_tensors; }
  double load_seconds() const { return load_seconds_; }
  const std::vector<std::pair<std::string, int64_t>>& skipped() const { return w_.skipped; }

  // The final normed hidden state after all layers, norm(hc_head(streams)):
  // (1, L, hidden), in the streams' dtype.
  //
  // `drafting` additionally keeps the MTP draft rows in step with the trunk's
  // positions (one row per token of the chunk) and records what rewinding a
  // partly accepted verify batch needs - see rewind() and kMaxRewindRows. The
  // engine sets it exactly when it may run speculative rounds.
  mx::array forward_hidden(const std::vector<int>& ids, int start_pos, bool drafting = false);

  // Full forward: (L, vocab) float32 logits, the logits of the position AFTER
  // each fed token. `start_pos` must equal position(), or be 0, which resets the
  // state first (a recurrent state cannot be rewound to an arbitrary position).
  mx::array forward_tokens(const std::vector<int>& ids, int start_pos);

  // Convenience wrapper: only the last position's logits, as a host vector.
  std::vector<float> forward_last_logits(const std::vector<int>& ids, int start_pos);

  // The EXL3 lm_head on top of a forward_hidden() result: (1, L, hidden) ->
  // (L, vocab) float32. Exposed so the engine can run the head on the last
  // position only (a prefill does not need the intermediate rows).
  mx::array apply_lm_head(const mx::array& hidden) const;

  // ---- MTP draft block (the checkpoint's layer cfg_.num_layers) ------------
  //
  // While `forward_hidden(..., drafting=true)` is used, every chunk appends the
  // draft rows it covers, so the committed state always holds exactly the rows
  // 1..position()-1 and rewind() can drop the rows of a partly accepted verify
  // batch. A wider chunk than kMaxRewindRows still appends its rows but cannot
  // be rewound (the recurrence's prefix states are not kept for it).
  static constexpr int kMaxRewindRows = 4;
  bool has_mtp() const { return w_.mtp.present; }
  // Keeps the draft rows in step with the trunk's positions. Only the MTP
  // drafter reads them, so only installing it turns this on (a model-free
  // drafter would pay for an extra layer per chunk for nothing).
  void set_mtp_drafting(bool on) { mtp_drafting_ = on; }
  bool mtp_drafting() const { return mtp_drafting_; }
  // Highest committed draft row (row j belongs to the token at position j).
  int mtp_rows() const { return mtp_.rows; }
  // A copy of the committed draft state. Appending to the copy leaves the
  // committed rows alone (an mx::array value is immutable), which is what the
  // draft chain needs: rows past the round's first are provisional until the
  // target accepts the token they were built for.
  Glm5MtpState mtp_state() const { return mtp_; }
  // The trunk's final-normed hidden at the last committed position - the
  // predecessor of the next draft row.
  const mx::array& last_hidden() const { return last_hidden_; }
  // One draft row for `token` at position `pos` whose predecessor hidden is
  // `prev` (1,1,hidden); returns `row_j`, what the shared head consumes.
  mx::array mtp_row(int token, const mx::array& prev, int pos, Glm5MtpState& st);
  // The trunk's lm_head on one draft row: (vocab) float32 logits.
  std::vector<float> mtp_logits(const mx::array& row);
  // Drops every draft row at or past `rows`.
  void mtp_trim(int rows) { mtp_.trim(rows); }
  // Rewinds the whole model (KDA state and window, compressed MLA cache, DSA
  // indexer rows, draft rows and the last hidden) to `pos` committed tokens.
  // `pos` must lie inside the last chunk; throws otherwise (a chunk wider than
  // kMaxRewindRows cannot be rewound - the speculative loop must not be used
  // with it).
  void rewind(int pos);

  struct LayerState {
    KdaState kda;      // linear_attention layers
    MlaState mla;      // deepseek_sparse_attention layers
    IndexerState idx;  // ditto
  };

  // The loaded weights (read-only access for the tests and benchmarks).
  const Glm5Weights& weights() const { return w_; }

  ModelConfig cfg_;
  Glm5Weights w_;
  G5HcConfig hc_;
  std::vector<LayerState> states_;
  int pos_ = 0;
  int vocab_ = 0;
  double load_seconds_ = 0.0;

 private:
  // Appends the draft rows of a chunk: row j for every token of the chunk whose
  // row is not there yet (row start_pos takes the stored last hidden, the rest
  // take the chunk's own hidden states).
  void mtp_extend(const mx::array& hidden, const std::vector<int>& ids, int start_pos);

  Glm5MtpState mtp_;
  bool mtp_drafting_ = false;              // Model::make_mtp_drafter() installed it
  Glm5Profile* profile_ = nullptr;         // set by set_profile() (bench only)
  std::vector<KdaRewind> kda_rewind_;      // per trunk layer, for rewind()
  mx::array last_hidden_ = mx::array(0.0f);  // (1,1,hidden) at pos_-1, compact
  mx::array batch_hidden_ = mx::array(0.0f); // (1,L,hidden) of the last rewound-able chunk
  int batch_start_ = 0;
  int batch_rows_ = 0;
  bool batch_captured_ = false;
};

} // namespace exl3
