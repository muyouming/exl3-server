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
#include "model.h"

namespace exl3 {

namespace mx = mlx::core;

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
  mx::array forward_hidden(const std::vector<int>& ids, int start_pos);

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

  struct LayerState {
    KdaState kda;      // linear_attention layers
    MlaState mla;      // deepseek_sparse_attention layers
    IndexerState idx;  // ditto
  };

  ModelConfig cfg_;
  Glm5Weights w_;
  G5HcConfig hc_;
  std::vector<LayerState> states_;
  int pos_ = 0;
  int vocab_ = 0;
  double load_seconds_ = 0.0;
};

} // namespace exl3
