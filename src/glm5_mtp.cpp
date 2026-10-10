#include "glm5_mtp.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "engine.h"
#include "glm5_forward.h"
#include "glm5_kda.h"
#include "glm5_moe.h"

namespace exl3 {

void Glm5MtpState::trim(int n) {
  if (n >= rows) return;
  mla.trim(n);
  idx.trim(n);
  rows = std::max(n, 0);
}

mx::array mtp_block_rows(const G5Mtp& m, const ModelConfig& cfg, const mx::array& embed,
                   const mx::array& tokens, const mx::array& prev, Glm5MtpState& st,
                   int start_pos) {
  if (!m.present) throw std::runtime_error("glm5 MTP: the checkpoint carries no draft block");
  const int R = (int)tokens.shape(1);
  if (R <= 0) throw std::runtime_error("glm5 MTP: empty row list");
  if ((int)prev.shape(1) != R) {
    throw std::runtime_error("glm5 MTP: " + std::to_string(R) + " rows but " +
                             std::to_string(prev.shape(1)) + " predecessor hiddens");
  }
  if (st.rows + 1 != start_pos) {
    throw std::runtime_error("glm5 MTP: the state holds " + std::to_string(st.rows) +
                             " rows but the chunk starts at position " + std::to_string(start_pos));
  }

  // Draft block input: eh_proj(cat[enorm(emb(t_j)), hnorm(h_{j-1})]).
  mx::array emb = mx::take(embed, tokens, 0); // (1, R, hidden) f16
  mx::array h = mx::concatenate({rms_norm(emb, m.enorm, cfg.rms_norm_eps),
                                 rms_norm(mx::astype(prev, emb.dtype()), m.hnorm,
                                          cfg.rms_norm_eps)},
                                -1);
  mx::array x = g5_linear(m.eh_proj, h); // (1, R, hidden) f16

  // One plain decoder layer: MLA + DSA indexer, then MoE. The draft block has no
  // hyper-connection weights, so both sub-layers are residual adds.
  const G5Layer& l = m.layer;
  mx::array residual = x;
  mx::array hn = rms_norm(x, l.input_norm, cfg.rms_norm_eps);
  mx::array q_resid = mla_q_resid(l, hn);
  mx::array topk = indexer_topk(l, cfg, hn, q_resid, st.idx, start_pos);
  x = mx::add(residual, mla_layer(l, cfg, hn, st.mla, start_pos, topk, q_resid));

  residual = x;
  hn = rms_norm(x, l.post_norm, cfg.rms_norm_eps);
  x = mx::add(residual, l.moe ? moe_layer(l, cfg, hn) : dense_mlp(l, cfg, hn));

  // shared_head.norm: what the trunk's lm_head - and the next chain row - eats.
  mx::array row = rms_norm(x, m.head_norm, cfg.rms_norm_eps); // (1, R, hidden)
  st.rows += R;
  return row;
}

namespace {

// MTP drafter. Greedy by construction: the targets the engine verifies against
// are argmaxes, and the target's own argmax re-scores every proposal.
class MtpDrafter : public Drafter {
 public:
  explicit MtpDrafter(Glm5Model& model) : model_(model) {}

  std::vector<int> draft(const std::vector<int>& context, int cur, int k,
                         const std::vector<float>& taps) override {
    (void)context;
    (void)taps;
    std::vector<int> out;
    const int pos = model_.position(); // committed token positions
    if (k <= 0 || pos <= 0 || !model_.has_mtp()) return out;
    // The committed state is in step with `pos` already (the trunk appends one
    // draft row per token it processes, and rewind() drops the rows of a partly
    // accepted chunk); the chain below runs on a scratch copy because its deeper
    // rows are provisional until the target accepts what they were built for.
    Glm5MtpState st = model_.mtp_state();
    st.trim(pos - 1);
    mx::array prev = model_.last_hidden(); // h_{pos-1}
    int token = cur;
    for (int i = 0; i < k; ++i) {
      mx::array row = model_.mtp_row(token, prev, pos + i, st);
      std::vector<float> logits = model_.mtp_logits(row);
      int best = 0;
      for (size_t v = 1; v < logits.size(); ++v) {
        if (logits[v] > logits[(size_t)best]) best = (int)v;
      }
      out.push_back(best);
      token = best;
      prev = row;
    }
    return out;
  }

 private:
  Glm5Model& model_;
};

} // namespace

std::shared_ptr<Drafter> make_mtp_drafter(Glm5Model& model) {
  if (!model.has_mtp()) return nullptr;
  return std::make_shared<MtpDrafter>(model);
}

} // namespace exl3
