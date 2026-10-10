#include "glm5_forward.h"

#include <cstdio>
#include <stdexcept>

#include "glm5_moe.h"
#include "util.h"

namespace exl3 {

bool Glm5Model::load(const std::string& dir, const ModelConfig& cfg, std::string& err,
                     int threads, bool verbose) {
  cfg_ = cfg;
  const int64_t t0 = now_ms();
  if (!load_glm5(dir, cfg_, w_, err, threads, verbose)) return false;
  if (w_.layers.size() != (size_t)cfg_.num_layers) {
    err = "glm5: loader returned " + std::to_string(w_.layers.size()) + " layers, config says " +
          std::to_string(cfg_.num_layers);
    return false;
  }
  if (cfg_.hc_mult <= 0) {
    err = "glm5: hc_mult must be positive (config mhc/hc_mult)";
    return false;
  }
  hc_ = g5_hc_config(cfg_);
  vocab_ = w_.lm_head.out_features;
  if (vocab_ <= 0) {
    err = "glm5: lm_head has no output width";
    return false;
  }
  if (cfg_.vocab_size != vocab_) {
    std::fprintf(stderr, "glm5: config vocab_size %d != lm_head width %d, using the head's\n",
                 cfg_.vocab_size, vocab_);
  }
  cfg_.vocab_size = vocab_;
  states_.assign((size_t)cfg_.num_layers, LayerState{});
  pos_ = 0;
  load_seconds_ = (double)(now_ms() - t0) / 1000.0;
  return true;
}

void Glm5Model::reset() {
  for (auto& s : states_) {
    s.kda.conv_window = mx::array(0.0f);
    s.kda.recurrent = mx::array(0.0f);
    s.mla.k_pass = mx::array(0.0f);
    s.mla.len = 0;
    s.idx.packed = mx::array(0.0f);
    s.idx.len = 0;
  }
  pos_ = 0;
}

mx::array Glm5Model::forward_hidden(const std::vector<int>& ids, int start_pos) {
  const int L = (int)ids.size();
  if (L <= 0) throw std::runtime_error("Glm5Model::forward_hidden: empty token list");
  if (start_pos != pos_) {
    if (start_pos != 0) {
      throw std::runtime_error("Glm5Model: start_pos " + std::to_string(start_pos) +
                               " does not continue the state at position " + std::to_string(pos_) +
                               " (the KDA state cannot be rewound)");
    }
    reset(); // fresh sequence
  }

  mx::array ids_a(ids.data(), mx::Shape{1, L}, mx::int32);
  mx::array emb = mx::take(w_.embed, ids_a, 0);        // (1, L, hidden) f16
  mx::array streams = g5_hc_expand(emb, hc_.mult);     // (1, L, N, hidden)

  for (int i = 0; i < cfg_.num_layers; ++i) {
    const G5Layer& l = w_.layers[(size_t)i];
    LayerState& st = states_[(size_t)i];

    // ---- attention sub-layer (reference 1312-1337) -------------------------
    mx::array residual = streams;
    G5HcActivation a =
        g5_hc_prepare(streams, l.hc_attn_base, l.hc_attn_fn, l.hc_attn_scale, hc_);
    mx::array h = rms_norm(a.collapsed, l.input_norm, cfg_.rms_norm_eps);
    mx::array out = mx::array(0.0f);
    if (l.plan.linear_attention) {
      out = kda_layer(l, cfg_, h, st.kda);
    } else {
      // The indexer shares q_resid with the MLA (reference 1185-1188, 1221).
      mx::array q_resid = mla_q_resid(l, h);
      mx::array topk = indexer_topk(l, cfg_, h, q_resid, st.idx, start_pos);
      out = mla_layer(l, cfg_, h, st.mla, start_pos, topk, q_resid);
    }
    streams = g5_hc_mix(out, a.post, a.comb, residual);

    // ---- MLP sub-layer (reference 1339-1346) -------------------------------
    residual = streams;
    G5HcActivation f = g5_hc_prepare(streams, l.hc_ffn_base, l.hc_ffn_fn, l.hc_ffn_scale, hc_);
    h = rms_norm(f.collapsed, l.post_norm, cfg_.rms_norm_eps);
    out = l.moe ? moe_layer(l, cfg_, h) : dense_mlp(l, cfg_, h);
    streams = g5_hc_mix(out, f.post, f.comb, residual);
  }

  // norm(hc_head(streams)): reference 1508.
  mx::array hidden = rms_norm(g5_hc_head(streams), w_.final_norm, cfg_.rms_norm_eps);
  pos_ = start_pos + L;
  return hidden;
}

mx::array Glm5Model::apply_lm_head(const mx::array& hidden) const {
  const int L = (int)hidden.shape(1);
  const int hidden_size = (int)hidden.shape(2);
  if (w_.lm_head.in_features != hidden_size) {
    throw std::runtime_error("Glm5Model: lm_head expects " +
                             std::to_string(w_.lm_head.in_features) + " inputs, got " +
                             std::to_string(hidden_size));
  }
  // The trellis kernels are fp16-only.
  mx::array y = g5_linear(w_.lm_head, mx::astype(hidden, mx::float16));
  mx::array logits = mx::reshape(mx::astype(y, mx::float32), {L, vocab_});
  mx::eval(logits);
  return logits;
}

mx::array Glm5Model::forward_tokens(const std::vector<int>& ids, int start_pos) {
  return apply_lm_head(forward_hidden(ids, start_pos));
}

std::vector<float> Glm5Model::forward_last_logits(const std::vector<int>& ids, int start_pos) {
  mx::array logits = forward_tokens(ids, start_pos);
  const size_t n = (size_t)logits.shape(0);
  const float* p = logits.data<float>();
  return std::vector<float>(p + (n - 1) * (size_t)vocab_, p + n * (size_t)vocab_);
}

} // namespace exl3
