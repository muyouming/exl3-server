#include "glm5_forward.h"

#include <cstdio>
#include <initializer_list>
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
  kda_rewind_.assign((size_t)cfg_.num_layers, KdaRewind{});
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
  mtp_ = Glm5MtpState{};
  for (auto& r : kda_rewind_) {
    r.captured = false;
    r.conv_in = mx::array(0.0f);
    r.prefix_states.clear();
  }
  last_hidden_ = mx::array(0.0f);
  batch_hidden_ = mx::array(0.0f);
  batch_start_ = 0;
  batch_rows_ = 0;
  batch_captured_ = false;
  pos_ = 0;
}

mx::array Glm5Model::forward_hidden(const std::vector<int>& ids, int start_pos, bool drafting) {
  const int L = (int)ids.size();
  if (L <= 0) throw std::runtime_error("Glm5Model::forward_hidden: empty token list");
  if (start_pos != pos_) {
    if (start_pos != 0) {
      throw std::runtime_error("Glm5Model: start_pos " + std::to_string(start_pos) +
                               " does not continue the state at position " + std::to_string(pos_) +
                               " (the KDA state cannot be rewound; use rewind() inside the last "
                               "chunk)");
    }
    reset(); // fresh sequence
  }
  // The chunk's KDA prefix states are only kept when the caller may rewind it:
  // a prefill (start_pos 0) never is, and a chunk wider than kMaxRewindRows
  // costs more state than the speculative loop can justify.
  const bool capture = drafting && start_pos > 0 && L <= kMaxRewindRows;

  mx::array ids_a(ids.data(), mx::Shape{1, L}, mx::int32);
  mx::array emb = mx::take(w_.embed, ids_a, 0);        // (1, L, hidden) f16
  mx::array streams = g5_hc_expand(emb, hc_.mult);     // (1, L, N, hidden)

  // Bench instrumentation: when a profile is installed, every stage boundary is
  // an eval() and the elapsed wall time is charged to that stage.
  const bool prof = profile_ != nullptr;
  int64_t t_stage = prof ? now_ms() : 0;
  const int64_t t_begin = t_stage;
  auto prof_stage = [&](double& acc, std::initializer_list<mx::array> outs) {
    if (!prof) return;
    if (outs.size() != 0) mx::eval(std::vector<mx::array>(outs));
    acc += (double)(now_ms() - t_stage);
    t_stage = now_ms();
  };

  for (int i = 0; i < cfg_.num_layers; ++i) {
    const G5Layer& l = w_.layers[(size_t)i];
    LayerState& st = states_[(size_t)i];

    // ---- attention sub-layer (reference 1312-1337) -------------------------
    mx::array residual = streams;
    G5HcActivation a =
        g5_hc_prepare(streams, l.hc_attn_base, l.hc_attn_fn, l.hc_attn_scale, hc_);
    prof_stage(profile_->hc_prepare_ms, {a.collapsed, a.post, a.comb});
    mx::array h = rms_norm(a.collapsed, l.input_norm, cfg_.rms_norm_eps);
    prof_stage(profile_->rms_norm_ms, {h});
    mx::array out = mx::array(0.0f);
    if (l.plan.linear_attention) {
      out = kda_layer(l, cfg_, h, st.kda, capture ? &kda_rewind_[(size_t)i] : nullptr);
      prof_stage(profile_->kda_ms, {out});
    } else {
      // The indexer shares q_resid with the MLA (reference 1185-1188, 1221).
      mx::array q_resid = mla_q_resid(l, h);
      mx::array topk = indexer_topk(l, cfg_, h, q_resid, st.idx, start_pos);
      prof_stage(profile_->indexer_ms, {topk});
      out = mla_layer(l, cfg_, h, st.mla, start_pos, topk, q_resid);
      prof_stage(profile_->mla_ms, {out});
    }
    streams = g5_hc_mix(out, a.post, a.comb, residual);
    prof_stage(profile_->hc_mix_ms, {streams});

    // ---- MLP sub-layer (reference 1339-1346) -------------------------------
    residual = streams;
    G5HcActivation f = g5_hc_prepare(streams, l.hc_ffn_base, l.hc_ffn_fn, l.hc_ffn_scale, hc_);
    prof_stage(profile_->hc_prepare_ms, {f.collapsed, f.post, f.comb});
    h = rms_norm(f.collapsed, l.post_norm, cfg_.rms_norm_eps);
    prof_stage(profile_->rms_norm_ms, {h});
    out = l.moe ? moe_layer(l, cfg_, h) : dense_mlp(l, cfg_, h);
    prof_stage(l.moe ? profile_->moe_ms : profile_->dense_ms, {out});
    streams = g5_hc_mix(out, f.post, f.comb, residual);
    prof_stage(profile_->hc_mix_ms, {streams});
  }

  // norm(hc_head(streams)): reference 1508.
  mx::array hidden = rms_norm(g5_hc_head(streams), w_.final_norm, cfg_.rms_norm_eps);
  prof_stage(profile_->head_ms, {hidden});

  // The MTP draft rows of this chunk (row j for every token of the chunk whose
  // row is missing), then the bookkeeping a rewind needs. A compact copy of the
  // last hidden keeps the whole chunk's hidden array out of the live set.
  if (drafting && mtp_drafting_ && w_.mtp.present) mtp_extend(hidden, ids, start_pos);
  const int hs = (int)cfg_.hidden_size;
  last_hidden_ = mx::contiguous(
      mx::slice(hidden, {0, L - 1, 0}, {1, L, hs})); // (1, 1, hidden)
  batch_start_ = start_pos;
  batch_rows_ = L;
  batch_captured_ = capture;
  batch_hidden_ = capture ? hidden : mx::array(0.0f);
  mx::eval(last_hidden_);
  if (prof) {
    profile_->total_ms += (double)(now_ms() - t_begin);
    profile_->steps += 1;
  }

  pos_ = start_pos + L;
  return hidden;
}

void Glm5Model::mtp_extend(const mx::array& hidden, const std::vector<int>& ids, int start_pos) {
  const int L = (int)ids.size();
  const int hs = (int)cfg_.hidden_size;
  // Rows the chunk can build: row j needs the token at position j and the hidden
  // at j-1. The first row of a continuation chunk (j == start_pos) takes the
  // hidden stored by the previous chunk; the rest take this chunk's own.
  const int j0 = std::max(1, mtp_.rows + 1);
  const int j1 = start_pos + L - 1;
  if (j0 > j1) return;
  if (j0 < start_pos) {
    throw std::runtime_error("glm5 MTP: draft rows " + std::to_string(j0) + ".." +
                             std::to_string(start_pos - 1) + " are missing at position " +
                             std::to_string(start_pos));
  }
  const int R = j1 - j0 + 1;
  std::vector<int> toks((size_t)R);
  std::vector<mx::array> prev;
  prev.reserve((size_t)R);
  for (int r = 0; r < R; ++r) {
    const int j = j0 + r;
    toks[(size_t)r] = ids[(size_t)(j - start_pos)];
    prev.push_back(j == start_pos
                       ? last_hidden_
                       : mx::slice(hidden, {0, j - 1 - start_pos, 0},
                                   {1, j - start_pos, hs}));
  }
  mx::array tok_arr(toks.data(), mx::Shape{1, R}, mx::int32);
  // (R >= 1: j0 <= j1 above.)
  mx::array prev_arr = R == 1 ? prev[0] : mx::concatenate(prev, 1);
  mtp_block_rows(w_.mtp, cfg_, w_.embed, tok_arr, prev_arr, mtp_, j0);
}

mx::array Glm5Model::mtp_row(int token, const mx::array& prev, int pos, Glm5MtpState& st) {
  mx::array tok(token);
  tok = mx::reshape(mx::astype(tok, mx::int32), {1, 1});
  mx::array out = mtp_block_rows(w_.mtp, cfg_, w_.embed, tok, prev, st, pos);
  mx::eval(out);
  return out;
}

std::vector<float> Glm5Model::mtp_logits(const mx::array& row) {
  if (row.size() <= 1) throw std::runtime_error("glm5 MTP: empty draft row");
  mx::array logits = apply_lm_head(row); // (1, vocab) f32
  const float* p = logits.data<float>();
  return std::vector<float>(p, p + vocab_);
}

void Glm5Model::rewind(int pos) {
  if (pos == pos_) return;
  if (pos <= 0) {
    reset();
    return;
  }
  const int m = pos - batch_start_;
  if (!batch_captured_ || m <= 0 || m > batch_rows_) {
    throw std::runtime_error(
        "glm5: cannot rewind from position " + std::to_string(pos_) + " to " + std::to_string(pos) +
        " (outside the last chunk" +
        (batch_captured_ ? "" : "; its width exceeds Glm5Model::kMaxRewindRows") + ")");
  }
  if (m == batch_rows_) return; // nothing was drafted past the chunk
  const int hs = (int)cfg_.hidden_size;
  for (int i = 0; i < cfg_.num_layers; ++i) {
    LayerState& st = states_[(size_t)i];
    if (w_.layers[(size_t)i].plan.linear_attention) {
      kda_rewind(st.kda, kda_rewind_[(size_t)i], m);
    } else {
      st.mla.trim(batch_start_ + m);
      st.idx.trim(batch_start_ + m);
    }
  }
  if (w_.mtp.present) mtp_trim(batch_start_ + m - 1);
  last_hidden_ = mx::contiguous(mx::slice(batch_hidden_, {0, m - 1, 0}, {1, m, hs}));
  mx::eval(last_hidden_);
  batch_rows_ = 0;
  batch_captured_ = false;
  pos_ = pos;
  // Bookkeeping invariants: the rewound caches must end up holding exactly the
  // committed tokens (a trunk layer's latent and indexer caches are indexed by
  // position, and the KDA window by how many tokens the layer has seen).
  for (int i = 0; i < cfg_.num_layers; ++i) {
    const LayerState& st = states_[(size_t)i];
    if (w_.layers[(size_t)i].plan.linear_attention) {
      const int w = (int)st.kda.conv_window.shape(1);
      const mx::array& cw = w_.layers[(size_t)i].conv1d_w;
      const int want = std::min((int)cw.shape(cw.ndim() - 1) - 1, pos_);
      if (st.kda.conv_window.size() > 1 && w != want) {
        throw std::runtime_error("glm5 rewind: layer " + std::to_string(i) + " conv window " +
                                 std::to_string(w) + " rows, want " + std::to_string(want));
      }
    } else if (st.mla.len != pos_ || st.idx.len != pos_) {
      throw std::runtime_error("glm5 rewind: layer " + std::to_string(i) + " caches hold " +
                               std::to_string(st.mla.len) + "/" + std::to_string(st.idx.len) +
                               " rows, want " + std::to_string(pos_));
    }
  }
  if (w_.mtp.present && mtp_.rows != pos_ - 1) {
    throw std::runtime_error("glm5 rewind: draft rows " + std::to_string(mtp_.rows) + ", want " +
                             std::to_string(pos_ - 1));
  }
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
