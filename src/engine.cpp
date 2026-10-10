#include "engine.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

#include "util.h"

namespace exl3 {

// ---------------------------------------------------------------------------
// <think> routing
// ---------------------------------------------------------------------------

namespace {

constexpr const char* kThinkOpen = "<think>";
constexpr const char* kThinkClose = "</think>";

size_t partial_marker_suffix(const std::string& buf, const char* marker) {
  const size_t mlen = std::strlen(marker);
  const size_t max_check = std::min(buf.size(), mlen - 1);
  for (size_t l = max_check; l >= 1; --l) {
    if (buf.compare(buf.size() - l, l, marker, l) == 0) return l;
  }
  return 0;
}

class ThinkSplitter {
 public:
  std::string content, reasoning;

  void feed(const std::string& s, const DeltaFn& emit) {
    buf_ += s;
    for (;;) {
      if (!in_think_) {
        size_t i = buf_.find(kThinkOpen);
        if (i != std::string::npos) {
          push(buf_.substr(0, i), false, emit);
          buf_.erase(0, i + std::strlen(kThinkOpen));
          in_think_ = true;
          continue;
        }
        size_t hold = partial_marker_suffix(buf_, kThinkOpen);
        push(buf_.substr(0, buf_.size() - hold), false, emit);
        buf_.erase(0, buf_.size() - hold);
        break;
      }
      size_t i = buf_.find(kThinkClose);
      if (i != std::string::npos) {
        push(buf_.substr(0, i), true, emit);
        buf_.erase(0, i + std::strlen(kThinkClose));
        in_think_ = false;
        continue;
      }
      size_t hold = partial_marker_suffix(buf_, kThinkClose);
      push(buf_.substr(0, buf_.size() - hold), true, emit);
      buf_.erase(0, buf_.size() - hold);
      break;
    }
  }

  void start_in_think() { in_think_ = true; }

  void finish(const DeltaFn& emit) {
    if (!buf_.empty()) push(buf_, in_think_, emit);
    buf_.clear();
  }

 private:
  void push(const std::string& s, bool is_reasoning, const DeltaFn& emit) {
    if (s.empty()) return;
    if (is_reasoning) reasoning += s;
    else content += s;
    if (emit) emit(is_reasoning ? std::string() : s, is_reasoning ? s : std::string());
  }

  std::string buf_;
  bool in_think_ = false;
};

} // namespace

// ---------------------------------------------------------------------------
// N-gram draft
// ---------------------------------------------------------------------------

std::vector<int> NgramDrafter::draft(const std::vector<int>& context, int k,
                                     const std::vector<float>& /*taps*/) {
  std::vector<int> out;
  if (k <= 0 || context.size() < (size_t)min_suffix_ + 1) return out;
  const int n = (int)context.size();
  for (int m = std::min(max_suffix_, n - 1); m >= min_suffix_; --m) {
    // find the latest earlier occurrence of context[n-m .. n) that is followed
    // by at least one token
    for (int start = n - m - 1; start >= 0; --start) {
      bool match = true;
      for (int j = 0; j < m; ++j) {
        if (context[(size_t)(start + j)] != context[(size_t)(n - m + j)]) { match = false; break; }
      }
      if (!match) continue;
      const int cont_start = start + m;
      for (int j = cont_start; j < n && (int)out.size() < k; ++j) out.push_back(context[(size_t)j]);
      if (!out.empty()) return out;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

bool Engine::load(const std::string& model_path, std::string& err, int threads, bool fp16_weights) {
  if (!model_.load(model_path, err, threads, fp16_weights)) return false;
  if (!tok_.load(model_path, err)) return false;

  namespace fs = std::filesystem;
  std::string name = fs::path(model_path).filename().string();
  if (name.empty()) name = "exl3-model";
  model_name_ = name;
  cache_.init(model_.config().num_layers);
  return true;
}

void Engine::reset() {
  std::lock_guard<std::mutex> lk(mu_);
  cache_.init(model_.config().num_layers);
  prefix_.clear();
}

GenerateResult Engine::generate(const std::vector<int>& prompt, const GenerateOptions& opt,
                                const DeltaFn& on_delta, const CancelFn& cancelled) {
  std::lock_guard<std::mutex> lk(mu_);
  GenerateResult res;
  res.prompt_tokens = (int)prompt.size();
  if (prompt.empty()) {
    res.finish_reason = "stop";
    return res;
  }

  const bool speculative = drafter_ && opt.draft_tokens > 1 && opt.sample.temperature <= 0.0f;

  // Reuse the longest common prefix already in the KV cache.
  size_t common = 0;
  while (common < prefix_.size() && common < prompt.size() && prefix_[common] == prompt[common])
    common++;
  int restart = (int)common;
  if (restart >= (int)prompt.size()) restart = (int)prompt.size() - 1;
  if (restart != cache_.offset) cache_.trim(restart);
  prefix_.resize((size_t)restart);

  std::vector<int> pending_tokens(prompt.begin() + restart, prompt.end());
  int64_t t0 = now_ms();
  Model::ForwardResult fr = model_.forward_multi(pending_tokens, cache_, speculative);
  res.prefill_seconds = (double)(now_ms() - t0) / 1000.0;
  prefix_.insert(prefix_.end(), pending_tokens.begin(), pending_tokens.end());

  const auto& cfg = model_.config();
  std::vector<int> history = prompt;

  // ---- shared emission machinery -------------------------------------------
  ThinkSplitter think;
  if (opt.start_in_think) think.start_in_think();
  Tokenizer::Stream stream(&tok_);
  auto feed = [&](const std::string& s) {
    if (s.empty()) return;
    if (opt.split_think) think.feed(s, on_delta);
    else if (on_delta) on_delta(s, std::string());
  };
  std::string hold, full;
  size_t max_stop = 0;
  for (auto& st : opt.stop) max_stop = std::max(max_stop, st.size());
  const size_t holdback = max_stop > 0 ? max_stop - 1 : 0;

  // Returns false when generation must stop (EOS or a stop string was hit).
  auto emit = [&](int tok) -> bool {
    res.completion_tokens++;
    res.tokens.push_back(tok);
    if (!opt.ignore_eos &&
        std::find(cfg.eos_token_ids.begin(), cfg.eos_token_ids.end(), tok) !=
            cfg.eos_token_ids.end()) {
      res.finish_reason = "stop";
      return false;
    }
    std::string delta = stream.push(tok);
    if (!delta.empty()) {
      hold += delta;
      size_t stop_at = std::string::npos;
      for (auto& st : opt.stop) {
        if (st.empty()) continue;
        size_t k = hold.find(st);
        if (k != std::string::npos) stop_at = std::min(stop_at, k);
      }
      if (stop_at != std::string::npos) {
        std::string out = hold.substr(0, stop_at);
        feed(out);
        full += out;
        hold.clear();
        res.finish_reason = "stop";
        return false;
      }
      if (hold.size() > holdback) {
        std::string out = hold.substr(0, hold.size() - holdback);
        feed(out);
        full += out;
        hold.erase(0, hold.size() - holdback);
      }
    }
    return true;
  };

  uint64_t rng = opt.sample.seed;
  if (rng == 0) {
    std::random_device rd;
    rng = ((uint64_t)rd() << 32) ^ (uint64_t)rd() ^ (uint64_t)now_ms();
    if (rng == 0) rng = 0x9E3779B97F4A7C15ull;
  }

  const int64_t d0 = now_ms();
  bool stopped = false;

  if (speculative) {
    // Speculative loop: the cache holds everything except the token waiting to
    // be fed (`cur`); the drafter proposes k-1 tokens after it, the target
    // verifies the whole batch in one pass, and the longest matching prefix is
    // committed. Greedy-only, and exactly equivalent to plain greedy decoding.
    int cur = fr.greedy.back();
    std::vector<float> taps = fr.taps.empty() ? std::vector<float>() : fr.taps.back();
    if (!emit(cur)) {
      res.finish_reason = res.finish_reason.empty() ? "stop" : res.finish_reason;
    } else {
      for (;;) {
        if (cancelled && cancelled()) { res.finish_reason = "cancelled"; stopped = true; break; }
        if (res.completion_tokens >= opt.max_tokens) { res.finish_reason = "length"; break; }
        std::vector<int> props = drafter_->draft(history, opt.draft_tokens - 1, taps);
        std::vector<int> batch;
        batch.reserve(props.size() + 1);
        batch.push_back(cur);
        batch.insert(batch.end(), props.begin(), props.end());
        const int offset_before = cache_.offset;
        Model::ForwardResult vr = model_.forward_multi(batch, cache_, true);
        int accept = 0;
        while (accept < (int)props.size() && props[(size_t)accept] == vr.greedy[(size_t)accept + 1])
          accept++;
        // Commit: keep the feed token plus the accepted proposals.
        if (offset_before + 1 + accept < cache_.offset) cache_.trim(offset_before + 1 + accept);
        bool stop_now = false;
        for (int j = 0; j < accept && !stop_now; ++j) {
          history.push_back(props[(size_t)j]);
          if (cache_.offset > (int)prefix_.size()) prefix_.push_back(props[(size_t)j]);
          stop_now = !emit(props[(size_t)j]);
        }
        taps = vr.taps.empty() ? std::vector<float>() : vr.taps[(size_t)accept];
        cur = vr.greedy[(size_t)accept];
        if (stop_now) { stopped = true; break; }
        history.push_back(cur);
        if (!emit(cur)) { stopped = true; break; }
      }
    }
  } else {
    std::vector<float> logits = fr.logits_last;
    for (int step = 0; step < opt.max_tokens; ++step) {
      if (cancelled && cancelled()) { res.finish_reason = "cancelled"; stopped = true; break; }
      int tok = sample_token(logits, opt.sample, history, rng);
      history.push_back(tok);
      prefix_.push_back(tok);
      if (!emit(tok)) { stopped = true; break; }
      if (step + 1 >= opt.max_tokens) res.finish_reason = "length";
      logits = model_.forward({tok}, cache_);
      if (logits.empty()) { res.finish_reason = "stop"; break; }
    }
  }
  (void)stopped;
  if (res.finish_reason.empty()) res.finish_reason = "length";

  if (!hold.empty()) {
    feed(hold);
    full += hold;
  }
  if (opt.split_think) think.finish(on_delta);

  res.decode_seconds = (double)(now_ms() - d0) / 1000.0;
  res.text = full;
  if (opt.split_think) {
    res.content = think.content;
    res.reasoning = think.reasoning;
  } else {
    res.content = full;
    res.reasoning.clear();
  }
  return res;
}

} // namespace exl3
