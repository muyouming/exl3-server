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

  // Reuse the longest common prefix already in the KV cache.
  size_t common = 0;
  while (common < prefix_.size() && common < prompt.size() && prefix_[common] == prompt[common])
    common++;
  int restart = (int)common;
  if (restart >= (int)prompt.size()) restart = (int)prompt.size() - 1; // need logits for the last token
  if (restart != cache_.offset) cache_.trim(restart);
  prefix_.resize((size_t)restart);

  std::vector<int> pending_tokens(prompt.begin() + restart, prompt.end());
  int64_t t0 = now_ms();
  std::vector<float> logits = model_.forward(pending_tokens, cache_);
  res.prefill_seconds = (double)(now_ms() - t0) / 1000.0;
  prefix_.insert(prefix_.end(), pending_tokens.begin(), pending_tokens.end());

  const auto& cfg = model_.config();
  std::vector<int> context = prompt;

  ThinkSplitter think;
  if (opt.start_in_think) think.start_in_think();
  Tokenizer::Stream stream(&tok_);
  auto feed = [&](const std::string& s) {
    if (s.empty()) return;
    if (opt.split_think) think.feed(s, on_delta);
    else if (on_delta) on_delta(s, std::string());
  };
  std::string hold;            // text held back for stop-string matching
  std::string full;
  size_t max_stop = 0;
  for (auto& s : opt.stop) max_stop = std::max(max_stop, s.size());
  const size_t holdback = max_stop > 0 ? max_stop - 1 : 0;

  uint64_t rng = opt.sample.seed;
  if (rng == 0) {
    std::random_device rd;
    rng = ((uint64_t)rd() << 32) ^ (uint64_t)rd() ^ (uint64_t)now_ms();
    if (rng == 0) rng = 0x9E3779B97F4A7C15ull;
  }

  bool stopped = false;
  const int64_t d0 = now_ms();
  for (int step = 0; step < opt.max_tokens; ++step) {
    if (cancelled && cancelled()) {
      res.finish_reason = "cancelled";
      stopped = true;
      break;
    }
    int tok = sample_token(logits, opt.sample, context, rng);
    context.push_back(tok);
    prefix_.push_back(tok);
    res.completion_tokens++;

    if (!opt.ignore_eos && std::find(cfg.eos_token_ids.begin(), cfg.eos_token_ids.end(), tok) !=
                               cfg.eos_token_ids.end()) {
      res.finish_reason = "stop";
      stopped = true;
      break;
    }

    std::string delta = stream.push(tok);
    if (!delta.empty()) {
      hold += delta;
      size_t stop_at = std::string::npos;
      for (auto& s : opt.stop) {
        if (s.empty()) continue;
        size_t k = hold.find(s);
        if (k != std::string::npos) stop_at = std::min(stop_at, k);
      }
      if (stop_at != std::string::npos) {
        std::string emit = hold.substr(0, stop_at);
        feed(emit);
        full += emit;
        hold.clear();
        res.finish_reason = "stop";
        stopped = true;
        break;
      }
      if (hold.size() > holdback) {
        std::string emit = hold.substr(0, hold.size() - holdback);
        feed(emit);
        full += emit;
        hold.erase(0, hold.size() - holdback);
      }
    }
    if (step + 1 >= opt.max_tokens) res.finish_reason = "length";

    // Advance the model for the next step.
    logits = model_.forward({tok}, cache_);
    if (logits.empty()) {
      res.finish_reason = "stop";
      break;
    }
  }
  if (!stopped && res.finish_reason.empty()) res.finish_reason = "length";

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
