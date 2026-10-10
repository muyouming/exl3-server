#include "engine.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

#include "glm5_forward.h"
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

// The emission half of one request: token -> text, <think> routing, stop
// strings, and the running counters in GenerateResult. Extracted from
// Engine::generate so the serial loop and the batch scheduler share exactly one
// implementation of it (the stop-string holdback is subtle enough that a second
// copy would be a bug farm).
class Emitter {
 public:
  Emitter(const Tokenizer& tok, const std::vector<int>& eos_ids, const GenerateOptions& opt,
          GenerateResult& res)
      : eos_(eos_ids), opt_(opt), res_(res), stream_(&tok) {
    if (opt_.start_in_think) think_.start_in_think();
    size_t max_stop = 0;
    for (auto& st : opt_.stop) max_stop = std::max(max_stop, st.size());
    holdback_ = max_stop > 0 ? max_stop - 1 : 0;
  }

  // Appends the token, streams its text and reports whether generation must
  // stop (EOS or a stop string was hit).
  bool emit(int tok, const DeltaFn& on_delta) {
    res_.completion_tokens++;
    res_.tokens.push_back(tok);
    if (!opt_.ignore_eos && std::find(eos_.begin(), eos_.end(), tok) != eos_.end()) {
      res_.finish_reason = "stop";
      return false;
    }
    std::string delta = stream_.push(tok);
    if (!delta.empty()) {
      hold_ += delta;
      size_t stop_at = std::string::npos;
      for (auto& st : opt_.stop) {
        if (st.empty()) continue;
        size_t k = hold_.find(st);
        if (k != std::string::npos) stop_at = std::min(stop_at, k);
      }
      if (stop_at != std::string::npos) {
        std::string out = hold_.substr(0, stop_at);
        feed(out, on_delta);
        full_ += out;
        hold_.clear();
        res_.finish_reason = "stop";
        return false;
      }
      if (hold_.size() > holdback_) {
        std::string out = hold_.substr(0, hold_.size() - holdback_);
        feed(out, on_delta);
        full_ += out;
        hold_.erase(0, hold_.size() - holdback_);
      }
    }
    return true;
  }

  // Flushes the held-back text, closes the thinking stream and files the text
  // into the result. Call once, after the last emit().
  void finish(const DeltaFn& on_delta) {
    if (!hold_.empty()) {
      feed(hold_, on_delta);
      full_ += hold_;
      hold_.clear();
    }
    if (opt_.split_think) think_.finish(on_delta);
    res_.text = full_;
    if (opt_.split_think) {
      res_.content = think_.content;
      res_.reasoning = think_.reasoning;
    } else {
      res_.content = full_;
      res_.reasoning.clear();
    }
  }

 private:
  void feed(const std::string& s, const DeltaFn& on_delta) {
    if (s.empty()) return;
    if (opt_.split_think) think_.feed(s, on_delta);
    else if (on_delta) on_delta(s, std::string());
  }

  std::vector<int> eos_;
  GenerateOptions opt_;
  GenerateResult& res_;
  ThinkSplitter think_;
  Tokenizer::Stream stream_;
  std::string hold_, full_;
  size_t holdback_ = 0;
};

} // namespace

// ---------------------------------------------------------------------------
// N-gram draft
// ---------------------------------------------------------------------------

std::vector<int> NgramDrafter::draft(const std::vector<int>& context, int /*cur*/, int k,
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
  model_.reset_cache(cache_);
  return true;
}

void Engine::reset() {
  std::lock_guard<std::mutex> lk(mu_);
  model_.reset_cache(cache_);
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

  bool speculative = drafter_ && opt.draft_tokens > 1 && opt.sample.temperature <= 0.0f;
  // Some architectures can only undo the state of a bounded verify batch (the
  // GLM-5-Next recurrent state is rewound inside the chunk it was captured for).
  const int spec_row_cap = model_.max_speculative_rows();
  if (speculative && spec_row_cap > 0 && opt.draft_tokens > spec_row_cap) {
    std::fprintf(stderr,
                 "engine: speculative decoding needs %d rows per round but the model can only "
                 "rewind %d; decoding serially\n",
                 opt.draft_tokens, spec_row_cap);
    speculative = false;
  }

  // Reuse the longest common prefix already in the KV cache. GLM-5-Next keeps a
  // recurrent state (KDA delta matrices + a compressed KV/indexer cache) that
  // cannot be rewound to an arbitrary position, so it always restarts at 0 -
  // Model::forward_glm5 then resets the state and re-prefills.
  size_t common = 0;
  if (model_.config().arch != Arch::Glm5Next) {
    while (common < prefix_.size() && common < prompt.size() && prefix_[common] == prompt[common])
      common++;
  }
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

  // ---- emission machinery (one implementation, shared with the batch path) --
  Emitter em(tok_, cfg.eos_token_ids, opt, res);
  auto emit = [&](int tok) -> bool { return em.emit(tok, on_delta); };

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
        std::vector<int> props = drafter_->draft(history, cur, opt.draft_tokens - 1, taps);
        std::vector<int> batch;
        batch.reserve(props.size() + 1);
        batch.push_back(cur);
        batch.insert(batch.end(), props.begin(), props.end());
        const int offset_before = cache_.offset;
        Model::ForwardResult vr = model_.forward_multi(batch, cache_, true);
        // greedy[i] predicts the token after (cur, props[0..i-1]), so props[i]
        // is accepted exactly when it equals greedy[i].
        int accept = 0;
        while (accept < (int)props.size() && props[(size_t)accept] == vr.greedy[(size_t)accept])
          accept++;
        res.spec_rounds++;
        res.spec_proposals += (int64_t)props.size();
        res.spec_accepted += accept;
        // Emit at most what the token budget allows, and keep the KV cache in
        // step with exactly what was emitted.
        const int budget = opt.max_tokens - (int)res.completion_tokens;
        if (budget <= 0) { res.finish_reason = "length"; break; }
        const int wanted = accept + 1; // accepted proposals plus the target's own token
        const int n = std::min(wanted, budget);
        // The cache must keep `cur` plus the accepted proposals; the last
        // emitted token always stays out of the cache (it is fed next round).
        const int keep = offset_before + std::min(1 + accept, n);
        if (keep < cache_.offset) {
          cache_.trim(keep);
          // The model's own state (GLM-5-Next: the KDA recurrence, the caches
          // and the MTP draft rows) still covers the rejected rows.
          model_.rewind_state(keep);
        }
        bool stop_now = false;
        for (int j = 0; j < accept && j < n && !stop_now; ++j) {
          history.push_back(props[(size_t)j]);
          if (cache_.offset > (int)prefix_.size()) prefix_.push_back(props[(size_t)j]);
          stop_now = !emit(props[(size_t)j]);
        }
        const int bonus = vr.greedy[(size_t)accept];
        if (!stop_now && n > accept) {
          history.push_back(bonus);
          stop_now = !emit(bonus);
        }
        if (stop_now) { stopped = true; break; }
        if (n < wanted) { res.finish_reason = "length"; break; }
        taps = vr.taps.empty() ? std::vector<float>() : vr.taps[(size_t)accept];
        cur = bonus;
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

  em.finish(on_delta);
  res.decode_seconds = (double)(now_ms() - d0) / 1000.0;
  return res;
}

// ---------------------------------------------------------------------------
// Request-level batching
// ---------------------------------------------------------------------------

struct Engine::Pending {
  std::vector<int> prompt;
  GenerateOptions opt;
  DeltaFn on_delta;
  CancelFn cancelled;
  std::mutex m;
  std::condition_variable cv;
  bool done = false;
  GenerateResult result;
};

void Engine::start_batch(int max_batch) {
  stop_batch();
  if (max_batch < 0) return; // no scheduler: the CLI and the tests call generate()
  if (!model_.glm5()) {
    // Only the GLM-5-Next path keeps per-slot state; the Llama path has one
    // shared KV cache, so there is nothing to batch.
    std::fprintf(stderr, "engine: batching needs a GLM-5-Next checkpoint; staying serial\n");
    return;
  }
  batch_stop_.store(false);
  // 0 is the scheduler's serial mode: one request at a time through generate(),
  // but on this thread rather than on whichever HTTP worker picked the request
  // up (MLX streams are thread-local - see the header).
  batch_max_ = max_batch;
  worker_alive_.store(true);
  worker_ = std::thread([this] { batch_worker(); });
}

void Engine::stop_batch() {
  if (worker_.joinable()) {
    batch_stop_.store(true);
    qcv_.notify_all();
    worker_.join();
    worker_alive_.store(false);
  }
  batch_max_ = 0;
  std::deque<std::shared_ptr<Pending>> left;
  {
    std::lock_guard<std::mutex> lk(qmu_);
    left.swap(queue_);
  }
  for (auto& p : left) {
    std::lock_guard<std::mutex> lk(p->m);
    p->result.finish_reason = "cancelled";
    p->done = true;
    p->cv.notify_all();
  }
}

GenerateResult Engine::submit_batch(const std::vector<int>& prompt, const GenerateOptions& opt,
                                    const DeltaFn& on_delta, const CancelFn& cancelled) {
  // No scheduler (the CLI, the tests, or a checkpoint batching cannot serve):
  // the caller's thread runs the ordinary serial path.
  if (!worker_.joinable()) return generate(prompt, opt, on_delta, cancelled);
  auto p = std::make_shared<Pending>();
  p->prompt = prompt;
  p->opt = opt;
  p->on_delta = on_delta;
  p->cancelled = cancelled;
  p->result.prompt_tokens = (int)prompt.size();
  bool queued = false;
  {
    // The liveness check and the enqueue are one critical section against the
    // scheduler's own exit path (which clears the flag and swaps the queue under
    // the same lock), so a request either lands in a live queue - and is then
    // guaranteed an answer - or is served here, on the caller's thread.
    std::lock_guard<std::mutex> lk(qmu_);
    if (worker_alive_.load()) {
      queue_.push_back(p);
      queued = true;
    }
  }
  if (!queued) {
    std::fprintf(stderr, "engine: batch scheduler is not running; serving request serially\n");
    return generate(prompt, opt, on_delta, cancelled);
  }
  qcv_.notify_all();
  std::unique_lock<std::mutex> lk(p->m);
  p->cv.wait(lk, [&] { return p->done; });
  return p->result;
}

void Engine::batch_worker() {
  try {
    batch_worker_loop();
  } catch (const std::exception& e) {
    // batch_worker_loop answers everything it held before it returns; this is
    // only the last-resort net for a throw from outside its own cycle handling.
    std::fprintf(stderr, "engine: batch worker stopped: %s\n", e.what());
    batch_stop_.store(true);
  }
}

void Engine::batch_worker_loop() {
  struct Req {
    std::shared_ptr<Pending> p;
    int slot = -1;
    std::vector<int> history;
    uint64_t rng = 0;
    int next = -1;              // the token this slot feeds at the next step
    int64_t t_decode0 = 0;      // when this slot's decode phase began
    std::unique_ptr<Emitter> em;
    std::vector<float> logits;  // one row's logits, reused every step
    bool finished = false;
  };
  std::vector<Req> live;
  const int V = model_.glm5()->vocab_size();
  const std::vector<int> eos = config().eos_token_ids;

  auto finish_req = [](Req& r, const char* fallback) {
    r.finished = true;
    if (r.p->result.finish_reason.empty() && fallback) r.p->result.finish_reason = fallback;
  };
  auto publish = [&](Req& r) {
    r.em->finish(r.p->on_delta);
    r.p->result.decode_seconds = (double)(now_ms() - r.t_decode0) / 1000.0;
    if (r.p->result.finish_reason.empty()) r.p->result.finish_reason = "length";
    std::lock_guard<std::mutex> lk(r.p->m);
    r.p->done = true;
    r.p->cv.notify_all();
  };

  // Answer everything the queue still holds (as "cancelled"). When the
  // scheduler is going away rather than failing one cycle, also clear
  // worker_alive_ in the same critical section as the queue swap, so
  // submit_batch() can never enqueue into a scheduler that has already left.
  auto answer_queued = [&](bool stopping) {
    std::deque<std::shared_ptr<Pending>> left;
    {
      std::lock_guard<std::mutex> lk(qmu_);
      if (stopping) worker_alive_.store(false);
      left.swap(queue_);
    }
    for (auto& p : left) {
      std::lock_guard<std::mutex> lk(p->m);
      p->result.finish_reason = "cancelled";
      p->done = true;
      p->cv.notify_all();
    }
  };

  // A failed step (an allocation, a wrong state, a bug) must not strand anyone:
  // every live request is answered, the batch state is dropped, and the queued
  // requests are answered too so their submitters never hang. The scheduler
  // itself stays alive for the next request.
  auto fail_cycle = [&](const char* why) {
    std::fprintf(stderr, "engine: batch cycle failed: %s\n", why);
    for (auto& r : live) {
      finish_req(r, "cancelled");
      publish(r);
    }
    live.clear();
    {
      std::lock_guard<std::mutex> lk(mu_);
      model_.glm5()->batch_reset();
    }
    answer_queued(/*stopping=*/false);
  };

  // The scheduler's own exit path. A request in flight, or queued but not yet
  // admitted, must not be left waiting on a condition variable nobody will
  // notify - the previous version left the queue unanswered, so the next
  // submitter blocked in p->cv.wait() forever once the worker had stopped (this
  // is what made tools/batch_server_bench.sh produce nothing for 25 minutes:
  // one request failed, the scheduler was gone, every later one hung).
  auto shutdown = [&](const char* why) {
    std::fprintf(stderr, "engine: batch scheduler stopped: %s\n", why);
    for (auto& r : live) {
      finish_req(r, "cancelled");
      publish(r);
    }
    live.clear();
    // The live slots' state has to go with the scheduler: a shutdown must leave
    // the model as clean as a finished request does, or the next user of
    // batch_step() sees a batch whose slot count no longer matches its feed
    // (reproduced by this file's own serving-thread gate: without this reset the
    // benchmark's first batch_step throws "expected one token per live slot").
    {
      std::lock_guard<std::mutex> lk(mu_);
      model_.glm5()->batch_reset();
    }
    answer_queued(/*stopping=*/true);
  };

  // The scheduler's serial mode (`--batch 0`): one request at a time through the
  // ordinary generation path, but built and evaluated on THIS thread. An MTP
  // request (its draft-row state is per sequence) and an empty prompt also come
  // here - both used to run on whichever thread called submit_batch().
  auto serve_serial = [&](const std::shared_ptr<Pending>& p) {
    GenerateResult r;
    try {
      r = generate(p->prompt, p->opt, p->on_delta, p->cancelled);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "engine: serial request failed: %s\n", e.what());
      r.finish_reason = "cancelled";
      r.prompt_tokens = (int)p->prompt.size();
    }
    std::lock_guard<std::mutex> lk(p->m);
    p->result = std::move(r);
    p->done = true;
    p->cv.notify_all();
  };

  try {
  while (!batch_stop_.load()) {
    if (batch_max_ <= 0) {
      // Serial mode: take one request and finish it before the next.
      std::shared_ptr<Pending> p;
      {
        std::unique_lock<std::mutex> lk(qmu_);
        qcv_.wait(lk, [&] { return batch_stop_.load() || !queue_.empty(); });
        if (!queue_.empty()) {
          p = queue_.front();
          queue_.pop_front();
        }
      }
      if (!p) continue; // stopped with an empty queue
      serve_serial(p);
      continue;
    }
    try {
    // ---- admit waiting requests (prefill + first token, as the serial path) --
    std::vector<std::shared_ptr<Pending>> incoming;
    {
      std::unique_lock<std::mutex> lk(qmu_);
      if (live.empty()) {
        qcv_.wait(lk, [&] { return batch_stop_.load() || !queue_.empty(); });
        if (batch_stop_.load()) break;
      }
      while (!queue_.empty() && (int)(live.size() + incoming.size()) < batch_max_) {
        incoming.push_back(queue_.front());
        queue_.pop_front();
      }
    }
    for (auto& p : incoming) {
      if (p->opt.draft_tokens > 1 || p->prompt.empty()) {
        // Not batchable: speculative decoding keeps the draft rows (and the
        // rewind bookkeeping) per sequence, and an empty prompt has nothing to
        // prefill. Served alone, on this thread.
        serve_serial(p);
        continue;
      }
      Req r;
      r.p = p;
      r.history = p->prompt;
      r.rng = p->opt.sample.seed;
      if (r.rng == 0) {
        std::random_device rd;
        r.rng = ((uint64_t)rd() << 32) ^ (uint64_t)rd() ^ (uint64_t)now_ms();
        if (r.rng == 0) r.rng = 0x9E3779B97F4A7C15ull;
      }
      r.em = std::make_unique<Emitter>(tok_, eos, p->opt, p->result);
      std::vector<float> lg;
      const int64_t t0 = now_ms();
      {
        std::lock_guard<std::mutex> lk(mu_);
        r.slot = model_.glm5()->batch_join(p->prompt, &lg);
      }
      p->result.prefill_seconds = (double)(now_ms() - t0) / 1000.0;
      r.t_decode0 = now_ms();
      // The prefill's own row is the first token's distribution, exactly as the
      // serial loop takes it.
      int tok = sample_token(lg, p->opt.sample, r.history, r.rng);
      r.history.push_back(tok);
      if (!r.em->emit(tok, p->on_delta)) {
        finish_req(r, "stop");
      } else if (p->result.completion_tokens >= p->opt.max_tokens) {
        finish_req(r, "length");
      } else {
        r.next = tok;
      }
      live.push_back(std::move(r));
    }

    // ---- retire finished slots -------------------------------------------
    for (;;) {
      int idx = -1;
      for (int i = 0; i < (int)live.size(); ++i)
        if (live[(size_t)i].finished) {
          idx = i;
          break;
        }
      if (idx < 0) break;
      Req r = std::move(live[(size_t)idx]);
      live.erase(live.begin() + idx);
      {
        std::lock_guard<std::mutex> lk(mu_);
        model_.glm5()->batch_leave(r.slot);
      }
      for (auto& o : live)
        if (o.slot > r.slot) o.slot -= 1;
      publish(r);
    }
    if (live.empty()) continue;

    // ---- one fused step for every live request ---------------------------
    std::vector<int> feed;
    feed.reserve(live.size());
    for (auto& r : live) feed.push_back(r.next);
    mx::array logits = mx::array(0.0f);
    {
      std::lock_guard<std::mutex> lk(mu_);
      logits = model_.glm5()->batch_step(feed);
    }
    const float* lp = logits.data<float>();
    for (size_t i = 0; i < live.size(); ++i) {
      Req& r = live[i];
      r.logits.assign(lp + i * (size_t)V, lp + (i + 1) * (size_t)V);
      if (r.p->cancelled && r.p->cancelled()) {
        finish_req(r, "cancelled");
        continue;
      }
      int tok = sample_token(r.logits, r.p->opt.sample, r.history, r.rng);
      r.history.push_back(tok);
      if (!r.em->emit(tok, r.p->on_delta)) {
        finish_req(r, "stop");
      } else if (r.p->result.completion_tokens >= r.p->opt.max_tokens) {
        finish_req(r, "length");
      } else {
        r.next = tok;
      }
    }
    } catch (const std::exception& e) {
      fail_cycle(e.what());
    }
  }
  shutdown("the batch scheduler stopped");
  } catch (const std::exception& e) {
    // A throw from the loop's own bookkeeping (not from a step) must leave
    // nobody behind either.
    shutdown(e.what());
  }
}

} // namespace exl3
