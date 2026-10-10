#pragma once
// Generation loop shared by the CLI and the HTTP server: prefix-aware KV cache
// reuse, streaming detokenization, stop strings, and <think>/</think> routing.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "model.h"
#include "tokenizer.h"

namespace exl3 {

// Speculative drafters propose tokens that the target model verifies in one
// pass. Greedy-only: proposal acceptance compares against the target's argmax.
class Drafter {
 public:
  virtual ~Drafter() = default;
  // Propose up to `k` tokens that must follow `cur` (the token the engine just
  // emitted and has not fed to the target yet). `context` is the emitted-token
  // history that ends before `cur`; `taps` holds the target's hidden taps for
  // the last processed position (kNumTaps * hidden floats on the Llama/DSpark
  // path) or is empty when the model was not asked to capture them. The MTP
  // drafter ignores both and reads the model's own hidden and draft-row state.
  virtual std::vector<int> draft(const std::vector<int>& context, int cur, int k,
                                 const std::vector<float>& taps) = 0;
};

// Model-free drafter: continues the longest context suffix that occurred
// earlier in the sequence (prompt lookup / n-gram). Lossless with verification.
class NgramDrafter : public Drafter {
 public:
  explicit NgramDrafter(int max_suffix = 12, int min_suffix = 3)
      : max_suffix_(max_suffix), min_suffix_(min_suffix) {}
  std::vector<int> draft(const std::vector<int>& context, int cur, int k,
                         const std::vector<float>& taps) override;

 private:
  int max_suffix_, min_suffix_;
};

struct GenerateOptions {
  SampleOptions sample;
  int max_tokens = 512;
  int draft_tokens = 0; // >0 enables speculative decoding (greedy only)
  std::vector<std::string> stop; // stop strings, matched against decoded text
  bool ignore_eos = false;
  bool split_think = true;      // route <think>...</think> into a separate stream
  bool start_in_think = false;  // the prompt already opened a <think> block
};

struct GenerateResult {
  std::vector<int> tokens; // generated token ids (includes EOS/stop token)
  std::string text;      // full generated text, thinking markers included
  std::string content;   // text outside <think>...</think>
  std::string reasoning; // text inside <think>...</think>
  std::string finish_reason; // "stop" | "length" | "cancelled"
  int prompt_tokens = 0;
  int completion_tokens = 0;
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
  // Speculative decoding accounting (all zero when it did not run): verified
  // rounds, the proposals the drafter made and how many of them the target
  // accepted. acceptance = spec_accepted / spec_proposals, tokens per round =
  // 1 + spec_accepted / spec_rounds.
  int64_t spec_rounds = 0;
  int64_t spec_proposals = 0;
  int64_t spec_accepted = 0;
};

using DeltaFn = std::function<void(const std::string& content, const std::string& reasoning)>;
using CancelFn = std::function<bool()>;

class Engine {
 public:
  Engine() = default;
  // Stops the batch scheduler (a joinable worker thread outliving main would
  // have std::thread call std::terminate).
  ~Engine() { stop_batch(); }

  bool load(const std::string& model_path, std::string& err, int threads = 0,
            bool fp16_weights = false);
  const ModelConfig& config() const { return model_.config(); }
  // The loaded model, for tests/benchmarks that drive forward_multi directly.
  const Model& model() const { return model_; }
  Model& model() { return model_; }
  const Tokenizer& tokenizer() const { return tok_; }
  const std::string& model_name() const { return model_name_; }
  void set_model_name(const std::string& n) { model_name_ = n; }

  // One generation at a time (the KV cache is shared state).
  GenerateResult generate(const std::vector<int>& prompt, const GenerateOptions& opt,
                          const DeltaFn& on_delta = nullptr,
                          const CancelFn& cancelled = nullptr);

  void reset();
  void set_drafter(std::shared_ptr<Drafter> d) { drafter_ = std::move(d); }
  const std::shared_ptr<Drafter>& drafter() const { return drafter_; }

  // ---- request-level batching ---------------------------------------------
  //
  // With a batch scheduler running, concurrent requests share one fused decode
  // step (rows = sequences) instead of queueing behind each other. The
  // scheduler owns a worker thread that pulls requests off a queue, prefills
  // the new ones through the ordinary single-sequence path and then advances
  // every live request by one token per step. `start_batch(0)` disables it.
  //
  // Batching is GLM-5-Next only (the Llama path has one shared KV cache and no
  // per-slot state): for any other architecture submit_batch() is generate().
  void start_batch(int max_batch);
  void stop_batch();
  bool batch_running() const { return batch_max_ > 0; }
  int batch_max() const { return batch_max_; }
  // Submits one request and blocks until it is done. `on_delta` is called from
  // the scheduler thread while this call is blocked (the HTTP handlers rely on
  // that: the sink is only ever written by this thread).
  GenerateResult submit_batch(const std::vector<int>& prompt, const GenerateOptions& opt,
                              const DeltaFn& on_delta = nullptr,
                              const CancelFn& cancelled = nullptr);

 private:
  struct Pending;
  void batch_worker();
  void batch_worker_loop();

  std::shared_ptr<Drafter> drafter_;
  Model model_;
  Tokenizer tok_;
  KVCache cache_;
  std::vector<int> prefix_; // tokens currently resident in the cache
  std::mutex mu_;
  std::string model_name_;

  std::mutex qmu_;
  std::condition_variable qcv_;
  std::deque<std::shared_ptr<Pending>> queue_;
  std::thread worker_;
  std::atomic<bool> batch_stop_{false};
  int batch_max_ = 0;
};

} // namespace exl3
