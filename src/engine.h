#pragma once
// Generation loop shared by the CLI and the HTTP server: prefix-aware KV cache
// reuse, streaming detokenization, stop strings, and <think>/</think> routing.

#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "model.h"
#include "tokenizer.h"

namespace exl3 {

struct GenerateOptions {
  SampleOptions sample;
  int max_tokens = 512;
  std::vector<std::string> stop; // stop strings, matched against decoded text
  bool ignore_eos = false;
  bool split_think = true;      // route <think>...</think> into a separate stream
  bool start_in_think = false;  // the prompt already opened a <think> block
};

struct GenerateResult {
  std::string text;      // full generated text, thinking markers included
  std::string content;   // text outside <think>...</think>
  std::string reasoning; // text inside <think>...</think>
  std::string finish_reason; // "stop" | "length" | "cancelled"
  int prompt_tokens = 0;
  int completion_tokens = 0;
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
};

using DeltaFn = std::function<void(const std::string& content, const std::string& reasoning)>;
using CancelFn = std::function<bool()>;

class Engine {
 public:
  bool load(const std::string& model_path, std::string& err, int threads = 0,
            bool fp16_weights = false);
  const ModelConfig& config() const { return model_.config(); }
  const Tokenizer& tokenizer() const { return tok_; }
  const std::string& model_name() const { return model_name_; }
  void set_model_name(const std::string& n) { model_name_ = n; }

  // One generation at a time (the KV cache is shared state).
  GenerateResult generate(const std::vector<int>& prompt, const GenerateOptions& opt,
                          const DeltaFn& on_delta = nullptr,
                          const CancelFn& cancelled = nullptr);

  void reset();

 private:
  Model model_;
  Tokenizer tok_;
  KVCache cache_;
  std::vector<int> prefix_; // tokens currently resident in the cache
  std::mutex mu_;
  std::string model_name_;
};

} // namespace exl3
