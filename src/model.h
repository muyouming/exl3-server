#pragma once
// MiniCPM5-2B style Llama-family model: config, EXL3 weight loading into MLX
// float16 arrays, forward pass with a KV cache, and sampling.

#include <mlx/mlx.h>

#include "trellis_gemm.h"

#include <cstdint>
#include <string>
#include <vector>

namespace exl3 {

namespace mx = mlx::core;

struct ModelConfig {
  int hidden_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  int intermediate_size = 0;
  int vocab_size = 0;
  int max_position_embeddings = 0;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10000.0f;
  int bos_token_id = -1;
  std::vector<int> eos_token_ids;

  static bool load(const std::string& dir, ModelConfig& out, std::string& err);
};

struct KVCache {
  std::vector<mx::array> keys, values; // per layer (1, n_kv_heads, S, head_dim) f16
  int offset = 0;                      // cached positions
  void init(int layers) {
    keys.assign(layers, mx::array(0.0f));
    values.assign(layers, mx::array(0.0f));
    offset = 0;
  }
  void trim(int pos);
};

class Model {
 public:
  // Load config + safetensors from a model directory (or a .safetensors file).
  bool load(const std::string& path, std::string& err, int threads = 0,
            bool fp16_weights = false);
  const ModelConfig& config() const { return cfg_; }
  size_t weight_bytes() const { return weight_bytes_; }
  double load_seconds() const { return load_seconds_; }

  // Run `tokens` through the model, appending to `cache`, and return the
  // float32 logits of the last position.
  std::vector<float> forward(const std::vector<int>& tokens, KVCache& cache);

  // Hidden-state taps used by speculative drafters (DSpark reads layers
  // 1, 10, 20, 30, 39 of the target).
  static constexpr int kNumTaps = 5;
  static const int kTapLayers[kNumTaps];

  struct ForwardResult {
    std::vector<float> logits_last;      // (vocab) at the last position
    std::vector<int> greedy;             // argmax token at every position
    std::vector<std::vector<float>> taps; // (positions, 5 * hidden) when requested
  };

  // Like forward(), but also returns the greedy token at every position (needed
  // to verify speculative proposals) and, optionally, the per-position hidden
  // taps of kTapLayers.
  ForwardResult forward_multi(const std::vector<int>& tokens, KVCache& cache,
                              bool capture_taps = false);

  // Reset the KV cache (keeps allocations).
  void reset_cache(KVCache& cache) const { cache.init((int)cfg_.num_layers); }

 private:
  struct Linear {
    mx::array w = mx::array(0.0f);    // f16 (in, out), when !quantized
    mx::array bias = mx::array(0.0f); // f16 (out) or empty
    bool has_bias = false;
    bool quantized = false;           // weights stay in the packed trellis
    TrellisWeight tw;                 // valid when quantized
  };
  struct Layer {
    Linear q, k, v, o, gate, up, down;
    mx::array input_norm = mx::array(0.0f);
    mx::array post_norm = mx::array(0.0f);
  };

  mx::array load_linear(const struct SafetensorsFile& st, const std::string& key,
                        std::string& err, int threads);
  bool load_trellis(const struct SafetensorsFile& st, const std::string& key, TrellisWeight& out,
                    std::string& err);
  mx::array load_vector(const struct SafetensorsFile& st, const std::string& name,
                        std::string& err, bool transpose2d = false);

  bool fp16_weights_ = false;
  ModelConfig cfg_;
  std::vector<Layer> layers_;
  mx::array embed_ = mx::array(0.0f);      // f16 (vocab, hidden)
  mx::array final_norm_ = mx::array(0.0f); // f16 (hidden)
  Linear lm_head_;
  size_t weight_bytes_ = 0;
  double load_seconds_ = 0;
};

struct SampleOptions {
  float temperature = 0.0f; // <= 0 => greedy
  float top_p = 1.0f;
  int top_k = 0;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  uint64_t seed = 0x9E3779B97F4A7C15ull;
};

// Apply penalties/filters and pick the next token.
int sample_token(std::vector<float>& logits, const SampleOptions& opt,
                 const std::vector<int>& context, uint64_t& rng_state);

} // namespace exl3
