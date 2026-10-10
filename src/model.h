#pragma once
// MiniCPM5-2B style Llama-family model: config, EXL3 weight loading into MLX
// float16 arrays, forward pass with a KV cache, and sampling.

#include <mlx/mlx.h>

#include "trellis_gemm.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace exl3 {

namespace mx = mlx::core;

enum class Arch { Llama, Glm5Next };

class Glm5Model;

struct ModelConfig {
  Arch arch = Arch::Llama;
  std::string tensor_prefix = "model."; // "model.language_model." for GLM-5 VL checkpoints
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

  // --- GLM-5-Next / GLM-5.3-Flash ---
  struct LayerPlan {
    bool linear_attention = false; // KDA recurrent layer
    bool sparse_attention = false; // MLA + DSA indexer
    bool moe = false;
  };
  std::vector<LayerPlan> layer_plan;
  int first_k_dense_replace = 0;
  int n_routed_experts = 0;
  int num_experts_per_tok = 0;
  int moe_intermediate_size = 0;
  int n_shared_experts = 0;
  float routed_scaling_factor = 1.0f;
  float swiglu_limit = 0.0f;
  bool norm_topk_prob = false;
  int q_lora_rank = 0;
  int kv_lora_rank = 0;
  int qk_head_dim = 0;   // MLA (nope-only: no rope dims in this checkpoint)
  int v_head_dim = 0;
  int index_n_heads = 0; // DSA lightning indexer
  int index_head_dim = 0;
  int index_topk = 0;
  int index_kpool = 0;
  bool index_kpool_compress = false;
  bool index_kpool_always_select_tail = false;
  bool mhc = false;      // hyper-connections
  int hc_mult = 0;
  int hc_sinkhorn_iters = 0;
  float hc_eps = 1e-6f;  // hc_eps (reference default 1e-6)
  int n_group = 1;       // MoE routing groups
  int topk_group = 1;    // groups kept by the router
  // text_config.linear_attn_config.gate_lower_bound: non-zero selects fla's
  // USE_LOWER_BOUND forget gate (this checkpoint: -5.0).
  float linear_gate_lower_bound = 0.0f;
  int num_nextn_predict_layers = 0;

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
  Model();
  ~Model();
  // Load config + safetensors from a model directory (or a .safetensors file).
  bool load(const std::string& path, std::string& err, int threads = 0,
            bool fp16_weights = false);
  const ModelConfig& config() const { return cfg_; }
  size_t weight_bytes() const { return weight_bytes_; }
  double load_seconds() const { return load_seconds_; }
  // GLM-5-Next only (0 on the Llama path): the loader's tensor inventory.
  int64_t used_tensors() const;
  int64_t total_tensors() const;

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
  void reset_cache(KVCache& cache) const;

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
  // GLM-5-Next forward: `all_logits` runs the lm_head on every position (needed
  // only to verify speculative proposals), otherwise just on the last one.
  ForwardResult forward_glm5(const std::vector<int>& tokens, KVCache& cache, bool all_logits);
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
  // GLM-5-Next path (cfg_.arch == Arch::Glm5Next): all weights and the
  // incremental state live here; `layers_`/`embed_`/`lm_head_` stay empty.
  std::unique_ptr<Glm5Model> glm5_;
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
