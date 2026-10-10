#pragma once
// DSpark draft checkpoint (openbmb/MiniCPM5-2B-DSpark): a 5-layer Qwen3-style
// draft trained for exact pairing with MiniCPM5-2B. See docs/dspark.md for the
// full specification and the reference implementation (SGLang).
//
// This header covers the checkpoint structure and its loader; the block forward
// and the target-hidden-tap KV wiring are the remaining port work.

#include <mlx/mlx.h>

#include <string>
#include <vector>

namespace exl3 {

namespace mx = mlx::core;

struct DsparkLayer {
  mx::array q_w = mx::array(0.0f), k_w = mx::array(0.0f); // (in, out) f16
  mx::array v_w = mx::array(0.0f), o_w = mx::array(0.0f);
  mx::array gate_w = mx::array(0.0f), up_w = mx::array(0.0f), down_w = mx::array(0.0f);
  mx::array input_norm = mx::array(0.0f), post_norm = mx::array(0.0f);
  mx::array q_norm = mx::array(0.0f), k_norm = mx::array(0.0f);
};

struct DsparkCheckpoint {
  // config.json (subset)
  int hidden_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  int intermediate_size = 0;
  int vocab_size = 0;
  int block_size = 0;                  // draft tokens per forward pass
  int mask_token_id = -1;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10000.0f;
  int64_t markov_rank = 0;
  bool confidence_head_with_markov = false;
  std::vector<int> target_layer_ids;

  // weights (bf16 checkpoints are converted to f16 on load)
  mx::array fc = mx::array(0.0f);           // (5 * target_hidden, hidden)
  mx::array hidden_norm = mx::array(0.0f);  // (hidden)
  mx::array final_norm = mx::array(0.0f);   // (hidden)
  std::vector<DsparkLayer> layers;
  mx::array markov_w1 = mx::array(0.0f);    // (vocab, markov_rank)
  mx::array markov_w2 = mx::array(0.0f);    // (vocab, markov_rank) transposed on load
  mx::array conf_w = mx::array(0.0f), conf_b = mx::array(0.0f);

  size_t bytes = 0;

  // Loads config.json + model.safetensors from a directory (or a .safetensors
  // file). Missing optional tensors (confidence head) are tolerated.
  bool load(const std::string& path, std::string& err);
};

} // namespace exl3
