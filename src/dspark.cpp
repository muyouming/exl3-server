#include "dspark.h"

#include <cstdio>
#include <filesystem>
#include <map>

#include "fp16.h"
#include "nlohmann/json.hpp"
#include "safetensors.h"
#include "util.h"

namespace exl3 {

namespace {

std::string join(const std::string& a, const std::string& b) { return a + "." + b; }

// Load one tensor as f16 in (in, out) order when `transpose` is set (nn.Linear
// stores (out, in) in PyTorch-style checkpoints).
mx::array load_tensor(const SafetensorsFile& st, const std::string& name, bool transpose,
                      std::string& err, bool required = true) {
  auto it = st.tensors.find(name);
  if (it == st.tensors.end()) {
    if (required) err = "missing tensor " + name;
    return mx::array(0.0f);
  }
  const STTensor& info = it->second;
  int64_t numel = 1;
  for (auto d : info.shape) numel *= d;
  std::vector<uint16_t> buf((size_t)numel);
  if (info.dtype == DT::F16 || info.dtype == DT::BF16) {
    const uint8_t* p = st.map_base + st.header_len + info.offset;
    for (int64_t i = 0; i < numel; ++i) {
      uint16_t raw = ((const uint16_t*)p)[i];
      buf[(size_t)i] = info.dtype == DT::F16 ? raw : fp16_bits_from_bf16(raw);
    }
  } else if (info.dtype == DT::F32) {
    const float* p = (const float*)(st.map_base + st.header_len + info.offset);
    for (int64_t i = 0; i < numel; ++i) buf[(size_t)i] = fp16::from_double_rne(p[i]).to_bits();
  } else {
    err = "unsupported dtype for " + name;
    return mx::array(0.0f);
  }
  mx::Shape shape;
  for (auto d : info.shape) shape.push_back((int)d);
  if (transpose && shape.size() == 2) {
    const int64_t out_f = shape[0], in_f = shape[1];
    std::vector<uint16_t> t(buf.size());
    for (int64_t o = 0; o < out_f; ++o) {
      for (int64_t i = 0; i < in_f; ++i) t[(size_t)(i * out_f + o)] = buf[(size_t)(o * in_f + i)];
    }
    buf.swap(t);
    shape = mx::Shape{(int)in_f, (int)out_f};
  }
  auto* holder = new std::vector<uint16_t>(std::move(buf));
  return mx::array((void*)holder->data(), shape, mx::float16, [holder](void*) { delete holder; });
}

} // namespace

bool DsparkCheckpoint::load(const std::string& path, std::string& err) {
  namespace fs = std::filesystem;
  const std::string dir = fs::is_directory(path) ? path : fs::path(path).parent_path().string();
  const std::string weights = fs::is_directory(path) ? dir + "/model.safetensors" : path;

  nlohmann::json cfg;
  if (!parse_json_file(dir + "/config.json", cfg, err)) return false;
  try {
    hidden_size = cfg.at("hidden_size").get<int>();
    num_layers = cfg.at("num_hidden_layers").get<int>();
    num_heads = cfg.at("num_attention_heads").get<int>();
    num_kv_heads = cfg.at("num_key_value_heads").get<int>();
    head_dim = cfg.at("head_dim").get<int>();
    intermediate_size = cfg.at("intermediate_size").get<int>();
    vocab_size = cfg.at("vocab_size").get<int>();
    block_size = cfg.value("block_size", 7);
    mask_token_id = cfg.value("mask_token_id", -1);
    rms_norm_eps = cfg.value("rms_norm_eps", 1e-6f);
    markov_rank = cfg.value("markov_rank", 0);
    confidence_head_with_markov = cfg.value("confidence_head_with_markov", false);
    if (cfg.contains("rope_parameters")) {
      rope_theta = cfg["rope_parameters"].value("rope_theta", 10000.0f);
    }
    if (cfg.contains("target_layer_ids")) {
      for (auto& v : cfg["target_layer_ids"]) target_layer_ids.push_back(v.get<int>());
    }
  } catch (const std::exception& e) {
    err = std::string("config.json: ") + e.what();
    return false;
  }
  if ((int)target_layer_ids.size() != 5) {
    err = "DSpark: expected 5 target_layer_ids, got " + std::to_string(target_layer_ids.size());
    return false;
  }

  SafetensorsFile st;
  if (!SafetensorsFile::load(weights, st, err)) return false;

  auto take = [&](const std::string& name, bool transpose, bool required = true) {
    return load_tensor(st, name, transpose, err, required);
  };
  bytes += st.map_size; // rough: the checkpoint file is the memory we own afterwards

  fc = take("fc.weight", true);
  hidden_norm = take("hidden_norm.weight", false);
  final_norm = take("norm.weight", false);
  markov_w1 = take("markov_head.markov_w1.weight", false);
  markov_w2 = take("markov_head.markov_w2.weight", true);
  conf_w = take("confidence_head.proj.weight", false, false);
  conf_b = take("confidence_head.proj.bias", false, false);

  layers.resize((size_t)num_layers);
  for (int i = 0; i < num_layers; ++i) {
    const std::string p = "layers." + std::to_string(i);
    DsparkLayer& l = layers[(size_t)i];
    l.q_w = take(join(p, "self_attn.q_proj.weight"), true);
    l.k_w = take(join(p, "self_attn.k_proj.weight"), true);
    l.v_w = take(join(p, "self_attn.v_proj.weight"), true);
    l.o_w = take(join(p, "self_attn.o_proj.weight"), true);
    l.gate_w = take(join(p, "mlp.gate_proj.weight"), true);
    l.up_w = take(join(p, "mlp.up_proj.weight"), true);
    l.down_w = take(join(p, "mlp.down_proj.weight"), true);
    l.input_norm = take(join(p, "input_layernorm.weight"), false);
    l.post_norm = take(join(p, "post_attention_layernorm.weight"), false);
    l.q_norm = take(join(p, "self_attn.q_norm.weight"), false);
    l.k_norm = take(join(p, "self_attn.k_norm.weight"), false);
    if (!err.empty()) return false;
  }
  st.unmap();
  if (!err.empty()) return false;

  std::fprintf(stderr,
               "dspark: loaded draft: %d layers, hidden %d, block %d, markov rank %lld, "
               "taps [%d,%d,%d,%d,%d]\n",
               num_layers, hidden_size, block_size, (long long)markov_rank,
               target_layer_ids[0], target_layer_ids[1], target_layer_ids[2], target_layer_ids[3],
               target_layer_ids[4]);
  return true;
}

} // namespace exl3
