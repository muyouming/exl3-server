#include "glm5.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unordered_map>

#include "fp16.h"
#include "nlohmann/json.hpp"
#include "safetensors.h"
#include "util.h"

namespace exl3 {

namespace {

namespace fs = std::filesystem;

struct Store {
  std::vector<SafetensorsFile> files;
  std::unordered_map<std::string, int> where;
  int64_t bytes = 0;

  bool open(const std::string& dir, std::string& err) {
    std::vector<std::string> shards;
    if (fs::exists(dir + "/model.safetensors")) {
      shards.push_back(dir + "/model.safetensors");
    } else {
      nlohmann::json j;
      if (!parse_json_file(dir + "/model.safetensors.index.json", j, err)) return false;
      for (auto& [name, file] : j["weight_map"].items()) {
        std::string s = dir + "/" + file.get<std::string>();
        if (std::find(shards.begin(), shards.end(), s) == shards.end()) shards.push_back(s);
      }
      std::sort(shards.begin(), shards.end());
    }
    for (auto& s : shards) {
      SafetensorsFile f;
      if (!SafetensorsFile::load(s, f, err)) return false;
      bytes += f.map_size;
      int idx = (int)files.size();
      for (auto& [name, t] : f.tensors) where.emplace(name, idx);
      files.push_back(std::move(f));
    }
    return true;
  }

  const SafetensorsFile* file_for(const std::string& name) const {
    auto it = where.find(name);
    return it == where.end() ? nullptr : &files[(size_t)it->second];
  }

  bool has(const std::string& name) const { return where.count(name) != 0; }
};

// Loads any tensor as f16 or f32 in (in, out) order for 2-D tensors.
mx::array load_any(const Store& st, const std::string& name, bool transpose, bool as_f32,
                   std::string& err, bool required = true) {
  const SafetensorsFile* f = st.file_for(name);
  if (!f) {
    if (required) err = "missing tensor " + name;
    return mx::array(0.0f);
  }
  const STTensor& info = f->tensor(name);
  int64_t numel = 1;
  for (auto d : info.shape) numel *= d;
  const uint8_t* src = f->map_base + f->header_len + info.offset;
  std::vector<uint16_t> h16;
  std::vector<float> h32;
  auto to_f16 = [&](int64_t i) -> uint16_t {
    switch (info.dtype) {
      case DT::F16: return ((const uint16_t*)src)[i];
      case DT::BF16: return fp16_bits_from_bf16(((const uint16_t*)src)[i]);
      case DT::F32: return fp16::from_double_rne(((const float*)src)[i]).to_bits();
      default: return 0;
    }
  };
  if (as_f32) {
    h32.resize((size_t)numel);
    for (int64_t i = 0; i < numel; ++i) {
      switch (info.dtype) {
        case DT::F16: h32[(size_t)i] = (float)fp16::from_bits(((const uint16_t*)src)[i]).to_double(); break;
        case DT::BF16: h32[(size_t)i] = (float)fp16::from_bits(fp16_bits_from_bf16(((const uint16_t*)src)[i])).to_double(); break;
        case DT::F32: h32[(size_t)i] = ((const float*)src)[i]; break;
        default: break;
      }
    }
  } else {
    h16.resize((size_t)numel);
    for (int64_t i = 0; i < numel; ++i) h16[(size_t)i] = to_f16(i);
  }
  mx::Shape shape;
  for (auto d : info.shape) shape.push_back((int)d);
  if (transpose && shape.size() == 2) {
    const int64_t out_f = shape[0], in_f = shape[1];
    if (as_f32) {
      std::vector<float> t((size_t)numel);
      for (int64_t o = 0; o < out_f; ++o)
        for (int64_t i = 0; i < in_f; ++i) t[(size_t)(i * out_f + o)] = h32[(size_t)(o * in_f + i)];
      h32.swap(t);
    } else {
      std::vector<uint16_t> t((size_t)numel);
      for (int64_t o = 0; o < out_f; ++o)
        for (int64_t i = 0; i < in_f; ++i) t[(size_t)(i * out_f + o)] = h16[(size_t)(o * in_f + i)];
      h16.swap(t);
    }
    shape = mx::Shape{(int)in_f, (int)out_f};
  }
  if (as_f32) {
    auto* holder = new std::vector<float>(std::move(h32));
    return mx::array((void*)holder->data(), shape, mx::float32, [holder](void*) { delete holder; });
  }
  auto* holder = new std::vector<uint16_t>(std::move(h16));
  return mx::array((void*)holder->data(), shape, mx::float16, [holder](void*) { delete holder; });
}

// EXL3 linear (trellis + suh/svh + codebook marker) or a plain .weight tensor.
G5Linear load_linear(const Store& st, const std::string& key, std::string& err) {
  G5Linear lin;
  if (st.has(key + ".trellis")) {
    const SafetensorsFile* f = st.file_for(key + ".trellis");
    const STTensor& info = f->tensor(key + ".trellis");
    if (info.shape.size() != 3) { err = key + ".trellis must be 3-dimensional"; return lin; }
    const uint16_t* src = (const uint16_t*)f->i16_data(key + ".trellis", err);
    if (!src) return lin;
    const int64_t kt = info.shape[0], nt = info.shape[1], words16 = info.shape[2];
    if (words16 % 16 != 0) { err = key + ": trellis width not a multiple of 16"; return lin; }
    lin.quantized = true;
    lin.tw.K = (int)(words16 / 16);
    lin.tw.codebook = st.has(key + ".mul1") ? 2 : (st.has(key + ".mcg") ? 1 : 0);
    lin.in_features = (int)(kt * 16);
    lin.out_features = (int)(nt * 16);
    lin.tw.in_features = lin.in_features;
    lin.tw.out_features = lin.out_features;
    auto* holder = new std::vector<uint32_t>((size_t)(kt * nt * words16 / 2));
    std::memcpy(holder->data(), src, holder->size() * 4);
    lin.tw.words = mx::array((void*)holder->data(),
                             mx::Shape{(int)kt, (int)nt, (int)(words16 / 2)}, mx::uint32,
                             [holder](void*) { delete holder; });
    lin.tw.suh = load_any(st, key + ".suh", false, false, err);
    lin.tw.svh = load_any(st, key + ".svh", false, false, err);
  } else if (st.has(key + ".weight")) {
    lin.w = load_any(st, key + ".weight", true, false, err);
    if (!err.empty()) return lin;
    lin.in_features = (int)lin.w.shape(0);
    lin.out_features = (int)lin.w.shape(1);
    lin.quantized = false;
  } else {
    err = "missing linear " + key;
    return lin;
  }
  if (st.has(key + ".bias")) {
    lin.bias = load_any(st, key + ".bias", false, false, err);
    lin.has_bias = true;
  }
  return lin;
}

} // namespace

bool load_glm5(const std::string& dir, const ModelConfig& cfg, Glm5Weights& out, std::string& err,
               int threads, bool verbose) {
  (void)threads;
  if (cfg.arch != Arch::Glm5Next) {
    err = "load_glm5: config architecture is not glm5_next";
    return false;
  }
  Store st;
  if (!st.open(dir, err)) return false;

  const std::string p = cfg.tensor_prefix; // "model.language_model."
  out.embed = load_any(st, p + "embed_tokens.weight", false, false, err);
  out.final_norm = load_any(st, p + "norm.weight", false, false, err);
  out.lm_head = load_linear(st, "lm_head", err);
  if (!err.empty()) return false;

  out.layers.resize((size_t)cfg.num_layers);
  int64_t tensors_before = 0;
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string lp = p + "layers." + std::to_string(i) + ".";
    G5Layer& l = out.layers[(size_t)i];
    l.plan = cfg.layer_plan[(size_t)i];
    l.input_norm = load_any(st, lp + "input_layernorm.weight", false, false, err);
    l.post_norm = load_any(st, lp + "post_attention_layernorm.weight", false, false, err);
    l.hc_attn_base = load_any(st, lp + "hc_attn_base", false, true, err);
    l.hc_attn_fn = load_any(st, lp + "hc_attn_fn", false, true, err);
    l.hc_attn_scale = load_any(st, lp + "hc_attn_scale", false, true, err);
    l.hc_ffn_base = load_any(st, lp + "hc_ffn_base", false, true, err);
    l.hc_ffn_fn = load_any(st, lp + "hc_ffn_fn", false, true, err);
    l.hc_ffn_scale = load_any(st, lp + "hc_ffn_scale", false, true, err);
    if (!err.empty()) return false;

    if (l.plan.linear_attention) {
      const std::string a = lp + "linear_attn.";
      l.kda_q = load_linear(st, a + "q_proj", err);
      l.kda_k = load_linear(st, a + "k_proj", err);
      l.kda_v = load_linear(st, a + "v_proj", err);
      l.kda_b = load_linear(st, a + "b_proj", err);
      l.kda_g_a = load_linear(st, a + "g_a_proj", err);
      l.kda_g_b = load_linear(st, a + "g_b_proj", err);
      l.kda_o = load_linear(st, a + "o_proj", err);
      l.kda_f_a = load_linear(st, a + "forget_gate.f_a_proj", err);
      l.kda_f_b = load_linear(st, a + "forget_gate.f_b_proj", err);
      l.conv1d_w = load_any(st, a + "conv1d.weight", false, false, err);
      l.conv1d_b = load_any(st, a + "conv1d.bias", false, false, err, false);
      l.o_norm = load_any(st, a + "o_norm.weight", false, false, err);
      l.dt_bias = load_any(st, a + "forget_gate.dt_bias", false, true, err);
      l.a_log = load_any(st, a + "forget_gate.A_log", false, true, err);
    } else {
      const std::string a = lp + "self_attn.";
      l.q_a = load_linear(st, a + "q_a_proj", err);
      l.q_a_norm = load_any(st, a + "q_a_layernorm.weight", false, false, err);
      l.q_b = load_linear(st, a + "q_b_proj", err);
      l.kv_a = load_linear(st, a + "kv_a_proj_with_mqa", err);
      l.kv_a_norm = load_any(st, a + "kv_a_layernorm.weight", false, false, err);
      l.kv_b = load_linear(st, a + "kv_b_proj", err);
      l.attn_o = load_linear(st, a + "o_proj", err);
      const std::string ix = a + "indexer.";
      l.idx_wq_b = load_linear(st, ix + "wq_b", err);
      l.idx_wk = load_any(st, ix + "wk.weight", true, false, err);
      l.idx_k_norm_w = load_any(st, ix + "k_norm.weight", false, false, err);
      l.idx_k_norm_b = load_any(st, ix + "k_norm.bias", false, false, err);
      l.idx_weights_proj = load_any(st, ix + "weights_proj.weight", true, false, err);
      l.idx_ape = load_any(st, ix + "index_kpool_compress_ape", false, true, err);
      l.idx_gate = load_any(st, ix + "index_kpool_compress_gate", true, false, err);
    }
    if (!err.empty()) return false;

    l.moe = l.plan.moe;
    if (l.moe) {
      const std::string m = lp + "mlp.";
      l.router_w = load_any(st, m + "gate.weight", false, false, err);
      l.router_bias = load_any(st, m + "gate.e_score_correction_bias", false, true, err);
      l.shared_gate = load_linear(st, m + "shared_experts.gate_proj", err);
      l.shared_up = load_linear(st, m + "shared_experts.up_proj", err);
      l.shared_down = load_linear(st, m + "shared_experts.down_proj", err);
      l.experts.resize((size_t)cfg.n_routed_experts * 3);
      for (int e = 0; e < cfg.n_routed_experts; ++e) {
        const std::string ep = m + "experts." + std::to_string(e) + ".";
        l.experts[(size_t)(e * 3 + 0)] = load_linear(st, ep + "gate_proj", err);
        l.experts[(size_t)(e * 3 + 1)] = load_linear(st, ep + "up_proj", err);
        l.experts[(size_t)(e * 3 + 2)] = load_linear(st, ep + "down_proj", err);
        if (!err.empty()) return false;
      }
    } else {
      const std::string m = lp + "mlp.";
      l.dense_gate = load_linear(st, m + "gate_proj", err);
      l.dense_up = load_linear(st, m + "up_proj", err);
      l.dense_down = load_linear(st, m + "down_proj", err);
    }
    if (!err.empty()) return false;
    tensors_before += 1;
    if (verbose && (i % 8 == 7 || i + 1 == cfg.num_layers)) {
      std::fprintf(stderr, "glm5: loaded %d/%d layers (%zu experts)\n", i + 1, cfg.num_layers,
                   out.expert_count());
    }
  }
  out.bytes = st.bytes;
  return true;
}

} // namespace exl3
