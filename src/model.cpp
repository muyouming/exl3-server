#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unordered_map>

#include "exl3.h"
#include "fp16.h"
#include "safetensors.h"
#include "util.h"

namespace exl3 {

// ---------------------------------------------------------------------------
// Weight store: one or more safetensors shards, mmap'd.
// ---------------------------------------------------------------------------

namespace {

struct TensorStore {
  std::vector<SafetensorsFile> files;
  std::unordered_map<std::string, int> where;

  bool open(const std::string& dir, std::string& err) {
    namespace fs = std::filesystem;
    std::vector<std::string> shards;
    std::string index = dir + "/model.safetensors.index.json";
    if (fs::exists(dir + "/model.safetensors")) {
      shards.push_back(dir + "/model.safetensors");
    } else if (fs::exists(index)) {
      nlohmann::json j;
      if (!parse_json_file(index, j, err)) return false;
      for (auto& [name, file] : j["weight_map"].items()) {
        std::string s = dir + "/" + file.get<std::string>();
        if (std::find(shards.begin(), shards.end(), s) == shards.end()) shards.push_back(s);
      }
      std::sort(shards.begin(), shards.end());
    } else {
      err = "no model.safetensors or model.safetensors.index.json in " + dir;
      return false;
    }
    for (auto& s : shards) {
      SafetensorsFile f;
      if (!SafetensorsFile::load(s, f, err)) return false;
      int idx = (int)files.size();
      for (auto& [name, t] : f.tensors) where.emplace(name, idx);
      files.push_back(std::move(f));
    }
    return true;
  }

  bool has(const std::string& name) const { return where.count(name) != 0; }

  const SafetensorsFile* file_for(const std::string& name) const {
    auto it = where.find(name);
    if (it == where.end()) return nullptr;
    return &files[it->second];
  }
};

mx::array wrap_f16(std::vector<uint16_t>& buf, mx::Shape shape) {
  // Hand the buffer to MLX without a copy (unified memory, so Metal reads these
  // pages directly). The deleter frees the vector once the array is destroyed.
  auto* holder = new std::vector<uint16_t>(std::move(buf));
  return mx::array((void*)holder->data(), std::move(shape), mx::float16,
                   [holder](void*) { delete holder; });
}

} // namespace

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

bool ModelConfig::load(const std::string& dir, ModelConfig& out, std::string& err) {
  nlohmann::json j;
  if (!parse_json_file(dir + "/config.json", j, err)) return false;
  try {
    out.hidden_size = j.at("hidden_size").get<int>();
    out.num_layers = j.at("num_hidden_layers").get<int>();
    out.num_heads = j.at("num_attention_heads").get<int>();
    out.num_kv_heads = j.value("num_key_value_heads", out.num_heads);
    out.head_dim = j.value("head_dim", out.hidden_size / std::max(1, out.num_heads));
    out.intermediate_size = j.at("intermediate_size").get<int>();
    out.vocab_size = j.at("vocab_size").get<int>();
    out.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    out.rope_theta = j.value("rope_theta", 10000.0f);
    out.max_position_embeddings = j.value("max_position_embeddings", 0);
    out.bos_token_id = j.value("bos_token_id", -1);
    if (j.contains("eos_token_id")) {
      if (j["eos_token_id"].is_array()) {
        for (auto& e : j["eos_token_id"]) out.eos_token_ids.push_back(e.get<int>());
      } else {
        out.eos_token_ids.push_back(j["eos_token_id"].get<int>());
      }
    }
    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
      err = "rope_scaling is not supported by this engine";
      return false;
    }
  } catch (const std::exception& e) {
    err = std::string("config.json: ") + e.what();
    return false;
  }
  if (out.hidden_size % out.num_heads != 0) {
    err = "hidden_size is not divisible by num_attention_heads";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

mx::array Model::load_vector(const SafetensorsFile& st, const std::string& name,
                             std::string& err, bool transpose2d) {
  const STTensor& info = st.tensor(name);
  std::vector<uint16_t> buf((size_t)st.numel(name));
  if (info.dtype == DT::BF16) {
    const uint16_t* p = st.bf16_data(name, err);
    if (!p) return mx::array(0.0f);
    for (size_t i = 0; i < buf.size(); ++i) {
      // bf16 -> f16 with round-to-nearest-even (shift the 16-bit payload, keep
      // the top 10 mantissa bits, round with the 3 dropped bits + sticky)
      uint32_t b = p[i];
      buf[i] = fp16_bits_from_bf16((uint16_t)b);
    }
  } else if (info.dtype == DT::F16) {
    const uint16_t* p = st.f16_data(name, err);
    if (!p) return mx::array(0.0f);
    std::memcpy(buf.data(), p, buf.size() * 2);
  } else {
    err = "tensor " + name + " has unsupported dtype " + dtype_name(info.dtype);
    return mx::array(0.0f);
  }
  mx::Shape shape;
  for (auto d : info.shape) shape.push_back((int)d);
  if (transpose2d) {
    // Unquantized checkpoints store nn.Linear weights as (out, in); the engine
    // expects (in, out).
    if (info.shape.size() != 2) {
      err = "tensor " + name + " is not 2-dimensional";
      return mx::array(0.0f);
    }
    const int64_t out_f = info.shape[0], in_f = info.shape[1];
    std::vector<uint16_t> t(buf.size());
    for (int64_t o = 0; o < out_f; ++o) {
      for (int64_t i = 0; i < in_f; ++i) t[(size_t)(i * out_f + o)] = buf[(size_t)(o * in_f + i)];
    }
    buf.swap(t);
    shape = mx::Shape{(int)in_f, (int)out_f};
  }
  return wrap_f16(buf, std::move(shape));
}

// Load a packed EXL3 tensor for the on-the-fly decode path. The Metal kernels
// read these arrays directly, so they must live in memory we own (MLX's no-copy
// path is not safe for read-only file mappings).
bool Model::load_trellis(const SafetensorsFile& st, const std::string& key, TrellisWeight& out,
                         std::string& err) {
  const STTensor& info = st.tensor(key + ".trellis");
  if (info.shape.size() != 3) {
    err = key + ".trellis must be 3-dimensional";
    return false;
  }
  const uint16_t* src = (const uint16_t*)st.i16_data(key + ".trellis", err);
  if (!src) return false;
  const int64_t k_tiles = info.shape[0], n_tiles = info.shape[1], words16 = info.shape[2];
  if (words16 % 16 != 0) {
    err = key + ": trellis width is not a multiple of 16";
    return false;
  }
  out.K = (int)(words16 / 16);
  out.codebook = st.has(key + ".mul1") ? 2 : (st.has(key + ".mcg") ? 1 : 0);
  out.in_features = (int)(k_tiles * 16);
  out.out_features = (int)(n_tiles * 16);
  if (st.tensor(key + ".suh").shape[0] != k_tiles * 16 ||
      st.tensor(key + ".svh").shape[0] != n_tiles * 16) {
    err = key + ": trellis shape does not match the scale vectors";
    return false;
  }
  {
    auto* holder = new std::vector<uint32_t>((size_t)(k_tiles * n_tiles * words16 / 2));
    std::memcpy(holder->data(), src, holder->size() * 4);
    out.words = mx::array((void*)holder->data(),
                          mx::Shape{(int)k_tiles, (int)n_tiles, (int)(words16 / 2)}, mx::uint32,
                          [holder](void*) { delete holder; });
  }
  out.suh = load_vector(st, key + ".suh", err, false);
  out.svh = load_vector(st, key + ".svh", err, false);
  if (!err.empty()) return false;
  return true;
}

mx::array Model::load_linear(const SafetensorsFile& st, const std::string& key,
                             std::string& err, int threads) {
  const STTensor& info = st.tensor(key + ".trellis");
  if (info.shape.size() != 3) {
    err = key + ".trellis must be 3-dimensional";
    return mx::array(0.0f);
  }
  Exl3Tensor t;
  t.trellis = (const uint16_t*)st.i16_data(key + ".trellis", err);
  if (!t.trellis) return mx::array(0.0f);
  t.K = (int)(info.shape[2] / 16);
  if (t.K * 16 != info.shape[2]) {
    err = key + ": trellis width is not a multiple of 16";
    return mx::array(0.0f);
  }
  if (st.has(key + ".mul1")) {
    t.codebook = Codebook::Mul1;
  } else if (st.has(key + ".mcg")) {
    t.codebook = Codebook::McG;
  } else {
    t.codebook = Codebook::k3Inst;
  }
  const STTensor& suh = st.tensor(key + ".suh");
  const STTensor& svh = st.tensor(key + ".svh");
  t.in_features = suh.shape.at(0);
  t.out_features = svh.shape.at(0);
  t.suh = st.f16_data(key + ".suh", err);
  t.svh = st.f16_data(key + ".svh", err);
  if (!t.suh || !t.svh) return mx::array(0.0f);
  if (t.in_features != info.shape[0] * 16 || t.out_features != info.shape[1] * 16) {
    err = key + ": trellis shape does not match the scale vectors";
    return mx::array(0.0f);
  }
  if (t.in_features % 128 || t.out_features % 128) {
    err = key + ": both dimensions must be multiples of 128";
    return mx::array(0.0f);
  }

  std::vector<uint16_t> buf((size_t)t.in_features * (size_t)t.out_features);
  reconstruct(t, buf.data(), threads);
  weight_bytes_ += buf.size() * 2;
  return wrap_f16(buf, mx::Shape{(int)t.in_features, (int)t.out_features});
}

bool Model::load(const std::string& path, std::string& err, int threads, bool fp16_weights) {
  namespace fs = std::filesystem;
  int64_t t0 = now_ms();
  fp16_weights_ = fp16_weights;
  std::string dir = path;
  if (fs::is_regular_file(path)) dir = fs::path(path).parent_path().string();
  if (!ModelConfig::load(dir, cfg_, err)) return false;

  TensorStore store;
  if (!store.open(dir, err)) return false;

  const int hidden = cfg_.hidden_size;
  const std::string lm_head_key = "lm_head"; // resolves to .trellis or .weight

  std::fprintf(stderr, "exl3-server: loading %d layers from %s\n", cfg_.num_layers, dir.c_str());
  layers_.resize((size_t)cfg_.num_layers);

  auto load_any_linear = [&](const std::string& key, Linear& lin) -> mx::array {
    const SafetensorsFile* f = store.file_for(key + ".trellis");
    if (f) {
      if (!fp16_weights_) {
        if (!load_trellis(*f, key, lin.tw, err)) return mx::array(0.0f);
        lin.quantized = true;
        weight_bytes_ += (size_t)lin.tw.bytes();
        return mx::array(0.0f);
      }
      return load_linear(*f, key, err, threads);
    }
    f = store.file_for(key + ".weight");
    if (!f) {
      err = "missing tensor " + key;
      return mx::array(0.0f);
    }
    return load_vector(*f, key + ".weight", err, /*transpose2d=*/true);
  };

  auto load_bias = [&](const std::string& key, Linear& lin) {
    const SafetensorsFile* f = store.file_for(key + ".bias");
    if (f) {
      lin.bias = load_vector(*f, key + ".bias", err);
      lin.has_bias = true;
    }
  };

  {
    const SafetensorsFile* f = store.file_for("model.embed_tokens.weight");
    if (!f) { err = "missing model.embed_tokens.weight"; return false; }
    embed_ = load_vector(*f, "model.embed_tokens.weight", err);
  }
  {
    const SafetensorsFile* f = store.file_for("model.norm.weight");
    if (!f) { err = "missing model.norm.weight"; return false; }
    final_norm_ = load_vector(*f, "model.norm.weight", err);
  }

  int64_t layer0 = now_ms();
  for (int i = 0; i < cfg_.num_layers; ++i) {
    Layer& l = layers_[(size_t)i];
    const std::string p = "model.layers." + std::to_string(i);
    l.q.w = load_any_linear(p + ".self_attn.q_proj", l.q);
    l.k.w = load_any_linear(p + ".self_attn.k_proj", l.k);
    l.v.w = load_any_linear(p + ".self_attn.v_proj", l.v);
    l.o.w = load_any_linear(p + ".self_attn.o_proj", l.o);
    l.gate.w = load_any_linear(p + ".mlp.gate_proj", l.gate);
    l.up.w = load_any_linear(p + ".mlp.up_proj", l.up);
    l.down.w = load_any_linear(p + ".mlp.down_proj", l.down);
    if (!err.empty()) return false;
    load_bias(p + ".self_attn.q_proj", l.q);
    load_bias(p + ".self_attn.k_proj", l.k);
    load_bias(p + ".self_attn.v_proj", l.v);
    load_bias(p + ".self_attn.o_proj", l.o);
    load_bias(p + ".mlp.gate_proj", l.gate);
    load_bias(p + ".mlp.up_proj", l.up);
    load_bias(p + ".mlp.down_proj", l.down);
    const SafetensorsFile* f = store.file_for(p + ".input_layernorm.weight");
    if (f) l.input_norm = load_vector(*f, p + ".input_layernorm.weight", err);
    f = store.file_for(p + ".post_attention_layernorm.weight");
    if (f) l.post_norm = load_vector(*f, p + ".post_attention_layernorm.weight", err);
    if (!err.empty()) return false;
  }
  const int64_t layer_ms = now_ms() - layer0;

  lm_head_.w = load_any_linear(lm_head_key, lm_head_);
  if (!err.empty()) return false;
  if (lm_head_.quantized) {
    if (lm_head_.tw.in_features != hidden) {
      err = "lm_head input dimension does not match hidden_size";
      return false;
    }
    cfg_.vocab_size = lm_head_.tw.out_features;
  } else {
    if (lm_head_.w.shape(0) != hidden) {
      err = "lm_head input dimension does not match hidden_size";
      return false;
    }
    if (lm_head_.w.shape(1) != cfg_.vocab_size) {
      // tolerate padded vocab: keep the model's own output width
      cfg_.vocab_size = (int)lm_head_.w.shape(1);
    }
  }

  // Release the file mappings; all weights now live in MLX arrays.
  for (auto& f : store.files) f.unmap();

  load_seconds_ = (double)(now_ms() - t0) / 1000.0;
  std::fprintf(stderr,
               "exl3-server: loaded %.2f GB of weights in %.1fs (%.1fs for %d layers)\n",
               (double)weight_bytes_ / 1e9, load_seconds_, (double)layer_ms / 1000.0,
               cfg_.num_layers);
  return true;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

std::vector<float> Model::forward(const std::vector<int>& tokens, KVCache& cache) {
  const int L = (int)tokens.size();
  const int hidden = cfg_.hidden_size;
  const int n_heads = cfg_.num_heads, n_kv = cfg_.num_kv_heads, hd = cfg_.head_dim;
  const int S = cache.offset + L;
  if (L <= 0) return {};

  if ((int)cache.keys.size() != cfg_.num_layers) cache.init(cfg_.num_layers);

  mx::array ids(tokens.data(), mx::Shape{1, L}, mx::int32);
  mx::array h = mx::take(embed_, ids, 0); // (1, L, hidden)

  const bool need_mask = L > 1;
  mx::array mask(0.0f);
  if (need_mask) {
    auto qpos = mx::reshape(mx::arange(cache.offset, cache.offset + L), {L, 1});
    auto kpos = mx::reshape(mx::arange(0, S), {1, S});
    auto allow = mx::less_equal(kpos, qpos); // key position <= query position
    mask = mx::where(allow, mx::array(0.0f, mx::float16),
                     mx::array(-INFINITY, mx::float16));
  }
  const float attn_scale = 1.0f / std::sqrt((float)hd);
  auto linear = [](const Linear& lin, const mx::array& x) -> mx::array {
    mx::array y = lin.quantized ? trellis_linear(lin.tw, x) : mx::matmul(x, lin.w);
    return lin.has_bias ? mx::add(y, lin.bias) : y;
  };

  for (int i = 0; i < cfg_.num_layers; ++i) {
    Layer& l = layers_[(size_t)i];
    auto r = mx::fast::rms_norm(h, l.input_norm, cfg_.rms_norm_eps);

    auto q = linear(l.q, r);
    auto k = linear(l.k, r);
    auto v = linear(l.v, r);

    q = mx::transpose(mx::reshape(q, {1, L, n_heads, hd}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {1, L, n_kv, hd}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {1, L, n_kv, hd}), {0, 2, 1, 3});

    q = mx::fast::rope(q, hd, /*traditional=*/false, cfg_.rope_theta, 1.0f, cache.offset);
    k = mx::fast::rope(k, hd, /*traditional=*/false, cfg_.rope_theta, 1.0f, cache.offset);

    if (cache.offset > 0) {
      k = mx::concatenate({cache.keys[(size_t)i], k}, 2);
      v = mx::concatenate({cache.values[(size_t)i], v}, 2);
    }
    cache.keys[(size_t)i] = k;
    cache.values[(size_t)i] = v;

    mx::array attn = mx::fast::scaled_dot_product_attention(
        q, k, v, attn_scale, /*mask_mode=*/"",
        need_mask ? std::optional<mx::array>(mask) : std::nullopt);
    attn = mx::reshape(mx::transpose(attn, {0, 2, 1, 3}), {1, L, n_heads * hd});
    h = mx::add(h, linear(l.o, attn));

    r = mx::fast::rms_norm(h, l.post_norm, cfg_.rms_norm_eps);
    auto g = linear(l.gate, r);
    auto u = linear(l.up, r);
    auto act = mx::multiply(g, mx::sigmoid(g)); // SiLU
    h = mx::add(h, linear(l.down, mx::multiply(act, u)));
  }

  h = mx::fast::rms_norm(h, final_norm_, cfg_.rms_norm_eps);
  auto last = mx::slice(h, {0, L - 1, 0}, {1, L, hidden});
  mx::array logits = linear(lm_head_, last);
  logits = mx::reshape(mx::astype(logits, mx::float32), {cfg_.vocab_size});

  std::vector<mx::array> to_eval;
  to_eval.reserve(2 * layers_.size() + 1);
  to_eval.push_back(logits);
  for (auto& a : cache.keys) to_eval.push_back(a);
  for (auto& a : cache.values) to_eval.push_back(a);
  mx::eval(to_eval);

  cache.offset = S;
  std::vector<float> out((size_t)cfg_.vocab_size);
  std::memcpy(out.data(), logits.data<float>(), out.size() * sizeof(float));
  return out;
}

void KVCache::trim(int pos) {
  if (pos >= offset) return;
  if (pos <= 0) {
    for (auto& a : keys) a = mx::array(0.0f);
    for (auto& a : values) a = mx::array(0.0f);
    offset = 0;
    return;
  }
  for (auto& a : keys) {
    if (a.ndim() == 4) a = mx::slice(a, {0, 0, 0, 0}, {1, a.shape(1), pos, a.shape(3)});
  }
  for (auto& a : values) {
    if (a.ndim() == 4) a = mx::slice(a, {0, 0, 0, 0}, {1, a.shape(1), pos, a.shape(3)});
  }
  {
    std::vector<mx::array> to_eval(keys.begin(), keys.end());
    to_eval.insert(to_eval.end(), values.begin(), values.end());
    mx::eval(to_eval);
  }
  offset = pos;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

static inline uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static inline double uniform01(uint64_t& state) {
  return (double)(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0);
}

int sample_token(std::vector<float>& logits, const SampleOptions& opt,
                 const std::vector<int>& context, uint64_t& rng_state) {
  const int n = (int)logits.size();

  if (opt.repetition_penalty != 1.0f || opt.presence_penalty != 0.0f ||
      opt.frequency_penalty != 0.0f) {
    std::unordered_map<int, int> counts;
    counts.reserve(context.size() * 2);
    for (int t : context) {
      if (t >= 0 && t < n) counts[t]++;
    }
    for (auto& [t, c] : counts) {
      float& x = logits[(size_t)t];
      if (opt.repetition_penalty != 1.0f) {
        x = x > 0 ? x / opt.repetition_penalty : x * opt.repetition_penalty;
      }
      if (opt.presence_penalty != 0.0f) x -= opt.presence_penalty;
      if (opt.frequency_penalty != 0.0f) x -= opt.frequency_penalty * (float)c;
    }
  }

  if (opt.temperature <= 0.0f) {
    int best = 0;
    float best_v = -INFINITY;
    for (int i = 0; i < n; ++i) {
      if (logits[(size_t)i] > best_v) { best_v = logits[(size_t)i]; best = i; }
    }
    logits[(size_t)best] = best_v;
    return best;
  }

  const int candidate_cap = std::max(opt.top_k > 0 ? opt.top_k : 0, 2048);
  int m = std::min(n, candidate_cap);
  std::vector<int32_t> idx((size_t)n);
  for (int i = 0; i < n; ++i) idx[(size_t)i] = i;

  auto cmp = [&](int32_t a, int32_t b) { return logits[(size_t)a] > logits[(size_t)b]; };
  if (m < n) {
    std::partial_sort(idx.begin(), idx.begin() + m, idx.end(), cmp);
  } else {
    std::sort(idx.begin(), idx.end(), cmp);
  }

  // temperature
  const float inv_t = 1.0f / opt.temperature;
  float maxl = logits[(size_t)idx[0]] * inv_t;
  // probabilities over the candidate set
  std::vector<float> probs((size_t)m);
  double sum = 0.0;
  for (int i = 0; i < m; ++i) {
    float p = std::exp(logits[(size_t)idx[(size_t)i]] * inv_t - maxl);
    probs[(size_t)i] = p;
    sum += p;
  }
  if (!(sum > 0.0)) return idx[0];

  // min_p
  int keep = m;
  if (opt.min_p > 0.0f) {
    const double thresh = (double)opt.min_p * (probs[0] / sum);
    keep = 0;
    for (int i = 0; i < m; ++i) {
      if (probs[(size_t)i] / sum >= thresh) keep = i + 1;
      else break;
    }
    if (keep == 0) keep = 1;
  }
  // top_k
  if (opt.top_k > 0 && opt.top_k < keep) keep = opt.top_k;
  // top_p
  if (opt.top_p < 1.0f) {
    double acc = 0.0;
    int k = 0;
    for (int i = 0; i < keep; ++i) {
      acc += probs[(size_t)i];
      ++k;
      if (acc / sum >= (double)opt.top_p) break;
    }
    keep = std::max(1, k);
  }

  double total = 0.0;
  for (int i = 0; i < keep; ++i) total += probs[(size_t)i];
  double r = uniform01(rng_state) * total;
  int chosen = keep - 1;
  for (int i = 0; i < keep; ++i) {
    r -= probs[(size_t)i];
    if (r <= 0.0) { chosen = i; break; }
  }
  return idx[(size_t)chosen];
}

} // namespace exl3
