#include "glm5.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "fp16.h"
#include "nlohmann/json.hpp"
#include "safetensors.h"
#include "util.h"

namespace exl3 {

namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Zero-copy plumbing
// ---------------------------------------------------------------------------

// A private, writable mapping of a file range. It is kept alive by the MLX
// arrays that view it (each array's deleter owns a shared_ptr to it), so the
// pages are unmapped only once the last view is gone. PROT_WRITE + MAP_PRIVATE
// means MLX never sees a read-only mapping (see the "MLX pitfalls" note in
// glm5.h) and a hypothetical write would land in a private copy of the page,
// never in the checkpoint.
struct Mapping {
  void* base = nullptr;
  size_t len = 0;
  ~Mapping() {
    if (base) ::munmap(base, len);
  }
};

int64_t page_bytes() {
  static const int64_t p = [] {
    long v = ::sysconf(_SC_PAGESIZE);
    return v > 0 ? (int64_t)v : (int64_t)16384;
  }();
  return p;
}

struct Store {
  std::vector<SafetensorsFile> files; // header + mapping used for addressing
  std::vector<int> fds;               // O_RDONLY fds, used to create views
  std::unordered_map<std::string, int> where;
  int64_t bytes = 0; // total bytes mapped across all shards
  float gate_lower_bound = 0.0f; // linear_attn_config.gate_lower_bound

  // Accounting: which tensors were consumed, and how many bytes they hold.
  mutable std::unordered_set<std::string> used;
  mutable int64_t used_tensors = 0;
  mutable int64_t used_bytes = 0;

  ~Store() {
    for (int fd : fds)
      if (fd >= 0) ::close(fd);
  }

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
      int fd = ::open(s.c_str(), O_RDONLY);
      if (fd < 0) {
        err = "cannot open " + s;
        return false;
      }
      fds.push_back(fd);
      files.push_back(std::move(f));
    }
    return true;
  }

  const SafetensorsFile* file_for(const std::string& name) const {
    auto it = where.find(name);
    return it == where.end() ? nullptr : &files[(size_t)it->second];
  }

  int shard_of(const std::string& name) const {
    auto it = where.find(name);
    return it == where.end() ? -1 : it->second;
  }

  bool has(const std::string& name) const { return where.count(name) != 0; }

  // Records a consumed tensor (and its byte count).
  void mark(const std::string& name) const {
    if (!used.insert(name).second) return;
    const SafetensorsFile* f = file_for(name);
    if (!f) return;
    ++used_tensors;
    used_bytes += f->tensor(name).nbytes;
  }
};

// Wraps [off, off + n) of shard `fi` in a single uint8 MLX array. The mapping
// is page aligned (required for MLX's no-copy path: a foreign pointer that is
// not page aligned is copied into a Metal buffer instead) and padded up to a
// page, which is why the caller slices the tensor out of it.
bool map_range(const Store& st, int fi, int64_t off, int64_t n, mx::array& out, std::string& err) {
  const int64_t page = page_bytes();
  const int64_t start = off - (off % page);
  const int64_t end = ((off + n + page - 1) / page) * page;
  const int64_t len = end - start;
  if (len <= 0 || len > (int64_t)INT_MAX) {
    err = "range too large for a zero-copy view";
    return false;
  }
  auto mp = std::make_shared<Mapping>();
  void* base = ::mmap(nullptr, (size_t)len, PROT_READ | PROT_WRITE, MAP_PRIVATE, st.fds[(size_t)fi],
                      (off_t)start);
  if (base == MAP_FAILED) {
    err = "mmap failed for " + st.files[(size_t)fi].path;
    return false;
  }
  mp->base = base;
  mp->len = (size_t)len;
  // The deleter owns the mapping: MLX runs it when the last array (and every
  // view that shares the buffer) is destroyed.
  out = mx::array(base, mx::Shape{(int)len}, mx::uint8, [mp](void*) {});
  return true;
}

// Zero-copy typed view of a tensor: (in, out) is not applied here, the caller
// passes the target dtype and shape, whose byte size must match the tensor.
bool tensor_view(const Store& st, const std::string& name, mx::Dtype dt, const mx::Shape& shape,
                 mx::array& out, std::string& err) {
  const int fi = st.shard_of(name);
  if (fi < 0) {
    err = "missing tensor " + name;
    return false;
  }
  const SafetensorsFile& f = st.files[(size_t)fi];
  const STTensor& info = f.tensor(name);
  const int64_t off = f.header_len + info.offset;
  int64_t numel = 1;
  for (auto d : info.shape) numel *= d;
  if (numel <= 0 || info.nbytes % numel != 0) {
    err = "unsupported byte layout for " + name;
    return false;
  }
  const int64_t align = info.nbytes / numel; // bytes per source element (2 for f16, 4 for f32)
  if (info.nbytes % align != 0 || off % align != 0) {
    err = "unaligned data for " + name;
    return false;
  }
  // The view is created inside a page-aligned mapping, so the alignment that
  // matters is the one of the *target* dtype: MLX stores an array offset in
  // elements of that dtype, and a mismatch would silently truncate it instead
  // of failing (which is why the packed trellis words are read as bytes).
  const int64_t req = mx::size_of(dt);
  if (req <= 0 || off % req != 0) {
    err = "unaligned data for " + name;
    return false;
  }
  int64_t want = 1;
  for (auto d : shape) want *= d;
  // The view covers the same *bytes*; a dtype reinterpretation (the I16 trellis
  // words read as uint32) halves the element count, so compare bytes, not
  // elements.
  if (want <= 0 || info.nbytes % want != 0) {
    err = "view shape does not match " + name;
    return false;
  }
  mx::array raw = mx::array(0.0f);
  if (!map_range(st, fi, off, info.nbytes, raw, err)) return false;
  const int pad = (int)(off % page_bytes());
  const int nbytes = (int)info.nbytes;
  mx::array bytes = mx::slice(raw, {pad}, {pad + nbytes});
  out = mx::reshape(mx::view(bytes, dt), shape);
  return true;
}

// Loads any tensor as f16 or f32 in (in, out) order for 2-D tensors. Tensors
// whose file bytes already have the requested dtype and layout are loaded as
// zero-copy views; everything else (bf16, transposes) is materialized.
mx::array load_any(const Store& st, const std::string& name, bool transpose, bool as_f32,
                   std::string& err, bool required = true) {
  const SafetensorsFile* f = st.file_for(name);
  if (!f) {
    if (required) err = "missing tensor " + name;
    return mx::array(0.0f);
  }
  const STTensor& info = f->tensor(name);
  st.mark(name);
  mx::Shape shape;
  for (auto d : info.shape) shape.push_back((int)d);

  const bool exact = !transpose && ((as_f32 && info.dtype == DT::F32) ||
                                    (!as_f32 && info.dtype == DT::F16));
  if (exact) {
    mx::array out = mx::array(0.0f);
    std::string view_err;
    if (tensor_view(st, name, as_f32 ? mx::float32 : mx::float16, shape, out, view_err)) {
      mx::eval(out);
      return out;
    }
    // not viewable (unaligned / oversized): fall through to a copy
  }

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
    mx::array out((void*)holder->data(), shape, mx::float32, [holder](void*) { delete holder; });
    mx::eval(out);
    return out;
  }
  auto* holder = new std::vector<uint16_t>(std::move(h16));
  mx::array out((void*)holder->data(), shape, mx::float16, [holder](void*) { delete holder; });
  mx::eval(out);
  return out;
}

// EXL3 linear (trellis + suh/svh + codebook marker) or a plain .weight tensor.
G5Linear load_linear(const Store& st, const std::string& key, std::string& err) {
  G5Linear lin;
  if (st.has(key + ".trellis")) {
    const SafetensorsFile* f = st.file_for(key + ".trellis");
    const STTensor& info = f->tensor(key + ".trellis");
    if (info.shape.size() != 3) {
      err = key + ".trellis must be 3-dimensional";
      return lin;
    }
    if (st.has(key + ".mul1")) st.mark(key + ".mul1");
    if (st.has(key + ".mcg")) st.mark(key + ".mcg");
    st.mark(key + ".trellis");
    const int64_t kt = info.shape[0], nt = info.shape[1], words16 = info.shape[2];
    if (words16 % 16 != 0) {
      err = key + ": trellis width not a multiple of 16";
      return lin;
    }
    lin.quantized = true;
    lin.tw.K = (int)(words16 / 16);
    lin.tw.codebook = st.has(key + ".mul1") ? 2 : (st.has(key + ".mcg") ? 1 : 0);
    lin.in_features = (int)(kt * 16);
    lin.out_features = (int)(nt * 16);
    lin.tw.in_features = lin.in_features;
    lin.tw.out_features = lin.out_features;
    // The packed words stay a zero-copy view of the shard when the tensor's
    // byte offset allows it: a page-aligned mmap plus the tensor's own offset
    // is 4-byte aligned whenever that offset is, which is what lets the GEMV
    // read each tile word with one aligned 32-bit load. A shard's JSON header
    // has an arbitrary length, so some tensors (the lm_head here) start 2 bytes
    // in: those get an aligned materialised copy instead (trellis_pack), so the
    // kernel's word loads are aligned for every tensor and its value is the
    // file's bytes either way.
    // The words stay a zero-copy view of the shard, and `aligned` says whether
    // the tensor's byte offset lets the GEMV read a tile word as one aligned
    // 32-bit load. A shard's JSON header has an arbitrary length and 11 of this
    // checkpoint's 13 shards have header_len % 4 != 0, i.e. 96 of its 102 GB of
    // trellis weights, so most tensors take the four-byte spelling of the word
    // (~20% slower on the 4096x24576 K4 shape, measured).
    //
    // Materialising those tensors with trellis_pack() instead - a byte-exact
    // copy into an aligned, engine-owned buffer - makes every word load aligned
    // but was measured a 2x *loss* end to end on this 128 GB machine (the MoE
    // stage: 123 -> 257 ms per step): 96 GB of anonymous weight memory is
    // subject to the compressor, where the file mapping's clean pages are
    // simply reclaimable. The pack is therefore used only where the view is not
    // possible; trellis_pack() remains the way to obtain an aligned buffer (and
    // is what the tests and the benchmark build their weights with).
    mx::array packed = mx::array(0.0f);
    std::string view_err;
    if (tensor_view(st, key + ".trellis", mx::uint8,
                     mx::Shape{(int)kt, (int)nt, (int)(words16 * 2)}, packed, view_err)) {
      lin.tw.aligned = ((f->header_len + info.offset) % 4) == 0;
    } else {
      const uint8_t* src = f->data(key + ".trellis", err);
      if (!src) return lin;
      packed = trellis_pack((const uint16_t*)src, (int)kt, (int)nt, lin.tw.K);
      lin.tw.aligned = true; // trellis_pack's buffer is 16-byte aligned
    }
    if (packed.size() == 0) {
      err = key + ": empty trellis";
      return lin;
    }
    lin.tw.packed = packed;
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

// --- metadata assertions: the checkpoint must match the loader's model ---------

bool expect_lin(const G5Linear& l, int in, int out, const std::string& what, std::string& err) {
  if (l.in_features != in || l.out_features != out) {
    err = what + ": expected (in=" + std::to_string(in) + ", out=" + std::to_string(out) +
          "), got (in=" + std::to_string(l.in_features) + ", out=" + std::to_string(l.out_features) +
          ")";
    return false;
  }
  return true;
}

bool expect_vec(const mx::array& a, int64_t n, const std::string& what, std::string& err) {
  if ((int64_t)a.size() != n) {
    err = what + ": expected " + std::to_string(n) + " elements, got " + std::to_string(a.size());
    return false;
  }
  return true;
}

// `text_config.linear_attn_config.gate_lower_bound` (a checkpoint property the
// engine needs to pick the forget-gate arm; see G5Layer::kda_gate_lower_bound)
// is parsed into ModelConfig::linear_gate_lower_bound by ModelConfig::load.

// One GLM-5-Next decoder layer: shared norms, the attention sub-layer (KDA or
// MLA + indexer, per `l.plan`) and the MLP sub-layer (MoE or dense). `with_hc`
// loads the hyper-connection parameters, which every trunk layer has and the
// MTP draft layer does not (its checkpoint tensors do not exist).
bool load_decoder_layer(const Store& st, const ModelConfig& cfg, const std::string& lp,
                        bool with_hc, G5Layer& l, std::string& err) {
  l.input_norm = load_any(st, lp + "input_layernorm.weight", false, false, err);
  l.post_norm = load_any(st, lp + "post_attention_layernorm.weight", false, false, err);
  l.hc_attn_base = load_any(st, lp + "hc_attn_base", false, true, err, with_hc);
  l.hc_attn_fn = load_any(st, lp + "hc_attn_fn", false, true, err, with_hc);
  l.hc_attn_scale = load_any(st, lp + "hc_attn_scale", false, true, err, with_hc);
  l.hc_ffn_base = load_any(st, lp + "hc_ffn_base", false, true, err, with_hc);
  l.hc_ffn_fn = load_any(st, lp + "hc_ffn_fn", false, true, err, with_hc);
  l.hc_ffn_scale = load_any(st, lp + "hc_ffn_scale", false, true, err, with_hc);
  if (!err.empty()) return false;

  if (l.plan.linear_attention) {
    // KDA: the module is `self_attn` (not `linear_attn`), q|k|v are fused into
    // `qkv_proj`, and A_log/dt_bias hang directly off `self_attn`.
    const std::string a = lp + "self_attn.";
    l.kda_qkv = load_linear(st, a + "qkv_proj", err);
    l.kda_b = load_linear(st, a + "b_proj", err);
    l.kda_g_a = load_linear(st, a + "g_a_proj", err);
    l.kda_g_b = load_linear(st, a + "g_b_proj", err);
    l.kda_f_a = load_linear(st, a + "f_a_proj", err);
    l.kda_f_b = load_linear(st, a + "f_b_proj", err);
    l.kda_o = load_linear(st, a + "o_proj", err);
    l.conv1d_w = load_any(st, a + "conv1d.weight", false, false, err);
    l.conv1d_b = load_any(st, a + "conv1d.bias", false, false, err, /*required=*/false);
    l.o_norm = load_any(st, a + "o_norm.weight", false, false, err);
    l.dt_bias = load_any(st, a + "dt_bias", false, true, err);
    l.a_log = load_any(st, a + "A_log", false, true, err);
    if (!err.empty()) return false;
    l.kda_gate_lower_bound = st.gate_lower_bound;
    // Geometry comes from the checkpoint itself (A_log/dt_bias).
    l.kda_heads = (int)l.a_log.size();
    l.kda_head_dim = l.kda_heads > 0 ? (int)(l.dt_bias.size() / (size_t)l.kda_heads) : 0;
    const int H = l.kda_heads, kd = l.kda_head_dim, qkv = H * kd;
    l.conv1d_has_bias = (int64_t)l.conv1d_b.size() == (int64_t)3 * qkv;
    if (!expect_lin(l.kda_qkv, cfg.hidden_size, 3 * qkv, a + "qkv_proj", err) ||
        !expect_lin(l.kda_b, cfg.hidden_size, H, a + "b_proj", err) ||
        !expect_lin(l.kda_g_a, cfg.hidden_size, kd, a + "g_a_proj", err) ||
        !expect_lin(l.kda_g_b, kd, qkv, a + "g_b_proj", err) ||
        !expect_lin(l.kda_f_a, cfg.hidden_size, kd, a + "f_a_proj", err) ||
        !expect_lin(l.kda_f_b, kd, qkv, a + "f_b_proj", err) ||
        !expect_lin(l.kda_o, qkv, cfg.hidden_size, a + "o_proj", err) ||
        !expect_vec(l.o_norm, kd, a + "o_norm.weight", err) ||
        !expect_vec(l.dt_bias, qkv, a + "dt_bias", err)) {
      return false;
    }
    if (l.conv1d_w.ndim() != 3 || l.conv1d_w.shape(0) != 3 * qkv || l.conv1d_w.shape(2) < 1) {
      err = a + "conv1d.weight: expected (" + std::to_string(3 * qkv) + ", 1, kernel)";
      return false;
    }
  } else {
    // MLA + DSA indexer.
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
    if (!err.empty()) return false;
    const int qk = cfg.qk_head_dim, vd = cfg.v_head_dim, nh = cfg.num_heads;
    if (!expect_lin(l.q_a, cfg.hidden_size, cfg.q_lora_rank, a + "q_a_proj", err) ||
        !expect_lin(l.q_b, cfg.q_lora_rank, nh * qk, a + "q_b_proj", err) ||
        !expect_lin(l.kv_a, cfg.hidden_size, cfg.kv_lora_rank, a + "kv_a_proj_with_mqa", err) ||
        !expect_lin(l.kv_b, cfg.kv_lora_rank, nh * (qk + vd), a + "kv_b_proj", err) ||
        !expect_lin(l.attn_o, nh * vd, cfg.hidden_size, a + "o_proj", err) ||
        !expect_vec(l.q_a_norm, cfg.q_lora_rank, a + "q_a_layernorm.weight", err) ||
        !expect_vec(l.kv_a_norm, cfg.kv_lora_rank, a + "kv_a_layernorm.weight", err) ||
        !expect_lin(l.idx_wq_b, cfg.q_lora_rank, cfg.index_n_heads * cfg.index_head_dim,
                    ix + "wq_b", err) ||
        !expect_vec(l.idx_k_norm_w, cfg.index_head_dim, ix + "k_norm.weight", err) ||
        !expect_vec(l.idx_k_norm_b, cfg.index_head_dim, ix + "k_norm.bias", err)) {
      return false;
    }
    if (l.idx_wk.ndim() != 2 || l.idx_wk.shape(0) != cfg.hidden_size ||
        l.idx_wk.shape(1) != cfg.index_head_dim ||
        l.idx_weights_proj.ndim() != 2 || l.idx_weights_proj.shape(0) != cfg.hidden_size ||
        l.idx_weights_proj.shape(1) != cfg.index_n_heads ||
        l.idx_ape.ndim() != 2 || l.idx_ape.shape(0) != cfg.index_kpool ||
        l.idx_ape.shape(1) != cfg.index_head_dim ||
        l.idx_gate.ndim() != 2 || l.idx_gate.shape(0) != cfg.hidden_size ||
        l.idx_gate.shape(1) != cfg.index_head_dim) {
      err = a + "indexer: unexpected shape for weights_proj/ape/gate";
      return false;
    }
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
  return err.empty();
}

} // namespace

bool load_glm5(const std::string& dir, const ModelConfig& cfg, Glm5Weights& out, std::string& err,
               int threads, bool verbose) {
  (void)threads;
  if (cfg.arch != Arch::Glm5Next) {
    err = "load_glm5: config architecture is not glm5_next";
    return false;
  }
  const int64_t t0 = now_ms();
  Store st;
  if (!st.open(dir, err)) return false;
  st.gate_lower_bound = cfg.linear_gate_lower_bound;
  if (verbose) {
    std::fprintf(stderr, "glm5: %zu shards, %zu tensors, %.1f GB mapped in %.1fs\n",
                 st.files.size(), st.where.size(), st.bytes / 1e9, (now_ms() - t0) / 1000.0);
  }

  const std::string p = cfg.tensor_prefix; // "model.language_model."
  out.embed = load_any(st, p + "embed_tokens.weight", false, false, err);
  out.final_norm = load_any(st, p + "norm.weight", false, false, err);
  out.lm_head = load_linear(st, "lm_head", err);
  if (!err.empty()) return false;
  if (verbose) {
    std::fprintf(stderr, "glm5: embed + lm_head loaded in %.1fs\n", (now_ms() - t0) / 1000.0);
  }

  out.layers.resize((size_t)cfg.num_layers);
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string lp = p + "layers." + std::to_string(i) + ".";
    G5Layer& l = out.layers[(size_t)i];
    l.plan = cfg.layer_plan[(size_t)i];
    const int64_t layer_bytes0 = st.used_bytes;
    const int64_t layer_t0 = now_ms();
    if (!load_decoder_layer(st, cfg, lp, /*with_hc=*/true, l, err)) return false;
    if (verbose) {
      std::fprintf(stderr, "glm5: layer %2d %-18s %-17s %8.1f MB %6.0f ms\n", i,
                   l.plan.linear_attention ? "linear_attention" : "sparse_attention",
                   l.moe ? (std::to_string(l.experts.size() / 3) + " MoE experts").c_str() : "dense MLP",
                   (st.used_bytes - layer_bytes0) / 1048576.0, (double)(now_ms() - layer_t0));
    }
  }

  // --- MTP draft block (the checkpoint's extra layer `num_layers`) -----------
  if (cfg.num_nextn_predict_layers > 0) {
    const std::string lp = p + "layers." + std::to_string(cfg.num_layers) + ".";
    G5Mtp& m = out.mtp;
    const int64_t mtp_bytes0 = st.used_bytes;
    const int64_t mtp_t0 = now_ms();
    // The draft layer is one more sparse-attention MoE layer; it carries no
    // hyper-connection parameters (the reference builds the MTP block without
    // them), so its residuals are plain.
    ModelConfig::LayerPlan mp;
    mp.linear_attention = false;
    mp.sparse_attention = true;
    mp.moe = true;
    m.layer.plan = mp;
    if (!load_decoder_layer(st, cfg, lp, /*with_hc=*/false, m.layer, err)) return false;
    m.enorm = load_any(st, lp + "enorm.weight", false, false, err);
    m.hnorm = load_any(st, lp + "hnorm.weight", false, false, err);
    m.eh_proj = load_linear(st, lp + "eh_proj", err);
    m.head_norm = load_any(st, lp + "shared_head.norm.weight", false, false, err);
    if (!err.empty()) return false;
    if (!expect_lin(m.eh_proj, 2 * cfg.hidden_size, cfg.hidden_size, lp + "eh_proj", err) ||
        !expect_vec(m.enorm, cfg.hidden_size, lp + "enorm.weight", err) ||
        !expect_vec(m.hnorm, cfg.hidden_size, lp + "hnorm.weight", err) ||
        !expect_vec(m.head_norm, cfg.hidden_size, lp + "shared_head.norm.weight", err)) {
      return false;
    }
    m.present = true;
    m.layer_index = cfg.num_layers;
    if (verbose) {
      std::fprintf(stderr, "glm5: MTP draft layer %d (MLA + %zu MoE experts) %8.1f MB %6.0f ms\n",
                   cfg.num_layers, m.layer.experts.size() / 3,
                   (st.used_bytes - mtp_bytes0) / 1048576.0, (double)(now_ms() - mtp_t0));
    }
  }

  // --- explicit skip sets ---------------------------------------------------
  const int64_t t_skip = now_ms();
  std::map<std::string, int64_t> skipped;
  std::vector<std::string> unexpected;
  const std::string layers_p = p + "layers.";
  for (auto& [name, fi] : st.where) {
    (void)fi;
    if (st.used.count(name)) continue;
    if (name.rfind("model.visual.", 0) == 0) {
      skipped["model.visual. (vision tower, text-only engine)"]++;
      continue;
    }
    if (name.rfind("mtp.", 0) == 0 || name.rfind("model.mtp.", 0) == 0) {
      // The `mtp.*` export layout is not wired (this checkpoint uses the extra
      // layer layout instead); fail loudly rather than dropping it silently.
      unexpected.push_back(name);
      continue;
    }
    if (name.rfind(layers_p, 0) == 0) {
      const char* num = name.c_str() + layers_p.size();
      char* end = nullptr;
      long idx = std::strtol(num, &end, 10);
      const long loaded = cfg.num_layers + (out.mtp.present ? 1 : 0);
      if (end != num && *end == '.' && idx >= loaded) {
        skipped[layers_p + "{i>=" + std::to_string(loaded) +
                "} (draft layers beyond the loaded trunk + MTP block)"]++;
        continue;
      }
    }
    unexpected.push_back(name);
  }
  if (!unexpected.empty()) {
    std::sort(unexpected.begin(), unexpected.end());
    err = "load_glm5: " + std::to_string(unexpected.size()) +
          " checkpoint tensors are not handled (first: " + unexpected.front() + ")";
    return false;
  }
  for (auto& [label, count] : skipped) {
    std::fprintf(stderr, "glm5: skipped %lld tensors: %s\n", (long long)count, label.c_str());
  }

  out.bytes = st.bytes;
  out.used_bytes = st.used_bytes;
  out.used_tensors = st.used_tensors;
  out.total_tensors = (int64_t)st.where.size();
  for (auto& [label, count] : skipped) out.skipped.emplace_back(label, count);
  if (verbose) {
    int kda = 0, dsa = 0, moe = 0;
    for (auto& l : out.layers) {
      kda += l.plan.linear_attention;
      dsa += l.plan.sparse_attention;
      moe += l.moe;
    }
    std::fprintf(stderr,
                 "glm5: %d layers (%d KDA / %d DSA, %d dense + %d MoE)%s, %lld tensors of %lld "
                 "loaded, %.1f GB used of %.1f GB mapped, index scan %.1fs, total %.1fs\n",
                 cfg.num_layers, kda, dsa, cfg.num_layers - moe, moe,
                 out.mtp.present ? " + 1 MTP draft layer" : "", (long long)out.used_tensors,
                 (long long)out.total_tensors, st.used_bytes / 1e9, st.bytes / 1e9,
                 (now_ms() - t_skip) / 1000.0, (now_ms() - t0) / 1000.0);
  }
  return true;
}

} // namespace exl3
