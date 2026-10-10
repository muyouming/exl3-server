#include "trellis_gemm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace exl3 {

namespace {

const float kHadScale = 1.0f / 11.313708498984761f; // 1/sqrt(128)

// ---------------------------------------------------------------------------
// Metal kernel source (MLX generates the signature; this is the body).
//
// One threadgroup owns QMV_NT output tiles x QMV_MB rows and covers the whole
// k range of its split. Each lane decodes 8 stream positions (two groups of
// four, sharing one 64-bit window each) and accumulates them into eight
// partial sums: positions 0..3 belong to output column c = lane/4, positions
// 4..7 to column c + 8, with rows 2*(lane%4) + {0,1,8,9}. A quadgroup shuffle
// reduction (4 lanes) completes the 16 rows of the tile.
// ---------------------------------------------------------------------------

const char* kQmvSource = R"METAL(
  uint tid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup;
  uint simd = simdgroup_index_in_threadgroup;
  uint tile_n = threadgroup_position_in_grid.x * uint(QMV_NT);
  uint row_base = threadgroup_position_in_grid.y * uint(QMV_MB);
  uint split = threadgroup_position_in_grid.z;

  uint word0[2];
  uint word1[2];
  uint shift[2];
  for (uint g = 0u; g < 2u; ++g) {
    uint first = lane * 8u + g * 4u;
    int end = int(first + 4u) * K + 256 * K;
    int last_word = (end - 1) / 32;
    word1[g] = uint(last_word) % uint(PACKED_U32);
    word0[g] = uint(last_word - 1) % uint(PACKED_U32);
    shift[g] = uint((last_word + 1) * 32 - end);
  }
  // The lane's eight positions span at most four 32-bit words, so one aligned
  // 4-word load (a single instruction instead of four) feeds every group.
  // Lanes whose span wraps the ring end fall back to per-group loads.

  float acc[QMV_MB][QMV_NT][8];
  for (uint b = 0u; b < uint(QMV_MB); ++b) {
    for (uint ot = 0u; ot < uint(QMV_NT); ++ot) {
      for (uint j = 0u; j < 8u; ++j) acc[b][ot][j] = 0.0f;
    }
  }

  uint tiles_per_split = (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
  uint tile_begin = split * tiles_per_split;
  uint tile_end = min(tile_begin + tiles_per_split, uint(TILES_K));

  for (uint tk = tile_begin + simd; tk < tile_end; tk += uint(QMV_SG)) {
    float xv[QMV_MB][4];
    for (uint b = 0u; b < uint(QMV_MB); ++b) {
      uint row = row_base + b;
      const device half2* xpairs =
          reinterpret_cast<const device half2*>(xhat + row * uint(INPUT_DIMS) + tk * 16u);
      half2 xp = xpairs[lane & 7u];
      half2 xa = simd_shuffle(xp, ushort(lane & 3u));
      half2 xb = simd_shuffle(xp, ushort((lane & 3u) + 4u));
      xv[b][0] = float(xa.x);
      xv[b][1] = float(xa.y);
      xv[b][2] = float(xb.x);
      xv[b][3] = float(xb.y);
    }
    for (uint ot = 0u; ot < uint(QMV_NT); ++ot) {
      const device uint* words =
          trellis + (tk * uint(TILES_N) + tile_n + ot) * uint(PACKED_U32);
      for (uint g = 0u; g < 2u; ++g) {
        ulong merged = (ulong(words[word0[g]]) << 32) | ulong(words[word1[g]]);
        uint sh = shift[g];
        uint win = uint(merged >> sh);
        uint cw3 = win & 0xffffu;
        uint cw2 = (win >> K) & 0xffffu;
        uint cw1 = (win >> (2u * K)) & 0xffffu;
        uint cw0 = K <= 5 ? ((win >> (3u * K)) & 0xffffu)
                          : (uint(merged >> (sh + 3u * K)) & 0xffffu);
        float v0 = qmv_decode(cw0);
        float v1 = qmv_decode(cw1);
        float v2 = qmv_decode(cw2);
        float v3 = qmv_decode(cw3);
        uint j = g * 4u;
        for (uint b = 0u; b < uint(QMV_MB); ++b) {
          acc[b][ot][j + 0u] = fma(xv[b][0], v0, acc[b][ot][j + 0u]);
          acc[b][ot][j + 1u] = fma(xv[b][1], v1, acc[b][ot][j + 1u]);
          acc[b][ot][j + 2u] = fma(xv[b][2], v2, acc[b][ot][j + 2u]);
          acc[b][ot][j + 3u] = fma(xv[b][3], v3, acc[b][ot][j + 3u]);
        }
      }
    }
  }

  threadgroup float partials[QMV_SG][QMV_MB * QMV_NT * 16];
  for (uint b = 0u; b < uint(QMV_MB); ++b) {
    for (uint ot = 0u; ot < uint(QMV_NT); ++ot) {
      float c0 = quad_sum(acc[b][ot][0] + acc[b][ot][1]) +
                 quad_sum(acc[b][ot][2] + acc[b][ot][3]);
      float c1 = quad_sum(acc[b][ot][4] + acc[b][ot][5]) +
                 quad_sum(acc[b][ot][6] + acc[b][ot][7]);
      if ((lane & 3u) == 0u) {
        uint base = (b * uint(QMV_NT) + ot) * 16u + (lane >> 2u);
        partials[simd][base] = c0;
        partials[simd][base + 8u] = c1;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint out_per_tg = uint(QMV_MB * QMV_NT * 16);
  for (uint idx = tid; idx < out_per_tg; idx += uint(QMV_SG * 32)) {
    uint b = idx / uint(QMV_NT * 16);
    uint local = idx % uint(QMV_NT * 16);
    uint ot = local >> 4u;
    uint col = local & 15u;
    float sum = 0.0f;
    for (uint g = 0u; g < uint(QMV_SG); ++g) sum += partials[g][local + b * uint(QMV_NT * 16)];
    uint row = row_base + b;
    uint index = row * uint(OUTPUT_DIMS) + (tile_n + ot) * 16u + col;
#if N_SPLITS > 1
    index += split * uint(PAD_ROWS) * uint(OUTPUT_DIMS);
#endif
    yhat[index] = sum;
  }
)METAL";

// Fixed helper prologue (codebook + reduction helpers) generated per codebook.
std::string codebook_prologue(int cb) {
  static const char* kSel4 =
      "inline uint sel4(uint4 v, uint i) {\n"
      "  return i == 0u ? v.x : (i == 1u ? v.y : (i == 2u ? v.z : v.w));\n"
      "}\n";
  if (std::getenv("EXL3_QMV_NODECODE")) {
    return "#include <metal_stdlib>\nusing namespace metal;\n"
           "inline float quad_sum(float v) { v += simd_shuffle_xor(v, ushort(1));"
           " v += simd_shuffle_xor(v, ushort(2)); return v; }\n"
           "inline float qmv_decode(uint x) { return float(x & 0xffu); }\n";
  }
  std::string s =
      std::string("#include <metal_stdlib>\n")
      + "using namespace metal;\n" + kSel4 +
      "inline float quad_sum(float v) {\n"
      "  v += simd_shuffle_xor(v, ushort(1));\n"
      "  v += simd_shuffle_xor(v, ushort(2));\n"
      "  return v;\n"
      "}\n"
      "inline float qmv_decode(uint x) {\n"
      "  x &= 0xffffu;\n";
  if (cb == 2) {
    s +=
        "  uint bits = x * 0x83DCD12Du;\n"
        "  uint pairs = (bits & 0x00ff00ffu) + ((bits >> 8) & 0x00ff00ffu);\n"
        "  uint sum = 0x6400u + (pairs & 0xffffu) + (pairs >> 16);\n"
        "  half value = as_type<half>(ushort(sum));\n"
        "  half inv = as_type<half>(ushort(0x1EEEu));\n"
        "  half bias = as_type<half>(ushort(0xC931u));\n"
        "  return float(fma(value, inv, bias));\n";
  } else if (cb == 1) {
    s +=
        "  uint bits = x * 0xCBAC1FEDu;\n"
        "  bits = 0x3B603B60u ^ (bits & 0x8FFF8FFFu);\n"
        "  half2 v = as_type<half2>(bits);\n"
        "  return float(v.x + v.y);\n";
  } else {
    s +=
        "  uint bits = x * 89226354u + 64248484u;\n"
        "  bits = 0x3B603B60u ^ (bits & 0x8FFF8FFFu);\n"
        "  half2 v = as_type<half2>(bits);\n"
        "  return float(v.x + v.y);\n";
  }
  s += "}\n";
  return s;
}

const char* kHadSource = R"METAL(
  uint tid = thread_position_in_threadgroup.x;
  uint block = threadgroup_position_in_grid.x;
  const device half* xp = x + block * 128u;
  const device half* sp = scale_vec + (block % uint(BLOCKS_PER_ROW)) * 128u;
  threadgroup float buf[128];
  threadgroup float buf2[128];
  float v = float(xp[tid]);
  if (HAD_PRE) v *= float(sp[tid]);
  buf[tid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint len = 1u; len < 128u; len <<= 1u) {
    uint in_group = tid % (2u * len);
    uint base = (tid / (2u * len)) * 2u * len;
    float a = buf[base + (in_group % len)];
    float b = buf[base + len + (in_group % len)];
    buf2[tid] = (in_group < len) ? (a + b) : (a - b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf[tid] = buf2[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float o = buf[tid] * HAD_SCALE;
  if (!HAD_PRE) o *= float(sp[tid]);
  y_out[block * 128u + tid] = HALF_OUT(o);
)METAL";

struct HadConfig {
  int blocks_per_row; // in_features / 128
  bool pre;           // scale before the transform (input side) or after
  bool operator==(const HadConfig& o) const {
    return blocks_per_row == o.blocks_per_row && pre == o.pre;
  }
};

std::string had_key(const HadConfig& c) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "had_%d_%d", c.blocks_per_row, (int)c.pre);
  return buf;
}

struct Config {
  int in, out, K, cb, nt, sg, mb, splits, pad_rows;
  bool operator==(const Config& o) const {
    return in == o.in && out == o.out && K == o.K && cb == o.cb && nt == o.nt && sg == o.sg &&
           mb == o.mb && splits == o.splits && pad_rows == o.pad_rows;
  }
};

std::string config_key(const Config& c) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%d_%d_%d_%d_%d_%d_%d_%d_%d", c.in, c.out, c.K, c.cb, c.nt,
                c.sg, c.mb, c.splits, c.pad_rows);
  return buf;
}

int env_int(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : 0;
}

std::mutex g_mutex;
std::unordered_map<std::string, mx::fast::CustomKernelFunction> g_kernels;

const mx::fast::CustomKernelFunction& kernel_for(const Config& c) {
  const std::string key = config_key(c);
  auto it = g_kernels.find(key);
  if (it != g_kernels.end()) return it->second;

  std::string header = codebook_prologue(c.cb);
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "#define K %d\n#define CBK %d\n#define INPUT_DIMS %d\n#define TILES_K %d\n"
                "#define TILES_N %d\n#define OUTPUT_DIMS %d\n#define PACKED_U32 %d\n"
                "#define QMV_NT %d\n#define QMV_MB %d\n#define QMV_SG %d\n#define N_SPLITS %d\n"
                "#define PAD_ROWS %d\n",
                c.K, c.cb, c.in, c.in / 16, c.out / 16, c.out, 8 * c.K, c.nt, c.mb, c.sg, c.splits,
                c.pad_rows);
  header += buf;

  char name[160];
  std::snprintf(name, sizeof(name), "exl3_qmv_%s", key.c_str());
  auto fn = mx::fast::metal_kernel(name, {"xhat", "trellis"}, {"yhat"}, kQmvSource, header, true,
                                   false, mx::CompileOptions{});
  return g_kernels.emplace(key, std::move(fn)).first->second;
}

std::unordered_map<std::string, mx::fast::CustomKernelFunction> g_had_kernels;

const mx::fast::CustomKernelFunction& had_kernel_for(const HadConfig& c) {
  const std::string key = had_key(c);
  auto it = g_had_kernels.find(key);
  if (it != g_had_kernels.end()) return it->second;
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "#define BLOCKS_PER_ROW %d\n#define HAD_PRE %d\n#define HAD_SCALE %a\n"
                "#define HALF_OUT(v) half(v)\n",
                c.blocks_per_row, (int)c.pre, (double)kHadScale);
  std::string header = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + buf;
  char name[96];
  std::snprintf(name, sizeof(name), "exl3_%s", key.c_str());
  auto fn = mx::fast::metal_kernel(name, {"x", "scale_vec"}, {"y_out"}, kHadSource, header, true,
                                   false, mx::CompileOptions{});
  return g_had_kernels.emplace(key, std::move(fn)).first->second;
}

// y = (H128 (x . s) ) * scale / sqrt(128)   (pre = true)
// y = (H128 x) * scale / sqrt(128) . s      (pre = false)
mx::array hadamard_scale(const mx::array& x, const mx::array& s, int blocks_per_row, bool pre) {
  const int64_t total = x.size();
  const int64_t blocks = total / 128;
  HadConfig cfg{blocks_per_row, pre};
  std::lock_guard<std::mutex> lk(g_mutex);
  const auto& kernel = had_kernel_for(cfg);
  auto outs = kernel({x, s}, {x.shape()}, {mx::float16}, {(int)blocks * 128, 1, 1}, {128, 1, 1},
                     std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                     false, mx::default_device());
  return outs[0];
}

int split_count(int k_tiles, int n_tiles) {
  int forced = env_int("EXL3_QMV_SPLITS");
  if (forced > 0) {
    while (k_tiles / forced < 1) forced /= 2;
    return forced > 0 ? forced : 1;
  }
  int warps = n_tiles * 4; // simdgroups (4 warps) per output tile
  int splits = 1;
  while (warps * splits < 1280 && k_tiles / (splits * 2) >= 4) splits *= 2;
  return splits;
}

} // namespace

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

static TrellisTuning& tuning_state() {
  static TrellisTuning t = [] {
    TrellisTuning out;
    out.nt = env_int("EXL3_QMV_NT");
    out.sg = env_int("EXL3_QMV_SG");
    out.mb = env_int("EXL3_QMV_MB");
    return out;
  }();
  return t;
}

void trellis_set_tuning(const TrellisTuning& t) { tuning_state() = t; }

TrellisTuning trellis_get_tuning() { return tuning_state(); }

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

mx::array trellis_matmul_raw(const TrellisWeight& w, const mx::array& xhat) {
  const int in = w.in_features;
  const int out = w.out_features;
  const int rows = (int)xhat.shape(0);
  const int k_tiles = in / 16;
  const int n_tiles = out / 16;

  TrellisTuning tn = tuning_state();
  const int sg = tn.sg > 0 ? tn.sg : 4;
  const int mb = tn.mb > 0 ? tn.mb : (rows >= 8 ? 8 : (rows >= 4 ? 4 : 1));
  // Tiles per threadgroup: enough threadgroups to fill the GPU, but large
  // enough to amortize the per-tile decode of the activations.
  int nt = tn.nt;
  if (nt <= 0) {
    nt = 1;
    while (n_tiles / nt > 1024) nt *= 2;
    while (n_tiles % nt != 0) nt /= 2;
    if (nt < 1) nt = 1;
  }
  const int splits = split_count(k_tiles, n_tiles);

  // Rows are padded to a whole number of row blocks so the kernel needs no row
  // bound, and large row counts are bucketed to limit kernel variants.
  int pad_rows = ((rows + mb - 1) / mb) * mb;
  if (pad_rows >= 64) pad_rows = ((pad_rows + 31) / 32) * 32;
  mx::array xin = xhat;
  if (pad_rows > rows) {
    xin = mx::concatenate(
        {xhat, mx::zeros(mx::Shape{pad_rows - rows, in}, mx::float16)}, 0);
  }

  Config cfg{in, out, w.K, w.codebook, nt, sg, mb, splits, pad_rows};
  std::vector<mx::Shape> shapes;
  if (splits > 1) shapes = {mx::Shape{splits, pad_rows, out}};
  else shapes = {mx::Shape{pad_rows, out}};
  mx::array y = mx::array(0.0f);
  {
    std::lock_guard<std::mutex> lk(g_mutex); // kernel map only; never held while launching more
    const auto& kernel = kernel_for(cfg);
    auto outs = kernel({xin, w.words}, shapes, {mx::float32},
                       {n_tiles / nt * sg * 32, pad_rows / mb, splits}, {sg * 32, 1, 1},
                       std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                       false, mx::default_device());
    y = outs[0];
  }
  if (splits > 1) y = mx::sum(y, std::vector<int>{0}, false);
  if (pad_rows > rows) y = mx::slice(y, {0, 0}, {rows, out});
  return y;
}

mx::array trellis_linear(const TrellisWeight& w, const mx::array& x) {
  const int in = w.in_features;
  const int out = w.out_features;
  const int64_t rows = x.size() / in;
  if (rows * in != (int64_t)x.size()) {
    throw std::invalid_argument("trellis_linear: input feature size mismatch");
  }
  // The kernels work on a flat (rows, in) view; any leading dimensions of x are
  // restored on the way out (mx::matmul semantics).
  mx::array x2 = x.ndim() == 2 ? x : mx::reshape(x, {(int)rows, in});
  // xhat = ((x . suh) H128 / sqrt(128)), one fused kernel, shapes preserved.
  mx::array xhat = hadamard_scale(x2, w.suh, in / 128, /*pre=*/true);

  // yhat = xhat @ W_hat, split rows padded so the kernel needs no row bound.
  const int k_tiles = in / 16;
  const int n_tiles = out / 16;
  TrellisTuning tn = tuning_state();
  const int sg = tn.sg > 0 ? tn.sg : 4;
  const int mb = tn.mb > 0 ? tn.mb : (rows >= 8 ? 8 : (rows >= 4 ? 4 : 1));
  int nt = tn.nt;
  if (nt <= 0) {
    nt = 1;
    while (n_tiles / nt > 1024) nt *= 2;
    while (nt > 1 && n_tiles % nt != 0) nt /= 2;
  }
  if (nt > n_tiles) nt = n_tiles;
  const int splits = split_count(k_tiles, n_tiles);
  int pad_rows = (int)(((rows + mb - 1) / mb) * mb);
  if (pad_rows >= 64) pad_rows = ((pad_rows + 31) / 32) * 32;
  mx::array xin = xhat;
  if (pad_rows > rows) {
    mx::Shape zshape(xhat.shape().begin(), xhat.shape().end());
    const int row_axis = (int)zshape.size() - 2;
    zshape[row_axis] = pad_rows - (int)rows;
    xin = mx::concatenate({xhat, mx::zeros(zshape, mx::float16)}, row_axis);
  }

  Config cfg{in, out, w.K, w.codebook, nt, sg, mb, splits, pad_rows};
  std::vector<mx::Shape> shapes;
  if (splits > 1) shapes = {mx::Shape{splits, pad_rows, out}};
  else shapes = {mx::Shape{pad_rows, out}};
  mx::array y = mx::array(0.0f);
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    const auto& kernel = kernel_for(cfg);
    auto outs = kernel({xin, w.words}, shapes, {mx::float32},
                       {n_tiles / nt * sg * 32, pad_rows / mb, splits}, {sg * 32, 1, 1},
                       std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                       false, mx::default_device());
    y = outs[0];
  }
  if (splits > 1) y = mx::sum(y, std::vector<int>{0}, false);
  if (pad_rows > rows) y = mx::slice(y, {0, 0}, {(int)rows, out});
  // y = (yhat H128 / sqrt(128)) . svh
  y = mx::astype(y, mx::float16);
  y = hadamard_scale(y, w.svh, out / 128, /*pre=*/false);
  // Rows above were flattened to x.size()/in, so restore x's leading dimensions
  // with the feature dim replaced by `out` (mx::matmul semantics) - a no-op for
  // the 2-D callers, and what the 3-D ones (KDA/MLA activations) need.
  if (x.ndim() != 2) {
    mx::Shape yshape(x.shape().begin(), x.shape().end());
    yshape[yshape.size() - 1] = out;
    y = mx::reshape(y, yshape);
  }
  return y;
}

} // namespace exl3
