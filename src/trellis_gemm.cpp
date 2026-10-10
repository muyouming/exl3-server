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
@GROUPHEAD@

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
          reinterpret_cast<const device half2*>(xhat + @XOFF@ row * uint(INPUT_DIMS) +
                                                tk * 16u);
      half2 xp = xpairs[lane & 7u];
      half2 xa = simd_shuffle(xp, ushort(lane & 3u));
      half2 xb = simd_shuffle(xp, ushort((lane & 3u) + 4u));
      xv[b][0] = float(xa.x);
      xv[b][1] = float(xa.y);
      xv[b][2] = float(xb.x);
      xv[b][3] = float(xb.y);
    }
    for (uint ot = 0u; ot < uint(QMV_NT); ++ot) {
      // `wsel` addresses this member's raw little-endian byte stream; word j of
      // a tile is 4 bytes starting at 4*j.
      const device uchar* tile =
          wsel + (tk * uint(TILES_N) + tile_n + ot) * uint(PACKED_BYTES);
      for (uint g = 0u; g < 2u; ++g) {
        // Two aligned 32-bit loads of the ring words that bracket this lane's
        // eight positions (the ring's wrap is why they are not one 8-byte
        // load). Byte-granular loads of the same two words measured 20% slower
        // on the 4096x24576 K4 shape, and fetching the neighbour word with a
        // simdgroup shuffle instead of the second load 19% slower still - the
        // second load is an L1 hit in the same cache line.
        ulong merged = (ulong(qmv_word(tile, word0[g])) << 32) | ulong(qmv_word(tile, word1[g]));
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
    yhat[@YOFF@index] = sum;
  }
)METAL";

// Fixed helper prologue (codebook + reduction helpers) generated per codebook.
std::string codebook_prologue(int cb, int aligned) {
  static const char* kSel4 =
      "inline uint sel4(uint4 v, uint i) {\n"
      "  return i == 0u ? v.x : (i == 1u ? v.y : (i == 2u ? v.z : v.w));\n"
      "}\n";
  // The packed trellis words are addressed as raw little-endian bytes. When the
  // buffer's base is 4-byte aligned (which trellis_pack guarantees, and which a
  // zero-copy view of a shard can promise only when its header length happens
  // to be a multiple of 4) word j is one aligned 32-bit load; otherwise it is
  // assembled from the same four bytes, which is ~20% slower on the 4096x24576
  // K4 shape but keeps the byte-stream contract exact.
  static const char* kWordAligned =
      "inline uint qmv_word(const device uchar* p, uint j) {\n"
      "  return *reinterpret_cast<const device uint*>(p + j * 4u);\n"
      "}\n";
  static const char* kWordBytes =
      "inline uint qmv_word(const device uchar* p, uint j) {\n"
      "  uint b = j * 4u;\n"
      "  return uint(p[b]) | (uint(p[b + 1u]) << 8) | (uint(p[b + 2u]) << 16) |\n"
      "         (uint(p[b + 3u]) << 24);\n"
      "}\n";
  const char* kWord = aligned ? kWordAligned : kWordBytes;
  // EXL3_QMV_NODECODE swaps the codebook decode for `x & 0xff`: the bandwidth
  // numbers then show what the memory path alone costs (the arithmetic is not
  // what limits this kernel - see docs/perf.md).
  if (std::getenv("EXL3_QMV_NODECODE")) {
    return std::string("#include <metal_stdlib>\nusing namespace metal;\n") + kWord +
           "inline float quad_sum(float v) { v += simd_shuffle_xor(v, ushort(1));"
           " v += simd_shuffle_xor(v, ushort(2)); return v; }\n"
           "inline float qmv_decode(uint x) { return float(x & 0xffu); }\n";
  }
  std::string s =
      std::string("#include <metal_stdlib>\n")
      + "using namespace metal;\n" + kSel4 + kWord +
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
  // With a stacked scale vector (one in_features-long vector per member, the
  // grid covering n x rows x blocks_per_row blocks in that order) the block's
  // member selects its slice. S_STRIDE is 0 otherwise, which folds the whole
  // term away.
  const device half* sp = scale_vec + (block / uint(ROWS_PER_MEMBER * BLOCKS_PER_ROW)) *
                                           uint(S_STRIDE) +
                           (block % uint(BLOCKS_PER_ROW)) * 128u;
  threadgroup float buf[128];
  threadgroup float buf2[128];
  // The input is rounded to f16 before the transform, which is where the
  // operator graph used to place its `astype` (the QMV writes f32): folding it
  // in removes one dispatch per linear and is exact, `half(half(x)) == x` for
  // the f16 input side.
  float v = float(half(x[block * 128u + tid]));
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
  bool stacked;       // the scale vector is packed per member (n blocks of it)
  int rows;           // rows per member (only needed when stacked)
  bool operator==(const HadConfig& o) const {
    return blocks_per_row == o.blocks_per_row && pre == o.pre && stacked == o.stacked &&
           rows == o.rows;
  }
};

std::string had_key(const HadConfig& c) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "had_%d_%d_%d_%d", c.blocks_per_row, (int)c.pre, (int)c.stacked,
                c.rows);
  return buf;
}

struct Config {
  int in, out, K, cb, nt, sg, mb, splits, pad_rows, group, aligned;
  bool operator==(const Config& o) const {
    return in == o.in && out == o.out && K == o.K && cb == o.cb && nt == o.nt && sg == o.sg &&
           mb == o.mb && splits == o.splits && pad_rows == o.pad_rows && group == o.group &&
           aligned == o.aligned;
  }
};

std::string config_key(const Config& c) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d", c.in, c.out, c.K, c.cb, c.nt,
                c.sg, c.mb, c.splits, c.pad_rows, c.group, c.aligned);
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

  std::string header = codebook_prologue(c.cb, c.aligned);
  char buf[704];
  std::snprintf(buf, sizeof(buf),
                "#define K %d\n#define CBK %d\n#define INPUT_DIMS %d\n#define TILES_K %d\n"
                "#define TILES_N %d\n#define OUTPUT_DIMS %d\n#define PACKED_U32 %d\n"
                "#define PACKED_BYTES %d\n"
                "#define QMV_NT %d\n#define QMV_MB %d\n#define QMV_SG %d\n#define N_SPLITS %d\n"
                "#define PAD_ROWS %d\n#define N_GROUP %d\n"
                "#define X_STRIDE (PAD_ROWS * INPUT_DIMS)\n"
                "#define Y_STRIDE (N_SPLITS * PAD_ROWS * OUTPUT_DIMS)\n",
                c.K, c.cb, c.in, c.in / 16, c.out / 16, c.out, 8 * c.K, 32 * c.K, c.nt, c.mb, c.sg,
                c.splits, c.pad_rows, c.group);
  header += buf;

  // One launch covers `group` tensors when group > 1: grid.z enumerates
  // (split, member) and each member reads its own weight buffer and its own
  // PAD_ROWS slice of the stacked activations/output.
  std::vector<std::string> inputs = {"xhat"};
  std::string body = kQmvSource;
  std::string head;
  if (c.group > 1) {
    for (int m = 0; m < c.group; ++m) inputs.push_back("trellis" + std::to_string(m));
    head = "  uint member = threadgroup_position_in_grid.z / uint(N_SPLITS);\n"
           "  uint split = threadgroup_position_in_grid.z % uint(N_SPLITS);\n"
           "  const device uchar* wsel = trellis0;\n";
    head += "  switch (member) {\n";
    for (int m = 1; m < c.group; ++m) {
      head += "    case " + std::to_string(m) + "u: wsel = trellis" + std::to_string(m) +
              "; break;\n";
    }
    head += "    default: break;\n  }\n";
  } else {
    inputs.push_back("trellis");
    head = "  uint split = threadgroup_position_in_grid.z;\n"
           "  const device uchar* wsel = trellis;\n";
  }
  auto subst = [&](const char* what, const std::string& with) {
    const size_t pos = body.find(what);
    if (pos == std::string::npos) throw std::runtime_error(std::string("qmv source: no ") + what);
    body.replace(pos, std::strlen(what), with);
  };
  subst("@GROUPHEAD@\n", head);
  subst("@XOFF@", c.group > 1 ? "member * uint(X_STRIDE) + " : "");
  subst("@YOFF@", c.group > 1 ? "member * uint(Y_STRIDE) + " : "");

  char name[160];
  std::snprintf(name, sizeof(name), "exl3_qmv_%s", key.c_str());
  auto fn = mx::fast::metal_kernel(name, inputs, {"yhat"}, body, header, true, false,
                                   mx::CompileOptions{});
  return g_kernels.emplace(key, std::move(fn)).first->second;
}

std::unordered_map<std::string, mx::fast::CustomKernelFunction> g_had_kernels;

const mx::fast::CustomKernelFunction& had_kernel_for(const HadConfig& c) {
  const std::string key = had_key(c);
  auto it = g_had_kernels.find(key);
  if (it != g_had_kernels.end()) return it->second;
  char buf[320];
  std::snprintf(buf, sizeof(buf),
                "#define BLOCKS_PER_ROW %d\n#define HAD_PRE %d\n#define HAD_SCALE %a\n"
                "#define HALF_OUT(v) half(v)\n#define S_STRIDE %d\n#define ROWS_PER_MEMBER %d\n",
                c.blocks_per_row, (int)c.pre, (double)kHadScale,
                c.stacked ? c.blocks_per_row * 128 : 0, c.stacked ? c.rows : 1);
  std::string header = std::string("#include <metal_stdlib>\nusing namespace metal;\n") + buf;
  char name[96];
  std::snprintf(name, sizeof(name), "exl3_%s", key.c_str());
  auto fn = mx::fast::metal_kernel(name, {"x", "scale_vec"}, {"y_out"}, kHadSource, header, true,
                                   false, mx::CompileOptions{});
  return g_had_kernels.emplace(key, std::move(fn)).first->second;
}

// The Hadamard on either side of a trellis linear, in one kernel:
//   input  side: y = (H128 (x . suh)) / sqrt(128)              [f16 in, f16 out]
//   output side: y = (H128 f16(y)) / sqrt(128) . svh           [f32 in, f16 out]
// The output side reads the QMV's f32 result and rounds it to f16 itself,
// which is where the operator graph used to put its `astype`: one dispatch
// fewer per linear (~1450 per decode token) with the same arithmetic.
// `s` is either one (in_features,) vector (the single-tensor form) or a packed
// concatenation of one per member (the stacked form, with `rows` rows per
// member and x's blocks in (member, row, block) order).
mx::array hadamard_run(const mx::array& x, const mx::array& s, int blocks_per_row, bool pre,
                       const mx::Shape& out_shape, int rows = 1, bool stacked = false) {
  // NOTE: mx::Shape is a std::vector<int>, so `.size()` is the rank, not the
  // element count (a zero grid silently skips the whole transform).
  int64_t total = 1;
  for (auto d : out_shape) total *= d;
  const int64_t blocks = total / 128;
  HadConfig cfg{blocks_per_row, pre, stacked, stacked ? rows : 1};
  std::lock_guard<std::mutex> lk(g_mutex);
  const auto& kernel = had_kernel_for(cfg);
  auto outs = kernel({x, s}, {out_shape}, {mx::float16}, {(int)blocks * 128, 1, 1}, {128, 1, 1},
                     std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                     false, mx::default_device());
  return outs[0];
}

} // namespace

mx::array trellis_hadamard(const mx::array& x, const mx::array& s, int blocks_per_row, bool pre,
                           int rows, bool stacked) {
  return hadamard_run(x, s, blocks_per_row, pre, x.shape(), rows, stacked);
}

namespace {

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
  // Rows per threadgroup: acc is QMV_MB*QMV_NT*8 floats, so the two together
  // must stay around 4 (32 accumulators) or the kernel spills - measured on the
  // 4096x24576 K4 shape: {mb,nt} = {4,2} at 4 rows is 1.9x slower than {4,1},
  // {8,2} at 8 rows 2.6x slower than {4,1}, and {1,4} at 1 row 1.1x slower than
  // {1,2}. Up to 4 rows a wider mb amortises the codebook decode over more rows
  // and wins (rows 2..26 all best at mb = min(rows,4)).
  const int mb = tn.mb > 0 ? tn.mb : (rows >= 4 ? 4 : rows);
  // Tiles per threadgroup: enough threadgroups to fill the GPU, but large
  // enough to amortize the per-tile decode of the activations.
  int nt = tn.nt;
  if (nt <= 0) {
    nt = 1;
    while (n_tiles / nt > 1024) nt *= 2;
    while (n_tiles % nt != 0) nt /= 2;
    if (nt < 1) nt = 1;
    if (mb >= 3) nt = 1; // mb*nt <= 4
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

  Config cfg{in, out, w.K, w.codebook, nt, sg, mb, splits, pad_rows, 1, w.aligned ? 1 : 0};
  std::vector<mx::Shape> shapes;
  if (splits > 1) shapes = {mx::Shape{splits, pad_rows, out}};
  else shapes = {mx::Shape{pad_rows, out}};
  mx::array y = mx::array(0.0f);
  {
    std::lock_guard<std::mutex> lk(g_mutex); // kernel map only; never held while launching more
    const auto& kernel = kernel_for(cfg);
    auto outs = kernel({xin, w.packed}, shapes, {mx::float32},
                       {n_tiles / nt * sg * 32, pad_rows / mb, splits}, {sg * 32, 1, 1},
                       std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                       false, mx::default_device());
    y = outs[0];
  }
  if (splits > 1) y = mx::sum(y, std::vector<int>{0}, false);
  if (pad_rows > rows) y = mx::slice(y, {0, 0}, {rows, out});
  return y;
}

// n linears in one launch. The members must share in_features/out_features/K
// and codebook (checked; they then need the same tiles/rows/splits geometry).
//   x       : (n, rows, in) f16, one activation block per member
//   returns : (n, rows, out) f16, y[m] = W_m @ x[m]
// Everything in between is batched: one Hadamard over the stacked input with a
// per-member scale vector, one QMV kernel whose grid.z enumerates
// (split, member), and one Hadamard over the stacked output.
namespace {
mx::array matmul_group_impl(const std::vector<TrellisWeight>& ws, const mx::array& x, int rows,
                            const mx::array& svh_stack);
}

mx::array trellis_matmul_group(const std::vector<TrellisWeight>& ws, const mx::array& x, int rows) {
  return matmul_group_impl(ws, x, rows, mx::array(0.0f));
}

namespace {
mx::array matmul_group_impl(const std::vector<TrellisWeight>& ws, const mx::array& x, int rows,
                            const mx::array& svh_stack) {
  const int n = (int)ws.size();
  if (n <= 0) return mx::array(0.0f);
  const int in = ws[0].in_features, out = ws[0].out_features;
  for (const auto& w : ws) {
    if (w.in_features != in || w.out_features != out || w.K != ws[0].K ||
        w.codebook != ws[0].codebook || w.aligned != ws[0].aligned) {
      throw std::invalid_argument("trellis_matmul_group: members must share their shape and word alignment");
    }
  }
  const int k_tiles = in / 16, n_tiles = out / 16;
  TrellisTuning tn = tuning_state();
  const int sg = tn.sg > 0 ? tn.sg : 4;
  const int mb = tn.mb > 0 ? tn.mb : (rows >= 4 ? 4 : rows);
  int nt = tn.nt;
  if (nt <= 0) {
    nt = 1;
    while (n_tiles / nt > 1024) nt *= 2;
    while (nt > 1 && n_tiles % nt != 0) nt /= 2;
    if (mb >= 3) nt = 1;
  }
  if (nt > n_tiles) nt = n_tiles;
  const int splits = split_count(k_tiles, n_tiles);
  int pad_rows = (int)(((rows + mb - 1) / mb) * mb);
  if (pad_rows >= 64) pad_rows = ((pad_rows + 31) / 32) * 32;

  mx::array xin = x;
  if (pad_rows > rows) xin = mx::concatenate({x, mx::zeros({n, pad_rows - rows, in}, mx::float16)}, 1);

  Config cfg{in, out, ws[0].K, ws[0].codebook, nt, sg, mb, splits, pad_rows, n,
             ws[0].aligned ? 1 : 0};
  std::vector<mx::Shape> shapes;
  if (splits > 1) shapes = {mx::Shape{n, splits, pad_rows, out}};
  else shapes = {mx::Shape{n, pad_rows, out}};
  std::vector<mx::array> args;
  args.reserve((size_t)n + 1);
  args.push_back(xin);
  for (const auto& w : ws) args.push_back(w.packed);
  mx::array y = mx::array(0.0f);
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    const auto& kernel = kernel_for(cfg);
    auto outs = kernel(args, shapes, {mx::float32},
                       {n_tiles / nt * sg * 32, pad_rows / mb, splits * n}, {sg * 32, 1, 1},
                       std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                       false, mx::default_device());
    y = outs[0];
  }
  if (splits > 1) y = mx::sum(y, std::vector<int>{1}, false); // (n, pad_rows, out)
  // One batched output Hadamard: the members' svh vectors concatenated give the
  // per-member scale (the kernel's block index runs member-major over rows).
  mx::array sstack = svh_stack;
  if ((int64_t)sstack.size() != (int64_t)n * out) {
    std::vector<mx::array> svh;
    svh.reserve((size_t)n);
    for (const auto& w : ws) svh.push_back(w.svh);
    sstack = mx::concatenate(svh, 0);
  }
  y = hadamard_run(mx::reshape(y, {n * pad_rows, out}), sstack, out / 128, /*pre=*/false,
                   mx::Shape{n * pad_rows, out}, pad_rows, /*stacked=*/true);
  y = mx::reshape(y, {n, pad_rows, out});
  if (pad_rows > rows) y = mx::slice(y, {0, 0, 0}, {n, rows, out});
  return y;
}

// xhat = ((x . suh_m) H128 / sqrt(128)) for all members in one launch, then the
// grouped QMV and one batched output Hadamard. `suh_stack` empty means "use the
// members' own scale vectors".
mx::array group_impl(const std::vector<TrellisWeight>& ws, const mx::array& xr, int rows,
                     const mx::array& suh_stack) {
  const int n = (int)ws.size();
  const int in = ws[0].in_features;
  mx::array sstack = suh_stack;
  if ((int64_t)sstack.size() != (int64_t)n * in) {
    std::vector<mx::array> suh;
    suh.reserve((size_t)n);
    for (const auto& w : ws) suh.push_back(w.suh);
    sstack = mx::concatenate(suh, 0);
  }
  mx::array xhat = hadamard_run(xr, sstack, in / 128, /*pre=*/true, mx::Shape{n, rows, in}, rows,
                                /*stacked=*/true);
  return matmul_group_impl(ws, xhat, rows, mx::array(0.0f));
}
} // namespace

mx::array trellis_linear_group(const std::vector<TrellisWeight>& ws, const mx::array& x) {
  // One activation block shared by every member: replicate it (the Hadamard
  // would materialise a broadcast view anyway) and take the stacked path.
  const int n = (int)ws.size();
  if (n <= 0) return mx::array(0.0f);
  const int in = ws[0].in_features;
  const int rows = (int)(x.size() / in);
  mx::array xr = mx::reshape(x, {1, rows, in});
  if (n > 1) xr = mx::reshape(mx::tile(xr, {n, 1, 1}), {n, rows, in});
  return group_impl(ws, xr, rows, mx::array(0.0f));
}

mx::array trellis_linear_group_rows(const std::vector<TrellisWeight>& ws, const mx::array& x,
                                    int rows) {
  const int n = (int)ws.size();
  if (n <= 0) return mx::array(0.0f);
  const int in = ws[0].in_features;
  if ((int64_t)x.size() != (int64_t)n * rows * in) {
    throw std::invalid_argument("trellis_linear_group_rows: activation size mismatch");
  }
  return group_impl(ws, x, rows, mx::array(0.0f));
}

// The scale-stack forms: the caller holds one (n, in) input-scale and one
// (n, out) output-scale stack (already flat in member order), which is what a
// device-side gather of the routed experts' scale vectors produces.
mx::array trellis_linear_group_scaled(const std::vector<TrellisWeight>& ws, const mx::array& x,
                                      const mx::array& suh_stack, const mx::array& svh_stack) {
  const int n = (int)ws.size();
  if (n <= 0) return mx::array(0.0f);
  const int in = ws[0].in_features;
  const int out = ws[0].out_features;
  const int rows = (int)(x.size() / in);
  mx::array xr = mx::reshape(x, {1, rows, in});
  if (n > 1) xr = mx::reshape(mx::tile(xr, {n, 1, 1}), {n, rows, in});
  mx::array sstack = mx::reshape(suh_stack, {n * in});
  mx::array xhat = hadamard_run(xr, sstack, in / 128, /*pre=*/true, mx::Shape{n, rows, in}, rows,
                                /*stacked=*/true);
  return matmul_group_impl(ws, xhat, rows, mx::reshape(svh_stack, {n * out}));
}

mx::array trellis_linear_group_rows_scaled(const std::vector<TrellisWeight>& ws, const mx::array& x,
                                           int rows, const mx::array& suh_stack,
                                           const mx::array& svh_stack) {
  const int n = (int)ws.size();
  if (n <= 0) return mx::array(0.0f);
  const int in = ws[0].in_features;
  const int out = ws[0].out_features;
  if ((int64_t)x.size() != (int64_t)n * rows * in) {
    throw std::invalid_argument("trellis_linear_group_rows_scaled: activation size mismatch");
  }
  mx::array sstack = mx::reshape(suh_stack, {n * in});
  mx::array xhat = hadamard_run(x, sstack, in / 128, /*pre=*/true, mx::Shape{n, rows, in}, rows,
                                /*stacked=*/true);
  return matmul_group_impl(ws, xhat, rows, mx::reshape(svh_stack, {n * out}));
}

mx::array trellis_pack(const uint16_t* words16, int k_tiles, int n_tiles, int K) {
  if (words16 == nullptr || k_tiles <= 0 || n_tiles <= 0 || K <= 0 || K > 8) return mx::array(0.0f);
  const int words = 8 * K;           // u32 ring words per 16x16 tile
  const int tile_bytes = 4 * words;  // == 32*K, the file's bytes per tile
  const size_t bytes = (size_t)k_tiles * (size_t)n_tiles * (size_t)tile_bytes;
  // The buffer is a fresh, 16-byte-aligned (malloc) allocation that the engine
  // owns, so tile (tk, tn) starts at a multiple of 32*K and word j at +4*j -
  // the alignment the kernels' 32-bit loads need. `new uint8_t[]` is
  // uninitialised: value-initialising a 100 GB checkpoint's worth of these
  // costs a full extra pass over memory.
  uint8_t* buf = new uint8_t[bytes];
  // The current mapping is the file's own: the tile's 8*K little-endian words
  // in ring order. A different (permuted) layout would be applied here, and the
  // kernel's tile/word indexing is the only other place that must agree.
  std::memcpy(buf, words16, bytes);
  mx::array out((void*)buf, mx::Shape{(int)k_tiles, (int)n_tiles, tile_bytes}, mx::uint8,
                [buf](void*) { delete[] buf; });
  mx::eval(out);
  return out;
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
  mx::array xhat = hadamard_run(x2, w.suh, in / 128, /*pre=*/true, x2.shape());

  // yhat = xhat @ W_hat, split rows padded so the kernel needs no row bound.
  const int k_tiles = in / 16;
  const int n_tiles = out / 16;
  TrellisTuning tn = tuning_state();
  const int sg = tn.sg > 0 ? tn.sg : 4;
  // See trellis_matmul_raw for why mb*nt must stay near 4 (accumulator
  // registers) and why mb = min(rows, 4) is the measured optimum.
  const int mb = tn.mb > 0 ? tn.mb : (rows >= 4 ? 4 : rows);
  int nt = tn.nt;
  if (nt <= 0) {
    nt = 1;
    while (n_tiles / nt > 1024) nt *= 2;
    while (nt > 1 && n_tiles % nt != 0) nt /= 2;
    if (mb >= 3) nt = 1; // mb*nt <= 4
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

  Config cfg{in, out, w.K, w.codebook, nt, sg, mb, splits, pad_rows, 1, w.aligned ? 1 : 0};
  std::vector<mx::Shape> shapes;
  if (splits > 1) shapes = {mx::Shape{splits, pad_rows, out}};
  else shapes = {mx::Shape{pad_rows, out}};
  mx::array y = mx::array(0.0f);
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    const auto& kernel = kernel_for(cfg);
    auto outs = kernel({xin, w.packed}, shapes, {mx::float32},
                       {n_tiles / nt * sg * 32, pad_rows / mb, splits}, {sg * 32, 1, 1},
                       std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                       false, mx::default_device());
    y = outs[0];
  }
  // The k-split partials are summed in f32 exactly as before (MLX's reduction
  // order is its own; reordering it here moved the model's logits), then
  // y = (H128 y / sqrt(128)) . svh with the f16 rounding inside the kernel.
  if (splits > 1) y = mx::sum(y, std::vector<int>{0}, false);
  y = hadamard_run(y, w.svh, out / 128, /*pre=*/false, mx::Shape{pad_rows, out});
  if (pad_rows > rows) y = mx::slice(y, {0, 0}, {(int)rows, out});
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
