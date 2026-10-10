#include "trellis_gemm.h"

#include <algorithm>
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

// ---------------------------------------------------------------------------
// The staged matrix path.
//
// The QMV above pays its decode once per (output tile, k-tile) *row block*:
// mb = min(rows, 4) rows share one pass of the packed words, so a linear with
// `rows` reads its whole tensor rows/mb times (and pays the decode - a mul1 hash
// and a half fma, ~10 issued lane-instructions per weight - that many times
// over). At the row counts a request-level batch reaches that is 4 passes at 16
// rows and 7 at 26, and the decode, not the weight bytes, is what the machine's
// issue width is spending (docs/perf.md, "Honest distance to 70 tok/s").
//
// This kernel decodes a (16 k x 16 n) weight tile ONCE per threadgroup into a
// threadgroup fp16 tile and then multiplies with simdgroup_matrix, so the MACs
// leave the issue pipe and one decode serves MAT_MB rows. It is a *scheduling*
// change plus one fp32 association change (the tensor unit sums the k range in
// its own order - see the exactness note in trellis_gemm.h), so it is gated off
// below MAT_MIN_ROWS rows: every single-sequence call and every rows<=8 batch
// keeps the byte-exact QMV.
//
// Lane mapping. A lane decodes the same eight stream positions as the QMV
// (`lane*8 + g*4 + m`, two groups of four), so the decoded values are the same
// bits. Where those positions land in the 16x16 tile is read off the QMV's own
// accumulator indices: position `lane*8 + g*4 + {0..3}` multiplies the x values
// at k = 2r, 2r+1, 8+2r, 9+2r (r = lane%4) and accumulates into output column
// q = lane/4 for g = 0 and q+8 for g = 1. So with p = lane*8 + g*4 + m,
//   n = q + 8*g            k = 2*r + (m & 1) + 8*(m >> 1)
// and the four values of a group are a (k, k+1) pair for one n.
//
// Staging layout. (n, k)-major with 16 halves per row, because (a) a lane's
// (2r, 2r+1) and (8+2r, 9+2r) k pairs are then contiguous - two half2 stores
// per group - and (b) the B fragment of simdgroup_multiply_accumulate is the
// (k, n) operand, which is what `simdgroup_load(..., elements_per_row = 16,
// transpose = true)` produces from an (n, k) source.
//
// Pipelining: two threadgroup buffers, one barrier per k-tile - stage the next
// tile before the current tile's fragments are read, and let the barrier order
// both.
// ---------------------------------------------------------------------------

const char* kQmvMatSource = R"METAL(
  uint tid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup;
  uint simd = simdgroup_index_in_threadgroup;
  uint tile_n = threadgroup_position_in_grid.x * uint(MAT_NT) + simd;
  uint row_base = threadgroup_position_in_grid.y * uint(MAT_MB);
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
  const uint q = lane >> 2u;
  const uint r = lane & 3u;

  // One decoded tile per simdgroup (MAT_SG == MAT_NT: simdgroup s owns output
  // tile s of this threadgroup), double buffered over the k-tile loop.
  threadgroup half stg[2][MAT_SG][256];

  uint tiles_per_split = (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
  uint tile_begin = split * tiles_per_split;
  uint tile_end = min(tile_begin + tiles_per_split, uint(TILES_K));

  // Per-lane staging, split so the words are loaded one tile AHEAD of their
  // decode: the device loads for tile tk+1 are issued before tile tk's MACs and
  // only consumed after them, which is the only way an in-order machine lets
  // them overlap. The alternative - decode the tile where it is loaded, before
  // the MACs - stalls the warp on the multiply-hash's operands with the tensor
  // unit idle and the other warps of the threadgroup queueing behind it:
  // measured 1.09x slower at 16 rows and 1.29x at 64 (4096x24576 K4, 64
  // iterations per graph, median of 3, single-warp threadgroups).
  uint nw0[2];
  uint nw1[2];
  auto load_tile = [&](uint tk) {
    const device uchar* tile = wsel + (tk * uint(TILES_N) + tile_n) * uint(PACKED_BYTES);
    for (uint g = 0u; g < 2u; ++g) {
      nw0[g] = qmv_word(tile, word0[g]);
      nw1[g] = qmv_word(tile, word1[g]);
    }
  };
  auto decode_tile = [&](uint buf) {
    threadgroup half* dst = &stg[buf][simd][0];
    for (uint g = 0u; g < 2u; ++g) {
      ulong merged = (ulong(nw0[g]) << 32) | ulong(nw1[g]);
      uint sh = shift[g];
      uint win = uint(merged >> sh);
      uint cw3 = win & 0xffffu;
      uint cw2 = (win >> K) & 0xffffu;
      uint cw1 = (win >> (2u * K)) & 0xffffu;
      uint cw0 = K <= 5 ? ((win >> (3u * K)) & 0xffffu)
                        : (uint(merged >> (sh + 3u * K)) & 0xffffu);
      uint base = (q + 8u * g) * 16u + 2u * r;
      *reinterpret_cast<threadgroup half2*>(dst + base) =
          half2(qmv_decode_h(cw0), qmv_decode_h(cw1));
      *reinterpret_cast<threadgroup half2*>(dst + base + 8u) =
          half2(qmv_decode_h(cw2), qmv_decode_h(cw3));
    }
  };

  simdgroup_matrix<float, 8, 8> acc[MAT_MB / 8][2];
  for (uint rb = 0u; rb < uint(MAT_MB / 8); ++rb) {
    for (uint nh = 0u; nh < 2u; ++nh) acc[rb][nh] = simdgroup_matrix<float, 8, 8>(0.0f);
  }

  if (tile_begin < tile_end) {
    load_tile(tile_begin);
    decode_tile(0u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint tk = tile_begin; tk < tile_end; ++tk) {
    const uint buf = (tk - tile_begin) & 1u;
    if (tk + 1u < tile_end) load_tile(tk + 1u);
    for (uint kh = 0u; kh < 2u; ++kh) {
      // The weight tile's two 8-column halves, as the (k, n) B fragments.
      simdgroup_matrix<half, 8, 8> bf[2];
      simdgroup_load(bf[0], &stg[buf][simd][kh * 8u], 16, ulong2(0, 0), true);
      simdgroup_load(bf[1], &stg[buf][simd][128u + kh * 8u], 16, ulong2(0, 0), true);
      for (uint rb = 0u; rb < uint(MAT_MB / 8); ++rb) {
        simdgroup_matrix<half, 8, 8> af;
        simdgroup_load(af, xhat + @XOFF@ (row_base + 8u * rb) * uint(INPUT_DIMS) +
                               tk * 16u + 8u * kh,
                       uint(INPUT_DIMS), ulong2(0, 0), false);
        simdgroup_multiply_accumulate(acc[rb][0], af, bf[0], acc[rb][0]);
        simdgroup_multiply_accumulate(acc[rb][1], af, bf[1], acc[rb][1]);
      }
    }
    // One barrier per k-tile, after both halves of the handoff: it orders this
    // tile's MAC reads against the next staging's stores (this iteration wrote
    // the *other* buffer, which the previous iteration read) and makes this
    // iteration's stores visible to the next iteration's MACs.
    if (tk + 1u < tile_end) decode_tile(buf ^ 1u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }


  for (uint rb = 0u; rb < uint(MAT_MB / 8); ++rb) {
    for (uint nh = 0u; nh < 2u; ++nh) {
      uint index = (row_base + 8u * rb) * uint(OUTPUT_DIMS) + tile_n * 16u + 8u * nh;
#if N_SPLITS > 1
      index += split * uint(PAD_ROWS) * uint(OUTPUT_DIMS);
#endif
      simdgroup_store(acc[rb][nh], yhat + @YOFF@ index, uint(OUTPUT_DIMS));
    }
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
           "inline float qmv_decode(uint x) { return float(x & 0xffu); }\n"
           "inline half qmv_decode_h(uint x) { return half(x & 0xffu); }\n";
  }
  // The decode body is emitted twice: `qmv_decode` returns the f16 value as a
  // float (the QMV's accumulator is f32), `qmv_decode_h` returns it as the half
  // it is - the staged matrix path stores that half into its threadgroup tile,
  // and the two spellings are the same bits by construction (the fma and the
  // subtraction both happen in half precision, so the widening is exact).
  std::string body = "  x &= 0xffffu;\n";
  std::string tail;
  if (cb == 2) {
    body +=
        "  uint bits = x * 0x83DCD12Du;\n"
        "  uint pairs = (bits & 0x00ff00ffu) + ((bits >> 8) & 0x00ff00ffu);\n"
        "  uint sum = 0x6400u + (pairs & 0xffffu) + (pairs >> 16);\n"
        "  half value = as_type<half>(ushort(sum));\n"
        "  half inv = as_type<half>(ushort(0x1EEEu));\n"
        "  half bias = as_type<half>(ushort(0xC931u));\n";
    tail = "fma(value, inv, bias)";
  } else if (cb == 1) {
    body +=
        "  uint bits = x * 0xCBAC1FEDu;\n"
        "  bits = 0x3B603B60u ^ (bits & 0x8FFF8FFFu);\n"
        "  half2 v = as_type<half2>(bits);\n";
    tail = "half(v.x + v.y)";
  } else {
    body +=
        "  uint bits = x * 89226354u + 64248484u;\n"
        "  bits = 0x3B603B60u ^ (bits & 0x8FFF8FFFu);\n"
        "  half2 v = as_type<half2>(bits);\n";
    tail = "half(v.x + v.y)";
  }
  return std::string("#include <metal_stdlib>\n") + "using namespace metal;\n" + kSel4 + kWord +
         "inline float quad_sum(float v) {\n"
         "  v += simd_shuffle_xor(v, ushort(1));\n"
         "  v += simd_shuffle_xor(v, ushort(2));\n"
         "  return v;\n"
         "}\n"
         "inline float qmv_decode(uint x) {\n" + body + "  return float(" + tail + ");\n}\n" +
         "inline half qmv_decode_h(uint x) {\n" + body + "  return " + tail + ";\n}\n";
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

// Build one QMV-family kernel: the codebook prologue + the per-config macros,
// the member-select head (grid.z enumerates (split, member) when group > 1, and
// each member then reads its own weight buffer and its own PAD_ROWS slice of
// the stacked activations/output), and the substituted body. Shared by the
// scalar QMV and the staged matrix kernel so the two cannot drift apart in the
// head, the buffer list or the substitution rules - `source` is the only
// difference besides the macros.
mx::fast::CustomKernelFunction make_qmv_kernel(const std::string& name, const char* source,
                                               const std::string& macros, int cb, int aligned,
                                               int group) {
  std::string header = codebook_prologue(cb, aligned) + macros;
  std::vector<std::string> inputs = {"xhat"};
  std::string body = source;
  std::string head;
  if (group > 1) {
    for (int m = 0; m < group; ++m) inputs.push_back("trellis" + std::to_string(m));
    head = "  uint member = threadgroup_position_in_grid.z / uint(N_SPLITS);\n"
           "  uint split = threadgroup_position_in_grid.z % uint(N_SPLITS);\n"
           "  const device uchar* wsel = trellis0;\n";
    head += "  switch (member) {\n";
    for (int m = 1; m < group; ++m) {
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
  subst("@XOFF@", group > 1 ? "member * uint(X_STRIDE) + " : "");
  subst("@YOFF@", group > 1 ? "member * uint(Y_STRIDE) + " : "");
  return mx::fast::metal_kernel(name, inputs, {"yhat"}, body, header, true, false,
                                mx::CompileOptions{});
}

const mx::fast::CustomKernelFunction& kernel_for(const Config& c) {
  const std::string key = config_key(c);
  auto it = g_kernels.find(key);
  if (it != g_kernels.end()) return it->second;

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
  char name[160];
  std::snprintf(name, sizeof(name), "exl3_qmv_%s", key.c_str());
  auto fn = make_qmv_kernel(name, kQmvSource, buf, c.cb, c.aligned, c.group);
  return g_kernels.emplace(key, std::move(fn)).first->second;
}

// The staged matrix kernel: same prologue, head and buffer list, its own
// geometry macros. `mat_mb` is rows per threadgroup, `mat_nt` output tiles per
// threadgroup and simdgroups per threadgroup at once (the kernel requires
// MAT_SG == MAT_NT: simdgroup s stages and owns output tile s), so a geometry
// key of its own is what keeps this kernel from colliding with the QMV's.
struct MatConfig {
  int in, out, K, cb, mb, nt, splits, pad_rows, group, aligned;
  bool operator==(const MatConfig& o) const {
    return in == o.in && out == o.out && K == o.K && cb == o.cb && mb == o.mb && nt == o.nt &&
           splits == o.splits && pad_rows == o.pad_rows && group == o.group &&
           aligned == o.aligned;
  }
};

std::string mat_config_key(const MatConfig& c) {
  char buf[160];
  std::snprintf(buf, sizeof(buf), "%d_%d_%d_%d_%d_%d_%d_%d_%d_%d", c.in, c.out, c.K, c.cb, c.mb,
                c.nt, c.splits, c.pad_rows, c.group, c.aligned);
  return buf;
}

std::unordered_map<std::string, mx::fast::CustomKernelFunction> g_mat_kernels;

const mx::fast::CustomKernelFunction& mat_kernel_for(const MatConfig& c) {
  const std::string key = mat_config_key(c);
  auto it = g_mat_kernels.find(key);
  if (it != g_mat_kernels.end()) return it->second;

  char buf[704];
  std::snprintf(buf, sizeof(buf),
                "#define K %d\n#define CBK %d\n#define INPUT_DIMS %d\n#define TILES_K %d\n"
                "#define TILES_N %d\n#define OUTPUT_DIMS %d\n#define PACKED_U32 %d\n"
                "#define PACKED_BYTES %d\n"
                "#define MAT_MB %d\n#define MAT_NT %d\n#define MAT_SG %d\n#define N_SPLITS %d\n"
                "#define PAD_ROWS %d\n#define N_GROUP %d\n"
                "#define X_STRIDE (PAD_ROWS * INPUT_DIMS)\n"
                "#define Y_STRIDE (N_SPLITS * PAD_ROWS * OUTPUT_DIMS)\n",
                c.K, c.cb, c.in, c.in / 16, c.out / 16, c.out, 8 * c.K, 32 * c.K, c.mb, c.nt, c.nt,
                c.splits, c.pad_rows, c.group);
  char name[176];
  std::snprintf(name, sizeof(name), "exl3_qmv_mat_%s", key.c_str());
  auto fn = make_qmv_kernel(name, kQmvMatSource, buf, c.cb, c.aligned, c.group);
  return g_mat_kernels.emplace(key, std::move(fn)).first->second;
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

static TrellisTuning& tuning_state(); // the knobs, defined with the tuning API below

namespace {

// The k-split count is a pure function of the tensor's own geometry (k_tiles,
// n_tiles) and must stay one: the split partials are summed by mx::sum over
// axis 0 in the caller, so a different split count is a different f32
// association and a different result. Only the thread mapping (nt) below is
// free to change without changing a single output bit.
int split_count_impl(int k_tiles, int n_tiles) {
  // Enough independent simdgroups to cover the machine's memory latency, as
  // long as every one of them still gets a few k-tiles.
  int warps = n_tiles * 4; // 4 simdgroups per output tile
  int splits = 1;
  while (warps * splits < 1280 && k_tiles / (splits * 2) >= 4) splits *= 2;
  return splits;
}

int split_count(int k_tiles, int n_tiles) {
  int forced = env_int("EXL3_QMV_SPLITS");
  if (forced > 0) {
    while (k_tiles / forced < 1) forced /= 2;
    return forced > 0 ? forced : 1;
  }
  return split_count_impl(k_tiles, n_tiles);
}

// One geometry for every entry point of this file: rows per threadgroup (mb),
// output tiles per threadgroup (nt), k-splits and the padded row count. nt and
// mb are pure scheduling and are proven inert by tests/glm5_gemv_nt_test.cpp
// (every (nt, mb) pair produces the same bytes as any other, and the grouped
// and both scale-stack forms equal the per-expert path); the k-split count is
// part of the arithmetic and is not this function's to choose.
//
// The accumulator array is acc[mb][nt][8] floats and a product above 4 spills
// it: measured 2.7x for {1,16} at 1 row - which is what the old
// "n_tiles/nt > 1024" rule picked for the lm_head's 9680 output tiles (224 GB/s
// against 476 at nt=2/4 on the real K6 tensor) - and 5.8x for {2,16} at 2 rows
// (36 GB/s against 387).
//
// nt>1 is also what keeps the SIMD pipes fed, and this is a scheduling change:
// a (16x16 output tile, k-tile) simdgroup-iteration retires ~110
// warp-instructions (8 codeword decodes and their 8 FMAs per lane, 4 packed-word
// loads, the window shift) and 393216 of them in 0.168 ms on the 4096x24576
// tensor is 256e9/s - the machine's whole issue width. The tile rate that comes
// out is 2.4e9 tiles/s (300 GB/s of 128-byte K4 tiles, i.e. 2.4e9 * K/2 bytes/s
// for any K) on the shapes whose grid is large enough, and it is the
// per-iteration activation load, its two shuffles and the loop's address math
// that nt shares. The small grids measure 2.0e9 tiles/s at nt=1 against 2.3e9 at
// nt=4 (the 8-member grouped launch of the 4096x2048 K3 experts: 185 against 219
// GB/s), while on the 4096x24576 K4 tensor nt=4 loses 17% (266 against 317 GB/s)
// because 384 threadgroups is one wave.
struct Geom {
  int mb, nt, sg, splits, pad_rows;
};

Geom pick_geom(int k_tiles, int n_tiles, int rows, int members = 1) {
  TrellisTuning tn = tuning_state();
  const int sg = tn.sg > 0 ? tn.sg : 4;
  // Rows per threadgroup: up to 4 a wider mb amortises the codebook decode over
  // more rows and wins (rows 2..26 all best at mb = min(rows,4)).
  int mb = tn.mb > 0 ? tn.mb : (rows >= 4 ? 4 : rows);
  if (mb > 4) mb = 4;
  // Tiles per threadgroup: the largest power of two that (a) keeps mb*nt inside
  // the accumulator register budget, (b) divides n_tiles (the kernel writes
  // tile_n + ot for every ot < nt without a bound check) and (c) still leaves
  // ~2048 simdgroups in flight - nt amortises the per-iteration activation
  // path, but it is paid for in threadgroups, and on the 4096x24576 K4 tensor
  // nt=4 leaves 384 threadgroups (one wave) and measures 17% slower than
  // nt=2's 768. Like mb, nt cannot move an output bit: the per-tile
  // accumulation order, the split count and the per-output k order are all
  // unchanged by which threadgroup covers which output tile.
  const int splits = split_count(k_tiles, n_tiles);
  int nt = 1;
  const int cap = 4 / mb > 0 ? 4 / mb : 1;
  if (tn.nt > 0) {
    nt = tn.nt;
  } else {
    for (int cand = 2; cand <= cap; cand *= 2) {
      if (n_tiles % cand != 0) continue;
      const int64_t in_flight = (int64_t)(n_tiles / cand) * splits * 4 * members;
      if (in_flight < 2048) break;
      nt = cand;
    }
  }
  while (nt > 1 && (n_tiles % nt != 0 || n_tiles / nt < 2)) nt /= 2;
  // Rows padded to a whole number of row blocks so the kernel needs no row
  // bound; large row counts are bucketed to limit kernel variants.
  int pad_rows = ((rows + mb - 1) / mb) * mb;
  if (pad_rows >= 64) pad_rows = ((pad_rows + 31) / 32) * 32;
  return Geom{mb, nt, sg, splits, pad_rows};
}

// ---------------------------------------------------------------------------
// The staged matrix path: gate, geometry, and the two QMV launches.
//
// The kernel is described above kQmvMatSource. What it changes is ONE thing
// besides scheduling: the k range inside a split is summed by the tensor unit's
// own tree instead of the QMV's (quad shuffle + per-lane partials), so a staged
// output is not byte-identical to a QMV one - it is the same decoded fp16
// weights and the same fp32 accumulator, read in a different association.
//
// The decision, made here and enforced by the gate below:
//
//   * The staged path is OPT-IN and GATED, never silent: it runs only when the
//     caller's tuning allows it AND the row count reaches the threshold
//     (`TrellisTuning.mat_min_rows`, default 16 or EXL3_QMV_MAT_MIN_ROWS). The
//     single-sequence path is rows = 1 and every request-level batch below 16
//     rows stays on the QMV, so the byte-exact kernel is what serves the serial
//     path, the B=1..8 batch cells and every prefill chunk narrower than the
//     threshold.
//   * The deviation is reported, not loosened: tests/glm5_gemv_nt_test.cpp
//     hashes both paths and prints the staged path's max|diff| against the QMV
//     for the real tensor classes, and tests/glm5_batch_test.cpp's token
//     identity gates (the hard ones) run with the staged path on at B=16.
//   * Bit-identity was not attempted: the tensor unit's 16-deep k reduction is
//     not expressible as the QMV's four-lane shuffle tree (the QMV's partials
//     are `quad_sum(a0+a1) + quad_sum(a2+a3)` per 8 stream positions, and its
//     per-tile k order is a 4-lane interleave), so reproducing the order would
//     mean not using the matrix units at all. The staged path also picks its own
//     k-split count (one warp per output tile needs a different one - see
//     mat_split_count), which is a second, smaller f32 association change of the
//     same kind: the split partials are still summed by the caller's mx::sum.
// ---------------------------------------------------------------------------

// Output tiles per threadgroup == simdgroups per threadgroup: simdgroup s
// stages tile s of the threadgroup and accumulates the same tile's columns.
//
// The default is 1 - a threadgroup is one warp and the per-k-tile barrier costs
// nothing. Measured across the 15 real tensor classes at 26 rows (median of 3,
// same session): 1 beats 4 by 5% (lm_head K6) to 41% (kda o_proj) and never
// loses; the barrier is what the extra tiles per threadgroup buy nothing for
// (the staging and the MACs of a tile are one warp's work either way, and the
// barrier serialises them across the threadgroup's warps).
// EXL3_QMV_MAT_NT overrides it for experiments; every value >= 1 is legal as
// long as it divides the tile count (checked by the gate).
constexpr int kMatNT = 1;

int mat_nt_knob() {
  const int v = env_int("EXL3_QMV_MAT_NT");
  return v > 0 ? v : kMatNT;
}

// Threadgroups the staged kernel aims for. It is LATENCY-bound, not issue-bound
// (its staging phase is a device load plus a dependent multiply-hash chain, with
// one barrier per k-tile after it), so the threadgroup count is the rate: on
// mla o_proj (16384x4096 K4, 256 output tiles, 16 rows) the isolated kernel
// measures 2 splits 30.8 GB/s, 4 -> 60.2, 8 -> 105.5, 16 -> 126.7, i.e. it wants
// thousands of threadgroups.
//
// ... and the STEP wants hundreds. The isolated kernel is not the right
// instrument for this knob: the same pair of builds measured end to end at 16
// rows (tests/glm5_batch_test, interleaved rounds) reads 0.83x with a 4096 target
// and 1.19x with 512, because what the split buys in the isolated kernel (more
// threadgroups) it pays for in the step (splits f32 output planes per linear,
// each summed by its own dispatch, and thousands of 32-thread threadgroups
// competing with the MoE's expert kernels for the SMs). 512 is the shipped
// number; EXL3_QMV_MAT_TG overrides it.
constexpr int kMatSplitTarget = 512;
// ... and a split still has to get enough k-tiles that the staging pipeline's
// prologue and drain do not dominate.
constexpr int kMatSplitMinTiles = 8;

int mat_split_count(int k_tiles, int n_tiles, int nt) {
  const int forced = env_int("EXL3_QMV_SPLITS");
  if (forced > 0) {
    int f = forced;
    while (k_tiles / f < 1) f /= 2;
    return f > 0 ? f : 1;
  }
  const int target = env_int("EXL3_QMV_MAT_TG") > 0 ? env_int("EXL3_QMV_MAT_TG") : kMatSplitTarget;
  const int64_t tiles = std::max(1, n_tiles / nt);
  int splits = 1;
  while (tiles * splits < target && k_tiles / (splits * 2) >= kMatSplitMinTiles) splits *= 2;
  return splits;
}

struct MatGeom {
  int mb, nt, splits, pad_rows;
};

// A decode-shaped forward: a 3-D (or higher) activation whose sequence axis is
// one, so its row axis is the batch. The request-level batch passes (B, 1,
// hidden) and is the only shape that qualifies; the benchmarks pass (rows, 1,
// in) for the same reason.
//
// The scope has to be this narrow, and it is why it is a *shape* rule rather
// than a row count alone: a prefill reaches the same row counts through several
// shapes - (1, L, hidden) for the trunk, and (tokens_per_expert, hidden) flat 2-D
// for the general MoE's per-expert path (measured: the first forward of
// tests/glm5_forward_test, a 23-token prefill, dispatches the routed experts at
// 16-21 rows that way). Those must stay on the byte-exact QMV - the ticket's
// hard gate is that the single-sequence path, its prefill included, is
// bit-identical - and they are the right calls to keep there anyway: the staged
// kernel would stream the weights rows/MAT_MB times over a prefill, where the
// answer is a K-blocked GEMM over the batch.
bool decode_shaped(const mx::array& x) {
  return x.ndim() >= 3 && x.shape(x.ndim() - 2) == 1;
}

bool mat_path_ok(int rows, const TrellisWeight& w) {
  const TrellisTuning tn = tuning_state();
  if (tn.mat_min_rows <= 0 || rows < tn.mat_min_rows) return false;
  if (tn.mat_max_rows > 0 && rows > tn.mat_max_rows) return false;
  if (w.codebook != 1 && w.codebook != 2 && w.codebook != 3) return false;
  if (w.in_features < 16 || w.in_features % 16 != 0) return false;
  // The kernel writes MAT_NT output tiles per threadgroup with no bound check,
  // exactly like the QMV's nt, so MAT_NT must divide the tile count. Every
  // tensor class of the checkpoint has an out_features that is a multiple of
  // 128, so this only excludes shapes nobody packs.
  if ((w.out_features / 16) % mat_nt_knob() != 0) return false;
  return true;
}

MatGeom pick_mat_geom(int k_tiles, int n_tiles, int rows) {
  // Rows per threadgroup: 16 is the narrowest width a matrix op pays for, 32
  // the widest whose accumulator array (MAT_MB/8 x 2 fragments, 2 f32 per lane
  // each) still sits comfortably beside the staging loop's live values. A wider
  // block stages the tile once for more rows, which is the whole point.
  int mb = rows > 16 ? 32 : 16;
  // Output tiles per threadgroup == simdgroups per threadgroup. Both are
  // experimentation knobs (EXL3_QMV_MAT_MB / EXL3_QMV_MAT_NT), never load-bearing
  // for the arithmetic: the staged tile a simdgroup owns and the accumulator
  // fragments it keeps are the same work in a different arrangement.
  const int nt = mat_nt_knob();
  if (env_int("EXL3_QMV_MAT_MB") > 0) mb = env_int("EXL3_QMV_MAT_MB");
  if (mb < 8) mb = 8;
  if (mb > 32) mb = 32;
  // The staged kernel's own k-split count: one warp per output tile means the
  // QMV's count (tuned for `n_tiles * 4` warps) leaves it with a quarter of the
  // threadgroups it needs (see mat_split_count).
  const int splits = mat_split_count(k_tiles, n_tiles, nt);
  int pad_rows = ((rows + mb - 1) / mb) * mb;
  if (pad_rows >= 64) pad_rows = ((pad_rows + 31) / 32) * 32;
  return MatGeom{mb, nt, splits, pad_rows};
}

// Pad the activation's row axis (axis -2, which is axis 0 of the flat 2-D view a
// fused single-tensor call passes in) up to `pad_rows` with zeros, so no kernel
// needs a row bound.
mx::array pad_rows_zeros(const mx::array& x, int rows, int pad_rows) {
  if (pad_rows <= rows) return x;
  mx::Shape zshape(x.shape().begin(), x.shape().end());
  zshape[zshape.size() - 2] = pad_rows - rows;
  return mx::concatenate({x, mx::zeros(zshape, mx::float16)}, (int)zshape.size() - 2);
}

// The QMV launch: one member (the fused single-tensor form) or n (the grouped
// form). Returns (pad_rows, out) or (splits, pad_rows, out) fp32 for one member
// and the same with a leading n when grouped; the caller sums the split axis.
mx::array qmv_scalar_run(const std::vector<TrellisWeight>& ws, const mx::array& xin, const Geom& g) {
  const int n = (int)ws.size();
  const int in = ws[0].in_features, out = ws[0].out_features;
  const int n_tiles = out / 16;
  Config cfg{in, out, ws[0].K, ws[0].codebook, g.nt, g.sg, g.mb, g.splits, g.pad_rows, n,
             ws[0].aligned ? 1 : 0};
  std::vector<mx::Shape> shapes;
  if (n > 1) {
    shapes = g.splits > 1 ? std::vector<mx::Shape>{mx::Shape{n, g.splits, g.pad_rows, out}}
                          : std::vector<mx::Shape>{mx::Shape{n, g.pad_rows, out}};
  } else {
    shapes = g.splits > 1 ? std::vector<mx::Shape>{mx::Shape{g.splits, g.pad_rows, out}}
                          : std::vector<mx::Shape>{mx::Shape{g.pad_rows, out}};
  }
  std::vector<mx::array> args;
  args.reserve((size_t)n + 1);
  args.push_back(xin);
  for (const auto& w : ws) args.push_back(w.packed);
  std::lock_guard<std::mutex> lk(g_mutex); // kernel map only; never held while launching more
  const auto& kernel = kernel_for(cfg);
  auto outs = kernel(args, shapes, {mx::float32},
                     {n_tiles / g.nt * g.sg * 32, g.pad_rows / g.mb, g.splits * n},
                     {g.sg * 32, 1, 1},
                     std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                     false, mx::default_device());
  return outs[0];
}

// The same convention for the staged matrix kernel (one member only - its only
// callers are the single-tensor entry points; see matmul_group_impl for why the
// grouped forms stay on the QMV). Its own geometry is rows per threadgroup
// (mg.mb) and mg.nt output tiles per threadgroup.
mx::array qmv_mat_run(const TrellisWeight& w, const mx::array& xin, const MatGeom& mg) {
  const int out = w.out_features;
  const int n_tiles = out / 16;
  MatConfig cfg{w.in_features, out, w.K, w.codebook, mg.mb, mg.nt, mg.splits, mg.pad_rows, 1,
                w.aligned ? 1 : 0};
  std::vector<mx::Shape> shapes =
      mg.splits > 1 ? std::vector<mx::Shape>{mx::Shape{mg.splits, mg.pad_rows, out}}
                    : std::vector<mx::Shape>{mx::Shape{mg.pad_rows, out}};
  std::lock_guard<std::mutex> lk(g_mutex);
  const auto& kernel = mat_kernel_for(cfg);
  auto outs = kernel({xin, w.packed}, shapes, {mx::float32},
                     {n_tiles / mg.nt * mg.nt * 32, mg.pad_rows / mg.mb, mg.splits},
                     {mg.nt * 32, 1, 1},
                     std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt,
                     false, mx::default_device());
  return outs[0];
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
    // The staged path is on by default at the row counts a request-level batch
    // reaches (16, the width a simdgroup matrix op wants); EXL3_QMV_MAT_MIN_ROWS
    // moves or disables it (0), EXL3_QMV_MAT_MAX_ROWS caps it.
    const char* mat = std::getenv("EXL3_QMV_MAT_MIN_ROWS");
    out.mat_min_rows = mat ? std::atoi(mat) : 16;
    out.mat_max_rows = env_int("EXL3_QMV_MAT_MAX_ROWS");
    return out;
  }();
  return t;
}

void trellis_set_tuning(const TrellisTuning& t) { tuning_state() = t; }

TrellisTuning trellis_get_tuning() { return tuning_state(); }

TrellisPathInfo trellis_path_info(const TrellisWeight& w, int rows) {
  const int k_tiles = w.in_features / 16, n_tiles = w.out_features / 16;
  if (mat_path_ok(rows, w)) {
    const MatGeom mg = pick_mat_geom(k_tiles, n_tiles, rows);
    return TrellisPathInfo{true, mg.mb, mg.pad_rows, mg.pad_rows / mg.mb, mg.splits, mg.nt};
  }
  const Geom g = pick_geom(k_tiles, n_tiles, rows);
  return TrellisPathInfo{false, g.mb, g.pad_rows, g.pad_rows / g.mb, g.splits, g.nt};
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

// EXL3_QMV_MAT_DEBUG=1 traces every dispatch: which path, and the activation's
// shape that chose it (the shape is the whole scope rule, so a mis-scoped call
// - a prefill on the staged path - shows up here by name).
static void mat_debug(const mx::array& x, int rows, int in, int out, bool mat) {
  if (!std::getenv("EXL3_QMV_MAT_DEBUG")) return;
  std::fprintf(stderr, "MAT rows=%d in=%d out=%d ndim=%zu shape=", rows, in, out,
               (size_t)x.ndim());
  for (auto d : x.shape()) std::fprintf(stderr, "%d,", d);
  std::fprintf(stderr, " -> %s\n", mat ? "STAGED" : "qmv");
}

mx::array trellis_matmul_raw(const TrellisWeight& w, const mx::array& xhat) {
  const int in = w.in_features;
  const int out = w.out_features;
  // The activations are (…, rows, in); a flat 2-D call is the common case and
  // any leading dimensions are restored on the way out, like trellis_linear.
  const int64_t rows = xhat.size() / in;
  if (rows * in != (int64_t)xhat.size()) {
    throw std::invalid_argument("trellis_matmul_raw: input feature size mismatch");
  }
  const int k_tiles = in / 16;
  const int n_tiles = out / 16;

  const bool mat = decode_shaped(xhat) && mat_path_ok((int)rows, w);
  mat_debug(xhat, rows, in, out, mat);
  const Geom g = pick_geom(k_tiles, n_tiles, rows);
  const MatGeom mg = pick_mat_geom(k_tiles, n_tiles, rows);
  // The path's own splits and padded rows: the caller's split sum and its slice
  // are part of the path, not of the tensor (the two geometries differ).
  const int splits = mat ? mg.splits : g.splits;
  const int pad_rows = mat ? mg.pad_rows : g.pad_rows;
  mx::array xin = pad_rows_zeros(xhat, (int)rows, pad_rows);

  mx::array y = mat ? qmv_mat_run(w, xin, mg) : qmv_scalar_run({w}, xin, g);
  if (splits > 1) y = mx::sum(y, std::vector<int>{0}, false);
  if (pad_rows > rows) y = mx::slice(y, {0, 0}, {(int)rows, out});
  if (xhat.ndim() != 2) {
    mx::Shape yshape(xhat.shape().begin(), xhat.shape().end());
    yshape[yshape.size() - 1] = out;
    y = mx::reshape(y, yshape);
  }
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
  // The grouped forms stay on the QMV, deliberately: their activation is
  // (n, rows, in) with the sequence axis already flattened into `rows`, so the
  // decode-shaped test (the scope of the staged path - see decode_shaped) cannot
  // be made, and the two callers that reach rows >= the threshold with members
  // are the routed experts of the model's MoE (rows = 1 per member, which the
  // threshold excludes) and moe_layer's general path, whose (K, B*L, H) shaped
  // activation includes the single-sequence prefill. Guessing there would take
  // the staged path into every prefill through the back door.
  const Geom g = pick_geom(k_tiles, n_tiles, rows, n);
  const int splits = g.splits, pad_rows = g.pad_rows;

  mx::array xin = pad_rows_zeros(x, rows, pad_rows);
  mx::array y = qmv_scalar_run(ws, xin, g);
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
  const bool mat = decode_shaped(x) && mat_path_ok((int)rows, w);
  mat_debug(x, (int)rows, in, out, mat);
  const Geom g = pick_geom(k_tiles, n_tiles, (int)rows);
  const MatGeom mg = pick_mat_geom(k_tiles, n_tiles, (int)rows);
  const int splits = mat ? mg.splits : g.splits;
  const int pad_rows = mat ? mg.pad_rows : g.pad_rows;
  mx::array xin = pad_rows_zeros(xhat, (int)rows, pad_rows);

  // The k-split partials are summed in f32 exactly as before (MLX's reduction
  // order is its own; reordering it here moved the model's logits), then
  // y = (H128 y / sqrt(128)) . svh with the f16 rounding inside the kernel.
  mx::array y = mat ? qmv_mat_run(w, xin, mg) : qmv_scalar_run({w}, xin, g);
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
