#include "exl3.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "fp16.h"
#include "util.h"

namespace exl3 {

// ---------------------------------------------------------------------------
// Procedural codebooks (upstream quant/codebook.cuh).
// ---------------------------------------------------------------------------

// 16-bit codebook index -> fp16 bit pattern, as a 65536-entry table.
static std::vector<uint16_t> build_lut(Codebook cb) {
  std::vector<uint16_t> lut(65536);
  const double k_inv = fp16::from_bits(0x1eee).to_double(); // 1/147.7
  const double k_bias = fp16::from_bits(0xc931).to_double(); // -10.39
  for (uint32_t s = 0; s < 65536; ++s) {
    fp16 v;
    if (cb == Codebook::Mul1) {
      uint32_t x = s * 0x83DCD12Du; // wraps mod 2^32, as in CUDA
      uint32_t sum = 0x6400u + (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24);
      // sum is an exact integer 1024..2044; the fused multiply-add rounds once.
      v = fp16::from_double_rne(fp16::from_bits((uint16_t)sum).to_double() * k_inv + k_bias);
    } else {
      uint32_t x = cb == Codebook::McG ? s * 0xCBAC1FEDu : s * 89226354u + 64248484u;
      x = (x & 0x8fff8fffu) ^ 0x3b603b60u; // lop3
      // __hadd of the two fp16 halves (single rounding of the exact sum).
      v = fp16::from_double_rne(fp16::from_bits((uint16_t)x).to_double() +
                                fp16::from_bits((uint16_t)(x >> 16)).to_double());
    }
    lut[s] = v.to_bits();
  }
  return lut;
}

static const uint16_t* lut_for(Codebook cb) {
  static std::once_flag once[3];
  static std::vector<uint16_t> luts[3];
  int i = (int)cb;
  std::call_once(once[i], [&] { luts[i] = build_lut(cb); });
  return luts[i].data();
}

uint16_t codebook_value(uint16_t state, Codebook cb) { return lut_for(cb)[state]; }

// ---------------------------------------------------------------------------
// Packed bit stream.
//
// `stored` is the raw u16 array of one tile (16*K words). Reading it as 32-bit
// words (little-endian) recovers the upstream kernel's `ptr`: word i holds the
// two swapped halves, i.e. ptr[i] = stored[2i] | stored[2i+1] << 16, and the
// ring wraps at 8*K words. Position p's state is the 16-bit window ending at
// ring bit (p+1)*K, read with a funnel shift over the merged 64-bit window.
// ---------------------------------------------------------------------------

static inline uint32_t read_u32(const uint16_t* stored, int K, int64_t i) {
  const int64_t words = 8 * (int64_t)K;
  int64_t w = i % words;
  if (w < 0) w += words;
  return (uint32_t)stored[2 * w] | ((uint32_t)stored[2 * w + 1] << 16);
}

static inline uint16_t tile_state(const uint16_t* stored, int K, int p) {
  const int b0 = p * K + K - 16 + 256 * K; // start of the 16-bit window
  const int b1 = b0 + 16;                  // end of the window
  const int i0 = b0 / 32;
  const int i1 = (b1 - 1) / 32;
  const int shift = (i1 + 1) * 32 - b1;
  const uint64_t merged = ((uint64_t)read_u32(stored, K, i0) << 32) | read_u32(stored, K, i1);
  return (uint16_t)(merged >> shift);
}

// Stream position -> row-major element inside the 16x16 tile (upstream
// tensor_core_perm): p = lane*8 + i, row = 2*(lane%4) + [0,1,8,9][i%4],
// col = lane/4 + 8*(i/4).
static const int32_t* perm_table() {
  static int32_t pm[256];
  static std::once_flag once;
  std::call_once(once, [] {
    for (int lane = 0; lane < 32; ++lane) {
      const int r0 = (lane % 4) * 2;
      const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
      const int c0 = lane / 4;
      for (int i = 0; i < 4; ++i) {
        pm[lane * 8 + i] = rows[i] * 16 + c0;
        pm[lane * 8 + 4 + i] = rows[i] * 16 + c0 + 8;
      }
    }
  });
  return pm;
}

static void decode_tile(const uint16_t* stored, int K, Codebook cb, const uint16_t* lut,
                        const int32_t* pm, uint16_t* dst, int64_t dst_stride) {
  for (int p = 0; p < 256; ++p) {
    const int e = pm[p];
    dst[(e >> 4) * dst_stride + (e & 15)] = lut[tile_state(stored, K, p)];
  }
}

void decode_inner(const Exl3Tensor& t, uint16_t* out, int threads) {
  if (t.in_features % 16 || t.out_features % 16 || t.K < 1 || t.K > 8) {
    std::fprintf(stderr, "exl3: unsupported tensor geometry (%lld x %lld, K=%d)\n",
                 (long long)t.in_features, (long long)t.out_features, t.K);
    std::abort();
  }
  const int64_t rows = t.in_features / 16, cols = t.out_features / 16;
  const int64_t words = 16 * (int64_t)t.K;
  const uint16_t* lut = lut_for(t.codebook);
  const int32_t* pm = perm_table();
  const int nt = threads > 0 ? threads : default_threads();
  parallel_for(rows * cols, nt, [&](int64_t idx) {
    const int64_t rk = idx / cols, rc = idx % cols;
    decode_tile(t.trellis + (rk * cols + rc) * words, t.K, t.codebook, lut, pm,
                out + rk * 16 * t.out_features + rc * 16, (int64_t)t.out_features);
  });
}

// ---------------------------------------------------------------------------
// Hadamard rotations and scale folding.
//
// W = diag(svh) . (H128 . W_hat . H128) . diag(suh) / 128, with H128 applied per
// 128-row block from the left and per 128-column block from the right. Both sides
// are accumulated in fp32 (a 128-row block is staged at a time, so the working set
// stays at 128 * out_features floats) and rounded to fp16 once at the end.
// ---------------------------------------------------------------------------

static constexpr int kHad = 128;
static const float kHadScale = 1.0f / 11.313708498984761f; // 1/sqrt(128)

// In-place radix-2 butterfly over a contiguous run of 128 floats (Sylvester order).
static inline void had128(float* x) {
  for (int len = 1; len < kHad; len <<= 1) {
    for (int i = 0; i < kHad; i += 2 * len) {
      for (int j = i; j < i + len; ++j) {
        const float a = x[j], b = x[j + len];
        x[j] = a + b;
        x[j + len] = a - b;
      }
    }
  }
}

static void finish_rotations(uint16_t* w, int64_t in_f, int64_t out_f, const uint16_t* suh,
                            const uint16_t* svh, int threads) {
  const int64_t blocks = in_f / kHad;
  const int64_t col_blocks = out_f / kHad;
  parallel_for(blocks, threads, [&](int64_t b) {
    std::vector<float> buf((size_t)kHad * (size_t)out_f);
    uint16_t* base = w + b * kHad * out_f;
    for (int i = 0; i < kHad; ++i) {
      const uint16_t* src = base + (int64_t)i * out_f;
      float* dst = buf.data() + (size_t)i * out_f;
      for (int64_t c = 0; c < out_f; ++c) dst[c] = (float)fp16::from_bits(src[c]).to_double();
    }
    // Left: butterfly across the 128 rows, then scale rows by suh / sqrt(128).
    for (int len = 1; len < kHad; len <<= 1) {
      for (int i = 0; i < kHad; i += 2 * len) {
        for (int j = i; j < i + len; ++j) {
          float* ra = buf.data() + (size_t)j * out_f;
          float* rb = buf.data() + (size_t)(j + len) * out_f;
          for (int64_t c = 0; c < out_f; ++c) {
            const float a = ra[c], d = rb[c];
            ra[c] = a + d;
            rb[c] = a - d;
          }
        }
      }
    }
    for (int i = 0; i < kHad; ++i) {
      const float s = (float)fp16::from_bits(suh[b * kHad + i]).to_double() * kHadScale;
      float* r = buf.data() + (size_t)i * out_f;
      for (int64_t c = 0; c < out_f; ++c) r[c] *= s;
    }
    // Right: per row and column block, butterfly then scale by svh / sqrt(128).
    for (int i = 0; i < kHad; ++i) {
      float* r = buf.data() + (size_t)i * out_f;
      for (int64_t cb = 0; cb < col_blocks; ++cb) {
        float* seg = r + cb * kHad;
        had128(seg);
        for (int j = 0; j < kHad; ++j) {
          const float s = (float)fp16::from_bits(svh[cb * kHad + j]).to_double() * kHadScale;
          seg[j] *= s;
        }
      }
      uint16_t* dst = base + (int64_t)i * out_f;
      for (int64_t c = 0; c < out_f; ++c) dst[c] = fp16::from_double_rne((double)r[c]).to_bits();
    }
  });
}

void reconstruct(const Exl3Tensor& t, uint16_t* out, int threads) {
  if (t.in_features % kHad || t.out_features % kHad) {
    std::fprintf(stderr, "exl3: %lld x %lld is not 128-divisible; cannot apply Hadamard rotation\n",
                 (long long)t.in_features, (long long)t.out_features);
    std::abort();
  }
  const int nt = threads > 0 ? threads : default_threads();
  decode_inner(t, out, nt);
  finish_rotations(out, t.in_features, t.out_features, t.suh, t.svh, nt);
}

} // namespace exl3
