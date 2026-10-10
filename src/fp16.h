#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

// Exact IEEE binary16 helpers, portable (no reliance on _Float16 arithmetic
// semantics). Used by the EXL3 procedural-codebook decoder, which requires
// fused single-rounding fp16 arithmetic.

struct fp16 {
  uint16_t bits;

  fp16() : bits(0) {}
  static fp16 from_bits(uint16_t b) { fp16 h; h.bits = b; return h; }
  uint16_t to_bits() const { return bits; }

  // Exact value as double.
  double to_double() const {
    unsigned s = bits >> 15, e = (bits >> 10) & 0x1F, m = bits & 0x3FF;
    double v;
    if (e == 0) v = (double)m * 0x1p-24;
    else if (e == 31) v = (m == 0) ? INFINITY : (double)NAN;
    else v = (1.0 + (double)m / 1024.0) * exp2((double)e - 15.0);
    return s ? -v : v;
  }

  // Round to nearest (ties to even) half from a double (e.g. a fused fma result).
  static fp16 from_double_rne(double x) {
    if (std::isnan(x)) return from_bits(0x7e00);
    if (x == 0.0) return from_bits(std::signbit(x) ? 0x8000 : 0x0000);
    if (std::isinf(x)) return from_bits(std::signbit(x) ? 0xFC00 : 0x7C00);
    const uint16_t sign = std::signbit(x) ? (uint16_t)0x8000 : (uint16_t)0x0000;
    const double m = std::fabs(x);
    if (m >= 65536.0) return from_bits(sign | 0x7C00); // overflow saturates to inf
    if (m < 0x1p-25) return from_bits(sign);           // 2^-25 ties to even -> +-0
    int e;
    const double frac = std::frexp(m, &e); // m = frac * 2^e with frac in [0.5, 1)
    if (e >= -13) {
      // Normal: value = (1 + man/1024) * 2^(exp-15) with exp = e + 14.
      const double scaled = frac * 2048.0; // in [1024, 2048)
      const double fl = std::floor(scaled);
      double r = fl;
      const double rem = scaled - fl;
      if (rem > 0.5 || (rem == 0.5 && std::fmod(r, 2.0) != 0.0)) r += 1.0;
      int exp = e + 14;
      if (r >= 2048.0) { // mantissa carry out
        r = 1024.0;
        exp += 1;
      }
      if (exp > 30) return from_bits(sign | 0x7C00);
      const uint32_t man = (uint32_t)r - 1024u;
      return from_bits((uint16_t)(sign | ((uint32_t)exp << 10) | man));
    } else {
      // Subnormal: value = man * 2^-24.
      const double scaled = m * 16777216.0; // m * 2^24
      const double fl = std::floor(scaled);
      double r = fl;
      const double rem = scaled - fl;
      if (rem > 0.5 || (rem == 0.5 && std::fmod(r, 2.0) != 0.0)) r += 1.0;
      if (r >= 1024.0) return from_bits(sign | 0x0400); // rounds up to the smallest normal
      return from_bits((uint16_t)(sign | (uint32_t)r));
    }
  }

  bool is_zero() const { return (bits & 0x7FFF) == 0; }
};

// bf16 (16-bit pattern) -> f16 bits, round to nearest even. bf16 is the top
// half of an f32, so widen through an f32 (exact) and round once to half.
inline uint16_t fp16_bits_from_bf16(uint16_t bf) {
  const uint32_t f32_bits = (uint32_t)bf << 16;
  float f;
  std::memcpy(&f, &f32_bits, sizeof(f));
  return fp16::from_double_rne((double)f).to_bits();
}
