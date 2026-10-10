#pragma once
// EXL3 (exllamav3) serialized trellis format.
//
// Storage, per quantized linear:
//   {key}.trellis  i16  [in/16, out/16, 16*K]   packed 16x16 tiles, K bits per weight
//   {key}.suh      f16  [in]                    input-side scale vector
//   {key}.svh      f16  [out]                   output-side scale vector
//   {key}.mul1     i32  scalar marker           selects the mul1 procedural codebook
//   {key}.mcg      i32  scalar marker           selects the mcg procedural codebook
//
// Tile layout: one tile holds 256 weights as a tail-biting ring of 256*K bits. Within
// a stored 32-bit word the two 16-bit halves are swapped (upstream SWAP16 at pack time).
// The state window for stream position p is the 16 bits ending at ring bit (p+1)*K, and
// p maps to a row-major tile element through the tensor-core permutation (upstream
// tensor_core_perm). Values are decoded by a procedural codebook, then rotated:
//
//   W = diag(svh) . (H128 . W_hat . H128) / 128 . diag(suh)
//
// with H128 the Sylvester Hadamard matrix applied independently to every 128-row block
// from the left and every 128-column block from the right (upstream had_k = had_n = 128,
// scale 1/sqrt(128) per side).

#include <cstdint>

namespace exl3 {

// Procedural codebooks; values match upstream `cb` indices.
enum class Codebook { k3Inst = 0, McG = 1, Mul1 = 2 };

struct Exl3Tensor {
  const uint16_t* trellis = nullptr; // raw i16 bit patterns
  int K = 0;                         // bits per weight (16*K words per tile)
  Codebook codebook = Codebook::Mul1;
  int64_t in_features = 0;  // rows (multiple of 128, Hadamard requirement)
  int64_t out_features = 0; // cols (multiple of 128)
  const uint16_t* suh = nullptr; // f16 [in_features]
  const uint16_t* svh = nullptr; // f16 [out_features]
};

// Decode a single 16-bit trellis state to an fp16 bit pattern.
uint16_t codebook_value(uint16_t state, Codebook cb);

// Decode trellis values only (rotated basis), f16 row-major [in, out].
void decode_inner(const Exl3Tensor& t, uint16_t* out, int threads = 0);

// Full reconstruction into f16 row-major [in_features, out_features].
// threads = 0 picks a hardware default.
void reconstruct(const Exl3Tensor& t, uint16_t* out, int threads = 0);

} // namespace exl3
