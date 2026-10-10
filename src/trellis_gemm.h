#pragma once
// Trellis-resident linear layers: instead of materializing fp16 weights, the
// packed EXL3 trellis is kept in memory and decoded inside a custom Metal GEMV
// kernel, so a decode step touches ~4x fewer bytes than an fp16 weight path.
//
//   y = x @ (diag(svh) . H128 . W_hat . H128 . diag(suh))
//
// is evaluated as:  xhat = ((x . suh) H128 / sqrt(128))  [MLX hadamard_transform]
//                   yhat = xhat @ W_hat                  [custom Metal kernel]
//                   y    = ((yhat H128 / sqrt(128)) . svh)

#include <mlx/mlx.h>

#include <string>
#include <vector>

namespace exl3 {

namespace mx = mlx::core;

struct TrellisWeight {
  // The packed trellis words as little-endian BYTES: one uint8 element per
  // byte, shaped (in/16, out/16, 16*K) - i.e. the raw file layout of the
  // tensor (each 32-bit word is two u16 halves, word i = b[4i] | b[4i+1]<<8 |
  // b[4i+2]<<16 | b[4i+3]<<24).
  //
  // Byte granularity is what makes a zero-copy mmap view safe: MLX divides a
  // buffer offset by the new element size when an array is *viewed* as a wider
  // dtype, and a safetensors tensor does not have to start on a 4-byte
  // boundary (the shard header length is arbitrary), so viewing the file bytes
  // as uint32 silently reads up to 3 bytes early.
  mx::array packed = mx::array(0.0f); // uint8 (in/16, out/16, 16*K)
  mx::array suh = mx::array(0.0f);    // f16 (in)
  mx::array svh = mx::array(0.0f);    // f16 (out)
  int in_features = 0;
  int out_features = 0;
  int K = 4;
  int codebook = 2; // mul1

  int64_t bytes() const {
    return (int64_t)packed.size() + 2 * ((int64_t)in_features + out_features);
  }
};

// y = x @ W in fp16, x is (rows, in_features).
mx::array trellis_linear(const TrellisWeight& w, const mx::array& x);

// Raw kernel output in the rotated basis: yhat = xhat @ W_hat (rows, out), fp32.
// xhat must already contain ((x . suh) H128 / sqrt(128)). Used by tests.
mx::array trellis_matmul_raw(const TrellisWeight& w, const mx::array& xhat);

// Tuning knobs (also used by the CLI/benchmark): output tiles and simdgroups per
// threadgroup, and rows per threadgroup block for batched prefill.
struct TrellisTuning {
  int nt = 0;   // 0 = pick automatically
  int sg = 0;   // 0 = pick automatically
  int mb = 0;   // 0 = pick automatically
};
void trellis_set_tuning(const TrellisTuning& t);
TrellisTuning trellis_get_tuning();

} // namespace exl3
