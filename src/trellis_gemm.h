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
  mx::array words = mx::array(0.0f); // uint32 (in/16, out/16, 8*K)
  mx::array suh = mx::array(0.0f);   // f16 (in)
  mx::array svh = mx::array(0.0f);   // f16 (out)
  int in_features = 0;
  int out_features = 0;
  int K = 4;
  int codebook = 2; // mul1

  int64_t bytes() const { return (int64_t)words.size() * 4 + 2 * ((int64_t)in_features + out_features); }
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
