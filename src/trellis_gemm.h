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
  // b[4i+2]<<16 | b[4i+3]<<24). Byte granularity is what makes the four-byte
  // spelling of a word explicit; a wider-dtype view is not safe (MLX divides a
  // buffer offset by the new element size) and is exactly how the byte-stream
  // contract was first broken.
  //
  // `aligned` says whether the buffer's base address is a multiple of 4, i.e.
  // whether every tile word can be read with one aligned 32-bit load. It is
  // true for trellis_pack()'s output, and for a zero-copy view only when the
  // tensor's file offset happens to be 4-byte aligned (a safetensors header
  // length is arbitrary); the kernel otherwise assembles the word from its four
  // bytes, ~20% slower on the 4096x24576 K4 shape but identical in value.
  mx::array packed = mx::array(0.0f); // uint8 (in/16, out/16, 16*K)
  bool aligned = true;
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

// Materialise the packed trellis words into the buffer layout the kernels read.
//
// `words16` points at the raw i16 array of the on-disk tensor, shaped
// (in/16, out/16, 16*K): 8*K u32 ring words per 16x16 tile, each stored as two
// little-endian u16 (word i = b[4i] | b[4i+1]<<8 | b[4i+2]<<16 | b[4i+3]<<24).
// The returned uint8 array is (in/16, out/16, 32*K) with the same bytes but on
// a fresh, 16-byte-aligned, engine-owned allocation, so the kernel may read
// word j of a tile as an aligned 32-bit load at byte 4*j (a zero-copy view of a
// shard cannot promise that: a safetensors header has an arbitrary length, so a
// tensor can start at an odd byte offset - the byte-granularity contract is
// kept by *reading* every word as those four bytes, in this order).
mx::array trellis_pack(const uint16_t* words16, int k_tiles, int n_tiles, int K);

// Raw kernel output in the rotated basis: yhat = xhat @ W_hat (rows, out), fp32.
// xhat must already contain ((x . suh) H128 / sqrt(128)). Used by tests.
mx::array trellis_matmul_raw(const TrellisWeight& w, const mx::array& xhat);

// n linears in ONE launch, for tensors that share their shape (the 8 routed
// experts of a decode step, the per-layer projections of one layer...). The
// result is bit-identical to n calls to trellis_linear: each member keeps its
// own tile/row/split geometry and its own scale vectors, the members differ
// only in which weight buffer and which slice of the packed activation they
// read (grid.z = split + N_SPLITS*member).
//   ws      : the member weights (same in_features/out_features/K/codebook)
//   x       : for trellis_linear_group        - (rows, in) shared by all members
//             for trellis_linear_group_rows   - (n, rows, in), one block each
//   returns : (n, rows, out) f16
mx::array trellis_linear_group(const std::vector<TrellisWeight>& ws, const mx::array& x);
mx::array trellis_linear_group_rows(const std::vector<TrellisWeight>& ws, const mx::array& x,
                                    int rows);
mx::array trellis_matmul_group(const std::vector<TrellisWeight>& ws, const mx::array& xhat, int rows);

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
