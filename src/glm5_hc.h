#pragma once
// GLM-5-Next mHC (manifold-constrained hyper-connections), the residual
// stream machinery wrapped around every attention and MLP sub-layer.
//
// Semantics pinned from the HF reference modeling_glm5_next.py
// (transformers/models/glm5_next/modeling_glm5_next.py):
//
//   class Glm5NextTextHyperConnection                              (line 218)
//     __init__                                                          (271-283)
//       N = hc_mult; the projection emits (2+N)*N = 24 weights:
//       fn    f32 [(2+N)*N, N*D]   (out, in) as stored on disk
//       base  f32 [(2+N)*N]
//       scale f32 [3]              one scale for pre / post / comb each
//     forward(hidden_streams) -> (post, comb, collapsed)                (285-329)
//       flattened = streams.view(B,S,-1).float()                          (300)
//       flattened = Glm5NextTextUnweightedRMSNorm(flattened)              (301)
//                   (x * rsqrt(mean(x^2) + eps) over the N*D axis, 209-214)
//       w = F.linear(flattened, fn.float())  ==  flattened @ fn.T         (303)
//       pre_w, post_w, comb_w = w.split([N, N, N*N])                      (305)
//       pre_b, post_b, comb_b = base.split([N, N, N*N])                   (306)
//       pre_scale, post_scale, comb_scale = scale.unbind(0)               (307)
//       comb_w/b are viewed (...,N,N) / (N,N) by row-major split           (309-310)
//       pre  = sigmoid(pre_w*pre_scale + pre_b) + hc_eps                  (313)
//       post = 2 * sigmoid(post_w*post_scale + post_b)                    (314)
//       comb = softmax(comb_w*comb_scale + comb_b, dim=-1) + hc_eps        (315)
//       comb = comb / (comb.sum(-2) + eps)                                (321)
//       repeat (hc_sinkhorn_iters - 1) times {                            (322-324)
//         comb = comb / (comb.sum(-1) + eps)     # row-normalise
//         comb = comb / (comb.sum(-2) + eps)     # column-normalise
//       }
//       collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(dtype)     (328)
//     NOTE: nothing is stored for the Sinkhorn iterations; they are a
//     config constant (hc_sinkhorn_iters, 20 for this checkpoint).
//
//   class Glm5NextTextHyperHead                                     (line 332)
//     forward: streams.mean(dim=2)  (unweighted collapse)              (335-336)
//
//   class Glm5NextTextDecoderLayer.forward                          (lines 1299+)
//     attention sub-layer, identical for the MLP sub-layer:
//       residual = hidden_states                       ([...,N,D])      (1312)
//       post, comb, h = self.attn_hc(hidden_states)                     (1313)
//       h = self.input_layernorm(h); h = self.self_attn(h)              (1315-1334)
//       hidden_states = post.to(dtype).unsqueeze(-1) * h.unsqueeze(-2)
//                     + matmul(comb.to(dtype).transpose(-1,-2), residual) (1335-1337)
//       -- note comb enters TRANSPOSED: out[i] = post_i*h + sum_j comb[j][i]*res[j]
//     ffn sub-layer repeats with ffn_hc / post_attention_layernorm / mlp (1339-1346)
//
//   model level: hidden = inputs_embeds.unsqueeze(2).expand(-1,-1,N,-1)  (1493)
//                logits = norm(hc_head(hidden))                           (1505)
//
// Usage (one hyper-connection per sub-layer; N = cfg.hc_mult, D = hidden_size):
//
//   G5HcConfig hc = g5_hc_config(cfg);
//   mx::array s = g5_hc_expand(embeds, hc.mult);      // [B,S,D] -> [B,S,N,D]
//   for (const G5Layer& l : layers) {
//     G5HcActivation a = g5_hc_prepare(s, l.hc_attn_base, l.hc_attn_fn,
//                                      l.hc_attn_scale, hc);
//     mx::array h = attn(rms_norm(a.collapsed, l.input_norm, eps));
//     s = g5_hc_mix(h, a.post, a.comb, s);
//     G5HcActivation f = g5_hc_prepare(s, l.hc_ffn_base, l.hc_ffn_fn,
//                                      l.hc_ffn_scale, hc);
//     h = mlp(rms_norm(f.collapsed, l.post_norm, eps));
//     s = g5_hc_mix(h, f.post, f.comb, s);
//   }
//   mx::array h = rms_norm(g5_hc_head(s), final_norm, eps);   // [B,S,D]
//
// Layouts (the reference uses [B,S]; any leading batch dims are supported):
//   streams  [..., N, D]                      dtype of `streams` is preserved
//   base     [(2+N)*N] f32   fn [(2+N)*N, N*D] f32 (out, in)  scale [3] f32
//   post     [..., N] f32    comb [..., N, N] f32   collapsed [..., D]

#include <mlx/mlx.h>

#include "model.h"

namespace exl3 {
namespace mx = mlx::core;

// Checkpoint hyper-connection constants (config: mhc, hc_mult, hc_sinkhorn_iters;
// hc_eps defaults to 1e-6 in the reference config).
struct G5HcConfig {
  int mult = 4;             // hc_mult: number of residual streams N
  int sinkhorn_iters = 20;  // hc_sinkhorn_iters
  float eps = 1e-6f;        // hc_eps
};

// Reads mult/sinkhorn_iters from the model config (eps is not in ModelConfig,
// the reference default is 1e-6).
G5HcConfig g5_hc_config(const ModelConfig& cfg, float eps = 1e-6f);

// Sinkhorn-Knopp on a strictly positive (..., N, N) matrix, exactly as the
// reference applies it to the post-softmax+eps combination matrix (321-324):
// one column normalisation, then `iters - 1` (row, column) pairs, each
// normalisation dividing by (sum + eps). The result is doubly stochastic up to
// the eps bias in the denominators.
mx::array g5_hc_sinkhorn(const mx::array& comb, int iters, float eps);

// The three hyper-connection weights produced for one sub-layer.
struct G5HcActivation {
  mx::array post = mx::array(0.0f);       // [..., N] f32,  2*sigmoid(...)      (314)
  mx::array comb = mx::array(0.0f);       // [..., N, N] f32, Sinkhorn-normalised (315, 321-324)
  mx::array collapsed = mx::array(0.0f);  // [..., D], streams dtype, pre-weighted sum (328)
};

// Computes (post, comb, collapsed) from the current residual streams, i.e. the
// whole Glm5NextTextHyperConnection.forward (300-328). "pre" is consumed here.
G5HcActivation g5_hc_prepare(const mx::array& streams, const mx::array& base,
                             const mx::array& fn, const mx::array& scale,
                             const G5HcConfig& cfg);

// Mixes a sub-layer output back into the streams (reference 1335-1337 / 1344-1346):
//   out[i] = post[i] * block_out + sum_j comb[j][i] * residual[j]
// block_out [..., D], post [..., N], comb [..., N, N], residual [..., N, D].
// post/comb are cast to block_out's dtype, all math in that dtype.
mx::array g5_hc_mix(const mx::array& block_out, const mx::array& post,
                    const mx::array& comb, const mx::array& residual);

// Expands a [..., D] tensor into `mult` identical streams [..., mult, D]
// (model level, reference 1493; materialised, like the reference's
// `.expand(-1,-1,N,-1).contiguous()`).
mx::array g5_hc_expand(const mx::array& x, int mult);

// Final unweighted collapse of the streams: mean over the stream axis
// (Glm5NextTextHyperHead, reference 335-336). Input [..., N, D] -> [..., D].
mx::array g5_hc_head(const mx::array& streams);

} // namespace exl3
