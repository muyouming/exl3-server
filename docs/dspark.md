# DSpark speculative decoding for MiniCPM5-2B (plan + findings)

Why speculation is the only path past ~200 tok/s on this class of hardware:

- Non-speculative decode must read every weight once per token:
  EXL3 4 bpw body (~0.99 GB) + 6 bpw `lm_head` (~0.20 GB) ≈ **1.2 GB/token**.
- Measured streaming bandwidth on the M4 Max (via a reduction kernel) is ~425 GB/s,
  so a *perfect* kernel could not exceed **~350 tok/s**, and our trellis GEMV kernels
  currently run at 180-365 GB/s per shape (load-instruction bound, not DRAM bound).
- Each verification pass, by contrast, reads the weights once for **k tokens**:
  with the published acceptance of ~5.5 tokens per round the traffic drops to
  ~0.35 GB/token, which puts 300+ tok/s within reach with the current kernels.

## The official draft

`openbmb/MiniCPM5-2B-DSpark` (Apache-2.0) is a draft checkpoint trained for exact
pairing with `MiniCPM5-2B` and its tokenizer.

| Property | Value |
|---|---|
| Draft layers | 5 (Qwen3-style: GQA 16/2 heads, head_dim 128, q/k RMSNorm, RoPE theta 5e6) |
| Draft parameters | 323.8 M (647 MB bf16, 62 tensors) |
| Tokens per draft pass | 7 (`block_size`) |
| Target hidden taps | layers `[1, 10, 20, 30, 39]` (concatenated 5x2048 -> `fc` -> 2048 -> `hidden_norm`) |
| Markov head | `markov_w1 [130560, 256]` + `markov_w2 [130560, 256]`, rank 256 |
| Confidence head | `confidence_head.proj [1, 2304]` (hidden 2048 + markov 256) |
| Language head | none - the draft projects through the **target's** embedding/lm_head |
| Acceptance (greedy) | Math 6.05, Code 6.11, General 4.16, aggregate **5.52** tokens/step |

Draft block sampling (from the reference implementation in SGLang
`python/sglang/srt/models/dspark.py`, `DSparkDraftMixin` + `VanillaMarkov`):

1. one transformer pass over the block (anchor token + mask tokens, `mask_token_id`
   75982) produces `base_logits` for all 7 positions;
2. a cheap semi-autoregressive rollout over the block refines them:
   `logits_i = base_logits[:, i, :] + markov_w2 @ markov_w1[prev_token]`, argmax,
   with `prev_token` starting at the last accepted token;
3. the confidence head ranks positions so verification width can be scheduled
   (static verification is a supported fallback).

The draft's context KV is taken from the target's own hidden states (DFlash design:
`write_target_hidden_kv` maps tap layer -> draft layer), so the draft never runs over
the prompt.

## Status in this repository

- Target side: done. `Model::forward_multi()` returns the greedy token at every
  position plus per-position hidden taps; `Engine` runs a speculative loop
  (draft -> single verify pass -> longest matching prefix -> KV trim), greedy-only,
  with plain decoding as a fallback. Verified **lossless**: speculative output is
  token-identical to plain greedy.
- Drafter interface: `Drafter` + `NgramDrafter` (model-free prompt lookup). The
  n-gram drafter is lossless but only pays off on repetitive text; measured
  acceptance on general prompts is low, so it is opt-in (`--draft-ngram N`).
- Not yet ported: the DSpark draft itself (5 layers + `fc` projector + markov head;
  draft KV from the target taps). Expected cost per round: draft pass ~0.65 GB +
  verify pass ~1.2 GB for ~5.5 tokens.

## Speed accounting (M4 Max, 128 GB, MiniCPM5-2B-EXL3)

| Path | Decode | Weight bytes per token |
| --- | ---: | ---: |
| fp16 weights, MLX matmul (`--fp16-weights`) | ~50 tok/s | 4.5 GB |
| trellis-resident in-kernel decode (default) | ~105 tok/s | 1.2 GB |
| + DSpark speculation (projected, acceptance 5.5) | 250-400 tok/s | ~0.35 GB |
