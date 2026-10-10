#!/usr/bin/env python3
"""Build a tiny GLM-5-Next shaped safetensors fixture (mixed EXL3 + plain
tensors) used to test src/glm5.cpp without the real 95 GB checkpoint.

Usage: python3 tools/make_glm5_fixture.py [out_dir=/tmp/glm5_fake]
"""
import json, numpy as np, struct, os
import sys
out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/glm5_fake'
os.makedirs(out, exist_ok=True)
H, NH, NKV, HD = 64, 2, 2, 8          # tiny geometry
NL, NE, MOE_INTER, INTER = 4, 2, 32, 48
cfg = {
 "architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next_text",
 "text_config": {
   "hidden_size": H, "num_hidden_layers": NL, "num_attention_heads": NH, "num_key_value_heads": NKV,
   "head_dim": HD, "intermediate_size": INTER, "vocab_size": 128, "rms_norm_eps": 1e-5,
   "bos_token_id": 1, "eos_token_id": [2, 3], "max_position_embeddings": 4096,
   "first_k_dense_replace": 1, "n_routed_experts": NE, "num_experts_per_tok": 1,
   "moe_intermediate_size": MOE_INTER, "n_shared_experts": 1, "routed_scaling_factor": 2.5,
   "swiglu_limit": 10.0, "norm_topk_prob": True, "q_lora_rank": 16, "kv_lora_rank": 8,
   "qk_head_dim": HD, "qk_nope_head_dim": HD, "qk_rope_head_dim": 0, "v_head_dim": HD,
   "index_n_heads": 2, "index_head_dim": 8, "index_topk": 16, "index_kpool": 4,
   "index_kpool_compress": True, "index_kpool_always_select_tail": True,
   "mhc": True, "hc_mult": 4, "hc_sinkhorn_iters": 20, "num_nextn_predict_layers": 1,
   "layer_types": ["linear_attention","linear_attention","deepseek_sparse_attention","linear_attention"],
   "mlp_layer_types": ["dense","sparse","sparse","sparse"],
 }}
json.dump(cfg, open(out+'/config.json','w'), indent=1)

T={}
def exl3(key, in_f, out_f, K=4):
    T[f'{key}.trellis']=np.zeros((in_f//16, out_f//16, 16*K), np.int16)
    T[f'{key}.suh']=np.full(in_f, 0.02, np.float16); T[f'{key}.svh']=np.full(out_f, 0.03, np.float16)
    T[f'{key}.mul1']=np.array([-2082680531], np.int32)
def plain(key, shape, dt=np.float16, transpose_to_out_in=True):
    T[f'{key}.weight']= (np.zeros(shape[::-1], dt) if transpose_to_out_in else np.zeros(shape, dt))
def vec(key, n, dt=np.float16): T[key]=np.zeros(n, dt)

P='model.language_model.'
vec(P+'embed_tokens.weight', 128*H, np.uint16)  # bf16 raw
vec(P+'norm.weight', H, np.uint16)
T['lm_head.trellis']=np.zeros((H//16, 8, 16*6), np.int16); T['lm_head.suh']=np.full(H, .02, np.float16)
T['lm_head.svh']=np.full(128, .03, np.float16); T['lm_head.mul1']=np.array([-2082680531], np.int32)
for i, (lt, mt) in enumerate(zip(cfg['text_config']['layer_types'], cfg['text_config']['mlp_layer_types'])):
    lp=f'{P}layers.{i}.'; K=4
    vec(lp+'input_layernorm.weight', H, np.uint16); vec(lp+'post_attention_layernorm.weight', H, np.uint16)
    T[lp+'hc_attn_base']=np.zeros(4*H-4*H//4, np.float32); T[lp+'hc_attn_fn']=np.zeros((4*H, H*4), np.float32)
    T[lp+'hc_attn_scale']=np.zeros(3, np.float32); T[lp+'hc_ffn_base']=np.zeros(4*H-4*H//4, np.float32)
    T[lp+'hc_ffn_fn']=np.zeros((4*H, H*4), np.float32); T[lp+'hc_ffn_scale']=np.zeros(3, np.float32)
    if lt=='linear_attention':
        a=lp+'linear_attn.'
        for nm,(inf,outf) in {'q_proj':(H,H),'k_proj':(H,H//2),'v_proj':(H,H),'b_proj':(H,2),'g_a_proj':(H,8),'g_b_proj':(8,4),'o_proj':(H,H)}.items(): exl3(a+nm, inf, outf, K)
        exl3(a+'forget_gate.f_a_proj', H, 8, K); exl3(a+'forget_gate.f_b_proj', 8, 2, K)
        T[a+'conv1d.weight']=np.zeros((6*H//2, 1, 4), np.float16); T[a+'conv1d.bias']=np.zeros(6*H//2, np.float16)
        vec(a+'o_norm.weight', H)
        T[a+'forget_gate.dt_bias']=np.zeros(2, np.float32); T[a+'forget_gate.A_log']=np.zeros(2, np.float32)
    else:
        a=lp+'self_attn.'
        exl3(a+'q_a_proj', H, 16, K); vec(a+'q_a_layernorm.weight', 16)
        exl3(a+'q_b_proj', 16, NH*HD, K); exl3(a+'kv_a_proj_with_mqa', H, 8, K); vec(a+'kv_a_layernorm.weight', 8)
        exl3(a+'kv_b_proj', 8, NKV*(HD+HD), K); exl3(a+'o_proj', NH*HD, H, K)
        ix=a+'indexer.'
        exl3(ix+'wq_b', H, 2*8, K)
        T[ix+'wk.weight']=np.zeros((8,H), np.float16); T[ix+'k_norm.weight']=np.zeros(8, np.float16)
        T[ix+'k_norm.bias']=np.zeros(8, np.float16); T[ix+'weights_proj.weight']=np.zeros((2,H), np.float16)
        T[ix+'index_kpool_compress_ape']=np.zeros((4,8), np.float32); T[ix+'index_kpool_compress_gate']=np.zeros((8,H), np.float16)
    m=lp+'mlp.'
    if mt=='sparse':
        T[m+'gate.weight']=np.zeros((NE,H), np.float16); T[m+'gate.e_score_correction_bias']=np.zeros(NE, np.float32)
        for pj,(inf,outf) in {'shared_experts.gate_proj':(H,INTER),'shared_experts.up_proj':(H,INTER),'shared_experts.down_proj':(INTER,H)}.items(): exl3(m+pj, inf, outf, K)
        for e in range(NE):
            ep=m+f'experts.{e}.'
            exl3(ep+'gate_proj', H, MOE_INTER, 2); exl3(ep+'up_proj', H, MOE_INTER, 2); exl3(ep+'down_proj', MOE_INTER, H, 2)
    else:
        exl3(m+'gate_proj', H, INTER, K); exl3(m+'up_proj', H, INTER, K); exl3(m+'down_proj', INTER, H, K)
# write safetensors
offset=0; hdr={}
blobs=[]
for k,v in T.items():
    a=np.ascontiguousarray(v)
    hdr[k]={'dtype':{np.dtype(np.float16):'F16',np.dtype(np.float32):'F32',np.dtype(np.int16):'I16',np.dtype(np.int32):'I32',np.dtype(np.uint16):'BF16'}[a.dtype],
            'shape':list(a.shape),'data_offsets':[offset, offset+a.nbytes]}
    blobs.append(a.tobytes()); offset+=a.nbytes
js=json.dumps(hdr).encode()
with open(out+'/model.safetensors','wb') as f:
    f.write(struct.pack('<Q', len(js))); f.write(js)
    for b in blobs: f.write(b)
print('fake checkpoint:', len(T), 'tensors,', round(offset/1e6,2), 'MB')
