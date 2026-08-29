"""Micro cases: find the smallest breaking geometry on sm75."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(7)
dev = "cuda:0"


def run(n, nq, h, hk, d, tag):
    scale = d ** -0.5
    q = torch.randn(nq, h, d, device=dev).half()
    k = torch.randn(n, hk, d, device=dev).half()
    v = torch.randn(n, hk, d, device=dev).half()
    cuq = torch.tensor([0, nq], dtype=torch.int32, device=dev)
    cuk = torch.tensor([0, n], dtype=torch.int32, device=dev)
    out = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=nq, cu_seqlens_q=cuq, max_seqlen_k=n,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False, fa_version=2)
    qf = q.cpu().float().transpose(0, 1).unsqueeze(0)
    kf = k.cpu().float().repeat_interleave(h // hk, dim=1).transpose(0, 1).unsqueeze(0)
    vf = v.cpu().float().repeat_interleave(h // hk, dim=1).transpose(0, 1).unsqueeze(0)
    ref = torch.nn.functional.scaled_dot_product_attention(
        qf, kf, vf, scale=scale).squeeze(0).transpose(0, 1)
    err = (out.cpu().float() - ref).abs().max().item()
    print(f"{tag}: n={n} nq={nq} h={h}/{hk} d={d}: max abs err {err:.4f}")
    return err


# keys sweep, one query row, hdim 128
for n in (1, 2, 4, 8, 16, 32, 64, 65, 128):
    run(n, 1, 8, 2, 128, "keys ")

# query rows sweep, few keys
for nq in (1, 2, 8, 16, 64):
    run(16, nq, 8, 2, 128, "query")

# head dims
for d in (32, 64, 96, 128):
    run(64, 16, 8, 2, d, "hdim ")

# MHA (no GQA)
run(64, 16, 4, 4, 128, "mha  ")

# score-vs-placement probe: v[j] = j (all dims) -> out = attn-weighted index
n, nq, h, hk, d = 32, 4, 8, 2, 128
scale = d ** -0.5
q = torch.randn(nq, h, d, device=dev).half()
k = torch.randn(n, hk, d, device=dev).half()
v = (torch.arange(n, device=dev).float().unsqueeze(-1).unsqueeze(-1)
     .expand(n, hk, d)).half()
cuq = torch.tensor([0, nq], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, n], dtype=torch.int32, device=dev)
out = flash_attn_varlen_func(
    q, k, v, max_seqlen_q=nq, cu_seqlens_q=cuq, max_seqlen_k=n,
    cu_seqlens_k=cuk, softmax_scale=scale, causal=False, fa_version=2)
qf = q.cpu().float().transpose(0, 1).unsqueeze(0)
kf = k.cpu().float().repeat_interleave(h // hk, dim=1).transpose(0, 1).unsqueeze(0)
vf = v.cpu().float().repeat_interleave(h // hk, dim=1).transpose(0, 1).unsqueeze(0)
ref = torch.nn.functional.scaled_dot_product_attention(
    qf, kf, vf, scale=scale).squeeze(0).transpose(0, 1)
print("indexV got row0 head0 dims0-3:", out.cpu().float()[0, 0, :4].tolist())
print("indexV ref row0 head0 dims0-3:", ref[0, 0, :4].tolist())
print("indexV max abs err:", (out.cpu().float() - ref).abs().max().item())
