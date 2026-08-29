"""Map the effective V-read: which (row, dim) does the P*V GEMM really read?

q=0 -> uniform P. Probe 1: v[n,:,d] = n -> out[d] = mean source ROW per dim
(correct: 31.5 for every d). Probe 2: v[n,:,d] = d -> out[d] = mean source
DIM per dim (correct: d).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(7)
dev = "cuda:0"
H, HK, D = 8, 2, 128
N, NQ = 64, 16
scale = D ** -0.5

q = torch.zeros(NQ, H, D, device=dev).half()
k = torch.randn(N, HK, D, device=dev).half()
cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)


def fa(v):
    return flash_attn_varlen_func(
        q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
        fa_version=2).cpu().float()[0, 0]


v_row = (torch.arange(N, device=dev).float().view(N, 1, 1)
         .expand(N, HK, D) / 64.0).half()
out = fa(v_row) * 64.0
print("row probe (correct: 31.5 everywhere):")
print("  dims  0-15:", [round(x, 1) for x in out[:16].tolist()])
print("  dims 64-79:", [round(x, 1) for x in out[64:80].tolist()])
print("  min/max:", round(out.min().item(), 2), round(out.max().item(), 2))

v_dim = (torch.arange(D, device=dev).float().view(1, 1, D)
         .expand(N, HK, D) / 128.0).half()
out = fa(v_dim) * 128.0
print("dim probe (correct: out[d] = d):")
print("  dims  0-15:", [round(x, 1) for x in out[:16].tolist()])
print("  dims 16-31:", [round(x, 1) for x in out[16:32].tolist()])
print("  dims 64-79:", [round(x, 1) for x in out[64:80].tolist()])
err = (out - torch.arange(D).float()).abs()
bad = (err > 0.5).nonzero().flatten().tolist()
print(f"  wrong dims ({len(bad)}):", bad[:48])
