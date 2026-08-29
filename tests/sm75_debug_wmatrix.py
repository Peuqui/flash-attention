"""Recover the full effective weight matrix w[n, d] of the P*V product.

q=0 -> P uniform. V = one-hot at row n1 (all dims, all heads) => out[d]
= w[n1, d]. Correct: 1/64 everywhere. Print deviation pattern."""
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

w = torch.zeros(N, D)
for n1 in range(N):
    v = torch.zeros(N, HK, D, device=dev)
    v[n1] = 1.0
    out = flash_attn_varlen_func(
        q, k, v.half(), max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
        fa_version=2).cpu().float()
    w[n1] = out[0, 0] * N  # correct => 1.0

dev_mask = (w - 1.0).abs() > 0.05
print(f"cells != 1/N: {dev_mask.sum().item()} of {N * D}")
rows_bad = dev_mask.any(dim=1).nonzero().flatten().tolist()
print("rows with deviations:", rows_bad)
cols_bad = dev_mask.any(dim=0).nonzero().flatten().tolist()
print(f"dims with deviations ({len(cols_bad)}):", cols_bad[:40])
# print a few example rows compactly: unique values and where
for n1 in rows_bad[:6]:
    vals = w[n1]
    uniq = sorted(set(round(x, 2) for x in vals.tolist()))
    segs = []
    start = 0
    for d in range(1, D + 1):
        if d == D or abs(vals[d] - vals[start]) > 0.05:
            segs.append((start, d - 1, round(vals[start].item(), 2)))
            start = d
    print(f"row {n1}: uniq={uniq} segs={segs[:10]}")
