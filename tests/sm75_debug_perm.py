"""Extract the d-dependent row permutation pi_d(n) of the P*V read on sm75.

Make P one-hot at key n0 (q = 8*k[n0] -> dominant score), v[n,:,d] = n.
Then out[d] = pi_d(n0). Correct kernel: out[d] = n0 for every d.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(7)
dev = "cuda:0"
H, HK, D = 8, 2, 128
N = 64
scale = D ** -0.5

k = torch.randn(N, HK, D, device=dev).half()
v = (torch.arange(N, device=dev).float().view(N, 1, 1)
     .expand(N, HK, D) / 64.0).half()
cuq = torch.tensor([0, 1], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)

for n0 in (0, 1, 2, 3, 8, 9, 16, 33):
    q = (k[n0:n0 + 1, :, :].float().repeat_interleave(H // HK, 1) * 8).half()
    out = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=1, cu_seqlens_q=cuq, max_seqlen_k=N,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
        fa_version=2).cpu().float()[0, 0] * 64.0
    pi = [round(x) for x in out.tolist()]
    # compress: list dim ranges with constant pi
    runs = []
    start = 0
    for d in range(1, D + 1):
        if d == D or pi[d] != pi[start]:
            runs.append((start, d - 1, pi[start]))
            start = d
    print(f"n0={n0:2d}: " + " ".join(f"d{a}-{b}->{p}" for a, b, p in runs))
