"""V=identity probe: reveal which V rows the P*V GEMM actually reads.

With q=0 the softmax is uniform (1/N per key). With v[n, :, d] = (n == d)
the output out[q, h, d] equals the effective weight the kernel puts on key d.
Correct kernel: 1/64 = 0.015625 everywhere. Deviations map the mis-access.
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
v = torch.zeros(N, HK, D, device=dev).half()
for n in range(N):
    v[n, :, n] = 1.0

cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)
out = flash_attn_varlen_func(
    q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
    cu_seqlens_k=cuk, softmax_scale=scale, causal=False, fa_version=2)

w = out.cpu().float()[:, :, :N] * N  # normalize: correct => 1.0 each
print("multiplicity map, query row 0, head 0 (correct = all 1.0):")
for base in range(0, N, 16):
    print(" ", [round(x, 2) for x in w[0, 0, base:base + 16].tolist()])
print("query row 1, head 0:")
for base in range(0, N, 16):
    print(" ", [round(x, 2) for x in w[1, 0, base:base + 16].tolist()])
print("row 0 head 1:", [round(x, 2) for x in w[0, 1, :16].tolist()])
same = all(torch.allclose(w[i, j], w[0, 0], atol=0.01)
           for i in range(NQ) for j in range(H))
print("identical pattern across all rows/heads:", same)
