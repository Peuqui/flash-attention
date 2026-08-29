"""Does the m=0 corruption depend on M-tile padding (nq vs kBlockM)?"""
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
v = torch.zeros(N, HK, D, device=dev)
v[0] = 1.0
v = v.half()
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)

for nq in (1, 2, 8, 16, 17, 32, 64, 100, 128, 200, 256):
    q = torch.zeros(nq, H, D, device=dev).half()
    cuq = torch.tensor([0, nq], dtype=torch.int32, device=dev)
    out = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=nq, cu_seqlens_q=cuq, max_seqlen_k=N,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
        fa_version=2).cpu().float()
    w = out[:, 0, 0] * N  # weight of v-row0 per query row, correct 1.0
    bad = [(m, round(w[m].item(), 2)) for m in range(nq)
           if abs(w[m] - 1.0) > 0.05]
    print(f"nq={nq:3d}: bad query rows: {bad if bad else 'none'}")
