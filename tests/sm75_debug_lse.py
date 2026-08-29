"""Test: extra weight on (m=0, v-row0) should be 3*ln(N) if LSE leaks in."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(7)
dev = "cuda:0"
H, HK, D = 8, 2, 128
NQ = 16
scale = D ** -0.5

for N in (16, 32, 64, 128, 256):
    q = torch.zeros(NQ, H, D, device=dev).half()
    k = torch.randn(N, HK, D, device=dev).half()
    v = torch.zeros(N, HK, D, device=dev)
    v[0] = 1.0
    cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
    cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)
    out = flash_attn_varlen_func(
        q, k, v.half(), max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
        fa_version=2).cpu().float()
    extra = out[0, 0, 0].item() * N - 1.0
    print(f"N={N:3d}: extra={extra:7.3f}   3*ln(N)={3 * math.log(N):7.3f}   "
          f"2*ln(N)={2 * math.log(N):6.3f}   ln(N)={math.log(N):6.3f}")
