"""Check the returned LSE: is row 0's softmax statistic already wrong?"""
import math
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
v = torch.randn(N, HK, D, device=dev).half()
cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)
out, lse = flash_attn_varlen_func(
    q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
    cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
    return_softmax_lse=True, fa_version=2)
print("lse shape:", tuple(lse.shape), " correct value ln(64) =",
      round(math.log(N), 4))
l = lse.cpu().float()
# expect (H, NQ) for varlen
for h in range(2):
    print(f"head {h} lse per query row:",
          [round(x, 3) for x in l[h][:16].tolist()])
