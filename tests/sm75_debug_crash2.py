"""Sweep the contiguous varlen+seqused_k crash: which n, causal, nq?"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(7)
dev = "cuda:0"
H, HK, D = 8, 2, 128
scale = D ** -0.5

nq, causal = int(sys.argv[1]), sys.argv[2] == "1"
for n in (16, 64, 65, 128, 192, 256, 300):
    q = torch.randn(nq, H, D, device=dev).half()
    k = torch.randn(n, HK, D, device=dev).half()
    v = torch.randn(n, HK, D, device=dev).half()
    cuq = torch.tensor([0, nq], dtype=torch.int32, device=dev)
    seqused = torch.tensor([n], dtype=torch.int32, device=dev)
    try:
        out = flash_attn_varlen_func(
            q, k, v, max_seqlen_q=nq, cu_seqlens_q=cuq, max_seqlen_k=n,
            seqused_k=seqused, softmax_scale=scale, causal=causal,
            fa_version=2)
        torch.cuda.synchronize()
        print(f"nq={nq} causal={causal} n={n}: OK")
    except Exception:
        print(f"nq={nq} causal={causal} n={n}: CRASH")
        break
