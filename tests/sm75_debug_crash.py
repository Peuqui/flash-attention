"""Isolate the varlen decode crash: swapped-ngroups path vs plain."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(7)
dev = "cuda:0"
H, HK, D = 8, 2, 128
scale = D ** -0.5


def case(tag, nq, n, hk):
    try:
        q = torch.randn(nq, H, D, device=dev).half()
        k = torch.randn(n, hk, D, device=dev).half()
        v = torch.randn(n, hk, D, device=dev).half()
        cuq = torch.tensor([0, nq], dtype=torch.int32, device=dev)
        seqused = torch.tensor([n], dtype=torch.int32, device=dev)
        out = flash_attn_varlen_func(
            q, k, v, max_seqlen_q=nq, cu_seqlens_q=cuq, max_seqlen_k=n,
            seqused_k=seqused, softmax_scale=scale, causal=True,
            fa_version=2)
        torch.cuda.synchronize()
        print(f"{tag}: OK, out[0,0,0]={out[0, 0, 0].item():.4f}")
    except Exception as e:
        print(f"{tag}: CRASH {type(e).__name__}: {str(e)[:80]}")
        sys.exit(1)


# nq=2: no ngroups swap (swap requires seqlen_q==1)
case("nq=2 GQA  n=300 ", 2, 300, HK)
# nq=1 MHA (h==hk): no swap either (needs num_heads > num_heads_k)
case("nq=1 MHA  n=300 ", 1, 300, H)
# nq=1 GQA: the swapped path
case("nq=1 GQA  n=300 ", 1, 300, HK)
