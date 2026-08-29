"""Bit probes on the dim index: v[n,:,d] = bit_b(d), q=0 (uniform P).

Correct kernel: out[d] = bit_b(d) exactly (0.0 or 1.0). Fractional values
reveal a row-dependent dim permutation delta(n, d).
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

dimidx = torch.arange(D, device=dev)
for b in range(7):
    bits = ((dimidx >> b) & 1).float().view(1, 1, D).expand(N, HK, D)
    v = bits.half().contiguous()
    out = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
        fa_version=2).cpu().float()[0, 0]
    expect = ((torch.arange(D) >> b) & 1).float()
    bad = (out - expect).abs() > 0.01
    frac = out[bad]
    print(f"bit {b}: wrong dims {bad.sum().item():3d}  "
          f"first wrong: {[(d, round(out[d].item(), 3)) for d in bad.nonzero().flatten().tolist()[:6]]}")
