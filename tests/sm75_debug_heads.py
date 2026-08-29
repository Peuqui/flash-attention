"""Head-mixing probe: v[n, h, d] = h. Correct: out[q, h_q, d] = h_q // group."""
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
v = (torch.arange(HK, device=dev).float().view(1, HK, 1)
     .expand(N, HK, D)).half()
cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)
out = flash_attn_varlen_func(
    q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
    cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
    fa_version=2).cpu().float()

for h in range(H):
    lo = out[0, h, :64]
    hi = out[0, h, 64:]
    print(f"q-head {h} (KV-head {h // (H // HK)}): "
          f"dims<64 -> {lo.min():.2f}..{lo.max():.2f}, "
          f"dims>=64 -> {hi.min():.2f}..{hi.max():.2f}")
