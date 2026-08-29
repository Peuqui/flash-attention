"""Row-permutation probe with nq=16 (the geometry that fails)."""
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

k = torch.randn(N, HK, D, device=dev).half()
v = (torch.arange(N, device=dev).float().view(N, 1, 1)
     .expand(N, HK, D) / 64.0).half()
cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)

for n0 in (0, 1, 2, 3, 8, 9, 16, 33):
    q1 = (k[n0:n0 + 1, :, :].float().repeat_interleave(H // HK, 1) * 8).half()
    q = q1.expand(NQ, H, D).contiguous()
    out = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
        fa_version=2).cpu().float() * 64.0
    # per query row: which source row does dim 0 / dim 8 / dim 64 report?
    rows = [round(out[m, 0, 0].item()) for m in range(NQ)]
    d8 = [round(out[m, 0, 8].item()) for m in range(NQ)]
    same_dims = torch.allclose(out[:, 0, :].max(dim=1).values,
                               out[:, 0, :].min(dim=1).values, atol=0.6)
    print(f"n0={n0:2d}: per-queryrow dim0 {rows}  dim8 {d8}  "
          f"dim-const={bool(same_dims)}")
