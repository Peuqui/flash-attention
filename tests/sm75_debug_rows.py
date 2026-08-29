"""Localize wrong rows/heads in the sm75 standard fwd kernel."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(7)
dev = "cuda:0"
H, HK, D = 8, 2, 128
scale = D ** -0.5

n = 128
q = torch.randn(n, H, D, device=dev).half()
k = torch.randn(n, HK, D, device=dev).half()
v = torch.randn(n, HK, D, device=dev).half()
cu = torch.tensor([0, n], dtype=torch.int32, device=dev)

for causal in (True, False):
    out = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=n, cu_seqlens_q=cu, max_seqlen_k=n,
        cu_seqlens_k=cu, softmax_scale=scale, causal=causal, fa_version=2)
    qf = q.cpu().float().transpose(0, 1).unsqueeze(0)
    kf = k.cpu().float().repeat_interleave(H // HK, dim=1).transpose(0, 1).unsqueeze(0)
    vf = v.cpu().float().repeat_interleave(H // HK, dim=1).transpose(0, 1).unsqueeze(0)
    ref = torch.nn.functional.scaled_dot_product_attention(
        qf, kf, vf, scale=scale, is_causal=causal).squeeze(0).transpose(0, 1)
    diff = (out.cpu().float() - ref).abs()  # (n, H, D)
    print(f"causal={causal}: abs max {diff.max():.3f}, mean {diff.mean():.5f}")
    bad = (diff > 0.05).any(dim=-1)  # (n, H)
    rows = bad.any(dim=1).nonzero().flatten().tolist()
    print(f"  bad rows ({len(rows)}): {rows[:40]}")
    heads = bad.any(dim=0).nonzero().flatten().tolist()
    print(f"  bad heads: {heads}")
    if rows:
        r = rows[0]
        print(f"  row {r} per-head max abs: "
              f"{[round(diff[r, h].max().item(), 3) for h in range(H)]}")
