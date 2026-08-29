"""Algebraic probes: separate wrong-scores from wrong-P*V on sm75, d=128."""
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
cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)


def fa(q, k, v):
    return flash_attn_varlen_func(
        q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False, fa_version=2)


# Probe 1: q = 0 -> uniform softmax -> out must equal mean over keys of v
q = torch.zeros(NQ, H, D, device=dev).half()
k = torch.randn(N, HK, D, device=dev).half()
v = torch.randn(N, HK, D, device=dev).half()
out = fa(q, k, v)
ref = v.float().mean(dim=0)  # (HK, D)
err = (out.cpu().float() - ref.cpu().repeat_interleave(H // HK, 0)).abs().max()
print(f"P1 q=0 (uniform P * V): max abs err {err:.5f}")

# Probe 2: v = 1 -> out must be exactly 1 if P is normalized
q = torch.randn(NQ, H, D, device=dev).half()
v1 = torch.ones(N, HK, D, device=dev).half()
out = fa(q, k, v1)
err = (out.cpu().float() - 1.0).abs().max()
print(f"P2 v=1 (P normalized):  max abs err {err:.5f}")

# Probe 3: q = e_0 * c (only dim 0 nonzero) -> scores depend only on k[:, :, 0]
q3 = torch.zeros(NQ, H, D, device=dev).half()
q3[:, :, 0] = 4.0
out = fa(q3, k, v)
s = (4.0 * k.float()[:, :, 0]) * scale  # (N, HK)
p = s.T.softmax(dim=-1)  # (HK, N)
ref3 = torch.einsum("hn,nhd->hd", p, v.float())  # (HK, D)
err = (out.cpu().float() - ref3.cpu().repeat_interleave(H // HK, 0)).abs().max()
print(f"P3 q=c*e0 (scores via k dim0): max abs err {err:.5f}")

# Probe 4: like P3 but nonzero dim is 64 (second smem page)
q4 = torch.zeros(NQ, H, D, device=dev).half()
q4[:, :, 64] = 4.0
s = (4.0 * k.float()[:, :, 64]) * scale
p = s.T.softmax(dim=-1)
ref4 = torch.einsum("hn,nhd->hd", p, v.float())
out = fa(q4, k, v)
err = (out.cpu().float() - ref4.cpu().repeat_interleave(H // HK, 0)).abs().max()
print(f"P4 q=c*e64 (scores via k dim64): max abs err {err:.5f}")
