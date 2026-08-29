"""Find the minimal V structure that breaks (q=0, uniform P), plus
determinism check (race suspicion)."""
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


def fa(v):
    return flash_attn_varlen_func(
        q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
        cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
        fa_version=2).cpu().float()


def check(tag, v):
    out = fa(v)
    ref = v.float().mean(dim=0).repeat_interleave(H // HK, 0).cpu()
    err = (out - ref).abs().max().item()
    print(f"{tag}: max abs err {err:.5f}")
    return out


a = torch.randn(N, device=dev)
b = torch.randn(D, device=dev)

check("V = a[n]        ", a.view(N, 1, 1).expand(N, HK, D).half().contiguous())
check("V = b[d]        ", b.view(1, 1, D).expand(N, HK, D).half().contiguous())
check("V = a[n]*b[d]   ", (a.view(N, 1, 1) * b.view(1, 1, D))
      .expand(N, HK, D).half().contiguous())
vr = torch.randn(N, HK, D, device=dev).half()
o1 = check("V = random      ", vr)
o2 = fa(vr)
print("deterministic (two runs identical):", torch.equal(o1, o2))

# random only in one 8-dim group, constant elsewhere
v8 = torch.zeros(N, HK, D, device=dev)
v8[:, :, 0:8] = torch.randn(N, HK, 8, device=dev)
check("V random dims0-7", v8.half().contiguous())
v8b = torch.zeros(N, HK, D, device=dev)
v8b[:, :, 0:1] = torch.randn(N, HK, 1, device=dev)
check("V random dim0   ", v8b.half().contiguous())
