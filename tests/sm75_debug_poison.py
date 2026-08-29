"""Poison-buffer probe: does the kernel read out of bounds around V?

Place v inside a large buffer, poison everything around it with a marker
constant. If the marker leaks into the output, the kernel reads OOB.
Also check whether corruption depends on (query row, head)."""
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
POISON = 777.0

q = torch.zeros(NQ, H, D, device=dev).half()
k = torch.randn(N, HK, D, device=dev).half()
cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)

pad = N * HK * D * 4  # generous guard zones
buf = torch.full((pad + N * HK * D + pad,), POISON, device=dev,
                 dtype=torch.float16)
v = buf[pad:pad + N * HK * D].view(N, HK, D)
v.zero_()
v[0] = 1.0  # basis probe at row 0 (the row that showed 13.55x weight)

out = flash_attn_varlen_func(
    q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
    cu_seqlens_k=cuk, softmax_scale=scale, causal=False,
    fa_version=2).cpu().float()

w00 = out[0, 0, 0].item() * N
print(f"row-0 weight with poisoned guards: {w00:.3f} "
      f"(clean allocator run gave 13.55; correct is 1.0)")
print(f"poison leaked (|out| huge)? max |out| = {out.abs().max():.1f}")

# corruption localization across (m, h)
wm = out[:, :, 0] * N  # (NQ, H) weight of row 0 per query-row/head
print("weight of v-row0 per (query row, head), correct = 1.0:")
for m in range(NQ):
    print(f"  m={m:2d}:", [round(x, 2) for x in wm[m].tolist()])
