"""Microbench: FA2 varlen paged attention at 31k KV for q=1..8.

Measures whether the multi-token spec-verify is kernel-bound on sm75
and how it scales with num_splits (the vLLM fork forces num_splits<=1
for paged varlen).
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(7)
dev = "cuda:0"
H, HK, D = 4, 1, 128   # pro GPU bei TP2: 27B hat 8 q / 2 kv Heads gesamt
N = 31488
BLOCK = 16
scale = D ** -0.5

nb = N // BLOCK
k = torch.randn(nb + 1, BLOCK, HK, D, device=dev).half()
v = torch.randn(nb + 1, BLOCK, HK, D, device=dev).half()
bt = torch.arange(1, nb + 1, dtype=torch.int32, device=dev).unsqueeze(0)
seqused = torch.tensor([N], dtype=torch.int32, device=dev)

print(f"KV: {N} tok, {H}/{HK} heads, hdim {D} (pro-GPU-Geometrie TP2)")
for q_len in (1, 2, 3, 4, 8):
    q = torch.randn(q_len, H, D, device=dev).half()
    cuq = torch.tensor([0, q_len], dtype=torch.int32, device=dev)

    def run():
        return flash_attn_varlen_func(
            q, k, v, max_seqlen_q=q_len, cu_seqlens_q=cuq, max_seqlen_k=N,
            seqused_k=seqused, softmax_scale=scale, causal=True,
            block_table=bt, fa_version=2)

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 50
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"q={q_len}: {ms:.3f} ms/call  ({64 * ms:.1f} ms hochgerechnet "
          f"auf 64 Schichten)")
