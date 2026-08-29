"""Isolate which sm75 code path is broken: standard fwd vs splitkv vs paged."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

torch.manual_seed(7)
dev = "cuda:0"
H, HK, D = 8, 2, 128
scale = D ** -0.5


def sdpa_ref(q, k, v, causal):
    qf = q.float().transpose(0, 1).unsqueeze(0)
    kf = k.float().repeat_interleave(H // HK, dim=1).transpose(0, 1).unsqueeze(0)
    vf = v.float().repeat_interleave(H // HK, dim=1).transpose(0, 1).unsqueeze(0)
    o = torch.nn.functional.scaled_dot_product_attention(
        qf, kf, vf, scale=scale, is_causal=causal)
    return o.squeeze(0).transpose(0, 1)


def report(name, got, ref):
    rel = ((got.float() - ref).abs() / (ref.abs() + 1e-3)).max().item()
    print(f"{name}: max rel err {rel:.2e}  "
          f"got[0,0,:4]={got.float()[0, 0, :4].tolist()}  "
          f"ref[0,0,:4]={ref[0, 0, :4].tolist()}")


# --- A: standard fwd, contiguous varlen, q_len == seq_len (prefill) ---
for n in (64, 128, 500):
    q = torch.randn(n, H, D, device=dev).half()
    k = torch.randn(n, HK, D, device=dev).half()
    v = torch.randn(n, HK, D, device=dev).half()
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    out = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=n, cu_seqlens_q=cu, max_seqlen_k=n,
        cu_seqlens_k=cu, softmax_scale=scale, causal=True, fa_version=2)
    report(f"A prefill n={n}", out.cpu(), sdpa_ref(q.cpu(), k.cpu(), v.cpu(), True))

# --- B: decode q=1, contiguous KV (splitkv likely, no paging) ---
for n in (300, 5000):
    q = torch.randn(1, H, D, device=dev).half()
    k = torch.randn(n, HK, D, device=dev).half()
    v = torch.randn(n, HK, D, device=dev).half()
    cuq = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    seqused = torch.tensor([n], dtype=torch.int32, device=dev)
    out = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=1, cu_seqlens_q=cuq, max_seqlen_k=n,
        seqused_k=seqused, softmax_scale=scale, causal=True, fa_version=2)
    report(f"B decode n={n}", out.cpu(), sdpa_ref(q.cpu(), k.cpu(), v.cpu(), False))

# --- C: decode q=1, paged KV block_size 16 ---
for n in (300, 5000):
    q = torch.randn(1, H, D, device=dev).half()
    nb = (n + 15) // 16
    kc = torch.randn(nb, 16, HK, D, device=dev).half()
    vc = torch.randn(nb, 16, HK, D, device=dev).half()
    cuq = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    seqused = torch.tensor([n], dtype=torch.int32, device=dev)
    bt = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(0)
    out = flash_attn_varlen_func(
        q, kc, vc, max_seqlen_q=1, cu_seqlens_q=cuq, max_seqlen_k=n,
        seqused_k=seqused, softmax_scale=scale, causal=True,
        block_table=bt, fa_version=2)
    k_lin = kc.reshape(-1, HK, D)[:n].cpu()
    v_lin = vc.reshape(-1, HK, D)[:n].cpu()
    report(f"C paged n={n}", out.cpu(), sdpa_ref(q.cpu(), k_lin, v_lin, False))
