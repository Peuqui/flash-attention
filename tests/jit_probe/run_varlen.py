"""Varlen probe: bisect the contiguous varlen crash.

Usage: run_varlen.py <unpadded_lse 0|1> <use_seqused 0|1>
"""
import os
import sys

import torch
from torch.utils.cpp_extension import load

UNPAD, USESEQ = sys.argv[1] == "1", sys.argv[2] == "1"
D, BM, BN = 128, 128, 64
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
mod = load(
    name=f"fwd_probe_d{D}_m{BM}_n{BN}",
    sources=[os.path.join(here, "fwd_probe.cu")],
    extra_include_paths=[
        os.path.join(root, "csrc/flash_attn"),
        os.path.join(root, "csrc/flash_attn/src"),
        os.path.join(root, "csrc/common"),
        os.path.join(root, "csrc/cutlass/include"),
    ],
    extra_cuda_cflags=[
        "-O2", "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "--use_fast_math", "-std=c++20",
        f"-DPROBE_HEADDIM={D}", f"-DPROBE_BLOCK_M={BM}",
        f"-DPROBE_BLOCK_N={BN}",
        "-DFLASHATTENTION_DISABLE_DROPOUT",
        "-DFLASHATTENTION_DISABLE_BACKWARD",
        "-DFLASHATTENTION_DISABLE_APPENDKV", "-DFLASHATTENTION_DISABLE_BF16",
    ],
    extra_cflags=["-std=c++20"],
    verbose=False,
)

dev = "cuda:0"
NQ, N, H, HK = 2, 16, 8, 2
scale = D ** -0.5

torch.manual_seed(7)
q = torch.randn(NQ, H, D, device=dev).half()
k = torch.randn(N, HK, D, device=dev).half()
v = torch.randn(N, HK, D, device=dev).half()
cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)
seqused = torch.tensor([N], dtype=torch.int32, device=dev)

out, lse = mod.probe_varlen(q, k, v, cuq, cuk, seqused,
                            NQ, N, scale, UNPAD, USESEQ)
torch.cuda.synchronize()
qf = q.float().transpose(0, 1)
kf = k.float().repeat_interleave(H // HK, 1).transpose(0, 1)
vf = v.float().repeat_interleave(H // HK, 1).transpose(0, 1)
ref = torch.nn.functional.scaled_dot_product_attention(
    qf, kf, vf, scale=scale).transpose(0, 1)
err = (out.float() - ref).abs().max().item()
print(f"varlen unpadded_lse={UNPAD} seqused={USESEQ}: max abs err {err:.4f}")
