"""Test the splitkv path in isolation: decode q=1, contiguous KV.

Usage: run_splitkv.py <headdim> <blockM> <blockN> <num_splits>
"""
import os
import sys

import torch
from torch.utils.cpp_extension import load

D, BM, BN, NS = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
                 int(sys.argv[4]))
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
        "-DFLASHATTENTION_DISABLE_BACKWARD", "-DFLASHATTENTION_DISABLE_APPENDKV", "-DFLASHATTENTION_DISABLE_BF16",
    ],
    extra_cflags=["-std=c++20"],
    verbose=False,
)

dev = "cuda:0"
B, SQ, SK, H, HK = 1, 1, 320, 8, 2
scale = D ** -0.5

torch.manual_seed(7)
q = torch.randn(B, SQ, H, D, device=dev).half()
k = torch.randn(B, SK, HK, D, device=dev).half()
v = torch.randn(B, SK, HK, D, device=dev).half()

out, lse = mod.probe_splitkv(q, k, v, scale, NS)
torch.cuda.synchronize()

qf = q.float()[0].transpose(0, 1)
kf = k.float()[0].repeat_interleave(H // HK, 1).transpose(0, 1)
vf = v.float()[0].repeat_interleave(H // HK, 1).transpose(0, 1)
ref = torch.nn.functional.scaled_dot_product_attention(
    qf, kf, vf, scale=scale).transpose(0, 1)
err = (out[0].float() - ref).abs().max().item()
print(f"splitkv d={D} N={BN} splits={NS}: max abs err {err:.4f}")
