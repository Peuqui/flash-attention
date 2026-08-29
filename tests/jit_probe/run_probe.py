"""JIT-compile one flash_fwd instantiation with chosen traits and test it.

Usage: run_probe.py <headdim> <blockM> <blockN>
"""
import math
import os
import sys

import torch
from torch.utils.cpp_extension import load

D, BM, BN = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
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
B, SQ, SK, H, HK = 1, 16, 64, 8, 2
scale = D ** -0.5

torch.manual_seed(7)
q = torch.zeros(B, SQ, H, D, device=dev).half()
k = torch.randn(B, SK, HK, D, device=dev).half()
v = torch.zeros(B, SK, HK, D, device=dev)
v[:, 0] = 1.0
v = v.half()

out, lse = mod.probe_fwd(q, k, v, scale)
torch.cuda.synchronize()
w0 = out[0, :, 0, 0].float() * SK  # weight of v-row0 per query row, head 0
print(f"d={D} M={BM} N={BN}: w(v0) row0={w0[0]:.2f} row1={w0[1]:.2f} "
      f"(correct 1.0)")
l = lse[0, 0].float()
print(f"  lse row0={l[0]:.4f} row1={l[1]:.4f} correct={math.log(SK):.4f}")

# full random check
q2 = torch.randn(B, SQ, H, D, device=dev).half()
v2 = torch.randn(B, SK, HK, D, device=dev).half()
out, lse = mod.probe_fwd(q2, k, v2, scale)
qf = q2.float()[0].transpose(0, 1)
kf = k.float()[0].repeat_interleave(H // HK, 1).transpose(0, 1)
vf = v2.float()[0].repeat_interleave(H // HK, 1).transpose(0, 1)
ref = torch.nn.functional.scaled_dot_product_attention(
    qf, kf, vf, scale=scale).transpose(0, 1)
err = (out[0].float() - ref).abs().max().item()
print(f"  random q/v: max abs err {err:.4f}")
