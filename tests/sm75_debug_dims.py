"""Check whether only output dims >= 64 are wrong (d=128, q=0 uniform)."""
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
v = torch.randn(N, HK, D, device=dev).half()
cuq = torch.tensor([0, NQ], dtype=torch.int32, device=dev)
cuk = torch.tensor([0, N], dtype=torch.int32, device=dev)
out = flash_attn_varlen_func(
    q, k, v, max_seqlen_q=NQ, cu_seqlens_q=cuq, max_seqlen_k=N,
    cu_seqlens_k=cuk, softmax_scale=scale, causal=False, fa_version=2)

ref = v.float().mean(dim=0).repeat_interleave(H // HK, 0)  # (H, D)
diff = (out.cpu().float() - ref.cpu()).abs()  # (NQ, H, D)
err_per_dim = diff.max(dim=0).values.max(dim=0).values  # (D,)
print("max err dims  0- 63:", err_per_dim[:64].max().item())
print("max err dims 64-127:", err_per_dim[64:].max().item())
bad = (err_per_dim > 0.02).nonzero().flatten().tolist()
print(f"bad dims ({len(bad)}):", bad[:40])

# where do the wrong values come from? test: is out[:, :, 64:] equal to a
# permutation of ref? e.g. does out dim 64+i replicate ref dim i?
o = out.cpu().float()[0, 0]
r = ref.cpu()[0]
print("out[64:68]:", [round(x, 4) for x in o[64:68].tolist()])
print("ref[64:68]:", [round(x, 4) for x in r[64:68].tolist()])
print("ref[ 0: 4]:", [round(x, 4) for x in r[:4].tolist()])
# correlation of out-second-half against shifted ref halves
for shift, tag in ((0, "ref[64:]"), (-64, "ref[:64]")):
    seg = r[64 + shift:128 + shift]
    print(f"corr out[64:] vs {tag}: "
          f"{torch.corrcoef(torch.stack([o[64:], seg]))[0, 1].item():.4f}")
