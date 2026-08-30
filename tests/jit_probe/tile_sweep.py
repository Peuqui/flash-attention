"""Turing-Tile-Sweep fuer die SplitKV-Kernel (hdim 128).

Misst je Kachelkandidat (kBlockM, kBlockN, Warps):
  - Decode/Verify: q=1/2/8 @ 31k Paged-KV, bestes Ergebnis ueber mehrere
    Split-Zahlen (die Dispatch-Heuristik variiert ohnehin mit kBlockN)
  - Prefill-Chunk: q=2048 @ 31k KV, num_splits=1 (Align-Pfad-Analogon)
Korrektheit: Ausgabe je Kandidat gegen die aktuelle Kachel (64,64,4).

Usage: tile_sweep.py  (Kandidatenliste unten)
"""
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

CANDIDATES = [
    (64, 64, 4),   # Referenz (aktuelle sm75-Tabelle)
    (64, 32, 2),   # 32 KB, weniger Warps (Runde 1 durch Crash verpasst)
    (128, 64, 8),  # 64 KB, 8 Warps — Chunk-Kandidat verfeinern
]

def build(bm, bn, w):
    return load(
        name=f"tile_d128_m{bm}_n{bn}_w{w}",
        sources=[os.path.join(HERE, "fwd_probe.cu")],
        extra_include_paths=[
            os.path.join(ROOT, "csrc/flash_attn"),
            os.path.join(ROOT, "csrc/flash_attn/src"),
            os.path.join(ROOT, "csrc/common"),
            os.path.join(ROOT, "csrc/cutlass/include"),
        ],
        extra_cuda_cflags=[
            "-O2", "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "--use_fast_math", "-std=c++20",
            "-DPROBE_HEADDIM=128", f"-DPROBE_BLOCK_M={bm}",
            f"-DPROBE_BLOCK_N={bn}", f"-DPROBE_WARPS={w}",
            "-DFLASHATTENTION_DISABLE_DROPOUT",
            "-DFLASHATTENTION_DISABLE_BACKWARD",
            "-DFLASHATTENTION_DISABLE_APPENDKV",
            "-DFLASHATTENTION_DISABLE_BF16",
        ],
        extra_cflags=["-std=c++20"], verbose=False,
    )

dev = "cuda:0"
H, HK, D = 4, 1, 128      # pro-GPU-Geometrie des 27B bei TP2
N, BLOCK = 31488, 16
scale = D ** -0.5

torch.manual_seed(7)
nb = N // BLOCK
k = torch.randn(nb + 1, BLOCK, HK, D, device=dev).half()
v = torch.randn(nb + 1, BLOCK, HK, D, device=dev).half()
bt = torch.arange(1, nb + 1, dtype=torch.int32, device=dev).unsqueeze(0)
seqused = torch.tensor([N], dtype=torch.int32, device=dev)
qs = {ql: torch.randn(ql, H, D, device=dev).half() for ql in (1, 2, 8, 2048)}

def bench(mod, ql, splits, iters=50):
    q = qs[ql]
    def run():
        return mod.probe_splitkv_traits(q, k, v, bt, seqused, N, scale, splits)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

ref_out = {}
results = []
for bm, bn, w in CANDIDATES:
    smem = (bm * D * 2 + 2 * bn * D * 2) // 1024
    ctas = 64 // smem if smem <= 32 else 1
    tag = f"M{bm} N{bn} W{w} ({smem}KB, {min(2, max(1, 64 // smem))} CTA)"
    try:
        mod = build(bm, bn, w)
    except Exception as e:
        print(f"{tag}: BUILD FAILED ({str(e)[-120:]})")
        continue
    # Korrektheit gegen Referenzkachel
    ok = True
    for ql in (1, 2, 8):
        o, _ = mod.probe_splitkv_traits(qs[ql], k, v, bt, seqused, N, scale, 64)
        if (bm, bn, w) == CANDIDATES[0]:
            ref_out[ql] = o.float()
        else:
            err = (o.float() - ref_out[ql]).abs().max().item()
            if err > 2e-3 or torch.isnan(o).any():
                ok = False
                print(f"{tag}: NUMERIK ABWEICHEND q={ql} err={err:.2e}")
                break
    if not ok:
        continue
    row = {"tag": tag}
    for ql in (1, 2, 8):
        row[f"q{ql}"] = min(bench(mod, ql, s, 50) for s in (36, 72, 113))
    row["chunk2048"] = bench(mod, 2048, 1, 10)
    results.append(row)
    print(f"{tag}: q1={row['q1']:.3f}  q2={row['q2']:.3f}  "
          f"q8={row['q8']:.3f}  chunk2048={row['chunk2048']:.2f} ms")

print("\n=== ZUSAMMENFASSUNG (ms, kleiner=besser) ===")
for r in results:
    print(f"{r['tag']:28s} q2 {r['q2']:.3f}  q8 {r['q8']:.3f}  "
          f"chunk {r['chunk2048']:.2f}")
