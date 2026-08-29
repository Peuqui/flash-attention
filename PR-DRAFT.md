# PR-Entwurf für vllm-project/flash-attention (NICHT abgeschickt)

Absenden erst nach Freigabe (vereinbart: nach der Flash-Next-Kalibration).
Ziel-Repo: vllm-project/flash-attention, Basis: main (Branch muss vor dem
Absenden auf main rebased werden; aktuell auf 28e862d = v0.27.1-Pin).

Befehl zum Absenden (nach Rebase + Freigabe):

    gh pr create --repo vllm-project/flash-attention \
      --head Peuqui:sm75-enablement --draft \
      --title "Enable and fix the FA2 forward path for Turing (sm75)" \
      --body-file PR-DRAFT.md   # Abschnitt unterhalb der Trennlinie

---

## Enable and fix the FA2 forward path for Turing (sm75)

FlashAttention-2 has carried an sm75 code path in its kernel traits
(SM75_16x8x8 MMA atom, LDSM copy atoms, `Has_cp_async=false`) since the
beginning, but it was never enabled — and it turns out it computes wrong
results for head dims > 64. This PR enables the forward path for Turing
and fixes the underlying bug, giving vLLM a real FlashAttention backend
on sm75 (Quadro RTX 8000, RTX 2080 Ti, Tesla T4) instead of the
TRITON_ATTN fallback that collapses at long context.

### The bug

`utils.h::gemm()/gemm_rs()` index the smem→register operand copies with
the MMA K-step loop variable:

```cpp
for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
```

On sm80+ this is 1:1 — the MMA atom K (16) equals the ldmatrix tile
width. The sm75 atom has K=8 at the same 16-wide copy granularity, so
`size<2>(tCrA)` is twice `size<2>(tCsA)`, and the copy index runs past
both the smem tile and the register view. The out-of-bounds layout
arithmetic walks into the *next* smem tiles: for hdim 128, Q-operand
reads land in the K and V tiles (measured offsets: K-step 8 → +32 KiB,
K-step 12 → +48 KiB = the V tile), and the register-side writes clobber
neighbouring fragments (corrupting the LSE). Head dims ≤ 64 happen to
stay inside their own tile, which is why small-hdim smoke tests pass
and the path looks deceptively functional.

The fix decouples the copy loop from the MMA loop via the compile-time
ratio `kMmaPerCopy = size<2>(tCrA) / size<2>(tCrA_copy_view)` (2 on
sm75, 1 on sm80+ — a no-op there).

### Enablement around it

- CMake: let `7.5` through the arch filters; bf16 kernel sources are
  built only for Ampere+ (the sm75 traits force `half_t` and cannot
  instantiate them). Turing-only builds drop them and define
  `FLASHATTENTION_DISABLE_BF16`.
- `FLASHATTENTION_DISABLE_APPENDKV` (mirroring the FA3 flag): the
  in-kernel KV-append/rotary copies trip layout static_asserts against
  the non-cp.async copy atoms on sm75; vLLM appends KV via
  `reshape_and_cache` and applies RoPE outside attention, so the path
  is dropped for Turing-only builds instead of ported.
- Entry-point checks admit sm75 (fp16-only; bf16 is rejected — Turing
  has no bf16 tensor-core MMA), and the splitkv tile tables get an
  sm75 column respecting the 64 KiB shared-memory limit.

### Verification (2× Quadro RTX 8000, Qwen 27B W4A8, TP2)

- Paged-KV varlen numerics vs an fp32 reference: max abs err 2.4e-4
  across q_len 1–8, GQA 8:2, hdim 128, contexts up to 31k; splitkv
  with 1/2/4 splits: 1e-4.
- End-to-end coherence 3/3.
- Throughput vs the (tuned) Triton backend on the same rig:
  short-context decode 43 → 76 tok/s (with MTP k=3), 31k-context
  prefill 194 → 505 tok/s, 31k-context decode 23 → 34 tok/s.

### Notes

- Forward only (`FLASHATTENTION_DISABLE_BACKWARD`), fp16 only.
- hdim 256 fits the 64 KiB limit only with the reduced 64×32 tiles;
  correctness-first, no Turing-specific tile tuning yet.
- The vLLM-side gates (backend priority for sm75, fp16-only
  `supports_combination`) are a separate follow-up PR to vllm-project/vllm.
