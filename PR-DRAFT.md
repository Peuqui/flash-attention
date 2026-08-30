# PR-Entwürfe (NICHT abgeschickt — Freigabe durch Peuqui erforderlich)

Beschlossene Aufteilung (2026-08-30): drei getrennte PRs plus
1Cat-Kontakt. Absenden erst nach Flash-Next-Lauf und expliziter
Freigabe. Vorher: Branch auf upstream main rebasen und in saubere
Teil-Branches splitten (split-kv-fix, sm75-enablement).

## PR A — vllm-project/flash-attention: Split-KV für paged Verify

    gh pr create --repo vllm-project/flash-attention \
      --head Peuqui:splitkv-paged-verify \
      --title "Enable split-KV for paged multi-token queries (spec-decode verify)" \
      --body-file <Abschnitt A>

### Enable split-KV for paged multi-token queries (spec-decode verify)

`set_params_splitkv` only applies split-k in the
`seqlenq_ngroups_swapped` case (q_len == 1): "Only apply split-k for
decoding". Any paged varlen call with 1 < q_len ≤ 64 — exactly the
shape of a speculative-decode verify — is forced to `num_splits ≤ 1`
and walks the whole KV serially in one CTA per (batch, head). At long
context this is a kernel-level cliff:

- Ampere (RTX 3090 Ti, unmodified vLLM 0.13.0 wheel, 31k paged KV):
  verify q=2 is **22.8×** slower than the equivalent split decode call;
  q=4: 16.5×; q=8: 7.3×. The restriction is unchanged in today's main
  (`STD_TORCH_CHECK(num_splits <= 1, ...)`).
- The absolute verify time is context-bound, not head-bound (~1.4 ms at
  31k regardless of q_len and head count) — the signature of a serial
  KV walk.

The fix enables splits for `paged_KV && max_seqlen_q <= 64` and makes
the combine kernel varlen-correct, which the old gate had masked:

- final O write used `batch_idx * o_batch_stride` (unset in varlen) —
  now cu_seqlens_q packing with a row guard;
- unpadded LSE write assumed uniform sequence lengths — now cu_seqlens
  packing;
- `softmax_lse_accum`/`out_accum` are pre-initialized (-inf / 0) so
  early-exit splits cannot leak garbage into the combine.

Measured end to end (2× Quadro RTX 8000, 27B W4A8, MTP spec decode,
31k context): long-context decode 23 → 34 tok/s; speculation at long
context flips from a loss to a win. Numerics: max abs err 2.4e-4 vs
fp32 reference across q_len 1–8, splits 1/2/4 bitwise-consistent.

## PR B — vllm-project/vllm: kleine Begleit-Fixes

    (Branch aus fork_patches/ ableiten, Titel z.B.
     "Fix FA version detection for heterogeneous multi-GPU rigs")

- `get_flash_attn_version()` und das Capability-Gate leiten die
  FA-Version GLOBAL von Gerät 0 ab; in heterogenen PP-Rigs diktiert
  damit die schwächste Karte den Kernel aller Stufen (RTX-Worker stirbt
  mit "FlashAttention version not detected", wenn eine V100 auf
  Position 0 sitzt). Fix: `torch.cuda.current_device()` an beiden
  Stellen (fork_patches/fa_utils.py, flash_attn_interface.py).
- Triton-3D-Spec-Dispatch-Fix (siehe SPEC-LONGCTX-HUNT.md, Nachtrunde).

## PR C — vllm-project/flash-attention: sm75-Enablement (als Draft)

    gh pr create --repo vllm-project/flash-attention \
      --head Peuqui:sm75-enablement --draft \
      --title "Enable and fix the FA2 forward path for Turing (sm75)" \
      --body-file <Abschnitt C>

### Enable and fix the FA2 forward path for Turing (sm75)

FlashAttention-2 has carried an sm75 code path in its kernel traits
(SM75_16x8x8 MMA atom, LDSM copy atoms, `Has_cp_async=false`) since the
beginning, but it was never enabled — and it turns out it computes wrong
results for head dims > 64. This PR enables the forward path for Turing
and fixes the underlying bug, giving vLLM a real FlashAttention backend
on sm75 (Quadro RTX 8000, RTX 2080 Ti, Tesla T4) instead of the
TRITON_ATTN fallback that collapses at long context.

Why it matters: the sm75 fleet is large and cheap. T4s are still
everywhere in clouds and inference boxes; RTX 8000 (48 GB) and
2080 Ti cards flood the used market at a fraction of the price of any
current 48-GB-class accelerator. Enabling the FA2 path keeps that
hardware viable for long-context serving instead of forcing owners to
replace it.

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
- Measured sm75 tiles (RTX 8000, hdim 128, 31k paged KV): splitkv
  dispatch 64×32 (32 KiB smem = 2 CTAs/SM, 18 % faster than 64×64 for
  q ≤ 8), align path 128×64 (the standard kernel's tile, 37 % faster
  chunk prefill, bitwise identity with the standard kernel preserved).

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
- Depends on PR A (split-KV + combine varlen fixes).
- The vLLM-side gates (backend priority for sm75, fp16-only
  `supports_combination`) are the separate PR B.

## 1Cat-Kontakt (Issue/Diskussion im 1Cat-vLLM-Repo)

Ton: Dank + Zahlen als Geschenk. Inhalte:

1. Dank für den SM70-Stack (XQA-Verify trägt unsere Produktion).
2. Split-KV-Befund aus PR A — falls für künftige FA-Anleihen relevant.
3. Messungen an flash_attn_v100 (v1.3.0, V100, H=4/HK=1/D=128, 31k KV):
   - Verify skaliert linear mit q (0,153/0,222/0,632 ms für q=1/2/8):
     tokens-as-batch läuft den KV pro Token; ein gebündelter
     Mehr-Token-Verify wäre der nächste große Hebel.
   - Prefill-Kachel D=128: 64×80 statt 32×176 misst 14,06 statt
     16,10 ms (−13 %) am 2048er-Chunk; M muss Vielfaches von 32 sein,
     48×112/80×48 liefern falsche Ergebnisse.
4. Heterogener Betrieb (RTX-8000-Stufe + V100-Stufe im PP-Gitter)
   funktioniert mit per-Device-FA-Detection — Hinweis auf PR B.
