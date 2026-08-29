# SM75-Enablement: vllm-flash-attn für Turing (RTX 8000)

Stand: 2026-08-29 (Projektstart) · Fork von vllm-project/flash-attention,
Branch `sm75-enablement` auf `28e862d` (= GIT_TAG-Pin von vLLM v0.27.1,
der Basis des 1Cat-Forks 1.3.0).

## Warum

vLLM auf sm75 fällt mangels FlashAttention auf TRITON_ATTN zurück, und
der kollabiert bei langem Kontext (gemessen 2026-08-29, 27B TP2 auf
2× RTX 8000 @ ~30k Kontext):

| Konfiguration | kurz | lang @30k |
|---|---:|---:|
| k=6 Spekulation (Triton, Dispatch-Patch) | 60,8 | 6,3 |
| k=0 (Triton, 64 Segmente) | 41–42 | 16,4 |
| Vergleich 2× V100 (XQA-Backend, k=5) | 60,5 | ~28–60 live |

Prefill @30k: ~75 tok/s (V100: 573+). Triton-Hebel sind ausgereizt
(Dispatch-Fix +70 % auf Spec, Segmente +7 % auf Basis — beide in
v100-skinny/fork_patches). Ziel: echtes FA2-Backend für sm75 →
Decode-Basis, Multi-Token-Verify UND Prefill in einem Schlag.

## Der Befund, der es möglich macht

`csrc/flash_attn/src/kernel_traits.h` trägt den sm75-Pfad BEREITS
(FA2-Erbe): `SM75_16x8x8_F32F16F16F32_TN`-MMA, `SM75_*_LDSM`-ldmatrix,
`Has_cp_async=false`-Zweig, fp16-only unter sm80. Paged KV (block_table),
varlen, GQA, SplitKV und Multi-Token-Query liegen ÜBER dieser
arch-parametrisierten Schicht. Gesperrt ist sm75 nur durch Build-Filter
und Gates. Öffentliches Rezept: farnghwai/flash-attention-2080ti,
Commit 4fd43d9e (auf FA2 2.7.3; nicht 1:1 anwendbar, aber die Landkarte).

## Patch-Stellen (Reihenfolge)

1. **CMakeLists.txt**: `FA2_ARCHS` um `7.5` ergänzen; die
   `cuda_archs_loose_intersection(... "8.0+PTX" ...)`-Filter anpassen
   (sm75 wird sonst STILL herausgefiltert). bf16-Kernel-Instanziierungen
   für sm75 aus der Generierung nehmen (Turing kann kein bf16).
2. **`csrc/flash_attn/src/flash_fwd_launch_template.h`** (+bwd):
   `#if __CUDA_ARCH__ >= 800 → ARCH_SUPPORTS_FLASH` um `== 750`
   erweitern (nur fwd nötig — bwd brauchen wir nicht, ggf. ganz aus dem
   Build nehmen = Compile-Zeit).
3. **`csrc/flash_attn/flash_api.cpp`**: die `is_sm8x_min`-TORCH_CHECKs
   („Ampere or newer") in den Entrypoints um einen fp16-only-sm75-Zweig
   ergänzen (bf16-Eingaben auf sm75 hart ablehnen).
4. **Shared-Memory-Deckel 64 KiB**: alle
   `cudaFuncAttributeMaxDynamicSharedMemorySize`-Opt-ins auf sm75 bei
   64 KiB kappen; Kernel-Configs, die mehr verlangen, brauchen kleinere
   Tiles. farnghwai: SplitKV-`block_n = hs<=64?128:(hs<=160?64:32)`
   statt `256:128:64`; `set_params_splitkv` synchron halten.
   hdim 256 passt initial NICHT (Neu-Tiling, später; 27B hat hdim 128).
5. **Build**: mit der 1Cat-venv-Toolchain (ABI-Match!):
   `CUDA_HOME=/home/mp/Projekte/v100-skinny/.cuda-nvcc-deb/usr/local/cuda-12.8`,
   Python/torch aus `.venv-sm70-130` (torch 2.10), `TORCH_CUDA_ARCH_LIST=7.5`,
   Ziel-Artefakt `_vllm_fa2_C.abi3.so`. ccache/ninja; nur fwd + fp16 +
   sm75 instanziieren (Compile-Zeit!).
6. **Drop-in**: die .so neben die bestehende in
   `.venv-sm70-130/.../vllm/vllm_flash_attn/` (Backup!), dann im
   1Cat-Fork die Gates öffnen:
   - `vllm/v1/attention/backends/flash_attn.py`:
     `supports_compute_capability >= (8,0)` → `(7,5)` mit fp16-Bedingung
   - `vllm/platforms/cuda.py`: sm75-Prioritätsliste (derzeit hart
     `[TRITON_ATTN, FLEX]`, Merge-Projekt Session 3) um FLASH_ATTN vorn
     ergänzen — als fork_patch in v100-skinny.
7. **Verifikation** (Reihenfolge zwingend):
   a. Numerik: Kernel-Outputs gegen Triton-2D-Referenz (q=1…8, Paged-KV,
      30k-Geometrie — Testskript-Vorlage: AIfred-Scratchpad
      `test_3d_spec_numerics.py`),
   b. 27B-Kohärenz 3/3 auf TP2-RTX,
   c. Lang-Kontext-A/B gegen die Tabelle oben (Messskript
      `rtx_longctx_sweep.py`), Ziel: k=0 lang ≥ 40, Spec lang ≥ 50,
   d. Kurz-Kontext-Regression (66,1 @ k=3 nicht verlieren).

## Build-Befunde (Patch-Runde 1, 2026-08-29)

- **fp16-Kernel kompilieren auf sm75** (fwd, splitkv, sparse) — die
  SM75-Traits tragen wirklich.
- **bf16-Instanziierungen brechen** (Traits erzwingen auf sm75 half_t,
  elem_type=bf16 lebt in einzelnen Atomen weiter → Layout-Asserts).
  Fix: `FLASHATTENTION_DISABLE_BF16` (static_switch.h + flash_api.cpp)
  und CMake baut bf16-Quellen nur für Ampere+ (Turing-only: ganz raus).
- **Append_KV/Rotary bricht auch in fp16**: `copy_rotary_*` reißt
  static_asserts, weil das Nicht-cp.async-Copy-Atom
  (`AutoVectorizingCopyWithAssumedAlignment`) andere Partitions-Shapes
  liefert als das SM80-cp.async-Atom. vLLM nutzt den Pfad nie (KV-Append
  via reshape_and_cache, RoPE außerhalb) → `FLASHATTENTION_DISABLE_APPENDKV`
  (APPENDKV_SWITCH in static_switch.h, Guard in mha_fwd_kvcache), im
  Turing-only-Build gesetzt.
- Smem-Rechnung (fp16, Q=M·d·2 B, KV=2·N·d·2 B): hdim ≤ 96 passt ohnehin,
  hdim 128 = exakt 64 KiB (Opt-in-Maximum), hdim 192/256 → sm75-Zweige
  64×32; SplitKV sm75-Tiles nach farnghwai `N = hs≤64?128:(hs≤160?64:32)`,
  Align-Variante spiegelt die Standard-Kernel-Tiles.
- Build-Drossel Pflicht: Mini hat 30 GiB RAM → `-j 3` (16 parallele
  nvcc-Jobs haben am 29.08. die Kiste per Swap-Thrashing eingefroren).

## ROOT CAUSE gefunden und gefixt (2026-08-29, Patch-Runde 2)

Der schlafende sm75-Pfad von FA2 hat einen **latenten Upstream-Bug in
`utils.h::gemm()/gemm_rs()`**: Die Software-Pipeline indiziert die
Smem→Register-Kopien mit dem MMA-K-Schleifenindex. Auf sm80+ ist das 1:1
(Atom-K 16 = LDSM-Kachel 16). Auf sm75 hat das MMA-Atom K=8, die
LDSM-Kachel bleibt aber 16 breit → das Register-Fragment hat 2× so viele
K-Steps wie die Copy-View, und `tCsA(_,_,i+1)` läuft für i≥Hälfte ÜBER
DAS SMEM-TILE HINAUS (Q-Reads landen im K-/V-Tile, Register-Writes
clobbern Nachbar-Fragmente → LSE-Korruption). Deshalb: hdim ≤ 64 heil
(Überlauf bleibt zufällig im Tile), hdim 96/128 kaputt; deterministisch,
copy-atom-unabhängig, cutlass-versionsunabhängig (3.6 und 3.9 getestet).

**Fix:** Copy-Schleife von der MMA-Schleife entkoppelt —
`kMmaPerCopy = size<2>(tCrA) / size<2>(tCrA_copy_view)` (2 auf sm75,
1 auf sm80 = unverändertes Verhalten), Kopie nur alle kMmaPerCopy Steps
mit eigenem Index. Verifiziert im JIT-Harness (tests/jit_probe/):
d=128 Zufalls-Q/V max abs err 7,5e-1 → **4e-4**, LSE exakt.

Diagnose-Werkzeuge im Repo: tests/sm75_debug_*.py (algebraische Sonden:
Uniform-P, Basis-V, Gift-Puffer, Bit-Proben), tests/jit_probe/
(Einzel-Instanz-JIT-Build mit frei wählbaren Traits + printf-Sonden).
Der Fix ist upstream-PR-würdig (macht FA2-sm75 erstmals korrekt;
vermutlich derselbe Bug in allen FA2-sm75-Community-Ports mit hdim>64).

## Verifikation ABGESCHLOSSEN (2026-08-29 Abend)

Messmatrix 27B NVFP4 TP2 auf 2× RTX 8000, max-model-len 65.536, gleiche
Prompts, Langtest = 31.469 Token Prefill + prefix-gecachter 200-Token-
Decode (tests/rtx_fa2_sweep.py; Kohärenz 3/3 in JEDEM Lauf):

| Konfig | kurz | Prefill@31k | Decode@31k |
|---|---:|---:|---:|
| Triton k=0 (Dispatch+Segment-Patches) | 43,6 | 194 | 23,1 |
| FA2 k=0 | 43,1 | **505** | **33,9** |
| FA2 k=3 | **75,8** | 501 | 19,2 |
| FA2 k=6 | 68,6 | 496 | 19,7 |

- Kurz-Kontext: FA2 k=3 = **75,8 tok/s Absolut-Rekord** (Triton-Best 66,1,
  V100-Speed 60,5, llama.cpp Q8+MTP 51,9).
- Prefill: **2,6× vs Triton** (505 vs 194).
- Decode lang k=0: **+47 %** (33,9 vs 23,1).
- Spekulation verliert bei 31k gegen k=0 (Drafter macht k sequenzielle
  Full-Context-Pässe pro Step — strukturell, nicht FA2-spezifisch).

Betriebs-Erkenntnisse für die Integration:
- Backend-Erzwingung heißt jetzt `--attention-backend` (CLI), die alte
  Env VLLM_ATTENTION_BACKEND ist wirkungslos.
- `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1` aktiviert auf JEDER Karte
  MTP k=4 mit Volta-only-Drafter-Kernel → auf RTX ohne explizite
  speculative-config („attention_backend":"FLASH_ATTN") Crash
  („Kernel supports only Volta GPUs").
- Drop-in liegt im 1Cat-venv (Backup backups/2026-08-29-fa2-sm75/),
  Gates offen (flash_attn_interface ≥75, flash_attn.py (7,5)+fp16-only,
  cuda.py sm75-Liste mit FLASH_ATTN vorn), gespiegelt in fork_patches/.
- Offen: Neu-Kalibration der 27B-RTX-Betriebspunkte über AIfred
  (k-Sweep unter FA2; k=3 ist heißer Kandidat), Drafter-Backend im
  llama-swap-Eintrag von TRITON_ATTN auf FLASH_ATTN umstellen.

## Risiken

- Tiles sind auf Ampere+cp.async getunt — ohne Retuning bleibt
  Durchsatz liegen (ssiu misst 66 % Compute-Peak auf Turing als Bestwert).
- hdim 256 (Flash-Next!) braucht eigenes Tiling — Phase 2.
- ABI: die .so MUSS gegen das venv-torch gebaut sein.

## Kontext / Belege

- Recherche-Duo 2026-08-29 (AIfred-Session): Triton-Dispatch-Analyse +
  Repo-Ranking. Community-Referenzen: farnghwai/flash-attention-2080ti
  (Rezept), ssiu/flash-attention-turing (sm75-Traits-Referenz, KEINE
  Lizenz — nur lesen), 1CatAI/1Cat-vLLM `flash-attention-v100/`
  (Apache-2.0, grouped_verify_paged als Verify-Blaupause, aber
  H6/D256/96KB-hartkodiert).
- Nach Abschluss: 1CatAI ansprechen (Issue #412-Kontext: heterogene
  Turing+Volta-Rigs) + Dispatch-Patch als vLLM-upstream-PR.
