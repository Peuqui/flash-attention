"""27B TP2-RTX A/B sweep: FA2-sm75 vs Triton numbers, short + long context.

Boots the calibrated vLLM server config on the two RTX 8000 with a chosen
attention backend and MTP depth, then measures:
  - coherence: 3 German prompts at temp 0 (texts printed for review)
  - short ctx: 200-token completion wall clock
  - long ctx (~30k tok): prefill time, then prefix-cached 200-token decode

Usage: rtx_fa2_sweep.py <backend FLASH_ATTN|TRITON_ATTN> <k 0..7> [port]
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

BACKEND = sys.argv[1]
K = int(sys.argv[2])
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8123

VENV = "/home/mp/Projekte/v100-skinny/.venv-sm70-130"
MODEL = ("/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/"
         "snapshots/554ebba9b5f1b79dc11246341960360e6ef05ef4")
NAME = "sweep-27b"

ENV = dict(os.environ)
ENV.update({
    "PATH": f"{VENV}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_HOME": "/home/mp/Projekte/v100-skinny/.cuda-nvcc-deb/usr/local/cuda-12.8",
    "TORCH_CUDA_ARCH_LIST": "7.5",
    "NCCL_P2P_DISABLE": "1",
    "VLLM_SM70_E5_CACHE": "0",
    "VLLM_SM70_NVFP4_TURBOMIND": "0",
    "VLLM_SM70_QUANT_BACKEND": "marlin",
    "VLLM_SKINNY_NVFP4": "1",
    "VLLM_SKINNY_QPN": "1",
    "VLLM_SKINNY_QPN2": "1",
    "VLLM_SKINNY_NVFP4_SRC": "/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu",
    "TORCHINDUCTOR_CACHE_DIR": "/home/mp/.cache/torchinductor",
    "CUDA_VISIBLE_DEVICES": "0,2",
    # k=0 must really mean no speculation: the 1Cat MTP defaults would
    # otherwise auto-enable k=4 with the Volta-only drafter kernel.
    "VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS": "1" if K > 0 else "0",
    "HOME": "/home/mp",
})

args = [
    f"{VENV}/bin/python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL, "--served-model-name", NAME,
    "--trust-remote-code", "--dtype", "float16",
    "--disable-custom-all-reduce", "--tensor-parallel-size", "2",
    "--pipeline-parallel-size", "1", "--gpu-memory-utilization", "0.98",
    "--block-size", "16", "--max-model-len", "65536",
    "--max-num-seqs", "4", "--max-num-batched-tokens", "2048",
    "--host", "127.0.0.1", "--port", str(PORT), "--language-model-only",
    "--attention-backend", BACKEND,
]
if K > 0:
    spec = {"method": "mtp", "num_speculative_tokens": K,
            "draft_sample_method": "greedy",
            "use_local_argmax_reduction": True,
            "attention_backend": BACKEND}
    caps = sorted({1, 2, 4, K + 1, 8})
    args += ["--speculative-config", json.dumps(spec),
             "--compilation-config",
             json.dumps({"cudagraph_capture_sizes": caps})]

log = open(f"/tmp/sweep_{BACKEND}_k{K}.log", "w")
# cwd must NOT be the flash-attention repo root: its flash_attn/ package
# directory would shadow the venv package on sys.path.
proc = subprocess.Popen(args, env=ENV, stdout=log, stderr=subprocess.STDOUT,
                        cwd="/home/mp")
print(f"server pid {proc.pid}, backend={BACKEND}, k={K}; waiting for boot...")


def api(path, payload=None, timeout=600):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


try:
    for _ in range(360):
        if proc.poll() is not None:
            print("SERVER DIED during boot -- last log lines:")
            log.flush()
            os.system(f"tail -5 /tmp/sweep_{BACKEND}_k{K}.log")
            sys.exit(1)
        try:
            api("/v1/models", timeout=5)
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError("server did not come up")
    print("server up")

    def complete(prompt, max_tokens, ignore_eos=True):
        t0 = time.time()
        r = api("/v1/completions", {
            "model": NAME, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": ignore_eos})
        dt = time.time() - t0
        u = r["usage"]
        return r["choices"][0]["text"], u["prompt_tokens"], \
            u["completion_tokens"], dt

    # --- coherence ---
    for i, p in enumerate([
        "Erkläre in drei Sätzen, warum der Himmel blau ist.\n\nAntwort:",
        "Nenne die Hauptstadt von Frankreich und einen bekannten Fluss dort.\n\nAntwort:",
        "Was ist 17 mal 23? Rechne Schritt für Schritt.\n\nAntwort:",
    ]):
        text, _, _, _ = complete(p, 90, ignore_eos=False)
        print(f"COHERENCE {i + 1}: {text.strip()[:220]!r}")

    # --- short context ---
    prompt = ("Schreibe einen ausführlichen Aufsatz über die Geschichte der "
              "Dampfmaschine und ihre Bedeutung für die Industrialisierung.")
    _ = complete(prompt, 20)  # warmup
    _, ptok, ctok, dt = complete(prompt, 200)
    print(f"SHORT: {ctok} tok in {dt:.2f}s = {ctok / dt:.1f} tok/s "
          f"(prefill {ptok} tok)")

    # --- long context ~30k tok ---
    filler = ("Die Industrialisierung veränderte Europa grundlegend. " * 12
              + "\n")
    long_prompt = filler * 260 + "\nFasse den Kern in einem Satz zusammen:"
    t0 = time.time()
    _, ptok, _, dt1 = complete(long_prompt, 1)
    print(f"LONG prefill: {ptok} tok in {dt1:.2f}s = {ptok / dt1:.0f} tok/s")
    _, _, ctok, dt2 = complete(long_prompt, 200)
    print(f"LONG decode (prefix-cached): {ctok} tok in {dt2:.2f}s = "
          f"{ctok / dt2:.1f} tok/s")
finally:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("server stopped")
