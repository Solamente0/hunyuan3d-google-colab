"""Checks the GPU and prepares caches and tokens.

Run from the notebook with `%run -i` so it can read the notebook's settings.
Part of https://github.com/Solamente0/hunyuan3d-google-colab
"""
CACHE_WEIGHTS_IN_DRIVE = globals().get("CACHE_WEIGHTS_IN_DRIVE", False)

import os, subprocess

r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit("❌ No GPU found. Go to Runtime → Change runtime type → A100 GPU.")

name, mem, cap = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
mem_gb = int(mem) / 1024
print(f"GPU: {name} | VRAM: {mem_gb:.0f} GB | compute capability: {cap}")

if float(cap) >= 10:
    raise SystemExit("❌ This GPU (Blackwell) isn't supported by the PyTorch 2.5.1 build Hunyuan3D 2.1 uses. Pick A100 instead.")

LOW_VRAM = mem_gb < 40
GPU_VRAM_GB = mem_gb
if mem_gb < 20:
    print("⚠️ Under 20 GB VRAM: only shape generation (no texture) will fit. Use cell 5 with TEXTURE off.")
elif LOW_VRAM:
    print("⚠️ Under 40 GB VRAM: the web UI may run out of memory during texturing. Cell 5 handles this GPU better.")
else:
    print("✅ Enough VRAM for the full pipeline.")

os.environ["TORCH_CUDA_ARCH_LIST"] = cap  # compile the CUDA rasterizer only for this GPU (much faster)

if CACHE_WEIGHTS_IN_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["HF_HOME"] = "/content/drive/MyDrive/hunyuan3d_cache/hf"
else:
    os.environ["HF_HOME"] = "/content/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # multi-connection downloads (several times faster)

try:
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
        print("🔑 Using HF_TOKEN from Colab secrets")
except Exception:
    pass
