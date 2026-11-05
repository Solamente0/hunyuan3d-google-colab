#!/usr/bin/env bash
# Installs Hunyuan3D 2.1 into its own Python 3.10 env on Colab. Safe to rerun.
# 2️⃣ Install everything (safe to rerun: finished steps are skipped)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
cd /content

echo "▶ [1/8] System libraries (OpenGL/X libs for opencv and open3d)"
apt-get update -qq 2>&1 | grep -v "^W: " || true   # hide harmless warnings from Colab's extra apt repos
apt-get install -y -qq libxrender1 libxi6 libxkbcommon0 libsm6 libgl1 libegl1 libglib2.0-0 > /dev/null

echo "▶ [2/8] Cloning Hunyuan3D-2.1"
[ -d Hunyuan3D-2.1 ] || git clone -q --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd /content/Hunyuan3D-2.1

echo "▶ [3/8] Creating a Python 3.10 environment with uv"
# Colab presets uv variables that point it at the system Python (3.12/3.13), and torch 2.5.1 has no
# wheels for 3.13. Clear them so every uv command below uses only our 3.10 env.
unset UV_SYSTEM_PYTHON UV_PYTHON UV_BREAK_SYSTEM_PACKAGES UV_PROJECT_ENVIRONMENT \
      UV_CONSTRAINT UV_BUILD_CONSTRAINT UV_OVERRIDE UV_INDEX_URL UV_EXTRA_INDEX_URL UV_INDEX UV_DEFAULT_INDEX
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
  export PATH=$HOME/.local/bin:$PATH
fi
echo "  $(uv --version)"
ENV=/content/h3d-env
PY=$ENV/bin/python
# Rebuild the env if it's missing or was created with the wrong Python by an earlier run
if ! "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10)' 2>/dev/null; then
  rm -rf "$ENV"
  UV_PYTHON_PREFERENCE=only-managed uv venv -q "$ENV" --python 3.10   # uv downloads its own Python 3.10
fi
export VIRTUAL_ENV=$ENV
export PATH=$ENV/bin:$PATH
echo "  $($PY --version) at $ENV"
UVPIP="uv pip install -q --python $PY"   # always install into the 3.10 env, never the system Python

echo "▶ [4/8] PyTorch 2.5.1 + CUDA 12.4"
$UVPIP torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
$UVPIP setuptools wheel ninja pybind11 "numpy==1.24.4"

echo "▶ [5/8] Python requirements"
# Drop the China-hosted mirrors (slow from Colab), deepspeed (training only) and bpy: bpy==4.0 is gone
# from PyPI, and only the training-data renderer (hy3dshape/tools/render) uses it, not inference.
grep -v -e '^--extra-index-url' -e '^deepspeed' -e '^bpy' requirements.txt > /tmp/h3d_req.txt
# Keep the CUDA torch build and numpy from step 4 (stop the requirements from swapping them)
printf 'torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\nnumpy==1.24.4\n' > /tmp/h3d_pins.txt
$UVPIP -r /tmp/h3d_req.txt -c /tmp/h3d_pins.txt --no-build-isolation-package basicsr
$UVPIP hf_transfer hf_xet -c /tmp/h3d_pins.txt

echo "▶ [6/8] Compiling the CUDA rasterizer and mesh painter"
# torch 2.5.1 is built with CUDA 12.4, so the extensions need a CUDA 12.x compiler. A different 12.x minor
# version is fine (torch only refuses a different major version, e.g. Colab's newer CUDA 13).
nvcc_major() { "$1/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo 0; }
CUDA_HOME=""
for c in /usr/local/cuda $(ls -d /usr/local/cuda-12* 2>/dev/null | sort -V); do
  [ "$(nvcc_major "$c")" = "12" ] && CUDA_HOME=$c
done
if [ -z "$CUDA_HOME" ]; then
  # NVIDIA's apt repo for Ubuntu 24.04 starts at 12.5 (no 12.4). Take the oldest 12.x it offers, and
  # only the compiler + the headers torch extensions need (a fraction of the full ~4 GB toolkit).
  UBU=$(. /etc/os-release; echo ${VERSION_ID//./})
  if ! dpkg -s cuda-keyring > /dev/null 2>&1; then
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBU}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb > /dev/null
  fi
  apt-get update -qq 2>&1 | grep -v "^W: " || true
  CU=""
  for m in 4 5 6 8 9; do
    # (no `| grep -q`: under pipefail its early exit makes apt-cache fail with SIGPIPE)
    if [[ "$(apt-cache policy "cuda-nvcc-12-$m" 2>/dev/null)" == *"Candidate: "[0-9]* ]]; then CU=12-$m; break; fi
  done
  [ -n "$CU" ] || { echo "❌ No CUDA 12.x compiler in NVIDIA's apt repo for ubuntu${UBU}"; exit 1; }
  echo "  Colab's CUDA isn't 12.x; installing the CUDA ${CU/-/.} compiler + headers (a few minutes)"
  apt-get install -y -qq --no-install-recommends cuda-nvcc-$CU cuda-cudart-dev-$CU cuda-cccl-$CU \
    cuda-nvrtc-dev-$CU cuda-profiler-api-$CU cuda-nvtx-$CU libcublas-dev-$CU libcusparse-dev-$CU \
    libcusolver-dev-$CU libcurand-dev-$CU > /dev/null
  CUDA_HOME=/usr/local/cuda-${CU/-/.}
fi
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
echo "  using CUDA at $CUDA_HOME for GPU arch ${TORCH_CUDA_ARCH_LIST:-auto}"

if ! $PY -c "import torch, custom_rasterizer" 2>/dev/null; then
  (cd hy3dpaint/custom_rasterizer && $UVPIP --no-build-isolation .)
fi
(cd hy3dpaint/DifferentiableRenderer && \
  c++ -O3 -Wall -shared -std=c++11 -fPIC $($PY -m pybind11 --includes) mesh_inpaint_processor.cpp \
  -o mesh_inpaint_processor$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"))

echo "▶ [7/8] RealESRGAN checkpoint"
mkdir -p hy3dpaint/ckpt
[ -f hy3dpaint/ckpt/RealESRGAN_x4plus.pth ] || \
  wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt

echo "▶ [8/8] Pre-downloading all model weights (~20 GB, fast parallel download)"
HF_HUB_ENABLE_HF_TRANSFER=1 $PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("tencent/Hunyuan3D-2.1")
snapshot_download("facebook/dinov2-giant", allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("weights ready")
EOF

echo "▶ Self-test"
$PY - <<'EOF'
import torch, custom_rasterizer
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
EOF
echo "✅ Install finished. Run cell 3."
