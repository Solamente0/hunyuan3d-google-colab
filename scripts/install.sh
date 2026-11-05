#!/usr/bin/env bash
# Installs Hunyuan3D 2.1 into its own Python 3.10 env on Colab. Safe to rerun.
# 2️⃣ Install everything (safe to rerun: finished steps are skipped)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
cd /content

echo "▶ [1/8] System libraries (needed by Blender's bpy)"
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
# Drop the China-hosted mirrors (slow from Colab) and deepspeed (training only)
grep -v -e '^--extra-index-url' -e '^deepspeed' requirements.txt > /tmp/h3d_req.txt
# Keep the CUDA torch build and numpy from step 4 (stop the requirements from swapping them)
printf 'torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\nnumpy==1.24.4\n' > /tmp/h3d_pins.txt
$UVPIP -r /tmp/h3d_req.txt -c /tmp/h3d_pins.txt --no-build-isolation-package basicsr
$UVPIP hf_transfer hf_xet -c /tmp/h3d_pins.txt

echo "▶ [6/8] Compiling the CUDA rasterizer and mesh painter"
NVCC_MAJOR=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo 0)
if [ "$NVCC_MAJOR" = "12" ]; then
  export CUDA_HOME=/usr/local/cuda
else
  CAND=$(ls -d /usr/local/cuda-12* 2>/dev/null | tail -1 || true)
  if [ -n "$CAND" ]; then
    export CUDA_HOME=$CAND
  else
    echo "  System CUDA isn't 12.x; installing the CUDA 12.4 toolkit (a few minutes)"
    UBU=$(. /etc/os-release; echo ${VERSION_ID//./})
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBU}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb > /dev/null
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4 > /dev/null
    export CUDA_HOME=/usr/local/cuda-12.4
  fi
fi
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
import torch, custom_rasterizer, bpy
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
EOF
echo "✅ Install finished. Run cell 3."
