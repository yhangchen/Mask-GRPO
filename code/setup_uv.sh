#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# Mask-GRPO environment setup — B200 / CUDA 12.8, using uv
#
# 策略：PyTorch 升级到 2.8.0+cu128 以支持 B200，
#       其余依赖尽量沿用原 requirements.txt 的版本，
#       仅在与新 torch 不兼容时才升级。
##############################################################################

VENV_DIR=".venv"
PYTHON_VERSION="3.11"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

# ---------- uv paths ----------
export UV_PYTHON_INSTALL_DIR=/data/banyuanhao/uv/python
export UV_CACHE_DIR=/data/banyuanhao/cache
export UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=managed

if [ -f "$HOME/.local/bin/env" ]; then
  source "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

# ---------- Install uv if missing ----------
if ! command -v uv >/dev/null 2>&1; then
  echo "[INFO] uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "[ERROR] uv still not found in PATH"; exit 1; }

# ---------- 1) venv ----------
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------- 2) 基础工具 + 编译依赖 ----------
uv pip install -U pip setuptools wheel
# ninja: flash-attn 编译加速; hatchling: einops>=0.8 的构建后端
uv pip install ninja hatchling

# ---------- 3) PyTorch 2.8.0 + CUDA 12.8 ----------
# 原版: torch==2.2.1, torchvision==0.17.1, triton==2.2.0
# 升级原因: 2.2.x 不支持 B200 (SM100)
uv pip install --index-url "${TORCH_INDEX_URL}" \
  torch==2.8.0 \
  torchvision

# numpy: 原版 1.24.4 与 Python 3.11 兼容，但 torch 2.8 需要 >=1.26
uv pip install numpy==1.26.3

# ---------- 4) 预装 einops（锁版本，避免 flash-attn 拉新版触发 hatchling 问题）----------
uv pip install einops==0.6.0

# ---------- 5) flash-attn (B200 = SM 100) ----------
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="10.0"
export CUTLASS_NVCC_ARCHS=100
export MAX_JOBS="$(nproc)"
echo "[INFO] Setting MAX_JOBS=${MAX_JOBS} for flash-attn build"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

uv pip install --no-build-isolation --no-binary :all: \
  "flash-attn @ git+https://github.com/Dao-AILab/flash-attention.git@v2.8.0"

# ---------- 5) HuggingFace 生态 ----------
# transformers: 原版 4.41.1, 与 torch 2.8 基本兼容，保留
# diffusers: 原版 0.30.1, 保留
# accelerate: 原版 1.0.1, 保留
uv pip install \
  transformers==4.41.1 \
  diffusers==0.30.1 \
  accelerate==1.0.1 \
  datasets \
  huggingface_hub \
  safetensors==0.4.3 \
  "peft<0.14.0"

# ---------- 6) 训练 / 分布式 ----------
# deepspeed: 原版 0.14.2, 保留
# lightning/pytorch-lightning: 原版 2.2.3, 保留
uv pip install \
  deepspeed==0.16.7 \
  lightning==2.2.3 \
  pytorch-lightning==2.2.3

# ---------- 7) 视觉 / 数据处理 ----------
# 尽量沿用原版本
uv pip install \
  opencv-python-headless \
  pillow==10.3.0 \
  scikit-image==0.22.0 \
  scikit-learn==1.5.0 \
  timm==1.0.3 \
  kornia==0.7.2 \
  webdataset==0.2.86 \
  braceexpand==0.1.7 \
  open-clip-torch==2.24.0

# ---------- 8) 工具类 ----------
uv pip install \
  omegaconf==2.3.0 \
  wandb==0.17.0 \
  tqdm==4.66.4 \
  pandas==1.5.3 \
  requests==2.31.0 \
  pydantic==1.10.15 \
  jsonargparse==4.14.1 \
  packaging==22.0 \
  jaxtyping==0.2.28 \
  typeguard==2.13.3 \
  PyYAML==6.0.1 \
  regex==2024.5.15

# ---------- 9) tokenizer / NLP ----------
uv pip install \
  sentencepiece==0.2.0 \
  tokenizers==0.19.1

# ---------- 10) Qwen VL / CLIP ----------
uv pip install qwen-vl-utils
uv pip install git+https://github.com/openai/CLIP.git

# ---------- 11) 其它 ----------
uv pip install \
  tensorboardX==2.6.2.2 \
  ipykernel \
  matplotlib==3.5.3 \
  psutil==5.9.8 \
  filelock==3.14.0
uv pip install numpy==1.26.3
echo "[INFO] Mask-GRPO environment setup complete."
