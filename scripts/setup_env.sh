#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-rs_exp5}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu121}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

echo "Installing GPU PyTorch from ${CUDA_WHEEL_INDEX}"
python -m pip install --upgrade torch torchvision torchaudio --index-url "${CUDA_WHEEL_INDEX}"

python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo
echo "Environment ${ENV_NAME} is ready."
python - <<'PY'
import torch
print("Python environment check")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Check NVIDIA driver with nvidia-smi.")
PY
