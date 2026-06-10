#!/usr/bin/env bash
set -u

ENV_NAME="${ENV_NAME:-pytorch}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "[FAIL] conda not found"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}" || exit 1

echo "[INFO] repo: ${ROOT_DIR}"
echo "[INFO] python: $(which python)"
python --version

echo
echo "[INFO] NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "[WARN] nvidia-smi failed; CUDA may be unavailable until the driver is fixed."
  if [ ! -e /dev/nvidia0 ]; then
    echo "[INFO] /dev/nvidia0 is not visible here; this can happen inside a sandbox/container even when CUDA works in your normal terminal."
  fi
else
  echo "[WARN] nvidia-smi not found"
fi

echo
echo "[INFO] Python packages"
python - <<'PY'
import importlib.util as u

mods = ["torch", "cv2", "numpy", "pyrealsense2", "open3d", "transformers", "yaml"]
for mod in mods:
    print(f"{mod:14s}: {bool(u.find_spec(mod))}")

try:
    import torch
    print("torch version  :", torch.__version__)
    print("cuda available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device    :", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch error    :", exc)
PY

echo
if [ "${CHECK_REALSENSE:-0}" = "1" ]; then
  echo "[INFO] RealSense devices"
  if command -v rs-enumerate-devices >/dev/null 2>&1; then
    rs-enumerate-devices || echo "[WARN] No RealSense device detected or permission/udev is not ready."
  else
    echo "[WARN] rs-enumerate-devices not found"
  fi
else
  echo "[INFO] RealSense device check skipped. Set CHECK_REALSENSE=1 to enable it."
fi
