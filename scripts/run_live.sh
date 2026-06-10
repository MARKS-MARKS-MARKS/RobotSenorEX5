#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-rs_exp5}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${ROOT_DIR}"

if [ -d /usr/share/fonts/truetype/dejavu ]; then
  export QT_QPA_FONTDIR="${QT_QPA_FONTDIR:-/usr/share/fonts/truetype/dejavu}"
  CV2_QT_FONT_DIR="$(python - <<'PY'
from pathlib import Path
import cv2
print(Path(cv2.__file__).resolve().parent / "qt" / "fonts")
PY
)"
  mkdir -p "${CV2_QT_FONT_DIR}"
  find /usr/share/fonts/truetype/dejavu -maxdepth 1 -name '*.ttf' -exec ln -sf {} "${CV2_QT_FONT_DIR}" \;
fi

python -m exp5 --mode live --config configs/default.yaml "$@"
