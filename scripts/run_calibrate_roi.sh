#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-rs_exp5}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${ROOT_DIR}"

python -m exp5 --mode calibrate_roi --config configs/default.yaml "$@"
