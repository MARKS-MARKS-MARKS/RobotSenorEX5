#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-rs_exp5}"
SESSION="${1:-data/sessions/cabinet_02}"
if [ "$#" -gt 0 ]; then
  shift
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${ROOT_DIR}"

python -m exp5 --mode capture --config configs/default.yaml --session "${SESSION}" "$@"
