#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-pytorch}"
SESSION="${1:-data/sessions/demo}"
if [ "$#" -gt 0 ]; then
  shift
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${ROOT_DIR}"

if [ ! -f "${SESSION}/color.png" ]; then
  python -m exp5 --mode demo --session "${SESSION}"
fi

python -m exp5 --mode replay --config configs/default.yaml --session "${SESSION}" "$@"
