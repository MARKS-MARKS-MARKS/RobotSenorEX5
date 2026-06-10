#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-pytorch}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${ROOT_DIR}"

python -m exp5.web_server --host "${HOST}" --port "${PORT}" "$@"
