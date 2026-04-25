#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/common_env.sh"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
JUPYTER_BIN="${ROOT_DIR}/.venv/bin/jupyter"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_public_replication_notebook.py"

cd "${ROOT_DIR}/src"
"${JUPYTER_BIN}" nbconvert \
  --to notebook \
  --execute revisiting-uid-public.ipynb \
  --output revisiting-uid-public.executed.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3
