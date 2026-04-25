#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/common_env.sh"

TRAIN_TOKENS="${ROOT_DIR}/src/wikitext-103/wiki.train.tokens"
LMPLZ_BIN="${ROOT_DIR}/kenlm/build/bin/lmplz"
ARPA_PATH="${ROOT_DIR}/src/wiki.arpa"
CLEAN_TOKENS="${XDG_CACHE_HOME}/wiki.train.tokens.clean"

if [ ! -f "${TRAIN_TOKENS}" ]; then
  echo "Missing ${TRAIN_TOKENS}. Run make download-public-data first."
  exit 1
fi

if [ ! -x "${LMPLZ_BIN}" ]; then
  echo "Missing ${LMPLZ_BIN}. Run make build-kenlm first."
  exit 1
fi

awk '!/=\s*/' "${TRAIN_TOKENS}" | awk 'NF' > "${CLEAN_TOKENS}"
"${LMPLZ_BIN}" -o 5 --skip_symbols < "${CLEAN_TOKENS}" > "${ARPA_PATH}"

echo "Wrote ${ARPA_PATH}"
