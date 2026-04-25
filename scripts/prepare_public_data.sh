#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/common_env.sh"
SRC_DIR="${ROOT_DIR}/src"
DATA_DIR="${SRC_DIR}/corpora"
WIKITEXT_ARCHIVE="${SRC_DIR}/wikitext-103-v1.zip"
WIKITEXT_DIR="${SRC_DIR}/wikitext-103"
NATURAL_STORIES_DIR="${DATA_DIR}/naturalstories"
GDOWN_BIN=""
PYTHON_BIN="python3"

if [ -x "${ROOT_DIR}/.venv/bin/gdown" ]; then
  GDOWN_BIN="${ROOT_DIR}/.venv/bin/gdown"
elif command -v gdown >/dev/null 2>&1; then
  GDOWN_BIN="$(command -v gdown)"
fi

if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

download_file() {
  local url="$1"
  local output="$2"

  if [ -f "${output}" ]; then
    echo "Skipping $(basename "${output}"), already present."
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  else
    echo "Need either curl or wget to download ${url}."
    exit 1
  fi
}

zip_is_valid() {
  local archive="$1"
  unzip -tqq "${archive}" >/dev/null 2>&1
}

mkdir -p "${DATA_DIR}"
mkdir -p "${NATURAL_STORIES_DIR}"

download_file "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip" "${DATA_DIR}/cola.zip"
if [ ! -d "${DATA_DIR}/cola_public" ]; then
  unzip -oq "${DATA_DIR}/cola.zip" -d "${DATA_DIR}"
fi

download_file "https://osf.io/a32be/download" "${DATA_DIR}/provo.csv"
download_file "https://osf.io/e4a2m/download" "${DATA_DIR}/provo_norms.csv"

download_file "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip" "${DATA_DIR}/ucl.zip"
if [ ! -f "${DATA_DIR}/ucl/selfpacedreading.RT.txt" ]; then
  mkdir -p "${DATA_DIR}/ucl"
  unzip -oq "${DATA_DIR}/ucl.zip" -d "${DATA_DIR}/ucl"
fi

download_file "https://raw.githubusercontent.com/languageMIT/naturalstories/master/probs/all_stories_gpt3.csv" "${NATURAL_STORIES_DIR}/all_stories_gpt3.csv"
download_file "https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS/processed_RTs.tsv" "${NATURAL_STORIES_DIR}/processed_RTs.tsv"

download_file "https://gu-clasp.github.io/914a288ca1e127a7f1547412d9a7e056/bnc.csv" "${DATA_DIR}/bnc.csv"
if grep -q "<!DOCTYPE html>" "${DATA_DIR}/bnc.csv"; then
  echo "Primary BNC mirror returned HTML; using GitHub fallback."
  download_file "https://raw.githubusercontent.com/ancadumitrache/CrowdTruth-grammatical-correctness/14ff1aa7aa030fb538cc880b6306d220cce63664/MOP2_final.csv" "${DATA_DIR}/bnc_fallback_raw.csv"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/convert_bnc_fallback.py" "${DATA_DIR}/bnc_fallback_raw.csv" "${DATA_DIR}/bnc.csv"
fi

if [ ! -f "${DATA_DIR}/brown_spr.csv" ]; then
  if [ -n "${GDOWN_BIN}" ]; then
    set +e
    "${GDOWN_BIN}" --fuzzy "https://drive.google.com/file/d/1e-anJ4laGlTY-E0LNook1EzKBU2S1jI8/view?usp=sharing" -O "${DATA_DIR}/data.zip"
    brown_status=$?
    set -e

    if [ "${brown_status}" -eq 0 ]; then
      unzip -oq "${DATA_DIR}/data.zip" -d "${DATA_DIR}"
      if [ -d "${DATA_DIR}/data/corpora" ]; then
        mv -f "${DATA_DIR}/data/corpora/"*brown* "${DATA_DIR}/" 2>/dev/null || true
        rm -rf "${DATA_DIR}/data"
      fi
    else
      echo "Skipping Brown corpus because the upstream Google Drive download failed."
    fi
  else
    echo "Skipping Brown corpus because gdown is not installed yet."
  fi
fi

download_file "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip" "${WIKITEXT_ARCHIVE}"
if ! zip_is_valid "${WIKITEXT_ARCHIVE}"; then
  echo "Primary WikiText-103 archive is invalid; using DeepAI fallback."
  rm -f "${WIKITEXT_ARCHIVE}"
  download_file "https://data.deepai.org/wikitext-103.zip" "${WIKITEXT_ARCHIVE}"
fi

if unzip -Z1 "${WIKITEXT_ARCHIVE}" | grep -q "wiki.train.tokens"; then
  if [ ! -d "${WIKITEXT_DIR}" ]; then
    unzip -oq "${WIKITEXT_ARCHIVE}" -d "${SRC_DIR}"
  fi
else
  mkdir -p "${WIKITEXT_DIR}"
  rm -rf "${SRC_DIR}/wikitext-103-raw"
  unzip -oq "${WIKITEXT_ARCHIVE}" -d "${SRC_DIR}/wikitext-103-raw"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/convert_wikitext_csv.py" \
    "${SRC_DIR}/wikitext-103-raw/wikitext-103/train.csv" \
    "${WIKITEXT_DIR}/wiki.train.tokens"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/convert_wikitext_csv.py" \
    "${SRC_DIR}/wikitext-103-raw/wikitext-103/test.csv" \
    "${WIKITEXT_DIR}/wiki.test.tokens"
fi

"${PYTHON_BIN}" -m nltk.downloader -d "${NLTK_DATA}" punkt punkt_tab

cat <<'EOF'

Public data setup complete.

Still manual or optional:
- Brown corpus, because the original Google Drive file currently returns 404
- Dundee corpus access
- GECO Dutch materials

The public notebook reads Natural Stories from local cached files in `src/corpora/naturalstories/`.
EOF
