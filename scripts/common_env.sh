#!/usr/bin/env bash

if [ -n "${ROOT_DIR:-}" ]; then
  ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
elif [ -n "${BASH_SOURCE[0]:-}" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
elif [ -f "${PWD}/scripts/common_env.sh" ]; then
  ROOT_DIR="${PWD}"
elif [ -f "${PWD}/common_env.sh" ]; then
  ROOT_DIR="$(cd "${PWD}/.." && pwd)"
else
  echo "Unable to infer ROOT_DIR. Run this from the repository root or export ROOT_DIR first." >&2
  return 1 2>/dev/null || exit 1
fi

export R_LIBS_USER="${R_LIBS_USER:-${ROOT_DIR}/.Rlibs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
export NLTK_DATA="${NLTK_DATA:-${XDG_CACHE_HOME}/nltk_data}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
export LANG="${LANG:-en_US.UTF-8}"
export LC_CTYPE="${LC_CTYPE:-en_US.UTF-8}"

if [ -z "${R_HOME:-}" ] && command -v R >/dev/null 2>&1; then
  export R_HOME="$(R RHOME)"
fi

mkdir -p "${R_LIBS_USER}" "${HF_HOME}" "${MPLCONFIGDIR}" "${NLTK_DATA}"
