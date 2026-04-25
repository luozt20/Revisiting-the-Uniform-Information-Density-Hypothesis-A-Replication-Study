PYTHON ?= python3
VENV ?= .venv
ROOT_DIR := $(CURDIR)
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
JUPYTER := $(VENV)/bin/jupyter
CMAKE := $(if $(wildcard $(VENV)/bin/cmake),$(VENV)/bin/cmake,cmake)
R_LIBS_USER := $(ROOT_DIR)/.Rlibs
XDG_CACHE_HOME := $(ROOT_DIR)/.cache
HF_HOME := $(XDG_CACHE_HOME)/huggingface
MPLCONFIGDIR := $(XDG_CACHE_HOME)/matplotlib
NLTK_DATA := $(XDG_CACHE_HOME)/nltk_data
COMMON_ENV = R_LIBS_USER="$(R_LIBS_USER)" XDG_CACHE_HOME="$(XDG_CACHE_HOME)" HF_HOME="$(HF_HOME)" MPLCONFIGDIR="$(MPLCONFIGDIR)" NLTK_DATA="$(NLTK_DATA)" TOKENIZERS_PARALLELISM=false

.PHONY: check-env venv install install-r build-kenlm build-wiki-arpa download-public-data notebook public-notebook run-public-replication onestop-extension-notebook run-onestop-extension mega-acceptability-extension-notebook run-mega-acceptability-extension

check-env:
	$(COMMON_ENV) $(if $(wildcard $(PY)),$(PY),$(PYTHON)) scripts/check_replication_env.py

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

install-r:
	$(COMMON_ENV) Rscript scripts/install_r_dependencies.R

build-kenlm:
	$(CMAKE) -S kenlm -B kenlm/build
	$(CMAKE) --build kenlm/build -j 4

build-wiki-arpa:
	$(COMMON_ENV) bash scripts/build_wiki_arpa.sh

download-public-data:
	$(COMMON_ENV) bash scripts/prepare_public_data.sh

notebook:
	cd src && $(COMMON_ENV) ../$(JUPYTER) notebook revisiting-uid.ipynb

public-notebook:
	$(COMMON_ENV) $(PY) scripts/build_public_replication_notebook.py

run-public-replication:
	$(COMMON_ENV) bash scripts/run_public_replication.sh

onestop-extension-notebook:
	$(COMMON_ENV) $(PY) scripts/build_onestop_extension_notebook.py

run-onestop-extension: onestop-extension-notebook
	cd src && $(COMMON_ENV) ../$(JUPYTER) nbconvert \
	  --to notebook \
	  --execute additional_plan_onestop_uid_extension.ipynb \
	  --output additional_plan_onestop_uid_extension.executed.ipynb \
	  --ExecutePreprocessor.timeout=-1 \
	  --ExecutePreprocessor.kernel_name=python3

mega-acceptability-extension-notebook:
	$(COMMON_ENV) $(PY) scripts/build_mega_acceptability_extension_notebook.py

run-mega-acceptability-extension: mega-acceptability-extension-notebook
	cd src && $(COMMON_ENV) ../$(JUPYTER) nbconvert \
	  --to notebook \
	  --execute additional_plan_mega_acceptability_extension.ipynb \
	  --output additional_plan_mega_acceptability_extension.executed.ipynb \
	  --ExecutePreprocessor.timeout=-1 \
	  --ExecutePreprocessor.kernel_name=python3
