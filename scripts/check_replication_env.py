#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_PYTHON_PACKAGES = [
    "numpy",
    "pandas",
    "nltk",
    "matplotlib",
    "scipy",
    "torch",
    "transformers",
    "rpy2",
    "gdown",
    "mosestokenizer",
    "sklearn",
    "plotnine",
    "chardet",
]

REQUIRED_R_PACKAGES = [
    "lme4",
    "lmerTest",
    "ggplot2",
    "dplyr",
]

REQUIRED_COMMANDS = [
    "python3",
    "Rscript",
    "make",
]

ONE_OF_COMMANDS = {
    "downloader": ["curl", "wget"],
    "unzip": ["unzip"],
}

CORE_DATA_PATHS = {
    "CoLA": "src/corpora/cola_public/raw/in_domain_train.tsv",
    "Provo": "src/corpora/provo.csv",
    "Provo norms": "src/corpora/provo_norms.csv",
    "UCL self-paced": "src/corpora/ucl/selfpacedreading.RT.txt",
    "UCL eye-tracking": "src/corpora/ucl/eyetracking.RT.txt",
    "BNC": "src/corpora/bnc.csv",
    "Natural Stories GPT-3 probs": "src/corpora/naturalstories/all_stories_gpt3.csv",
    "Natural Stories RTs": "src/corpora/naturalstories/processed_RTs.tsv",
    "WikiText-103": "src/wikitext-103/wiki.train.tokens",
    "KenLM 5-gram arpa": "src/wiki.arpa",
    "KenLM binary": "kenlm/build/bin/lmplz",
}

OPTIONAL_DATA_PATHS = {
    "Brown": "src/corpora/brown_spr.csv",
    "Dundee corpus": "src/corpora/dundee",
    "GECO materials": "src/corpora/DutchMaterials.csv",
    "GECO reading data": "src/corpora/L1ReadingData.csv",
}


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def run_r_package_check() -> dict[str, bool]:
    if shutil.which("Rscript") is None:
        return {pkg: False for pkg in REQUIRED_R_PACKAGES}

    repo_root = Path(__file__).resolve().parents[1]
    package_vector = ", ".join(f'"{pkg}"' for pkg in REQUIRED_R_PACKAGES)
    script = (
        f"pkgs <- c({package_vector})\n"
        "status <- vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)\n"
        "writeLines(paste(names(status), status, sep='\\t'))\n"
    )
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(repo_root / ".Rlibs")
    proc = subprocess.run(
        ["Rscript", "-"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        return {pkg: False for pkg in REQUIRED_R_PACKAGES}

    status: dict[str, bool] = {}
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            continue
        name, value = line.split("\t", 1)
        status[name] = value.strip().upper() == "TRUE"

    for pkg in REQUIRED_R_PACKAGES:
        status.setdefault(pkg, False)
    return status


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("R_LIBS_USER", str(repo_root / ".Rlibs"))
    print(f"Repository root: {repo_root}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    print_section("Command-line tools")
    missing_commands: list[str] = []
    for command in REQUIRED_COMMANDS:
        found = shutil.which(command) is not None
        print(f"[{'OK' if found else 'MISSING'}] {command}")
        if not found:
            missing_commands.append(command)

    cmake_candidates = [
        shutil.which("cmake"),
        str(repo_root / ".venv/bin/cmake"),
    ]
    cmake_found = any(candidate and Path(candidate).exists() for candidate in cmake_candidates)
    print(f"[{'OK' if cmake_found else 'MISSING'}] cmake")
    if not cmake_found:
        missing_commands.append("cmake")

    for label, candidates in ONE_OF_COMMANDS.items():
        found = any(shutil.which(candidate) is not None for candidate in candidates)
        joined = " or ".join(candidates)
        print(f"[{'OK' if found else 'MISSING'}] {label}: {joined}")
        if not found:
            missing_commands.append(label)

    print_section("Python packages")
    missing_python: list[str] = []
    for package in REQUIRED_PYTHON_PACKAGES:
        found = has_module(package)
        print(f"[{'OK' if found else 'MISSING'}] {package}")
        if not found:
            missing_python.append(package)

    print_section("R packages")
    r_status = run_r_package_check()
    missing_r = [pkg for pkg, found in r_status.items() if not found]
    for package in REQUIRED_R_PACKAGES:
        found = r_status.get(package, False)
        print(f"[{'OK' if found else 'MISSING'}] {package}")

    print_section("Core replication files")
    missing_paths: list[str] = []
    for label, relative_path in CORE_DATA_PATHS.items():
        path = repo_root / relative_path
        found = path.exists()
        print(f"[{'OK' if found else 'MISSING'}] {label}: {relative_path}")
        if not found:
            missing_paths.append(label)

    print_section("Optional extension files")
    for label, relative_path in OPTIONAL_DATA_PATHS.items():
        path = repo_root / relative_path
        found = path.exists()
        print(f"[{'OK' if found else 'MISSING'}] {label}: {relative_path}")

    print_section("Next steps")
    if missing_python:
        print("Run `make install` to create `.venv` and install Python dependencies.")
    if missing_r:
        print("Run `make install-r` to install the R packages used by the notebook into `.Rlibs`.")
    if missing_paths:
        print("Run `make download-public-data` for the public datasets, Natural Stories cache, NLTK punkt data, and WikiText-103.")
    if not (repo_root / "kenlm/build/bin/lmplz").exists():
        print("Run `make build-kenlm` to compile the 5-gram language-model tooling.")
    if not (repo_root / "src/wiki.arpa").exists():
        print("Run `make build-wiki-arpa` to build the 5-gram `wiki.arpa` model used by the notebook.")
    print("Core project priority: reproduce acceptability first, then the public reading-time datasets.")
    print("Brown is currently optional because the original Google Drive archive returns 404.")
    print("Dundee and GECO are optional until the public-data pipeline is working.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
