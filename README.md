# Revisiting the Uniform Information Density Hypothesis: Replication Study

**Credit.** This replication study is based on and adapted from the original [`rycolab/revisiting-uid`](https://github.com/rycolab/revisiting-uid.git) repository for the EMNLP 2021 paper "Revisiting the Uniform Information Density Hypothesis."

## What This Project Is

This repository is a course-project replication of the main empirical analyses from "Revisiting the Uniform Information Density Hypothesis." The original paper asks whether human processing difficulty is best predicted by linear surprisal, as a strict UID account might suggest, or by transformations of surprisal that allow non-linear effects.

Our replication focuses on the part of the paper that can be rerun with publicly available data:

- acceptability judgments from CoLA and BNC
- sentence-level reading-time analyses for public datasets, especially Natural Stories and Provo
- comparison of sentence surprisal transformed by different exponents `k`
- comparison against the paper's baseline acceptability predictors, SLOR and NormLP

The practical target is not to reproduce every appendix number exactly. Instead, the goal is to recover the paper's main qualitative pattern under a modern local environment and document where data availability or package drift prevents exact reproduction.

In that sense, this repository should be read as a successful replication of a well-defined public subset of the original project, not as a complete reproduction of every dataset, model, and appendix analysis from the original repository.

## What We Changed

The original repository is preserved as the historical base, but this replication adds a more reproducible public-data workflow:

- a root-level `Makefile` for setup, data preparation, model building, and notebook execution
- `scripts/check_replication_env.py` to verify Python, R, KenLM, and data prerequisites
- `scripts/prepare_public_data.sh` to download and organize public datasets
- `scripts/build_wiki_arpa.sh` to build the WikiText-103 KenLM 5-gram model
- `scripts/build_public_replication_notebook.py` to generate `src/revisiting-uid-public.ipynb`
- a public notebook path that skips unavailable Brown/Dundee/GECO inputs when needed
- a modernized plotting style for the acceptability baseline figure

Generated data, checkpoints, figures, and the large KenLM ARPA file are intentionally ignored by git.

## Data Availability

The public workflow can use:

- CoLA
- BNC
- Natural Stories
- Provo
- UCL self-paced reading and eye-tracking files
- WikiText-103 for the 5-gram language model

The following original datasets are treated as unavailable or optional in this replication:

- Brown: the original Google Drive archive referenced by the upstream code currently returns `404`
- Dundee: the upstream README states that access requires contacting the original authors
- GECO: referenced by the notebook, but not downloaded by the provided public-data scripts

One original model branch is also not included in the public rerun:

- TransfoXL: the upstream notebook includes a `transxl` path, but the corresponding Hugging Face `TransfoXL` stack is deprecated and is not compatible with the current environment used for this replication. The public notebook therefore keeps GPT-2, BERT, and the KenLM 5-gram model, and skips TransfoXL.

Because of these constraints, the strongest claims in this repository are about the public subset that successfully executes locally: CoLA/BNC acceptability, the available public reading-time datasets, and the GPT-2/BERT/5-gram model comparisons. It is not a full reproduction of the original project's complete data and model coverage.

## How To Run

Clone the repository with submodules:

```bash
git clone --recursive git@github.com:luozt20/Revisiting-the-Uniform-Information-Density-Hypothesis-A-Replication-Study.git
cd Revisiting-the-Uniform-Information-Density-Hypothesis-A-Replication-Study
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

Set up the Python and R environment:

```bash
make check-env
make install
make install-r
```

Build KenLM and prepare the public datasets:

```bash
make build-kenlm
make download-public-data
make build-wiki-arpa
```

Generate and run the public replication notebook:

```bash
make public-notebook
make run-public-replication
```

The executed notebook is written to:

```text
src/revisiting-uid-public.executed.ipynb
```

Figures are written to:

```text
src/figures/
```

Checkpoints and intermediate tables are written to:

```text
src/checkpoints/
```

## Expected Outputs

The public notebook should produce the main replication artifacts:

- `figure_reading_acceptability_delta_loglik`: power-sweep comparison for reading time and acceptability
- `figure_acceptability_baselines`: acceptability power sweep compared with SLOR and NormLP baselines
- `figure_surprisal_acceptability_correlation`: correlation between transformed surprisal and acceptability
- `figure_case_study3_windows`: comparison of alternative context windows where public data supports it

The most useful CSV checkpoints for inspection are:

- `src/checkpoints/acceptability_cv.csv`
- `src/checkpoints/lau_acceptability_cv.csv`
- `src/checkpoints/reading_time_cv.csv`
- `src/checkpoints/case_study2_variance.csv`
- `src/checkpoints/case_study3_all_vars.csv`

These generated files are not committed because they are reproducible outputs rather than source files.

## Additional Extension: OneStop

The first additional-plan extension is implemented in:

```text
src/additional_plan_onestop_uid_extension.ipynb
```

This notebook is separate from the main replication notebook. It downloads the OneStop Ordinary Reading
word-level Interest Area Report, aggregates it into paragraph-level participant trials, and tests whether
the UID power-sweep pattern generalizes to OneStop Advanced vs Elementary text versions.

To generate and run it:

```bash
make run-onestop-extension
```

The extension writes:

- `src/checkpoints/onestop_ordinary_trial_features.csv`
- `src/checkpoints/onestop_ordinary_uid_cv.csv`
- `src/figures/figure_onestop_ordinary_uid_by_difficulty.png`

## Additional Extension: MegaAcceptability

The second additional-plan extension is implemented in:

```text
src/additional_plan_mega_acceptability_extension.ipynb
```

This notebook is separate from both the main replication notebook and the OneStop extension. It downloads
MegaAcceptability v1.2, uses the normalized clause-embedding acceptability scores, scores each sentence
with the existing WikiText-103 KenLM 5-gram model, and tests whether the acceptability-side super-linear
surprisal pattern generalizes to this new structured acceptability dataset.

To generate and run it:

```bash
make run-mega-acceptability-extension
```

The extension writes:

- `src/checkpoints/mega_acceptability_ngram_features.csv`
- `src/checkpoints/mega_acceptability_uid_correlations.csv`
- `src/checkpoints/mega_acceptability_uid_cv.csv`
- `src/figures/figure_mega_acceptability_uid_power_sweep.png`

## Results And Conclusions

The original paper's conclusion is nuanced: reading-time results are broadly compatible with earlier linear-surprisal findings, but also leave room for a weakly super-linear effect; acceptability judgments give clearer evidence that non-uniform information density predicts lower acceptability; and global, language-level operationalizations of UID tend to explain the psychometric data better than local alternatives.

Our public replication recovers the clearest part of that conclusion for the data that can be rerun locally. Transformed surprisal often improves prediction over the linear `k = 1` baseline in acceptability judgments, while the public reading-time results show smaller and less decisive differences across `k`.

In the local public-data run:

- CoLA shows best-performing exponents above `k = 1` for BERT, GPT-2, and the 5-gram model.
- BNC shows a strong super-linear pattern for BERT, while GPT-2 and the 5-gram model are less consistently super-linear in this local run.
- SLOR and NormLP remain competitive baselines in some acceptability settings, so the result is not simply that every transformed-surprisal model dominates every baseline.
- Public reading-time results are weaker than acceptability results. Natural Stories and Provo show only small differences across `k`, which is consistent with the paper's claim that reading-time evidence does not reject a linear surprisal account even though weak super-linearity remains plausible.

Overall, this replication supports the original paper's broad contrast for the public subset we could rerun: acceptability judgments provide clearer evidence for non-linear UID-related effects than reading-time data, while the available reading-time results remain compatible with both linear surprisal and weak super-linear UID interpretations. Exact numerical agreement with the complete original project is not expected because Brown/Dundee/GECO are not fully available in this workflow, TransfoXL is skipped for compatibility reasons, and modern Python/R/Hugging Face dependencies differ from the original release environment.

For the OneStop extension, the preliminary result is more cautious. In ordinary reading, adding paragraph-level GPT-2 `surprisal^k` predictors yields only very small improvements over length/frequency baselines, and the preferred exponent does not show the same clear super-linear pattern seen in the acceptability analyses. This should be interpreted as a boundary-condition result for the current extension design, not as a refutation of the original paper: OneStop is a new eye-tracking corpus, the analysis is paragraph-level rather than the original sentence-level setup, and the first extension currently uses only the ordinary-reading regime.

For the MegaAcceptability extension, the preliminary result is more directly supportive of the original acceptability conclusion. With the WikiText-103 KenLM 5-gram model, the simple correlation between negative `surprisal^k` and normalized acceptability peaks around `k = 1.25`, and the controlled 5-fold CV analysis peaks around `k = 1.5` after accounting for length, verb identity, and syntactic frame. This suggests that the acceptability-side super-linear pattern generalizes beyond CoLA/BNC to a fine-grained clause-embedding judgment dataset, with the caveat that this extension currently uses an n-gram model rather than GPT-2/BERT.

## Repository Map

- `src/revisiting-uid.ipynb`: original analysis notebook from the upstream repository
- `src/revisiting-uid-public.ipynb`: generated public-data replication notebook
- `src/language_modeling.py`: language-model scoring utilities
- `scripts/`: setup, data, environment, and notebook-generation helpers
- `Makefile`: end-to-end local workflow targets
- `requirements.txt`: Python dependencies
- `kenlm/`: KenLM submodule used for the 5-gram model

## Limitations

This repository should be read as a transparent public-data subset replication rather than a perfect archival rerun of every original experiment. The main limitations are missing Brown/Dundee/GECO inputs, the skipped TransfoXL branch, model-version drift in the `transformers` ecosystem, and the computational cost of rebuilding language-model scores from scratch.
