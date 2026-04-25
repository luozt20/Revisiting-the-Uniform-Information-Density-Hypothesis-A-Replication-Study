# Revisiting UID Replication Guide

**Credit.** This replication study is based on and adapted from the original [`rycolab/revisiting-uid`](https://github.com/rycolab/revisiting-uid.git) repository for "Revisiting the Uniform Information Density Hypothesis."

This guide translates the EMNLP 2021 paper "Revisiting the Uniform Information Density Hypothesis" into a local execution plan for the team project.

## Recommended Scope

For the class project, prioritize the paper's main empirical findings before any extensions:

1. Reproduce the acceptability experiments on CoLA and BNC.
2. Reproduce the sentence-level reading-time experiments on Natural Stories, Provo, and UCL first.
3. Add Brown only if you can recover the original data bundle from a working mirror; the original Google Drive link is now dead.
4. Treat Dundee and GECO as optional extensions because they are not fully bundled with the repo.
5. Treat Case Study 2 and Case Study 3 as stretch goals after the main psychometric predictions run.

This matches the proposal more closely than trying to finish every appendix result first.

## Main Claims To Replicate

The paper's central claims are:

- Acceptability judgments show clearer evidence for a super-linear surprisal effect than reading times.
- For the power sweep over sentence surprisal, values with `k > 1` outperform the linear baseline (`k = 1`) on acceptability.
- Reading-time results are weaker, but still compatible with a slight super-linear effect.
- A regression toward a language-level mean surprisal can outperform sentence-level or purely local variability measures.

## Paper To Notebook Map

- Data loading and surprisal scoring: notebook cells 19-76
- Main psychometric prediction experiments: notebook cells 78-94
- Alternative UID operationalizations: notebook cells 95-106
- Correlation appendix analyses: notebook cells 108-111

The practical "main result" for this project is the section around cells 89-94.

## Data Availability

Publicly obtainable in this repo workflow:

- CoLA
- Provo
- UCL self-paced reading and eye-tracking
- BNC
- Natural Stories, cached locally from the public GitHub repo
- WikiText-103 for the KenLM 5-gram model

Not bundled or access-constrained:

- Brown: the original Google Drive bundle used by the repo now returns `404`
- Dundee corpus: the upstream README states that the original authors must be contacted
- GECO Dutch materials: referenced by the notebook, but not downloaded by the provided scripts

## Local Workflow

From the repo root:

```bash
make check-env
make install
make install-r
make build-kenlm
make download-public-data
make build-wiki-arpa
make public-notebook
make run-public-replication
```

If you want the cleanest path to a first result, start with the acceptability analysis after CoLA and BNC are available, then return to the reading-time datasets.

`make run-public-replication` executes a derived notebook that:

- uses local `src/corpora/naturalstories/` files instead of live GitHub reads
- skips the stale `load_stats(...)` cells that expect precomputed pickle files
- skips `transxl` in the public notebook because the deprecated TransfoXL path is no longer compatible with the current `transformers` stack
- leaves Brown, Dundee, and GECO empty when the data is unavailable
- preserves the main acceptability and public reading-time analysis blocks

## What Counts As A Successful First Replication

A solid first milestone is:

- The notebook runs through data preparation without missing-file errors.
- The `acceptability` dataframe is created for CoLA and BNC.
- The `k` sweep in cells 91-94 peaks above the linear baseline for at least part of the acceptability setup.
- The direction of the trend agrees with the paper even if exact numbers differ.
- Any missing Brown/Dundee/GECO result is documented explicitly as a data-access limitation rather than silently omitted.

Only after that should you spend time on Dundee, GECO, or later case studies.

## Report-Facing Deliverables

For the final project write-up, collect:

1. A short methods section documenting models, datasets, and preprocessing choices actually used locally.
2. One figure reproducing the power sweep over `k` for acceptability.
3. One table or figure summarizing reading-time results on the public datasets you could run.
4. A short discrepancy section explaining any differences from the paper due to missing corpora, package drift, or hardware limits.
