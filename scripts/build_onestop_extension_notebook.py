#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NOTEBOOK = REPO_ROOT / "src" / "additional_plan_onestop_uid_extension.ipynb"


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def main() -> int:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Additional Plan: OneStop UID Extension

            This standalone notebook extends the replication study to a new public eye-tracking dataset:
            [OneStop Eye Movements](https://lacclab.github.io/OneStop-Eye-Movements/).

            The goal is not to modify the original replication notebook. Instead, this notebook asks whether
            the UID/surprisal power-sweep pattern generalizes to OneStop ordinary reading, and whether the
            pattern differs between the Advanced and Elementary versions of the same texts.

            OneStop is useful for this extension because it provides word-level Interest Area Reports,
            text difficulty labels, and precomputed GPT-2 surprisal annotations.
            """
        ),
        md(
            """
            ## Research Question

            The original paper's conclusion is deliberately nuanced. It reports clearer evidence for
            non-linear UID-related effects in acceptability judgments, while reading-time results remain
            compatible with earlier linear-surprisal accounts and also leave room for weak super-linearity.
            OneStop lets us ask a more targeted generalization question:

            **Do transformed GPT-2 surprisal predictors improve paragraph reading-time prediction in a new
            eye-tracking corpus, and does the preferred exponent differ for Advanced vs Elementary texts?**

            This is a boundary-condition test for UID: if the original pattern generalizes strongly, a similar
            weakly super-linear power sweep should appear in OneStop reading times; if the effect is task-,
            dataset-, or aggregation-sensitive, OneStop may show weaker or different patterns.
            """
        ),
        code(
            """
            from __future__ import annotations

            import zipfile
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import requests

            try:
                from IPython.display import Image, display
            except Exception:
                Image = display = None

            REPO_ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
            SRC_DIR = REPO_ROOT / "src"
            DATA_DIR = SRC_DIR / "corpora" / "onestop" / "ordinary"
            CHECKPOINT_DIR = SRC_DIR / "checkpoints"
            FIGURE_DIR = SRC_DIR / "figures"

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)

            ONESTOP_IA_URL = "https://osf.io/download/xkgfz"
            ONESTOP_ZIP = DATA_DIR / "ia_Paragraph_ordinary.csv.zip"
            TRIAL_FEATURES = CHECKPOINT_DIR / "onestop_ordinary_trial_features.csv"
            TEXT_FEATURES = CHECKPOINT_DIR / "onestop_ordinary_text_features.csv"
            CV_RESULTS = CHECKPOINT_DIR / "onestop_ordinary_uid_cv.csv"

            print("Repo root:", REPO_ROOT)
            print("OneStop zip:", ONESTOP_ZIP)
            """
        ),
        code(
            """
            def download_file(url: str, output: Path, chunk_size: int = 1024 * 1024) -> None:
                if output.exists():
                    print(f"Skipping download; already present: {output}")
                    return

                print(f"Downloading {url} -> {output}")
                with requests.get(url, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    with output.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)


            download_file(ONESTOP_IA_URL, ONESTOP_ZIP)
            print(f"Zip size: {ONESTOP_ZIP.stat().st_size / 1024**2:.1f} MB")
            """
        ),
        md(
            """
            ## Feature Construction

            The OneStop IA report is word-level. To match the sentence/paragraph-level analysis style in the
            replication, we aggregate each participant-trial paragraph into one row:

            - `time_sum`: total paragraph dwell time, winsorized at the 99.5th percentile at the word level
            - `len`: number of words in the paragraph
            - `ch_len`: total non-punctuation character length
            - `mean_wordfreq`: mean word frequency
            - `gpt2_power_k`: mean GPT-2 surprisal transformed by exponent `k`
            - `gpt2_power_k_len`: interaction-like predictor `gpt2_power_k * len`

            The text-level surprisal features are identical for all participants who read the same paragraph,
            while reading times vary by participant and trial.
            """
        ),
        code(
            """
            POWER_LABELS = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75", "2.0", "2.25", "2.5", "2.75"]
            POWERS = [float(x) for x in POWER_LABELS]

            USECOLS = [
                "participant_id",
                "TRIAL_INDEX",
                "IA_ID",
                "IA_LABEL",
                "IA_DWELL_TIME",
                "IA_FIRST_FIXATION_DURATION",
                "IA_FIXATION_COUNT",
                "IA_SKIP",
                "word_length_no_punctuation",
                "wordfreq_frequency",
                "gpt2_surprisal",
                "difficulty_level",
                "paragraph_id",
                "article_id",
                "practice_trial",
                "repeated_reading_trial",
                "is_correct",
                "PARAGRAPH_RT",
            ]


            def read_onestop_ia(zip_path: Path) -> pd.DataFrame:
                with zipfile.ZipFile(zip_path) as zf:
                    with zf.open("ia_Paragraph_ordinary.csv") as f:
                        return pd.read_csv(f, usecols=USECOLS, na_values=["."])


            def build_trial_features(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
                print("Reading selected OneStop IA columns...")
                df = read_onestop_ia(zip_path)
                print("Raw IA rows:", df.shape)

                df = df.loc[
                    (~df["practice_trial"])
                    & (~df["repeated_reading_trial"])
                    & (df["word_length_no_punctuation"] > 0)
                ].copy()

                numeric_cols = [
                    "IA_DWELL_TIME",
                    "IA_FIRST_FIXATION_DURATION",
                    "IA_FIXATION_COUNT",
                    "word_length_no_punctuation",
                    "wordfreq_frequency",
                    "gpt2_surprisal",
                    "PARAGRAPH_RT",
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                dwell_cap = df["IA_DWELL_TIME"].quantile(0.995)
                df["IA_DWELL_TIME_W"] = df["IA_DWELL_TIME"].clip(lower=0, upper=dwell_cap)

                keys = ["participant_id", "TRIAL_INDEX", "article_id", "paragraph_id", "difficulty_level"]
                text_keys = ["article_id", "paragraph_id", "difficulty_level"]

                trial = (
                    df.groupby(keys, sort=False)
                    .agg(
                        time_sum=("IA_DWELL_TIME_W", "sum"),
                        raw_time_sum=("IA_DWELL_TIME", "sum"),
                        first_fix_sum=("IA_FIRST_FIXATION_DURATION", "sum"),
                        fixation_count=("IA_FIXATION_COUNT", "sum"),
                        skipped_count=("IA_SKIP", "sum"),
                        len=("IA_ID", "count"),
                        ch_len=("word_length_no_punctuation", "sum"),
                        mean_wordfreq=("wordfreq_frequency", "mean"),
                        mean_gpt2_surprisal=("gpt2_surprisal", "mean"),
                        paragraph_rt=("PARAGRAPH_RT", "first"),
                        is_correct=("is_correct", "first"),
                    )
                    .reset_index()
                )

                text = (
                    df.groupby(text_keys, sort=False)
                    .agg(
                        text=("IA_LABEL", lambda x: " ".join(map(str, x))),
                        len=("IA_ID", "count"),
                        ch_len=("word_length_no_punctuation", "sum"),
                    )
                    .reset_index()
                )
                text_power = (
                    df.groupby(text_keys, sort=False)
                    .agg(
                        mean_wordfreq=("wordfreq_frequency", "mean"),
                        mean_gpt2_surprisal=("gpt2_surprisal", "mean"),
                    )
                    .reset_index()
                )

                for label, p in zip(POWER_LABELS, POWERS):
                    powered = df.assign(_pow=np.power(df["gpt2_surprisal"].clip(lower=0), p))
                    vals = (
                        powered.groupby(text_keys, sort=False)["_pow"]
                        .mean()
                        .rename(f"gpt2_power_{label}")
                        .reset_index()
                    )
                    text_power = text_power.merge(vals, on=text_keys, how="left")

                text = text.merge(text_power, on=text_keys, how="left")
                trial = trial.merge(text_power, on=text_keys, how="left", suffixes=("", "_text"))

                trial["log_time_sum"] = np.log1p(trial["time_sum"])
                trial["skip_rate"] = trial["skipped_count"] / trial["len"]
                for label in POWER_LABELS:
                    trial[f"gpt2_power_{label}_len"] = trial[f"gpt2_power_{label}"] * trial["len"]

                return trial, text


            if TRIAL_FEATURES.exists() and TEXT_FEATURES.exists():
                trial_features = pd.read_csv(TRIAL_FEATURES)
                text_features = pd.read_csv(TEXT_FEATURES)
                print("Loaded cached OneStop features.")
            else:
                trial_features, text_features = build_trial_features(ONESTOP_ZIP)
                trial_features.to_csv(TRIAL_FEATURES, index=False)
                text_features.to_csv(TEXT_FEATURES, index=False)
                print("Saved:", TRIAL_FEATURES)
                print("Saved:", TEXT_FEATURES)

            print("Trial features:", trial_features.shape)
            print("Text features:", text_features.shape)
            """
        ),
        code(
            """
            summary_frame = trial_features.assign(
                text_id=(
                    trial_features["article_id"].astype(str)
                    + "_"
                    + trial_features["paragraph_id"].astype(str)
                    + "_"
                    + trial_features["difficulty_level"].astype(str)
                )
            )
            summary = (
                summary_frame.groupby("difficulty_level")
                .agg(
                    trials=("TRIAL_INDEX", "count"),
                    participants=("participant_id", "nunique"),
                    articles=("article_id", "nunique"),
                    texts=("text_id", "nunique"),
                    mean_len=("len", "mean"),
                    mean_time_sum=("time_sum", "mean"),
                    mean_gpt2_surprisal=("mean_gpt2_surprisal", "mean"),
                )
                .round(3)
            )
            display(summary) if display else print(summary)
            """
        ),
        md(
            """
            ## Cross-Validated Mixed-Effects Analysis

            The model below follows the replication logic but keeps this extension lightweight:

            - baseline: `log_time_sum ~ len + ch_len + mean_wordfreq + (1 | participant)`
            - OneStop predictor model: baseline + `gpt2_power_k * len`

            We use 5-fold cross-validation and report mean held-out log-likelihood improvement per trial.
            `k = 1` is the linear surprisal reference point. The `k = 0` case is skipped in the model sweep
            because `gpt2_power_0 * len` is collinear with the baseline `len` predictor.
            """
        ),
        code(
            """
            import os
            import subprocess

            r_script = CHECKPOINT_DIR / "run_onestop_ordinary_uid_cv.R"
            r_code = r'''
            library(lme4)
            library(ggplot2)
            library(dplyr)

            CHECKPOINT_DIR_R <- "__CHECKPOINT_DIR__"
            FIGURE_DIR_R <- "__FIGURE_DIR__"

            data <- read.csv(file.path(CHECKPOINT_DIR_R, "onestop_ordinary_trial_features.csv"),
                             stringsAsFactors=FALSE,
                             check.names=FALSE)
            data$participant_id <- factor(data$participant_id)
            data$difficulty_level <- factor(data$difficulty_level, levels=c("Ele", "Adv"))
            data <- data[is.finite(data$log_time_sum) & data$len > 0,]

            cv_file <- file.path(CHECKPOINT_DIR_R, "onestop_ordinary_uid_cv.csv")

            cv_lmer <- function(formula, df, outcome="log_time_sum", num_folds=5){
                set.seed(42)
                df <- df[sample(nrow(df)),]
                folds <- cut(seq_len(nrow(df)), breaks=num_folds, labels=FALSE)
                estimates <- c()
                for(i in seq_len(num_folds)){
                    test_idx <- which(folds == i)
                    train <- df[-test_idx,]
                    test <- df[test_idx,]
                    model <- suppressWarnings(lmer(
                        formula,
                        data=train,
                        REML=FALSE,
                        control=lmerControl(optimizer="bobyqa", calc.derivs=FALSE)
                    ))
                    sigma <- sqrt(mean(residuals(model)^2, na.rm=TRUE))
                    pred <- predict(model, newdata=test, allow.new.levels=TRUE)
                    estimates <- c(estimates, dnorm(test[[outcome]], mean=pred, sd=sigma, log=TRUE))
                }
                estimates
            }

            run_subset <- function(label, df){
                message("Running ", label, " n=", nrow(df))
                base_terms <- "len + ch_len + mean_wordfreq"
                if(length(unique(df$difficulty_level)) > 1){
                    base_terms <- paste(base_terms, "+ difficulty_level")
                }
                baseline_formula <- as.formula(paste0("log_time_sum ~ ", base_terms, " + (1 | participant_id)"))
                baseline <- cv_lmer(baseline_formula, df)
                power_labels <- c("0.25","0.5","0.75","1.0","1.25","1.5","1.75","2.0","2.25","2.5","2.75")
                rows <- lapply(power_labels, function(k_label){
                    pred <- paste0("`gpt2_power_", k_label, "_len`")
                    formula <- as.formula(paste0("log_time_sum ~ ", pred, " + ", base_terms, " + (1 | participant_id)"))
                    cv <- cv_lmer(formula, df)
                    diff <- cv - baseline
                    data.frame(
                        subset=label,
                        k=as.numeric(k_label),
                        delta_loglik=mean(diff, na.rm=TRUE),
                        se=sd(diff, na.rm=TRUE)/sqrt(length(diff)),
                        mean_loglik=mean(cv, na.rm=TRUE),
                        n=length(diff)
                    )
                })
                bind_rows(rows)
            }

            if(file.exists(cv_file)){
                results <- read.csv(cv_file)
                message("Loaded cached CV results: ", cv_file)
            } else {
                results <- bind_rows(
                    run_subset("All Ordinary", data),
                    run_subset("Elementary", filter(data, difficulty_level == "Ele")),
                    run_subset("Advanced", filter(data, difficulty_level == "Adv"))
                )
                write.csv(results, cv_file, row.names=FALSE)
                message("Saved CV results: ", cv_file)
            }

            print(results %>% group_by(subset) %>% slice_max(delta_loglik, n=1, with_ties=FALSE))

            p <- ggplot(results, aes(x=k, y=delta_loglik, color=subset, fill=subset)) +
                geom_hline(yintercept=0, color="grey80", linewidth=0.25) +
                geom_vline(xintercept=1, linetype=2, color="grey45", linewidth=0.4) +
                geom_ribbon(aes(ymin=delta_loglik-se, ymax=delta_loglik+se), alpha=0.15, color=NA) +
                geom_line(linewidth=0.85) +
                geom_point(size=2.6) +
                theme_minimal(base_family="serif") +
                labs(
                    title="OneStop Ordinary Reading UID Extension",
                    x=expression(italic("k")),
                    y="Per-Trial Delta LogLik",
                    color="Subset"
                ) +
                guides(fill="none") +
                theme(
                    text=element_text(size=15),
                    title=element_text(size=18),
                    axis.text=element_text(size=10),
                    axis.title=element_text(size=16),
                    legend.position="bottom",
                    plot.margin=margin(6, 8, 6, 6)
                )

            ggsave(file.path(FIGURE_DIR_R, "figure_onestop_ordinary_uid_by_difficulty.png"),
                   p, width=10.5, height=4.8, dpi=300, bg="white")
            ggsave(file.path(FIGURE_DIR_R, "figure_onestop_ordinary_uid_by_difficulty.pdf"),
                   p, width=10.5, height=4.8, device=cairo_pdf, bg="white")
            '''
            r_code = (
                r_code
                .replace("__CHECKPOINT_DIR__", CHECKPOINT_DIR.as_posix())
                .replace("__FIGURE_DIR__", FIGURE_DIR.as_posix())
            )
            r_script.write_text(r_code)

            env = os.environ.copy()
            env.setdefault("R_LIBS_USER", str(REPO_ROOT / ".Rlibs"))
            subprocess.run(["Rscript", str(r_script)], check=True, env=env)
            """
        ),
        code(
            """
            cv_results = pd.read_csv(CV_RESULTS)
            best = cv_results.loc[cv_results.groupby("subset")["delta_loglik"].idxmax()].sort_values("subset")
            display(best) if display else print(best.to_string(index=False))

            figure_path = FIGURE_DIR / "figure_onestop_ordinary_uid_by_difficulty.png"
            if Image and figure_path.exists():
                display(Image(filename=str(figure_path)))
            else:
                print("Figure:", figure_path)
            """
        ),
        md(
            """
            ## Preliminary Interpretation

            This extension is intentionally framed as a generalization test. A positive result would mean the
            same power-sweep pattern from the original paper appears in a new eye-tracking corpus. A null or
            different result is still informative because it identifies a possible boundary condition for UID.

            In the first local run, the OneStop ordinary-reading effect is much smaller than the acceptability
            effect in the main replication. The Elementary subset peaks at a sub-linear exponent, while the
            Advanced and All Ordinary curves are close to zero. This suggests that the clear super-linear
            acceptability pattern may not transfer directly to paragraph-level ordinary reading in OneStop.

            This result should not be read as refuting the original paper. The paper's reading-time conclusion
            was already more cautious than its acceptability conclusion: linear surprisal remained plausible,
            while weak super-linearity also could not be ruled out. The OneStop result adds a new boundary
            condition: in this new corpus and with this paragraph-level ordinary-reading aggregation, the
            UID-style power predictor contributes little beyond length, word-frequency, and participant
            baselines.

            A natural next extension is to download the other OneStop regimes and test whether information
            seeking or repeated reading changes the preferred surprisal transformation.
            """
        ),
    ]

    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUTPUT_NOTEBOOK)
    print(f"Wrote {OUTPUT_NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
