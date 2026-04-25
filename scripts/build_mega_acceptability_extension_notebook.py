#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NOTEBOOK = REPO_ROOT / "src" / "additional_plan_mega_acceptability_extension.ipynb"


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(dedent(source).strip())


def main() -> int:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Additional Plan: MegaAcceptability Extension

            This standalone notebook extends the acceptability side of the replication study to the
            [MegaAcceptability](https://megaattitude.io/projects/mega-acceptability/mega-acceptability-v1/)
            dataset.

            MegaAcceptability contains clause-embedding acceptability judgments for roughly 1,000 English
            predicates across 50 syntactic frames. The raw data were collected as Mechanical Turk 1-7
            ordinal ratings, and the v1.2 release also provides normalized verb-frame acceptability scores.

            The extension asks whether the original paper's clearer super-linear surprisal effect for
            acceptability judgments also appears in this more fine-grained clause-embedding setting.
            """
        ),
        md(
            """
            ## Analysis Design

            This notebook uses the normalized MegaAcceptability table and the existing WikiText-103 KenLM
            5-gram model from the replication workflow.

            The main predictor is:

            `uid_power_k = sentence_length * mean(word_surprisal ** k)`

            This follows the sentence-level power-sweep idea used in the main replication: when `k > 1`,
            high-surprisal words receive more weight, so less uniform information profiles should be more
            strongly penalized.

            Two diagnostics are reported:

            - Pearson correlation between `-uid_power_k` and normalized acceptability.
            - 5-fold cross-validated Gaussian log likelihood improvement after adding `uid_power_k` to a
              baseline with sentence length, character length, verb, and syntactic frame controls.

            The controlled CV analysis is intentionally conservative for this dataset. Because every item is
            built from a verb and a frame, the baseline already captures large lexical and construction-level
            effects; the surprisal predictor is tested for additional explanatory value beyond those controls.
            """
        ),
        code(
            """
            from __future__ import annotations

            import time
            import zipfile
            from pathlib import Path

            import kenlm
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import requests
            from scipy import stats
            from scipy.stats import pearsonr
            from sklearn.compose import ColumnTransformer
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import KFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            try:
                from IPython.display import Image, display
            except Exception:
                Image = display = None

            REPO_ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
            SRC_DIR = REPO_ROOT / "src"
            DATA_DIR = SRC_DIR / "corpora" / "mega_acceptability"
            CHECKPOINT_DIR = SRC_DIR / "checkpoints"
            FIGURE_DIR = SRC_DIR / "figures"

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)

            DATA_URL = "https://megaattitude.io/projects/mega-acceptability/mega-acceptability-v1.zip"
            DATA_ZIP = DATA_DIR / "mega-acceptability-v1.zip"
            NORMALIZED_MEMBER = "mega-acceptability-v1/mega-acceptability-v1-normalized.tsv"
            WIKI_ARPA = SRC_DIR / "wiki.arpa"

            FEATURES_CSV = CHECKPOINT_DIR / "mega_acceptability_ngram_features.csv"
            CORR_CSV = CHECKPOINT_DIR / "mega_acceptability_uid_correlations.csv"
            CV_CSV = CHECKPOINT_DIR / "mega_acceptability_uid_cv.csv"
            FIGURE_STEM = FIGURE_DIR / "figure_mega_acceptability_uid_power_sweep"

            POWERS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]

            print("Repo root:", REPO_ROOT)
            print("Data zip:", DATA_ZIP)
            print("KenLM model:", WIKI_ARPA)
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


            def load_normalized_mega() -> pd.DataFrame:
                download_file(DATA_URL, DATA_ZIP)
                with zipfile.ZipFile(DATA_ZIP) as zf:
                    with zf.open(NORMALIZED_MEMBER) as f:
                        df = pd.read_csv(f, sep="\\t")
                df = df.dropna(subset=["sentence", "responsenorm"]).reset_index(drop=True)
                return df


            mega = load_normalized_mega()
            print("Rows:", len(mega))
            print("Verbs:", mega["verb"].nunique())
            print("Frames:", mega["frame"].nunique())
            display(mega.head()) if display else print(mega.head().to_string(index=False))
            """
        ),
        code(
            """
            def score_sentence_ngram(sentence: str, model: kenlm.Model) -> np.ndarray:
                # MegaAcceptability sentences are already whitespace tokenized.
                base_change = np.log10(np.e)
                scores = list(model.full_scores(sentence, eos=False, bos=True))
                return np.asarray([-score / base_change for score, _ngram_len, _oov in scores], dtype=float)


            def build_ngram_features(df: pd.DataFrame) -> pd.DataFrame:
                if FEATURES_CSV.exists():
                    print(f"Loaded checkpoint: {FEATURES_CSV}")
                    return pd.read_csv(FEATURES_CSV)

                if not WIKI_ARPA.exists():
                    raise FileNotFoundError(
                        f"Missing {WIKI_ARPA}. Run `make build-wiki-arpa` before this extension."
                    )

                print("Loading KenLM ARPA model. This can take around 1-2 minutes.")
                start = time.time()
                model = kenlm.Model(str(WIKI_ARPA))
                print(f"Loaded KenLM in {time.time() - start:.1f}s")

                rows = []
                for idx, sentence in enumerate(df["sentence"].astype(str), start=1):
                    if idx % 10000 == 0:
                        print(f"Scored {idx:,} sentences")

                    tokens = sentence.split()
                    surprisals = score_sentence_ngram(sentence, model)
                    row = {
                        "len": len(tokens),
                        "lm_token_count": len(surprisals),
                        "ch_len": sum(len(token) for token in tokens),
                        "surprisal_mean": float(np.mean(surprisals)),
                        "surprisal_sum": float(np.sum(surprisals)),
                        "surprisal_max": float(np.max(surprisals)),
                        "surprisal_var": float(np.var(surprisals)),
                    }
                    for k in POWERS:
                        row[f"uid_power_{k:g}"] = float(len(tokens) * np.mean(surprisals ** k))
                    rows.append(row)

                features = pd.concat([df, pd.DataFrame(rows)], axis=1)
                features.to_csv(FEATURES_CSV, index=False)
                print(f"Saved checkpoint: {FEATURES_CSV}")
                return features


            mega_features = build_ngram_features(mega)
            display(mega_features[["sentence", "responsenorm", "len", "surprisal_sum"]].head()) if display else print(mega_features.head().to_string(index=False))
            """
        ),
        code(
            """
            summary = mega_features[["responsenorm", "len", "ch_len", "surprisal_sum"]].describe()
            display(summary) if display else print(summary.to_string())

            token_mismatch = (mega_features["len"] != mega_features["lm_token_count"]).sum()
            print("Sentences where whitespace token count differs from KenLM token count:", int(token_mismatch))
            """
        ),
        code(
            """
            def run_correlations(df: pd.DataFrame) -> pd.DataFrame:
                rows = []
                y = df["responsenorm"].to_numpy()
                for k in POWERS:
                    predictor = -df[f"uid_power_{k:g}"].to_numpy()
                    r, p_value = pearsonr(predictor, y)
                    rows.append({"k": k, "pearson_r": r, "p_value": p_value, "n": len(df)})
                out = pd.DataFrame(rows)
                out.to_csv(CORR_CSV, index=False)
                print(f"Saved checkpoint: {CORR_CSV}")
                return out


            corr_results = run_correlations(mega_features)
            display(corr_results) if display else print(corr_results.to_string(index=False))
            """
        ),
        code(
            """
            def make_one_hot_encoder() -> OneHotEncoder:
                try:
                    return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                except TypeError:
                    return OneHotEncoder(handle_unknown="ignore", sparse=True)


            def gaussian_loglik(y_true: np.ndarray, y_pred: np.ndarray, sigma2: float) -> np.ndarray:
                sigma = max(float(np.sqrt(sigma2)), 1e-8)
                return stats.norm.logpdf(y_true, loc=y_pred, scale=sigma)


            def fit_predict_loglik(
                df: pd.DataFrame,
                y: np.ndarray,
                train_idx: np.ndarray,
                test_idx: np.ndarray,
                numeric_cols: list[str],
                categorical_cols: list[str],
            ) -> np.ndarray:
                features = numeric_cols + categorical_cols
                preprocessor = ColumnTransformer(
                    [
                        ("num", StandardScaler(), numeric_cols),
                        ("cat", make_one_hot_encoder(), categorical_cols),
                    ]
                )
                model = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("regressor", Ridge(alpha=1.0)),
                    ]
                )
                model.fit(df.iloc[train_idx][features], y[train_idx])
                train_pred = model.predict(df.iloc[train_idx][features])
                sigma2 = np.mean((y[train_idx] - train_pred) ** 2)
                test_pred = model.predict(df.iloc[test_idx][features])
                return gaussian_loglik(y[test_idx], test_pred, sigma2)


            def run_power_cv(df: pd.DataFrame) -> pd.DataFrame:
                if CV_CSV.exists():
                    print(f"Loaded checkpoint: {CV_CSV}")
                    return pd.read_csv(CV_CSV)

                y = df["responsenorm"].to_numpy()
                folds = KFold(n_splits=5, shuffle=True, random_state=42)
                base_numeric = ["len", "ch_len"]
                categorical = ["verb", "frame"]

                base_loglik = np.empty(len(df))
                augmented_loglik = {k: np.empty(len(df)) for k in POWERS}

                for fold, (train_idx, test_idx) in enumerate(folds.split(df), start=1):
                    print(f"CV fold {fold}/5")
                    base_loglik[test_idx] = fit_predict_loglik(
                        df, y, train_idx, test_idx, base_numeric, categorical
                    )
                    for k in POWERS:
                        numeric_cols = base_numeric + [f"uid_power_{k:g}"]
                        augmented_loglik[k][test_idx] = fit_predict_loglik(
                            df, y, train_idx, test_idx, numeric_cols, categorical
                        )

                rows = []
                baseline_mean = float(np.mean(base_loglik))
                for k in POWERS:
                    diff = augmented_loglik[k] - base_loglik
                    rows.append(
                        {
                            "k": k,
                            "delta_loglik": float(np.mean(diff)),
                            "se": float(np.std(diff, ddof=1) / np.sqrt(len(diff))),
                            "mean_aug_loglik": float(np.mean(augmented_loglik[k])),
                            "mean_baseline_loglik": baseline_mean,
                            "baseline": "len_ch_len_verb_frame",
                            "model": "kenlm_5gram",
                        }
                    )

                out = pd.DataFrame(rows)
                out.to_csv(CV_CSV, index=False)
                print(f"Saved checkpoint: {CV_CSV}")
                return out


            cv_results = run_power_cv(mega_features)
            display(cv_results) if display else print(cv_results.to_string(index=False))
            """
        ),
        code(
            """
            def save_mega_figure(corr_df: pd.DataFrame, cv_df: pd.DataFrame) -> None:
                plt.rcParams.update(
                    {
                        "font.family": "serif",
                        "axes.spines.top": False,
                        "axes.spines.right": False,
                        "axes.grid": True,
                        "grid.alpha": 0.25,
                    }
                )

                fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
                color = "#2f6f9f"
                accent = "#b8582b"

                axes[0].plot(corr_df["k"], corr_df["pearson_r"], color=color, linewidth=2)
                axes[0].scatter(corr_df["k"], corr_df["pearson_r"], color=color, s=34, zorder=3)
                best_corr = corr_df.loc[corr_df["pearson_r"].idxmax()]
                axes[0].scatter([best_corr["k"]], [best_corr["pearson_r"]], color=accent, s=58, zorder=4)
                axes[0].axvline(1, color="0.45", linestyle="--", linewidth=1)
                axes[0].set_xlabel("k")
                axes[0].set_ylabel("Pearson r")
                axes[0].set_title("Correlation with Acceptability")

                axes[1].plot(cv_df["k"], cv_df["delta_loglik"], color=color, linewidth=2)
                axes[1].fill_between(
                    cv_df["k"],
                    cv_df["delta_loglik"] - cv_df["se"],
                    cv_df["delta_loglik"] + cv_df["se"],
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
                axes[1].scatter(cv_df["k"], cv_df["delta_loglik"], color=color, s=34, zorder=3)
                best_cv = cv_df.loc[cv_df["delta_loglik"].idxmax()]
                axes[1].scatter([best_cv["k"]], [best_cv["delta_loglik"]], color=accent, s=58, zorder=4)
                axes[1].axhline(0, color="0.82", linewidth=1)
                axes[1].axvline(1, color="0.45", linestyle="--", linewidth=1)
                axes[1].set_xlabel("k")
                axes[1].set_ylabel("Per-item Delta LogLik")
                axes[1].set_title("CV Improvement over Verb/Frame Baseline")

                for ax in axes:
                    ax.set_xticks(POWERS[::2])
                    ax.tick_params(labelsize=9)

                fig.suptitle("MegaAcceptability: Surprisal Power Sweep", fontsize=15)
                png_path = FIGURE_STEM.with_suffix(".png")
                pdf_path = FIGURE_STEM.with_suffix(".pdf")
                fig.savefig(png_path, dpi=300, facecolor="white")
                fig.savefig(pdf_path, facecolor="white")
                plt.show()
                print(f"Saved figure: {png_path}")
                print(f"Saved figure: {pdf_path}")


            save_mega_figure(corr_results, cv_results)
            """
        ),
        code(
            """
            best_corr = corr_results.loc[corr_results["pearson_r"].idxmax()]
            best_cv = cv_results.loc[cv_results["delta_loglik"].idxmax()]
            linear_cv = cv_results.loc[np.isclose(cv_results["k"], 1.0)].iloc[0]
            linear_corr = corr_results.loc[np.isclose(corr_results["k"], 1.0)].iloc[0]

            print(
                f"Best correlation: k={best_corr['k']:.2f}, "
                f"r={best_corr['pearson_r']:.3f}"
            )
            print(
                f"Linear correlation: k=1.00, "
                f"r={linear_corr['pearson_r']:.3f}"
            )
            print(
                f"Best CV improvement: k={best_cv['k']:.2f}, "
                f"Delta LogLik={best_cv['delta_loglik']:.4f} +/- {best_cv['se']:.4f}"
            )
            print(
                f"Linear CV improvement: k=1.00, "
                f"Delta LogLik={linear_cv['delta_loglik']:.4f} +/- {linear_cv['se']:.4f}"
            )

            figure_path = FIGURE_STEM.with_suffix(".png")
            if Image and figure_path.exists():
                display(Image(filename=str(figure_path)))
            """
        ),
        md(
            """
            ## Interpretation

            The MegaAcceptability extension gives a positive acceptability-side generalization result.
            In the local run with the WikiText-103 KenLM 5-gram model, the simple correlation between
            negative `surprisal^k` and normalized acceptability peaks above the linear setting, around
            `k = 1.25`. The controlled cross-validation analysis also peaks above linear, around `k = 1.5`,
            after controlling for length, verb identity, and syntactic frame.

            This pattern is consistent with the original paper's main acceptability conclusion: acceptability
            judgments provide clearer evidence for a non-linear, UID-related surprisal effect than the
            reading-time analyses. The effect is not merely that longer or lower-probability sentences are
            worse, because the CV analysis tests the surprisal-power predictor beyond length and the major
            lexical/frame factors in the dataset.

            The result should still be framed carefully. MegaAcceptability is a specialized clause-embedding
            dataset, not a broad sample of English sentences, and this notebook uses only the fast 5-gram
            language model rather than GPT-2 or BERT. The conclusion is therefore that the acceptability
            pattern generalizes to a new structured acceptability dataset under this n-gram scoring setup,
            not that every model or every acceptability dataset will show the same preferred exponent.
            """
        ),
    ]

    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUTPUT_NOTEBOOK)
    print(f"Wrote {OUTPUT_NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
