#!/usr/bin/env python3
"""Run the three-model UID extension analyses.

This module keeps the extension analyses aligned with the main public
replication: every extension is scored with GPT-2, BERT, and a KenLM 5-gram
model, then evaluated with the same surprisal-power sweep.
"""

from __future__ import annotations

import argparse
import gc
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import kenlm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from scipy.stats import norm, pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import BertForMaskedLM, BertTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = SRC_DIR / "corpora"
CHECKPOINT_DIR = SRC_DIR / "checkpoints"
FIGURE_DIR = SRC_DIR / "figures"
WIKI_ARPA = SRC_DIR / "wiki.arpa"

POWERS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
MODEL_NAMES = ["gpt", "bert", "ngram"]
PLOT_MODEL_ORDER = ["bert", "gpt", "ngram"]
MODEL_LABELS = {"gpt": "GPT-2", "bert": "Bert", "ngram": r"$n$-gram"}
MODEL_COLORS = {"bert": "#F8766D", "gpt": "#00BA38", "ngram": "#619CFF"}


def ensure_dirs() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def fmt_power(k: float) -> str:
    return f"{k:g}"


def normalize_text(text: object) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def whitespace_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words = text.split()
    spans: list[tuple[int, int]] = []
    pos = 0
    for word in words:
        start = text.find(word, pos)
        if start < 0:
            start = pos
        end = start + len(word)
        spans.append((start, end))
        pos = end + 1
    return words, spans


def aggregate_token_scores_to_words(
    text: str,
    token_scores: Iterable[float],
    token_offsets: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Aggregate subword/token scores into whitespace-word scores.

    The main replication also evaluates sentence-level predictors after
    aggregating token-level surprisals to word/sentence units. For the
    extension datasets, whitespace words are the common unit available in all
    three corpora.
    """

    words, spans = whitespace_spans(text)
    if not words:
        return np.asarray([], dtype=float)

    scores = np.zeros(len(words), dtype=float)
    assigned = np.zeros(len(words), dtype=bool)
    word_idx = 0

    for raw_score, raw_offset in zip(token_scores, token_offsets):
        start, end = int(raw_offset[0]), int(raw_offset[1])
        if end <= start:
            continue

        while word_idx < len(spans) and start >= spans[word_idx][1]:
            word_idx += 1
        if word_idx >= len(spans):
            break

        # Byte-level GPT-2 offsets can include the preceding space, so the
        # token start may be before the current word. Token end is the reliable
        # boundary for deciding when a whitespace word is complete.
        scores[word_idx] += float(raw_score)
        assigned[word_idx] = True
        if end >= spans[word_idx][1]:
            word_idx += 1

    if not assigned.all():
        scores = scores[assigned]
    return scores


def features_from_word_scores(text: str, model_name: str, word_scores: np.ndarray) -> dict[str, object]:
    text = normalize_text(text)
    words, _spans = whitespace_spans(text)
    scores = np.asarray(word_scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    row: dict[str, object] = {
        "model_name": model_name,
        "model": model_name,
        "n_words_text": len(words),
        "n_words_scored": int(len(scores)),
        "alignment_ok": bool(len(words) == len(scores)),
    }

    if len(scores) == 0:
        row.update(
            {
                "mean_surprisal": np.nan,
                "surprisal_sum": np.nan,
                "surprisal_var": np.nan,
                "surprisal_max": np.nan,
            }
        )
        for k in POWERS:
            row[f"uid_power_{fmt_power(k)}"] = np.nan
        return row

    clipped = np.clip(scores, 0, None)
    row.update(
        {
            "mean_surprisal": float(np.mean(scores)),
            "surprisal_sum": float(np.sum(scores)),
            "surprisal_var": float(np.var(scores)),
            "surprisal_max": float(np.max(scores)),
        }
    )
    for k in POWERS:
        row[f"uid_power_{fmt_power(k)}"] = float(len(scores) * np.mean(clipped**k))
    return row


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


@dataclass
class SurprisalScorer:
    device_name: str = "auto"
    gpt_batch_size: int = 32
    bert_text_batch_size: int = 256
    bert_mask_batch_size: int = 64

    def __post_init__(self) -> None:
        self.device = self._choose_device(self.device_name)
        self._model_name: str | None = None
        self._model: object | None = None
        self._tokenizer: object | None = None
        print(f"Using torch device: {self.device}", flush=True)

    @staticmethod
    def _choose_device(name: str) -> torch.device:
        if name != "auto":
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def clear(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def load(self, model_name: str) -> None:
        if self._model_name == model_name:
            return
        self.clear()
        start = time.time()
        if model_name == "gpt":
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            model.eval()
            model.to(self.device)
            self._tokenizer = tokenizer
            self._model = model
        elif model_name == "bert":
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
            model = BertForMaskedLM.from_pretrained("bert-base-cased")
            model.eval()
            model.to(self.device)
            self._tokenizer = tokenizer
            self._model = model
        elif model_name == "ngram":
            if not WIKI_ARPA.exists():
                raise FileNotFoundError(f"Missing {WIKI_ARPA}. Run `make build-wiki-arpa` first.")
            self._tokenizer = None
            self._model = kenlm.Model(str(WIKI_ARPA))
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self._model_name = model_name
        print(f"Loaded {model_name} in {time.time() - start:.1f}s", flush=True)

    def score_texts(self, model_name: str, texts: list[str]) -> list[np.ndarray]:
        self.load(model_name)
        if model_name == "gpt":
            return self._score_gpt(texts)
        if model_name == "bert":
            return self._score_bert(texts)
        if model_name == "ngram":
            return self._score_ngram(texts)
        raise ValueError(model_name)

    def _score_gpt(self, texts: list[str]) -> list[np.ndarray]:
        assert isinstance(self._model, GPT2LMHeadModel)
        assert isinstance(self._tokenizer, GPT2TokenizerFast)
        out: list[np.ndarray] = []
        tokenizer = self._tokenizer
        model = self._model

        for start in range(0, len(texts), self.gpt_batch_size):
            batch = [normalize_text(t) for t in texts[start : start + self.gpt_batch_size]]
            enc = tokenizer(
                batch,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=1022,
                return_attention_mask=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")
            input_ids = enc["input_ids"].to(self.device)
            attention = enc["attention_mask"].to(self.device)
            bos = torch.full(
                (input_ids.shape[0], 1),
                tokenizer.bos_token_id,
                dtype=input_ids.dtype,
                device=self.device,
            )
            input_ids_bos = torch.cat([bos, input_ids], dim=1)
            attention_bos = torch.cat([torch.ones_like(bos), attention], dim=1)

            with torch.no_grad():
                logits = model(input_ids_bos, attention_mask=attention_bos).logits[:, :-1, :].contiguous()
                labels = input_ids_bos[:, 1:].contiguous()
                losses = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                ).view(labels.shape)

            lengths = attention.sum(dim=1).detach().cpu().tolist()
            losses_cpu = losses.detach().cpu()
            for i, length in enumerate(lengths):
                token_scores = losses_cpu[i, : int(length)].numpy()
                token_offsets = [tuple(map(int, x)) for x in offsets[i][: int(length)].tolist()]
                out.append(aggregate_token_scores_to_words(batch[i], token_scores, token_offsets))
        return out

    def _score_bert(self, texts: list[str]) -> list[np.ndarray]:
        assert isinstance(self._model, BertForMaskedLM)
        assert isinstance(self._tokenizer, BertTokenizerFast)
        tokenizer = self._tokenizer
        model = self._model
        mask_id = tokenizer.mask_token_id
        out: list[np.ndarray] = []

        for start in range(0, len(texts), self.bert_text_batch_size):
            batch = [normalize_text(t) for t in texts[start : start + self.bert_text_batch_size]]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")
            special = enc.pop("special_tokens_mask")
            input_ids_cpu = enc["input_ids"]
            attention_cpu = enc["attention_mask"]

            positions: list[tuple[int, int]] = []
            per_text_scores: list[list[float]] = [[] for _ in batch]
            per_text_offsets: list[list[tuple[int, int]]] = [[] for _ in batch]
            for row in range(input_ids_cpu.shape[0]):
                length = int(attention_cpu[row].sum().item())
                for pos in range(length):
                    if int(special[row, pos].item()) == 0:
                        positions.append((row, pos))
                        per_text_offsets[row].append(tuple(map(int, offsets[row, pos].tolist())))

            total_positions = len(positions)
            for chunk_start in range(0, total_positions, self.bert_mask_batch_size):
                chunk = positions[chunk_start : chunk_start + self.bert_mask_batch_size]
                rows = torch.tensor([p[0] for p in chunk], dtype=torch.long)
                cols = torch.tensor([p[1] for p in chunk], dtype=torch.long)
                masked = input_ids_cpu[rows].clone()
                targets = masked[torch.arange(len(chunk)), cols].clone()
                masked[torch.arange(len(chunk)), cols] = mask_id
                attn = attention_cpu[rows].clone()

                masked = masked.to(self.device)
                attn = attn.to(self.device)
                cols_device = cols.to(self.device)
                targets_device = targets.to(self.device)

                with torch.no_grad():
                    logits = model(masked, attention_mask=attn).logits
                    selected = logits[torch.arange(len(chunk), device=self.device), cols_device, :]
                    log_probs = torch.nn.functional.log_softmax(selected, dim=-1)
                    scores = -log_probs[torch.arange(len(chunk), device=self.device), targets_device]

                for (row, _pos), score in zip(chunk, scores.detach().cpu().tolist()):
                    per_text_scores[row].append(float(score))

                processed = min(chunk_start + self.bert_mask_batch_size, total_positions)
                if processed == total_positions or processed % 2048 < self.bert_mask_batch_size:
                    print(
                        f"  BERT masked positions {processed:,}/{total_positions:,} "
                        f"for current text batch",
                        flush=True,
                    )

            for text, token_scores, token_offsets in zip(batch, per_text_scores, per_text_offsets):
                out.append(aggregate_token_scores_to_words(text, token_scores, token_offsets))
        return out

    def _score_ngram(self, texts: list[str]) -> list[np.ndarray]:
        assert isinstance(self._model, kenlm.Model)
        base_change = np.log10(np.e)
        out: list[np.ndarray] = []
        for text in texts:
            text = normalize_text(text)
            scores = self._model.full_scores(text, eos=False, bos=True)
            out.append(np.asarray([-score / base_change for score, _length, _oov in scores], dtype=float))
        return out


def score_text_table(
    texts: pd.DataFrame,
    *,
    text_key_col: str,
    text_col: str,
    cache_path: Path,
    models: list[str],
    scorer: SurprisalScorer,
    flush_every: int = 100,
    force: bool = False,
) -> pd.DataFrame:
    ensure_dirs()
    if force and cache_path.exists():
        cache_path.unlink()

    texts = texts[[text_key_col, text_col]].drop_duplicates(text_key_col).copy()
    texts[text_key_col] = texts[text_key_col].astype(str)
    texts[text_col] = texts[text_col].map(normalize_text)
    cache = read_cache(cache_path)
    if not cache.empty and text_key_col in cache.columns:
        cache[text_key_col] = cache[text_key_col].astype(str)
    required = {text_key_col, "model_name", "mean_surprisal", "uid_power_1"}
    if not cache.empty and not required.issubset(cache.columns):
        cache_path.unlink()
        cache = pd.DataFrame()

    for model_name in models:
        if cache.empty:
            done: set[str] = set()
        else:
            done = set(cache.loc[cache["model_name"] == model_name, text_key_col].astype(str))

        remaining = texts.loc[~texts[text_key_col].astype(str).isin(done)].copy()
        print(f"{cache_path.name}: {model_name} remaining texts: {len(remaining):,}", flush=True)
        rows: list[dict[str, object]] = []
        if remaining.empty:
            continue

        if model_name == "ngram":
            batch_size = 512
        elif model_name == "gpt":
            batch_size = scorer.gpt_batch_size
        else:
            batch_size = scorer.bert_text_batch_size

        for start in range(0, len(remaining), batch_size):
            batch = remaining.iloc[start : start + batch_size]
            word_scores = scorer.score_texts(model_name, batch[text_col].tolist())
            for (_, item), scores in zip(batch.iterrows(), word_scores):
                row = {
                    text_key_col: item[text_key_col],
                    text_col: item[text_col],
                    **features_from_word_scores(item[text_col], model_name, scores),
                }
                rows.append(row)

            if rows and len(rows) >= flush_every:
                append_rows(cache_path, rows)
                rows = []
                print(f"  saved through {min(start + batch_size, len(remaining)):,}/{len(remaining):,}", flush=True)

        append_rows(cache_path, rows)
        cache = read_cache(cache_path)
        if not cache.empty and text_key_col in cache.columns:
            cache[text_key_col] = cache[text_key_col].astype(str)
    cache = read_cache(cache_path)
    if not cache.empty and text_key_col in cache.columns:
        cache[text_key_col] = cache[text_key_col].astype(str)
    return cache


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", encoder, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def gaussian_cv_loglik(
    df: pd.DataFrame,
    *,
    outcome: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    groups: pd.Series | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    cols = [outcome] + numeric_cols + categorical_cols
    if groups is not None:
        work = df[cols].copy()
        work["_group"] = groups.to_numpy()
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        groups_clean = work.pop("_group")
    else:
        work = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        groups_clean = None

    if len(work) < n_splits:
        return np.asarray([], dtype=float)

    x = work[numeric_cols + categorical_cols]
    y = work[outcome].astype(float).to_numpy()
    model = Pipeline(
        steps=[
            ("prep", make_preprocessor(numeric_cols, categorical_cols)),
            ("ridge", Ridge(alpha=1.0, solver="lsqr")),
        ]
    )

    if groups_clean is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(x)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(x, y, groups_clean)

    loglik: list[np.ndarray] = []
    for train_idx, test_idx in splits:
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        sigma = float(np.std(y_train - train_pred, ddof=1))
        sigma = max(sigma, 1e-6)
        pred = model.predict(x_test)
        loglik.append(norm.logpdf(y_test, loc=pred, scale=sigma))
    return np.concatenate(loglik)


def cv_power_sweep(
    df: pd.DataFrame,
    *,
    outcome: str,
    base_numeric: list[str],
    categorical_cols: list[str],
    groups: pd.Series | None,
    n_splits: int,
    seed: int = 42,
) -> list[dict[str, float]]:
    baseline = gaussian_cv_loglik(
        df,
        outcome=outcome,
        numeric_cols=base_numeric,
        categorical_cols=categorical_cols,
        groups=groups,
        n_splits=n_splits,
        seed=seed,
    )
    baseline_mean = float(np.mean(baseline)) if len(baseline) else np.nan
    rows: list[dict[str, float]] = []
    for k in POWERS:
        pred_col = f"uid_power_{fmt_power(k)}"
        augmented = gaussian_cv_loglik(
            df,
            outcome=outcome,
            numeric_cols=base_numeric + [pred_col],
            categorical_cols=categorical_cols,
            groups=groups,
            n_splits=n_splits,
            seed=seed,
        )
        n = min(len(baseline), len(augmented))
        diff = augmented[:n] - baseline[:n]
        rows.append(
            {
                "k": k,
                "delta_loglik": float(np.mean(diff)) if n else np.nan,
                "se": float(np.std(diff, ddof=1) / math.sqrt(n)) if n > 1 else np.nan,
                "mean_aug_loglik": float(np.mean(augmented)) if len(augmented) else np.nan,
                "mean_baseline_loglik": baseline_mean,
                "n": int(n),
            }
        )
    return rows


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {stem.with_suffix('.pdf')}")


def apply_extension_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "axes.titlesize": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.65,
        }
    )


def finish_power_axes(
    axes: Iterable[plt.Axes],
    *,
    x_label: str = r"$k$",
    y_label: str,
) -> None:
    axes = list(axes)
    for ax in axes:
        ax.axhline(0, color="#d9d9d9", linewidth=0.7, zorder=0)
        ax.axvline(1, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xlabel(x_label)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(y_label)


def add_bottom_legend(fig: plt.Figure, axes: Iterable[plt.Axes]) -> None:
    handles: list[object] = []
    labels: list[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
            handlelength=2.2,
            columnspacing=1.4,
        )


def plot_cv_line(
    ax: plt.Axes,
    data: pd.DataFrame,
    model_name: str,
    *,
    y_col: str = "delta_loglik",
    se_col: str = "se",
    label: str | None = None,
) -> None:
    data = data.sort_values("k")
    if data.empty:
        return
    color = MODEL_COLORS[model_name]
    x = data["k"].astype(float).to_numpy()
    y = data[y_col].astype(float).to_numpy()
    ax.plot(
        x,
        y,
        marker="o",
        linewidth=0.9,
        markersize=3.1,
        color=color,
        label=label or MODEL_LABELS[model_name],
    )
    if se_col in data.columns:
        se = data[se_col].astype(float).to_numpy()
        if np.isfinite(se).any():
            ax.fill_between(x, y - se, y + se, color=color, alpha=0.16, linewidth=0)


def download_file(url: str, output: Path, chunk_size: int = 1024 * 1024) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {output}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def build_onestop_base(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = DATA_DIR / "onestop" / "ordinary"
    zip_path = data_dir / "ia_Paragraph_ordinary.csv.zip"
    download_file("https://osf.io/download/xkgfz", zip_path)

    trial_path = CHECKPOINT_DIR / "onestop_ordinary_base_trial_features.csv"
    text_path = CHECKPOINT_DIR / "onestop_ordinary_base_texts.csv"
    if not force and trial_path.exists() and text_path.exists():
        return pd.read_csv(trial_path), pd.read_csv(text_path)

    usecols = [
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
        "difficulty_level",
        "paragraph_id",
        "article_id",
        "practice_trial",
        "repeated_reading_trial",
        "is_correct",
        "PARAGRAPH_RT",
    ]
    print("Reading OneStop IA report...")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("ia_Paragraph_ordinary.csv") as f:
            df = pd.read_csv(f, usecols=usecols, na_values=["."])

    df = df.loc[
        (~df["practice_trial"])
        & (~df["repeated_reading_trial"])
        & (pd.to_numeric(df["word_length_no_punctuation"], errors="coerce") > 0)
    ].copy()
    for col in [
        "IA_DWELL_TIME",
        "IA_FIRST_FIXATION_DURATION",
        "IA_FIXATION_COUNT",
        "word_length_no_punctuation",
        "wordfreq_frequency",
        "PARAGRAPH_RT",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["IA_SKIP"] = pd.to_numeric(df["IA_SKIP"], errors="coerce").fillna(0)
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
            paragraph_rt=("PARAGRAPH_RT", "first"),
            is_correct=("is_correct", "first"),
        )
        .reset_index()
    )
    trial["log_time_sum"] = np.log1p(trial["time_sum"])
    trial["skip_rate"] = trial["skipped_count"] / trial["len"]
    trial["text_key"] = trial[text_keys].astype(str).agg("::".join, axis=1)

    text_words = (
        df.sort_values(text_keys + ["IA_ID"])
        .drop_duplicates(text_keys + ["IA_ID"])
        .groupby(text_keys, sort=False)
        .agg(
            text=("IA_LABEL", lambda x: normalize_text(" ".join(map(str, x)))),
            len=("IA_ID", "count"),
            ch_len=("word_length_no_punctuation", "sum"),
            mean_wordfreq=("wordfreq_frequency", "mean"),
        )
        .reset_index()
    )
    text_words["text_key"] = text_words[text_keys].astype(str).agg("::".join, axis=1)

    trial.to_csv(trial_path, index=False)
    text_words.to_csv(text_path, index=False)
    print(f"Saved {trial_path}")
    print(f"Saved {text_path}")
    return trial, text_words


def run_onestop(scorer: SurprisalScorer, models: list[str], force: bool = False) -> pd.DataFrame:
    trial, texts = build_onestop_base(force=force)
    lm_features = score_text_table(
        texts[["text_key", "text"]],
        text_key_col="text_key",
        text_col="text",
        cache_path=CHECKPOINT_DIR / "onestop_ordinary_lm_text_features.csv",
        models=models,
        scorer=scorer,
        force=force,
    )
    model_trial = trial.merge(
        lm_features.drop(columns=["text"], errors="ignore"),
        on="text_key",
        how="inner",
    )
    model_trial.to_csv(CHECKPOINT_DIR / "onestop_ordinary_three_model_trial_features.csv", index=False)

    rows: list[dict[str, object]] = []
    subsets = {
        "All Ordinary": model_trial,
        "Advanced": model_trial.loc[model_trial["difficulty_level"].astype(str).str.lower().str.startswith("adv")],
        "Elementary": model_trial.loc[model_trial["difficulty_level"].astype(str).str.lower().str.startswith("ele")],
    }
    for subset_name, subset_df in subsets.items():
        for model_name in models:
            data = subset_df.loc[subset_df["model_name"] == model_name].copy()
            if data.empty:
                continue
            categorical = ["participant_id"]
            if subset_name == "All Ordinary":
                categorical = categorical + ["difficulty_level"]
            sweep = cv_power_sweep(
                data,
                outcome="log_time_sum",
                base_numeric=["len", "ch_len", "mean_wordfreq"],
                categorical_cols=categorical,
                groups=None,
                n_splits=5,
            )
            for row in sweep:
                rows.append(
                    {
                        "subset": subset_name,
                        "model_name": model_name,
                        "model": model_name,
                        "baseline": "len_ch_len_wordfreq_participant_difficulty",
                        "outcome": "log_time_sum",
                        **row,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(CHECKPOINT_DIR / "onestop_ordinary_uid_cv.csv", index=False)
    plot_onestop(out)
    return out


def plot_onestop(cv: pd.DataFrame) -> None:
    apply_extension_theme()
    subsets = ["All Ordinary", "Advanced", "Elementary"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.05), sharey=True)
    for ax, subset in zip(axes, subsets):
        data = cv.loc[cv["subset"] == subset]
        for model_name in PLOT_MODEL_ORDER:
            m = data.loc[data["model_name"] == model_name].sort_values("k")
            if m.empty:
                continue
            plot_cv_line(ax, m, model_name)
        ax.set_title(subset)
    finish_power_axes(axes, y_label="Per Trial \u2206LogLik")
    add_bottom_legend(fig, axes)
    fig.subplots_adjust(bottom=0.24, wspace=0.28)
    save_figure(fig, FIGURE_DIR / "figure_onestop_ordinary_uid_by_difficulty")


def load_mega() -> pd.DataFrame:
    data_dir = DATA_DIR / "mega_acceptability"
    data_zip = data_dir / "mega-acceptability-v1.zip"
    download_file("https://megaattitude.io/projects/mega-acceptability/mega-acceptability-v1.zip", data_zip)
    with zipfile.ZipFile(data_zip) as zf:
        with zf.open("mega-acceptability-v1/mega-acceptability-v1-normalized.tsv") as f:
            mega = pd.read_csv(f, sep="\t")
    mega = mega.dropna(subset=["sentence", "responsenorm"]).reset_index(drop=True)
    mega["text_key"] = mega.index.astype(str)
    mega["text"] = mega["sentence"].map(normalize_text)
    mega["len"] = mega["text"].str.split().str.len()
    mega["ch_len"] = mega["text"].str.replace(" ", "", regex=False).str.len()
    return mega


def run_mega(scorer: SurprisalScorer, models: list[str], force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    mega = load_mega()
    lm_features = score_text_table(
        mega[["text_key", "text"]],
        text_key_col="text_key",
        text_col="text",
        cache_path=CHECKPOINT_DIR / "mega_acceptability_lm_features.csv",
        models=models,
        scorer=scorer,
        flush_every=500,
        force=force,
    )
    features = mega.merge(lm_features.drop(columns=["text"], errors="ignore"), on="text_key", how="inner")
    features.to_csv(CHECKPOINT_DIR / "mega_acceptability_three_model_features.csv", index=False)

    corr_rows: list[dict[str, object]] = []
    for model_name in models:
        data = features.loc[features["model_name"] == model_name]
        y = data["responsenorm"].astype(float).to_numpy()
        for k in POWERS:
            x = -data[f"uid_power_{fmt_power(k)}"].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            r, p_value = pearsonr(x[mask], y[mask])
            corr_rows.append(
                {
                    "model_name": model_name,
                    "model": model_name,
                    "k": k,
                    "pearson_r": float(r),
                    "p_value": float(p_value),
                    "n": int(mask.sum()),
                }
            )
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(CHECKPOINT_DIR / "mega_acceptability_uid_correlations.csv", index=False)

    cv_rows: list[dict[str, object]] = []
    for model_name in models:
        data = features.loc[features["model_name"] == model_name].copy()
        sweep = cv_power_sweep(
            data,
            outcome="responsenorm",
            base_numeric=["len", "ch_len"],
            categorical_cols=["verb", "frame"],
            groups=None,
            n_splits=5,
        )
        for row in sweep:
            cv_rows.append(
                {
                    "model_name": model_name,
                    "model": model_name,
                    "baseline": "len_ch_len_verb_frame",
                    "outcome": "responsenorm",
                    **row,
                }
            )
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(CHECKPOINT_DIR / "mega_acceptability_uid_cv.csv", index=False)
    plot_mega(corr, cv)
    return corr, cv


def plot_mega(corr: pd.DataFrame, cv: pd.DataFrame) -> None:
    apply_extension_theme()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.05))
    for model_name in PLOT_MODEL_ORDER:
        c = corr.loc[corr["model_name"] == model_name].sort_values("k")
        if not c.empty:
            axes[0].plot(
                c["k"],
                c["pearson_r"],
                marker="o",
                linewidth=0.9,
                markersize=3.1,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
        v = cv.loc[cv["model_name"] == model_name].sort_values("k")
        if not v.empty:
            plot_cv_line(axes[1], v, model_name)

    axes[0].set_title("Correlation")
    axes[0].set_xlabel(r"$k$")
    axes[0].set_ylabel("Pearson r")
    axes[0].axvline(1, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[1].set_title("Controlled CV")
    finish_power_axes([axes[1]], y_label="Per Item \u2206LogLik")
    add_bottom_legend(fig, axes)
    fig.subplots_adjust(bottom=0.24, wspace=0.32)
    save_figure(fig, FIGURE_DIR / "figure_mega_acceptability_uid_power_sweep")


def build_emtec_base(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = DATA_DIR / "emtec"
    reading_path = data_dir / "reading_measures_corrected.csv"
    stimuli_path = data_dir / "stimuli.csv"
    download_file("https://osf.io/download/wa3ty", reading_path)
    download_file("https://osf.io/download/vgp9a", stimuli_path)

    trial_path = CHECKPOINT_DIR / "emtec_base_trial_features.csv"
    text_path = CHECKPOINT_DIR / "emtec_base_texts.csv"
    if not force and trial_path.exists() and text_path.exists():
        return pd.read_csv(trial_path), pd.read_csv(text_path)

    usecols = [
        "subject_id",
        "item_id",
        "model",
        "decoding_strategy",
        "TRIAL_ID",
        "word_id",
        "word",
        "TFT",
        "FPRT",
        "Fix",
        "word_length_without_punct",
        "zipf_freq",
        "type",
        "task",
        "subcategory",
    ]
    print("Reading EMTeC corrected reading measures...")
    reading = pd.read_csv(reading_path, sep="\t", usecols=usecols)
    reading = reading.rename(columns={"model": "generator_model"})
    for col in ["TFT", "FPRT", "Fix", "word_length_without_punct", "zipf_freq"]:
        reading[col] = pd.to_numeric(reading[col], errors="coerce")

    condition_cols = ["item_id", "generator_model", "decoding_strategy"]
    word_table = (
        reading.sort_values(condition_cols + ["word_id"])
        .drop_duplicates(condition_cols + ["word_id"])
        .copy()
    )
    texts = (
        word_table.groupby(condition_cols, sort=False)
        .agg(
            text=("word", lambda x: normalize_text(" ".join(map(str, x)))),
            n_words=("word_id", "count"),
            ch_len=("word_length_without_punct", "sum"),
            mean_word_length=("word_length_without_punct", "mean"),
            mean_zipf=("zipf_freq", "mean"),
            type=("type", "first"),
            task=("task", "first"),
            subcategory=("subcategory", "first"),
        )
        .reset_index()
    )
    texts["condition_id"] = texts[condition_cols].astype(str).agg("::".join, axis=1)

    trial = (
        reading.groupby(["subject_id"] + condition_cols, sort=False)
        .agg(
            TFT_sum=("TFT", "sum"),
            FPRT_sum=("FPRT", "sum"),
            Fix_mean=("Fix", "mean"),
            rows=("word_id", "count"),
        )
        .reset_index()
    )
    trial = trial.merge(texts.drop(columns=["text"]), on=condition_cols, how="left")
    trial["log_tft_sum"] = np.log1p(trial["TFT_sum"])

    trial.to_csv(trial_path, index=False)
    texts.to_csv(text_path, index=False)
    print(f"Saved {trial_path}")
    print(f"Saved {text_path}")
    return trial, texts


def run_emtec(scorer: SurprisalScorer, models: list[str], force: bool = False) -> pd.DataFrame:
    trial, texts = build_emtec_base(force=force)
    lm_features = score_text_table(
        texts[["condition_id", "text"]],
        text_key_col="condition_id",
        text_col="text",
        cache_path=CHECKPOINT_DIR / "emtec_lm_text_features.csv",
        models=models,
        scorer=scorer,
        flush_every=16,
        force=force,
    )
    text_lm = texts.merge(lm_features.drop(columns=["text"], errors="ignore"), on="condition_id", how="inner")
    text_lm.to_csv(CHECKPOINT_DIR / "emtec_text_uid_features.csv", index=False)
    model_trial = trial.merge(
        lm_features.drop(columns=["text"], errors="ignore"),
        on="condition_id",
        how="inner",
    )
    model_trial.to_csv(CHECKPOINT_DIR / "emtec_trial_features.csv", index=False)

    rows: list[dict[str, object]] = []
    for model_name in models:
        data = model_trial.loc[model_trial["model_name"] == model_name].copy()
        sweep = cv_power_sweep(
            data,
            outcome="log_tft_sum",
            base_numeric=["n_words", "ch_len", "mean_word_length", "mean_zipf"],
            categorical_cols=["subject_id", "generator_model", "decoding_strategy", "task"],
            groups=data["condition_id"],
            n_splits=5,
        )
        for row in sweep:
            rows.append(
                {
                    "model_name": model_name,
                    "model": model_name,
                    "predictor_lm": model_name,
                    "baseline": "length_frequency_subject_generator_decoding_task",
                    "split": "GroupKFold_by_generated_text_condition",
                    "outcome": "log_total_fixation_time",
                    **row,
                }
            )
    cv = pd.DataFrame(rows)
    cv.to_csv(CHECKPOINT_DIR / "emtec_uid_cv.csv", index=False)

    summary = (
        text_lm.groupby(["model_name", "generator_model", "decoding_strategy"], sort=False)
        .agg(
            n_texts=("condition_id", "nunique"),
            mean_surprisal=("mean_surprisal", "mean"),
            mean_surprisal_var=("surprisal_var", "mean"),
            mean_uid_k1=("uid_power_1", "mean"),
            mean_n_words=("n_words", "mean"),
        )
        .reset_index()
    )
    rt_summary = (
        model_trial.groupby(["model_name", "generator_model", "decoding_strategy"], sort=False)
        .agg(mean_log_tft=("log_tft_sum", "mean"))
        .reset_index()
    )
    summary = summary.merge(rt_summary, on=["model_name", "generator_model", "decoding_strategy"], how="left")
    summary["mean_surprisal_rel_pct"] = summary.groupby("model_name")["mean_surprisal"].transform(
        lambda x: 100 * (x / x.mean() - 1)
    )
    summary.to_csv(CHECKPOINT_DIR / "emtec_decoding_uid_summary.csv", index=False)
    plot_emtec(cv, summary)
    return cv


def plot_emtec(cv: pd.DataFrame, summary: pd.DataFrame) -> None:
    apply_extension_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.05))
    for model_name in PLOT_MODEL_ORDER:
        c = cv.loc[cv["model_name"] == model_name].sort_values("k")
        if not c.empty:
            plot_cv_line(axes[0], c, model_name)

        s = (
            summary.loc[summary["model_name"] == model_name]
            .groupby("decoding_strategy", as_index=False)
            .agg(mean_surprisal=("mean_surprisal", "mean"))
            .sort_values("decoding_strategy")
        )
        if not s.empty:
            s["mean_surprisal_rel_pct"] = 100 * (s["mean_surprisal"] / s["mean_surprisal"].mean() - 1)
            axes[1].plot(
                s["decoding_strategy"],
                s["mean_surprisal_rel_pct"],
                marker="o",
                linewidth=0.9,
                markersize=3.1,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )

    axes[0].set_title("Reading-time CV")
    finish_power_axes([axes[0]], y_label="Per Trial \u2206LogLik")
    axes[1].set_title("Generated-text Surprisal")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Relative surprisal (%)")
    axes[1].axhline(0, color="#d9d9d9", linewidth=0.7, zorder=0)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y")
    axes[1].grid(False, axis="x")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    add_bottom_legend(fig, axes)
    fig.subplots_adjust(bottom=0.26, wspace=0.32)
    save_figure(fig, FIGURE_DIR / "figure_emtec_uid_generated_text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", "onestop", "mega", "emtec"],
        default="all",
        help="Extension dataset to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Scoring models to run.",
    )
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, mps, or cuda.")
    parser.add_argument("--force", action="store_true", help="Rebuild base tables and LM caches.")
    parser.add_argument("--gpt-batch-size", type=int, default=32)
    parser.add_argument("--bert-text-batch-size", type=int, default=256)
    parser.add_argument("--bert-mask-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    scorer = SurprisalScorer(
        device_name=args.device,
        gpt_batch_size=args.gpt_batch_size,
        bert_text_batch_size=args.bert_text_batch_size,
        bert_mask_batch_size=args.bert_mask_batch_size,
    )
    datasets = ["onestop", "mega", "emtec"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        print(f"\n=== Running {dataset} extension with models: {', '.join(args.models)} ===")
        if dataset == "onestop":
            run_onestop(scorer, args.models, force=args.force)
        elif dataset == "mega":
            run_mega(scorer, args.models, force=args.force)
        elif dataset == "emtec":
            run_emtec(scorer, args.models, force=args.force)
        scorer.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
