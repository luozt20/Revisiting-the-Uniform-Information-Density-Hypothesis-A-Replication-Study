#!/usr/bin/env python3
"""Sanity checks for AI-assisted analysis code on simulated data.

The course AI policy requires analysis code that was substantially assisted by
AI tools to be verified on data with known ground-truth properties before being
used on real data. This script checks the core mechanics used by the extension
runner without downloading corpora or loading language models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from extension_three_model_analysis import POWERS, cv_power_sweep, features_from_word_scores, fmt_power


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-8) -> None:
    if not np.isclose(actual, expected, atol=tol, rtol=tol):
        raise AssertionError(f"{name}: expected {expected:.12g}, got {actual:.12g}")


def verify_uid_feature_math() -> None:
    text = "alpha beta gamma"
    scores = np.asarray([1.0, 2.0, 4.0])
    row = features_from_word_scores(text, "simulated", scores)

    if row["n_words_text"] != 3 or row["n_words_scored"] != 3 or not row["alignment_ok"]:
        raise AssertionError("word-count alignment failed for the simple simulated sentence")

    assert_close("mean_surprisal", float(row["mean_surprisal"]), 7.0 / 3.0)
    assert_close("surprisal_sum", float(row["surprisal_sum"]), 7.0)
    assert_close("uid_power_1", float(row["uid_power_1"]), 7.0)
    assert_close("uid_power_2", float(row["uid_power_2"]), 21.0)


def make_ground_truth_dataset(n_items: int = 260, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for item_id in range(n_items):
        n_words = int(rng.integers(6, 22))
        condition = "A" if item_id % 2 == 0 else "B"
        length = float(n_words)
        frequency = float(rng.normal(0.0, 1.0))

        # Profiles vary in both total surprisal and peakiness, so different
        # powers contain distinguishable information rather than the same rank.
        base = rng.gamma(shape=2.0, scale=1.3, size=n_words)
        if rng.random() < 0.45:
            base[int(rng.integers(0, n_words))] += rng.uniform(4.0, 8.0)

        features = features_from_word_scores(" ".join(f"w{j}" for j in range(n_words)), "simulated", base)
        rows.append(
            {
                "item_id": item_id,
                "length": length,
                "frequency": frequency,
                "condition": condition,
                **{f"uid_power_{fmt_power(k)}": features[f"uid_power_{fmt_power(k)}"] for k in POWERS},
            }
        )

    df = pd.DataFrame(rows)
    true_signal = df["uid_power_1.5"].astype(float)
    true_signal = (true_signal - true_signal.mean()) / true_signal.std(ddof=0)
    condition_effect = np.where(df["condition"].to_numpy() == "B", 0.25, -0.25)
    noise = rng.normal(0.0, 0.35, size=len(df))
    df["outcome"] = 0.05 * df["length"] - 0.15 * df["frequency"] + condition_effect + 0.9 * true_signal + noise
    return df


def verify_cv_recovers_known_signal() -> None:
    df = make_ground_truth_dataset()
    sweep = pd.DataFrame(
        cv_power_sweep(
            df,
            outcome="outcome",
            base_numeric=["length", "frequency"],
            categorical_cols=["condition"],
            groups=None,
            n_splits=5,
            seed=7,
        )
    )

    if sweep.empty or sweep["delta_loglik"].isna().any():
        raise AssertionError("CV sweep returned empty or NaN results on simulated data")

    delta_at_truth = float(sweep.loc[np.isclose(sweep["k"], 1.5), "delta_loglik"].iloc[0])
    if delta_at_truth <= 0.25:
        raise AssertionError(f"known uid_power_1.5 predictor should improve held-out log-likelihood; got {delta_at_truth:.4f}")

    best_k = float(sweep.loc[sweep["delta_loglik"].idxmax(), "k"])
    if best_k not in {1.25, 1.5, 1.75}:
        raise AssertionError(f"best recovered k should be near the simulated ground truth 1.5; got {best_k:g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-summary",
        type=Path,
        default=None,
        help="Optional path for a one-line PASS summary.",
    )
    args = parser.parse_args()

    verify_uid_feature_math()
    verify_cv_recovers_known_signal()

    message = "PASS: UID feature math and CV power sweep recover the simulated ground-truth signal."
    print(message)
    if args.write_summary is not None:
        args.write_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_summary.write_text(message + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
