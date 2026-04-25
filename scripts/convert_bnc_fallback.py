#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: convert_bnc_fallback.py <input_csv> <output_tsv>", file=sys.stderr)
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    fieldnames = ["mean_rating", "text", "length", "MOP", "language", "rating_list"]

    with input_path.open(newline="", encoding="utf-8") as infile, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in reader:
            if row.get("language") != "en":
                continue

            writer.writerow(
                {
                    "mean_rating": row.get("mean_rating", ""),
                    "text": row.get("text", "").strip(),
                    "length": row.get("length", ""),
                    "MOP": "MOP2",
                    "language": row.get("language", ""),
                    "rating_list": row.get("rating_list", ""),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
