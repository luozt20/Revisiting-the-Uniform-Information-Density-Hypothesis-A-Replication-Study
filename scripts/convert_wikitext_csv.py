#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: convert_wikitext_csv.py <input_csv> <output_tokens>", file=sys.stderr)
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10

    with input_path.open(newline="", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)
        for row in reader:
            if not row:
                continue
            text = row[0].strip()
            if not text:
                continue
            outfile.write(text)
            outfile.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
