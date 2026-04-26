#!/usr/bin/env python3
"""
parse_iamp2l.py — Parse iAMP-2L benchmark (S1) and independent (S3) datasets.

Both input files are plain-text copies of the supplementary PDFs, produced by
selecting all text in the PDF and pasting into a text editor.  The files contain
prose headers and sub-section labels mixed with FASTA records.

A line is kept if it starts with '>' (FASTA header) or consists entirely of
uppercase letters A-Z (sequence).  Everything else is prose and is discarded.
The cleaned text is then passed directly to Bio.SeqIO.parse.

AMP records have IDs starting with "AP"; all others are non-AMP.

Usage
-----
    python parse_iamp2l.py \
        --train  raw/train.txt  \
        --test   raw/test.txt   \
        --outdir processed/

Output (in <outdir>)
--------------------
    AMP_train.fasta
    nonAMP_train.fasta
    AMP_test.fasta
    nonAMP_test.fasta
"""

import argparse
import re
from io import StringIO
from pathlib import Path

from Bio import SeqIO

_SEQ_LINE = re.compile(r"^[A-Z]+$")


def clean_to_fasta(path: Path) -> StringIO:
    """Return a StringIO containing only the valid FASTA lines from *path*."""
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith(">") or _SEQ_LINE.match(line):
            lines.append(line)
    return StringIO("\n".join(lines))


def parse_file(path: Path, label: str, outdir: Path) -> None:
    print(f"\nProcessing {label} file: {path}")

    records = list(SeqIO.parse(clean_to_fasta(path), "fasta"))

    amps     = [r for r in records if r.id.startswith("AP")]
    non_amps = [r for r in records if not r.id.startswith("AP")]

    print(f"  found {len(amps)} AMP and {len(non_amps)} non-AMP records")

    for subset, name in [(amps, "AMP"), (non_amps, "nonAMP")]:
        out = outdir / f"{name}_{label}.fasta"
        SeqIO.write(subset, out, "fasta")
        print(f"  wrote {len(subset):>5} records → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse iAMP-2L S1 (train) and S3 (test) text dumps into FASTA files."
    )
    parser.add_argument("--train",  required=True, type=Path)
    parser.add_argument("--test",   required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    parse_file(args.train, "train", args.outdir)
    parse_file(args.test,  "test",  args.outdir)
    print("\nDone.")


if __name__ == "__main__":
    main()
