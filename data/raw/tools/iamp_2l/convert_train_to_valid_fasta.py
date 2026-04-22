import sys

"""
FASTA processing script.

Behavior:
- Ignores lines starting with "("
- Ignores lines containing ":"
- Concatenates multi-line sequences
- Splits sequences into two output files:
    1. sequences whose header starts with ">non-AMP"
    2. all other sequences
"""

with open(sys.argv[1], encoding="utf-8") as rf, \
     open(sys.argv[2], "w", encoding="utf-8") as non_amp, \
     open(sys.argv[3], "w", encoding="utf-8") as amp:

    sequence = ""
    header = ""

    for line in rf:
        line = line.strip()

        # skip unwanted lines
        if not line or line.startswith("(") or ":" in line:
            continue

        if line.startswith(">"):
            # write previous sequence before starting a new one
            if sequence:
                if header.startswith(">non-AMP"):
                    non_amp.write(header + "\n" + sequence + "\n")
                else:
                    amp.write(header + "\n" + sequence + "\n")
                sequence = ""

            header = line

        else:
            sequence += line

    # write the last sequence
    if sequence:
        if header.startswith(">non-AMP"):
            non_amp.write(header + "\n" + sequence + "\n")
        else:
            amp.write(header + "\n" + sequence + "\n")
