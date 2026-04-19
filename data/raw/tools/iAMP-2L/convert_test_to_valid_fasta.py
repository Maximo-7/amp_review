import sys

"""
Simple script to clean/correct a FASTA file.

Behavior:
- Keeps header lines (starting with ">")
- Concatenates multi-line sequences into a single line
- Writes each sequence when a new header is encountered
- Ensures the last sequence in the file is also written
"""

with open(sys.argv[1], encoding="utf-8") as rf, open(sys.argv[2], "w", encoding="utf-8") as of:
    sequence = ""

    for line in rf:
        line = line.strip()

        if line.startswith(">"):
            # If we already accumulated a sequence, write it before the new header
            if sequence:
                of.write(sequence + "\n")
                sequence = ""

            # Write the header line
            of.write(line + "\n")

        else:
            # Accumulate sequence lines
            sequence += line

    # After the loop ends, write the last sequence if present
    if sequence:
        of.write(sequence + "\n")
