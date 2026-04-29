#!/bin/bash
# Clean a manually concatenated AGRAMP output file:
#   - keep the header line (first occurrence only)
#   - remove all subsequent header duplicates
#   - remove blank lines
#   - ensure a single trailing newline

INPUT="$1"

if [[ -z "$INPUT" ]]; then
    echo "Usage: $0 <input.tsv>"
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "Error: input file '$INPUT' not found"
    exit 1
fi

HEADER="SeqID	Prob_AMP	Prob_NOAMP	AMP/NOAMP	AMP_ID	Peptide"
TMP=$(mktemp)

awk -v header="$HEADER" '
    /^$/             { next }                      # skip blank lines
    !header_printed && $0 == header {              # first real header found
        print; header_printed = 1; next
    }
    $0 == header     { next }                      # skip duplicate headers
                     { print }
' "$INPUT" | sed -e '$a\' > "$TMP"

mv "$TMP" "$INPUT"
echo "Done: $INPUT"
