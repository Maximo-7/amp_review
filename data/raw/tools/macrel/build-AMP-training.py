# Extracted from build-AMP-table.py (lines 1-21)
# Source: https://github.com/BigDataBiology/macrel
# Changes: FASTA routes
from macrel.fasta import fasta_iter
from macrel.AMP_features import fasta_features
from os import makedirs

makedirs('preproc/', exist_ok=True)
normalized_fname = 'preproc/AMP_NAMP.train.faa'

# The AmPEP data has duplicates! The same exact same sequences appear on both
# the positive and negative classes:
seen = set()
with open(normalized_fname, 'wt') as output:
    for i, (_, seq) in enumerate(fasta_iter('M_model_train_AMP_sequence.fasta')):
        output.write(f">AMP_{i}\n{seq}\n")
        seen.add(seq)
    for i, (_, seq) in enumerate(fasta_iter('M_model_train_nonAMP_sequence.fasta')):
        if seq in seen: continue
        output.write(f">NAMP_{i}\n{seq}\n")
        seen.add(seq)
fs = fasta_features(normalized_fname)
fs['group'] = fs.index.map(lambda ix: ix.split('_')[0])
fs.to_csv('preproc/AMP.train.tsv.gz', sep='\t')
