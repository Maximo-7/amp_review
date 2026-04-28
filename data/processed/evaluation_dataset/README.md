# Evaluation Dataset

All files in this folder are derived from `evaluation_dataset.csv`.

| File | Description |
|---|---|
| `evaluation_dataset.csv` | Base dataset with all columns |
| `x_test_maximo_wo_length.csv` | id and seq columns (required for DLFea4AMPGen) |
| `x_test_maximo.csv` | ID, Sequence and Sequence_length |
| `y_test_maximo.csv` | Target variable ABP_from_databases (0/1) |
| `evaluation_dataset.fasta` | Sequences in FASTA format |
| `evaluation_dataset_geq_10aa.fasta` | Sequences in FASTA format (length >= 10, required for AMP Scanner) |

## `splitted_fasta/` subdirectory

The AGRAMP web server enforces a maximum number of sequences per submission.
`evaluation_dataset.fasta` is therefore split into **9 roughly equal parts**
(`evaluation_dataset_part1.fasta` … `evaluation_dataset_part9.fasta`) so that
each file can be submitted to AGRAMP independently for manual prediction.

Sequences are never split mid-record: every file starts with a `>` header and
ends with a complete sequence.  The number of parts is controlled by the
`AGRAMP_SPLIT_N` variable in `build_base_dataset.py`.
