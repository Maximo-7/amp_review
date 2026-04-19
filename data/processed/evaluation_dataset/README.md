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