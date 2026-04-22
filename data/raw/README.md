# Raw Data — Download Instructions

This directory contains the raw sequence data used to build the evaluation dataset for the AMP prediction tool benchmark.
All files were downloaded on **April 21, 2026**.

```
data/raw/
├── abps/          # Antibacterial peptide sequences from AMP databases
├── non_amps/      # Non-AMP sequences from UniProt
└── tools/         # Training/test datasets from the evaluated tools (when available)
```

---

## 1. Antibacterial Peptide (ABP) Sequences

Files are stored in `abps/`. Only sequences with documented antibacterial activity (active against Gram-positive and/or Gram-negative bacteria) are downloaded from each database.

### 1.1 APD — Antimicrobial Peptide Database

- **URL:** https://aps.unmc.edu/database
- **Steps:**
  1. On the search page, check the options **Anti-Gram+ bacteria** and **Anti-Gram− bacteria** for the activity filter (only these two).
  2. Click **Search** at the bottom of the page.
  3. Scroll to the bottom of the results page and click **"Click here to download FASTA file"**.
- **Downloaded file:** `results.fasta`
- **Rename to:** `apd.fasta`
- **Downloaded entries:** 5,494

### 1.2 DRAMP — Data Repository of Antimicrobial Peptides

- **URL:** http://dramp.cpu-bioinfor.org/downloads/
- **Steps:**
  1. Locate the table that classifies datasets by **Activity**.
  2. In the **Fasta** column, click **Antibacterial.fasta**.
- **Downloaded file:** `Antibacterial_amps.fasta`
- **Rename to:** `dramp.fasta`
- **Downloaded entries:** 4,159

### 1.3 dbAMP

- **URL:** https://ycclab.cuhk.edu.cn/dbAMP/download2024.php
- **Steps:**
  1. Find the **Download Functional Activity Data** table.
  2. Locate the row where **NAME** = `Antibacterial`.
  3. Click the cloud/download icon in the **ACTION** column to download the FASTA file.
- **Downloaded file:** `dbAMP_Antibacterial_2024.fasta`
- **Rename to:** `dbamp.fasta`
- **Downloaded entries:** 7,625

### 1.4 DBAASP — Database of Antimicrobial Activity and Structure of Peptides

- **URL:** https://dbaasp.org/search
- **Steps:**
  1. In the left-hand filter panel, find **Target Group** → **Target Group (Multi select)**.
  2. Select **Gram+** from the dropdown, then also select **Gram−** (both should appear as tags in the selector).
  3. Click the **Search** button at the bottom of the left panel.
  4. Once results load, click **Export Data** (top right).
  5. In the export panel, click **Export FASTA Data**.
- **Downloaded file:** `peptides-fasta.txt`
- **Rename to:** `dbaasp.fasta`
- **Downloaded entries:** 1,977

### 1.5 AMPDB — Antimicrobial Peptide Database

- **URL:** https://bblserver.org.in/ampdb/ampdb-downloads
- **Steps:**
  1. Locate the rows where the **Dataset** column reads **Anti-gram-negative Dataset** and **Anti-gram-positive Dataset**.
  2. For each row, click the **FASTA** link under its **Download format** column.
- **Downloaded files:**
  - `Anti-gram-negative dataset.fasta` → rename to `ampdb_agn.fasta`
  - `Anti-gram-positive dataset.fasta` → rename to `ampdb_agp.fasta`
- **Downloaded entries:** 5,800 (Gram-negative) and 2,238 (Gram-positive), as reported on the website (8,038 combined).

---

## 2. Tool Training Datasets

Files are stored in `tools/`. These are the training (and, where available, test) datasets published alongside each evaluated tool. They are used to:

- Check for **data leakage** between the training data and the evaluation set (see `notebooks/testing_data_leakage.ipynb`).
- **Exclude training sequences** from the evaluation dataset to ensure a fair benchmark.

> When a tool provides separately labelled AMP and non-AMP files for training (e.g. AGRAMP, which ships three non-AMP training files), all of them are used together.

The following tools are included in the benchmark. Models for which no training data is publicly available are marked accordingly.

| Tool | Training dataset availbale |
|------|-------|
| AMP Scanner | Yes |
| Macrel | Yes |
| amPEPpy | Yes |
| LMPred | Yes |
| AMPlify | Yes |
| Ma et al. (2022) | No |
| CAMPR4 (ANN / RF / SVM) | No |
| AMP-BERT | Yes |
| AMPFinder | Yes |
| PyAMPA (AMPValidate) | Yes |
| AGRAMP | Yes |
| PepNet | Yes |
| KT-AMPpred (AMP Fine-tuned Model) | Yes |
| PLAPD | Yes |
| DLFea4AMPGen (ABP-MPB model) | Yes |
| MultiAMP | Yes |

Refer to each tool's GitHub repository or publication for the exact download location of its training data.

---

## 3. Non-AMP Sequences from UniProt

Files are stored in `non_amps/`. Non-AMP sequences are retrieved from UniProt using a query designed to select secreted peptides while explicitly excluding an extensive collection of antimicrobial-related keywords. Two subsets are downloaded to balance the evaluation dataset (ABP sequences outnumber non-AMP sequences from a single source).

### UniProt Query

```
(length:[5 TO 255]) NOT (keyword:KW-0929) NOT (keyword:KW-0211) NOT (keyword:KW-0044)
AND (keyword:KW-0964) NOT (keyword:KW-0930) NOT (keyword:KW-0295) NOT (keyword:KW-0878)
NOT (keyword:KW-0078) NOT (keyword:KW-0081) NOT (keyword:KW-0425)
```

**Keyword meanings:**

| Keyword | Description |
|---------|-------------|
| KW-0044 | Antibiotic |
| KW-0078 | Bacteriocin |
| KW-0081 | Bacteriolytic enzyme |
| KW-0211 | Defensin |
| KW-0295 | Fungicide |
| KW-0425 | Lantibiotic |
| KW-0878 | Amphibian defense peptide |
| KW-0929 | Antimicrobial |
| KW-0930 | Antiviral protein |
| KW-0964 | Secreted *(included — positive filter)* |

### 3.1 Reviewed sequences (Swiss-Prot)

- **URL:** https://www.uniprot.org/
- **Steps:**
  1. Paste the query above into the UniProt search bar and press Enter.
  2. In the **Status** panel (top left), click **Reviewed (Swiss-Prot)**.
  3. Click **Download** (above the results table).
  4. In the download panel: set **Download all**, format **FASTA (canonical)**, and **Compressed: No** (or decompress the file before placing it in `non_amps/`).
- **Downloaded file:** e.g. `uniprotkb_length_5_TO_255_NOT_keyword_K_2026_04_21.fasta` *(filename includes the download date)*
- **Rename to:** `uniprot_reviewed.fasta`
- **Downloaded entries:** 17,637

### 3.2 Unreviewed sequences (TrEMBL)

- **Steps:** Same as above, but in the **Status** panel deselect **Reviewed (Swiss-Prot)** and select **Unreviewed (TrEMBL)** instead.
- **Downloaded file:** e.g. `uniprotkb_length_5_TO_255_NOT_keyword_K_2026_04_21.fasta.gz` *(may be compressed)*
- **Rename to:** `uniprot_unreviewed.fasta`
- **Note:** Decompress the file before running the pipeline if it was downloaded as `.gz`.
- **Downloaded entries:** 743,594 *(the processing pipeline subsamples this file to balance the dataset)*

---

## 4. Sequence Counts After Preprocessing

After download, sequences are preprocessed prior to building the evaluation dataset. Preprocessing strips leading/trailing whitespace, uppercases all sequences, and removes exact duplicates:

```python
def strip_upper_unique_by_sequence(dataframe):
    """
    Strips whitespace, uppercases sequences and drops duplicate sequences.
    Returns a reset-indexed copy.
    """
    df = dataframe.copy()
    df["Sequence"] = df["Sequence"].str.strip().str.upper()
    df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
    return df
```

The table below summarises entry counts at each stage. Note that future downloads from the same sources may yield different numbers as databases are updated.

| Source | Downloaded entries | Distinct raw sequences | Sequences after preprocessing |
|--------|------------------:|----------------------:|-------------------------------:|
| APD | 5,494 | 5,492 | 5,492 |
| DRAMP | 4,159 | 4,049 | 4,010 |
| dbAMP | 7,625 | 7,622 | 7,619 |
| DBAASP | 1,977 | 1,860 | 1,775 |
| AMPDB | 8,038 | 5,593 | 5,593 |
| UniProt reviewed (Swiss-Prot) | 17,637 | 16,191 | 16,191 |
| UniProt unreviewed (TrEMBL) | 743,594 | 656,985 | 656,985 |
