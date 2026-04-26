# Raw Data — Download Instructions

This directory contains the raw sequence data used to build the evaluation dataset for the AMP prediction tool benchmark.
All files were downloaded on **April 26, 2026**.

```
data/raw/
├── abps/          # Antibacterial peptide sequences from AMP databases
├── non_amps/      # Non-AMP sequences from UniProt
└── tools/         # Training/test datasets from the evaluated tools (when available)
```

---

## 1. Antibacterial Peptide (ABP) Sequences

Files are stored in `data/raw/abps/`. Only sequences with documented antibacterial activity (active against Gram-positive and/or Gram-negative bacteria) are downloaded from each database.

### 1.1 APD — Antimicrobial Peptide Database

- **URL:** https://aps.unmc.edu/database
- **Steps:**
  1. On the search page, check the options **Anti-Gram+ bacteria** and **Anti-Gram− bacteria** for the activity filter (only these two).
  2. Click **Search** at the bottom of the page.
  3. Scroll to the bottom of the results page and click **"<u>Click here</u> to download FASTA file"**.
- **Downloaded file:** `results.fasta`
- **Rename to:** `apd.fasta`
- **Downloaded entries:** 5,496

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
  1. In the left-hand filter panel, scroll down and find **Targets** → **Target Group (Multi select)**.
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

Files are stored in `data/raw/tools/`. These are the training (and, where available, test) datasets published alongside each reviewed tool. They are used to:

- Check for **data leakage** between the training data and the evaluation set (see `notebooks/testing_data_leakage.ipynb`).
- **Exclude training sequences** from the evaluation dataset to ensure a fair benchmark (see `scripts/build_base_dataset.py`).

Beyond leakage checking and sequence exclusion, the datasets in `data/raw/tools/` are also used directly as training input by the Nextflow workflows that wrap the model training processes (`train_amp_bert.nf`, `train_lmpred.nf`, `train_kt_amppred.nf`, `train_plapd.nf`). Those scripts are included for reproducibility but are not necessary for the evaluation pipeline. They read the directory via the `tools_dir` parameter in `nextflow.config` (`tools_dir = 'data/raw/tools'`).

The following tools are covered in the benchmark. The table lists data availability and whether a `data/raw/tools/` subdirectory is present in this repository. Tools without any available data, or for which no data download was needed, do not have a subdirectory.

| Tool | Reference | Training data available | Test data available | `data/raw/tools/` directory | Evaluated |
|------|-----------|:-----------------------:|:-------------------:|:------------------:|-----------|
| iAMP-2L | Xiao et al. (2013) | Yes¹ | Yes¹ | `iamp_2l/` | No — leakage check only |
| iAMPpred | Meher et al. (2017) | No | No | — | No |
| AMP Scanner | Veltri et al. (2018) | Yes | Yes | `amp_scanner/` | Yes |
| AmPEP | Bhadra et al. (2018) | Yes | No | `ampep/` | No — leakage check only |
| Macrel | Santos-Júnior et al. (2020) | Yes | No | `macrel/` | Yes |
| amPEPpy (length/count balanced model) | Lawrence et al. (2021) | Yes | Yes | `ampeppy/` | Yes |
| LMPred (T5 UniRef50-based model) | Dee (2022) | Yes | Yes | `lmpred/` | Yes |
| AMPlify | Li et al. (2022) | Yes | Yes | `amplify/` | Yes |
| AMPpred-EL | Lv et al. (2022) | No | No | — | No |
| Ma et al. (2022) | Ma et al. (2022) | No | Yes | `ma_et_al/` | Yes |
| CAMPR4 prediction server | Gawde et al. (2023) | No | No | — | Yes |
| AMP-BERT | Lee et al. (2023) | Yes | Yes | `amp_bert/` | Yes |
| AMPFinder (stage 1 classifier) | Yang et al. (2023) | Yes | Yes | `ampfinder/` | Yes |
| GEU-AMP50 | Panwar et al. (2023) | No | No | — | No |
| AMP-GSM | Söylemez et al. (2023) | No | No | — | No |
| AMP-RNNpro | Shaon et al. (2024) | Yes | Yes | `amp_rnnpro/` | No — leakage check only |
| PyAMPA (AMPValidate) | Ramos-Llorens et al. (2024) | No² | No | — | Yes |
| AGRAMP (3-gram 9-letter model) | Shao et al. (2024) | Yes | Yes | `agramp/` | Yes |
| PepNet | Han et al. (2024) | Yes | Yes | `pepnet/` | Yes |
| Bhangu et al. (2025) | Bhangu et al. (2025) | No | No | — | No |
| KT-AMPpred (AMP Fine-tuned Model) | Liang et al. (2025) | Yes | Yes | `kt_amppred/` | Yes |
| MSCMamba | He et al. (2025) | No | No | — | No |
| PLAPD | Zhang et al. (2025) | Yes | Yes | `plapd/` | Yes |
| DLFea4AMPGen (ABP-MPB model) | Gao et al. (2025) | Yes | Yes | `dlfea4ampgen/` | Yes |
| AMP-CapsNet | Ghulam et al. (2026) | No | No | — | No |
| MultiAMP (sequence-only model) | Li et al. (2026) | Yes | Yes | `multiamp/` | Yes |

¹ iAMP-2L provides training and test sequences embedded in PDF supplementary files. Discrepancies were found between the dataset described in the paper and the supplementary file contents.  
² PyAMPA was trained on the AMPlify dataset (retrievable from `amplify/`), but does not provide its own data files.

Refer to each tool's GitHub repository or publication for the exact download location of its training data. For tools that can be downloaded automatically, a convenience script `data/raw/tools/download_all.sh` is provided (see end of this section). The following tools **cannot** be downloaded automatically and must be retrieved manually:

- **iAMP-2L** — sequences are embedded in PDF supplementary files; manual text extraction is required (§ 2.1).
- **AGRAMP** — the dataset server does not support direct download; files must be saved via right-click (§ 2.14).

### 2.1 iAMP-2L

iAMP-2L (Xiao et al., 2013) does not distribute its sequences as FASTA files; they are embedded in two PDF supplementary documents. Both PDFs are saved to `data/raw/tools/iamp_2l/raw/` and their text content is extracted manually before parsing.

> **Note on data provenance:** The Macrel repository suggests downloading two files named `Supp-S1.pdf` and `Supp-S2.pdf` directly from `http://www.jci-bioinfo.cn/iAMP/` —presumably old versions of iAMP-2L datasets— but that host is no longer reachable (`Name or service not known`). The supplementary files are now available through the journal publisher (Elsevier) at the URLs below. Note also that the current supplementary files appear to differ from the original dataset reported in the paper, with discrepant sequence counts.

**Benchmark dataset (training + test split) — Supporting Information S1**

- **URL:** https://ars.els-cdn.com/content/image/1-s2.0-S0003269713000390-mmc1.pdf
- **Save as:** `data/raw/tools/iamp_2l/raw/Supp-S1.pdf`
- **Steps:**
  1. Open the PDF.
  2. Select all text (Ctrl + A) and paste it into a plain-text file.
  3. Save as `data/raw/tools/iamp_2l/raw/train.txt`.

**Independent test dataset — Supporting Information S3**

- **URL:** https://ars.els-cdn.com/content/image/1-s2.0-S0003269713000390-mmc3.pdf
- **Save as:** `data/raw/tools/iamp_2l/raw/Supp-S3.pdf`
- **Steps:**
  1. Open the PDF.
  2. Select all text (Ctrl + A) and paste it into a plain-text file.
  3. Save as `data/raw/tools/iamp_2l/raw/test.txt`.

**Parsing**

Once both `.txt` files are in place, run the parsing script from inside the `data/raw/tools/iamp_2l/` directory:

```bash
python parse_iamp2l.py \
    --train  raw/train.txt  \
    --test   raw/test.txt   \
    --outdir processed/
```

The script produces four FASTA files in `processed/`:

```
processed/
├── AMP_train.fasta
├── nonAMP_train.fasta
├── AMP_test.fasta
└── nonAMP_test.fasta
```

### 2.2 AMP Scanner

AMP Scanner v2 (Veltri et al., 2018) distributes its original training and test splits directly in the repository. Files are stored in `data/raw/tools/amp_scanner/`.

- **URL:** https://github.com/dan-veltri/amp-scanner-v2/tree/main/original-dataset
- **Steps:** Download all files from that folder, preserving their original names. The repository also contains a `README.md` describing the dataset — keep it alongside the sequence files as useful provenance.

<details>
<summary>Optional: download via command line</summary>

A blobless sparse checkout fetches only the files in the target subfolder at the pinned commit:

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/dan-veltri/amp-scanner-v2.git /tmp/amp-scanner-v2
cd /tmp/amp-scanner-v2
git sparse-checkout set original-dataset
git checkout 933052e2365631fe93098892120ee535e0ba381a
cp original-dataset/* /path/to/tools/amp_scanner/
```

</details>

The resulting directory should look like:

```
data/raw/tools/amp_scanner/
├── AMP.eval.fa
├── AMP.te.fa
├── AMP.tr.fa
├── DECOY.eval.fa
├── DECOY.te.fa
├── DECOY.tr.fa
└── README.md
```

### 2.3 AmPEP

AmPEP (Bhadra et al., 2018) distributes its training sequences as two ZIP archives at the root of the repository. Files are stored in `data/raw/tools/ampep/`.

- **URL:** https://github.com/ShirleyWISiu/AmPEP
- **Files:** `M_model_train_AMP_sequence.zip`, `M_model_train_nonAMP_sequence.zip`
- **Steps:** Download both ZIP files, unzip them, and remove the archives.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/ShirleyWISiu/AmPEP.git /tmp/AmPEP
cd /tmp/AmPEP
git checkout 066e9c42dfebf9e08d67295b5a15493218ee194d -- \
    M_model_train_AMP_sequence.zip \
    M_model_train_nonAMP_sequence.zip
cp M_model_train_AMP_sequence.zip M_model_train_nonAMP_sequence.zip /path/to/tools/ampep/
cd /path/to/tools/ampep/
unzip M_model_train_AMP_sequence.zip
unzip M_model_train_nonAMP_sequence.zip
rm M_model_train_AMP_sequence.zip M_model_train_nonAMP_sequence.zip
```

</details>

The resulting directory should look like:

```
data/raw/tools/ampep/
├── M_model_train_AMP_sequence.fasta
└── M_model_train_nonAMP_sequence.fasta
```

### 2.4 Macrel

Macrel (Santos-Júnior et al., 2020) builds its training dataset from two sources: the AmPEP training sequences (reused from `data/raw/tools/ampep/`) and hemolytic peptide data from the HemoPI-1 database. Files are stored in `data/raw/tools/macrel/`.

**AmPEP sequences**

Copy the FASTA files from the AmPEP folder (§ 2.3):

```bash
cp data/raw/tools/ampep/M_model_train_AMP_sequence.fasta tools/macrel/
cp data/raw/tools/ampep/M_model_train_nonAMP_sequence.fasta tools/macrel/
```

**HemoPI-1 sequences**

<details>
<summary>Optional: download via command line</summary>

```bash
cd data/raw/tools/macrel/
wget -O hemo.training.pos.faa 'https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/main/pos.fa'
wget -O hemo.training.neg.faa 'https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/main/neg.fa'
wget -O hemo.validation.pos.faa 'https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/validation/pos.fa'
wget -O hemo.validation.neg.faa 'https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/validation/neg.fa'
```

</details>

**Preprocessing**

Once all source files are in place, run `build-AMP-training.py` from inside `data/raw/tools/macrel/` to produce the normalised training dataset. The script requires the Macrel conda environment:

```bash
conda create --name env_macrel -c bioconda macrel
conda activate env_macrel
python build-AMP-training.py
gunzip preproc/AMP.train.tsv.gz
```

The resulting directory should look like:

```
data/raw/tools/macrel/
├── M_model_train_AMP_sequence.fasta
├── M_model_train_nonAMP_sequence.fasta
├── hemo.training.pos.faa
├── hemo.training.neg.faa
├── hemo.validation.pos.faa
├── hemo.validation.neg.faa
└── preproc/
    ├── AMP_NAMP.train.faa
    └── AMP.train.tsv
```

### 2.5 amPEPpy

amPEPpy (Lawrence et al., 2021) provides its training data in the `training_data/` folder of the repository. Only two files are needed for leakage testing and the evaluation pipeline. Files are stored in `data/raw/tools/ampeppy/`.

- **URL:** https://github.com/tlawrence3/amPEPpy/tree/master/training_data
- **Files:** `M_model_train_AMP_sequence.numbered.fasta`, `M_model_train_nonAMP_sequence.numbered.proplen.subsample.fasta`
- **Steps:** Download those two files, preserving their original names.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/tlawrence3/amPEPpy.git /tmp/amPEPpy
cd /tmp/amPEPpy
git checkout 85aab3428b328d9fe4744052258746d8f4ba7bf6 -- \
    training_data/M_model_train_AMP_sequence.numbered.fasta \
    training_data/M_model_train_nonAMP_sequence.numbered.proplen.subsample.fasta
cp training_data/M_model_train_AMP_sequence.numbered.fasta \
   training_data/M_model_train_nonAMP_sequence.numbered.proplen.subsample.fasta \
   /path/to/tools/ampeppy/
```

</details>

The resulting directory should look like:

```
data/raw/tools/ampeppy/
├── M_model_train_AMP_sequence.numbered.fasta
└── M_model_train_nonAMP_sequence.numbered.proplen.subsample.fasta
```

### 2.6 LMPred

LMPred (Dee, 2022) distributes its dataset directly in the repository. Files are stored in `data/raw/tools/lmpred/`.

- **URL:** https://github.com/williamdee1/LMPred_AMP_Prediction/tree/main/LM_Pred_Dataset
- **Steps:** Download all files from that folder, preserving their original names.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/williamdee1/LMPred_AMP_Prediction.git /tmp/LMPred
cd /tmp/LMPred
git sparse-checkout set LM_Pred_Dataset
git checkout 30c7188ea5bf8e699eebfbb08cd1ae9aef1094e9
cp LM_Pred_Dataset/* /path/to/tools/lmpred/
```

</details>

The resulting directory should look like:

```
data/raw/tools/lmpred/
├── X_test.csv
├── X_train.csv
├── X_val.csv
├── y_test.csv
├── y_train.csv
├── y_val.csv
└── README.md
```

### 2.7 AMPlify

AMPlify (Li et al., 2022) distributes its training and test data in the `data/` folder of the repository. The imbalanced non-AMP files are not used in this benchmark. Files are stored in `data/raw/tools/amplify/`.

- **URL:** https://github.com/BirolLab/AMPlify/tree/master/data
- **Files:** `AMPlify_AMP_train_common.fa`, `AMPlify_AMP_test_common.fa`, `AMPlify_non_AMP_train_balanced.fa`, `AMPlify_non_AMP_test_balanced.fa`
- **Steps:** Download those four files, preserving their original names.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/BirolLab/AMPlify.git /tmp/AMPlify
cd /tmp/AMPlify
git checkout 3a07713c25b8a21ef66d31d10e121989d26d9320 -- \
    data/AMPlify_AMP_train_common.fa \
    data/AMPlify_AMP_test_common.fa \
    data/AMPlify_non_AMP_train_balanced.fa \
    data/AMPlify_non_AMP_test_balanced.fa
cp data/AMPlify_AMP_train_common.fa \
   data/AMPlify_AMP_test_common.fa \
   data/AMPlify_non_AMP_train_balanced.fa \
   data/AMPlify_non_AMP_test_balanced.fa \
   /path/to/tools/amplify/
```

</details>

The resulting directory should look like:

```
data/raw/tools/amplify/
├── AMPlify_AMP_train_common.fa
├── AMPlify_AMP_test_common.fa
├── AMPlify_non_AMP_train_balanced.fa
└── AMPlify_non_AMP_test_balanced.fa
```

### 2.8 Ma et al. (2022)

Ma et al. (2022) do not provide training data publicly; only the test sequences used in the paper are available. Files are stored in `data/raw/tools/ma_et_al/`.

- **URL:** https://github.com/mayuefine/c_AMPs-prediction/tree/master/Data
- **Files:** `AMPs.fa`, `Non-AMPs.fa`
- **Steps:** Download both files, preserving their original names.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/mayuefine/c_AMPs-prediction.git /tmp/c_AMPs-prediction
cd /tmp/c_AMPs-prediction
git checkout cf7658bc5d504ba6d996fa7b152270e38275dc46 -- \
    Data/AMPs.fa \
    Data/Non-AMPs.fa
cp Data/AMPs.fa Data/Non-AMPs.fa /path/to/tools/ma_et_al/
```

</details>

The resulting directory should look like:

```
data/raw/tools/ma_et_al/
├── AMPs.fa
└── Non-AMPs.fa
```

### 2.9 CAMPR4 prediction server

CAMPR4 does not provide any of the datasets used to train or test its models (ANN, RF, and SVM). No files are downloaded for this tool.

### 2.10 AMP-BERT

AMP-BERT (Lee et al., 2023) distributes its training data as CSV files at the root of the repository. Files are stored in `data/raw/tools/amp_bert/`.

- **URL:** https://github.com/GIST-CSBL/AMP-BERT
- **Files:** `non_amp_ampep_cdhit90.csv`, `veltri_dramp_cdhit_90.csv`, `all_veltri.csv`
- **Steps:** Download those three files, preserving their original names.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/GIST-CSBL/AMP-BERT.git /tmp/AMP-BERT
cd /tmp/AMP-BERT
git checkout b9ba228180b6edc04f39cac4724281af5f031db5 -- \
    non_amp_ampep_cdhit90.csv \
    veltri_dramp_cdhit_90.csv \
    all_veltri.csv
cp non_amp_ampep_cdhit90.csv \
   veltri_dramp_cdhit_90.csv \
   all_veltri.csv \
   /path/to/tools/amp_bert/
```

</details>

The resulting directory should look like:

```
data/raw/tools/amp_bert/
├── non_amp_ampep_cdhit90.csv
├── veltri_dramp_cdhit_90.csv
└── all_veltri.csv
```

### 2.11 AMPFinder (stage 1 classifier)

AMPFinder (Yang et al., 2023) is a two-stage classifier; only the stage 1 data (AMP vs. non-AMP discrimination) is relevant here. The stage 2 data (functional classification) is not considered. Files are stored in `data/raw/tools/ampfinder/`.

- **URL:** https://github.com/abcair/AMPFinder/tree/main/data/D1
- **Files:** `3594-Samp.fasta`, `3925-Snonamp.fasta` (two of the four files present in D1)
- **Steps:** Download those two files, preserving their original names.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/abcair/AMPFinder.git /tmp/AMPFinder
cd /tmp/AMPFinder
git checkout 666b173d62c59627ac37ea7508467ee82cd9ec67 -- \
    data/D1/3594-Samp.fasta \
    data/D1/3925-Snonamp.fasta
cp data/D1/3594-Samp.fasta \
   data/D1/3925-Snonamp.fasta \
   /path/to/tools/ampfinder/
```

</details>

The resulting directory should look like:

```
data/raw/tools/ampfinder/
├── 3594-Samp.fasta
└── 3925-Snonamp.fasta
```

### 2.12 AMP-RNNpro

AMP-RNNpro (Shaon et al., 2024) distributes its training and test sequences at the root of the repository. Files are stored in `data/raw/tools/amp_rnnpro/`.

- **URL:** https://github.com/Shazzad-Shaon3404/Antimicrobials_
- **Files:** `train_p.fasta`, `trainn_n.fasta`, `testp`, `testn`, `README.md`
- **Steps:** Download those five files, preserving their original names. Note that `Train_file` is also present in the repository but is an empty placeholder (0 bytes) and is not needed. Also note that on Windows, `testp` and `testn` may be saved with a `.txt` extension appended; rename them if so.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/Shazzad-Shaon3404/Antimicrobials_.git /tmp/Antimicrobials_
cd /tmp/Antimicrobials_
git checkout a2946913193422cc3fb05d7578eb37d09f646884 -- \
    train_p.fasta \
    trainn_n.fasta \
    testp \
    testn \
    README.md
cp train_p.fasta trainn_n.fasta testp testn README.md \
   /path/to/tools/amp_rnnpro/
```

</details>

The resulting directory should look like:

```
data/raw/tools/amp_rnnpro/
├── train_p.fasta
├── trainn_n.fasta
├── testp
├── testn
└── README.md
```

### 2.13 PyAMPA (AMPValidate)

The AMPValidate model from PyAMPA was trained on the AMPlify training dataset (§ 2.7) using a 60/20/20 train/val/test split. No separate data files are provided by the authors; no action is needed here.

### 2.14 AGRAMP (3-gram 9-letter model)

AGRAMP (Shao et al., 2024) distributes its training and test datasets through a dedicated web page. Files are stored in `data/raw/tools/agramp/`. Note that these files cannot be downloaded automatically: each link must be right-clicked and saved individually ("Save link as…"), as left-clicking opens the FASTA content in a new browser tab instead of downloading it.

- **URL:** http://omics.gmu.edu/agramp/datasets.php
- **Steps:** Right-click each link listed below, choose "Save link as…", and save the file under the name indicated.

| Link label on the page | Save as |
|------------------------|---------|
| Positive training set (1500) | `AMP_train.fasta` |
| Positive testing set (139) | `AMP_test.fasta` |
| Negative training set (1500), NOAMP1 | `NOAMP1_train.fasta` |
| Negative testing set (139), NOAMP1 | `NOAMP1_test.fasta` |
| Negative training set (1500), NOAMP2 | `NOAMP2_train.fasta` |
| Negative testing set (139), NOAMP2 | `NOAMP2_test.fasta` |
| Negative training set (1500), NOAMP3 | `NOAMP3_train.fasta` |
| Negative testing set (139), NOAMP3 | `NOAMP3_test.fasta` |

All three negative sets are used in the benchmark, as the authors do not specify which one was used for non-AMP sequences.

The resulting directory should look like:

```
data/raw/tools/agramp/
├── AMP_train.fasta
├── AMP_test.fasta
├── NOAMP1_train.fasta
├── NOAMP1_test.fasta
├── NOAMP2_train.fasta
├── NOAMP2_test.fasta
├── NOAMP3_train.fasta
└── NOAMP3_test.fasta
```

### 2.15 PepNet

PepNet (Han et al., 2024) distributes its datasets through Zenodo, which assigns a persistent DOI to each record — a better practice for reproducibility than GitHub, where content can be altered or removed without notice. Files are stored in `data/raw/tools/pepnet/`.

- **URL:** https://zenodo.org/records/13223516
- **Steps:** Download `datasets.tar.gz` from the record and extract it. The archive contains a `datasets/` folder with three subfolders (`AMP/`, `AIP/`, `Toxic/`) and a `properties.pkl` file, all of which are extracted directly inside `data/raw/tools/pepnet/`.
> **Note:** The `properties.pkl` file and the `AMP/checkpoints/` and `AMP/feature/` subfolders, required for PepNet predictions, are already integrated in the corresponding Docker image utilized in the evaluation pipeline. `AMP/checkpoints/`, `AMP/feature/`, `AIP/checkpoints/`, `AIP/feature/`, `Toxic/checkpoints/` and `Toxic/feature/` contain heavy files and are not included in the repository.

<details>
<summary>Optional: download via command line</summary>

```bash
wget -O /tmp/pepnet_datasets.tar.gz \
    'https://zenodo.org/records/13223516/files/datasets.tar.gz?download=1'
tar -xzf /tmp/pepnet_datasets.tar.gz -C data/raw/tools/pepnet/
```

</details>

The resulting directory should look like:

```
data/raw/tools/pepnet/
├── AMP/
├── AIP/
├── Toxic/
└── properties.pkl
```

### 2.16 KT-AMPpred

KT-AMPpred (Liang et al., 2025) distributes its training and test data in the `data/` folder of the repository. Files are stored in `data/raw/tools/kt_amppred/`.

- **URL:** https://github.com/liangxiaodata/AMPpred/tree/main/data
- **Steps:** Download all files from that folder, preserving their original names.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/liangxiaodata/AMPpred.git /tmp/AMPpred
cd /tmp/AMPpred
git sparse-checkout set data
git checkout b4a7276e73b212b1be98256e0c9aa236caa20540
cp data/* /path/to/tools/kt_amppred/
```

</details>

The resulting directory should look like:

```
data/raw/tools/kt_amppred/
├── ABP_DS_test.tsv
├── ABP_DS_train.tsv
├── AFP_DS_test.tsv
├── AFP_DS_train.tsv
├── AMP_DS_test.tsv
├── AMP_DS_train.tsv
├── AVP_DS_test.tsv
├── AVP_DS_train.tsv
└── README.md
```

### 2.17 PLAPD

PLAPD (Zhang et al., 2025) stores its data in the `data/datasets/AMP/` folder of the repository, which contains many files. Only two are needed for leakage testing and the evaluation pipeline; these were identified by manually verifying sequence counts. Files are stored in `data/raw/tools/plapd/`.

- **URL:** https://github.com/lichaozhang2/PLAPD/tree/main/data/datasets/AMP
- **Files:** `training_data.csv`, `val_data.csv`
- **Steps:** Download those two files, preserving their original names.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/lichaozhang2/PLAPD.git /tmp/PLAPD
cd /tmp/PLAPD
git checkout 5f5c6ef4b21adc9e297d240e41bc1e2ca065c54b -- \
    data/datasets/AMP/training_data.csv \
    data/datasets/AMP/val_data.csv
cp data/datasets/AMP/training_data.csv \
   data/datasets/AMP/val_data.csv \
   /path/to/tools/plapd/
```

</details>

The resulting directory should look like:

```
data/raw/tools/plapd/
├── training_data.csv
└── val_data.csv
```

### 2.18 DLFea4AMPGen

DLFea4AMPGen (Gao et al., 2025) distributes its dataset in the `Dataset/` folder of the repository, which includes four subfolders and an Excel file. Files are stored in `data/raw/tools/dlfea4ampgen/`.

- **URL:** https://github.com/hgao12345/DLFea4AMPGen/tree/main/Dataset
- **Steps:** Download all contents of that folder, preserving the folder structure.

<details>
<summary>Optional: download via command line</summary>

```bash
git clone --no-checkout --filter=blob:none \
    https://github.com/hgao12345/DLFea4AMPGen.git /tmp/DLFea4AMPGen
cd /tmp/DLFea4AMPGen
git sparse-checkout set Dataset
git checkout 6ec4a46a206f2501e0c29abf453ae6e0ddb5227e
cp -r Dataset/* /path/to/tools/dlfea4ampgen/
```

</details>

The resulting directory should look like:

```
data/raw/tools/dlfea4ampgen/
├── ABP/
├── AFP/
├── AOP/
├── Other/
├── AMP_from_5databases.xlsx
└── README.md
```

### 2.19 MultiAMP

MultiAMP (Li et al., 2026) distributes its dataset through Hugging Face, which — like Zenodo — uses persistent identifiers. Files are stored in `data/raw/tools/multiamp/`.

- **URL:** https://huggingface.co/jiayi11/multi_amp/blob/main/data.tar.gz
- **Steps:** Download `data.tar.gz` and extract it. The archive contains three folders (`structure/`, `test_amp/`, and `train_amp/`), which will be extracted directly inside `data/raw/tools/multiamp/`. The `structure/` folder is not used in the evaluation pipeline or leakage tests and is not included in the repository.

<details>
<summary>Optional: download via command line</summary>

```bash
wget -O /tmp/multiamp_data.tar.gz \
    'https://huggingface.co/jiayi11/multi_amp/resolve/7c2b1b86304b62e9d0ff4d186c6be6ca82e02e28/data.tar.gz'
tar -xzf /tmp/multiamp_data.tar.gz -C data/raw/tools/multiamp/
```

</details>

The resulting directory should look like:

```
data/raw/tools/multiamp/
├── structure/
├── test_amp/
└── train_amp/
```

### Convenience download script

`data/raw/tools/download_all.sh` automates all downloads that do not require manual steps. Run it from the root of the repository:

```bash
bash data/raw/tools/download_all.sh
```

The following tools are **not** handled by the script and must be set up manually using the instructions in their respective subsections:

- **iAMP-2L** (§ 2.1) — requires manual PDF text extraction.
- **AGRAMP** (§ 2.14) — requires manual right-click download from the dataset server.

---

## 3. Non-AMP Sequences from UniProt

Files are stored in `data/raw/non_amps/`. Non-AMP sequences are retrieved from UniProt using a query designed to select secreted peptides while explicitly excluding an extensive collection of antimicrobial-related keywords. Two subsets are downloaded to balance the evaluation dataset in case ABP sequences outnumber reviewed non-AMP sequences.

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
  4. In the download panel: set **Download all**, format **FASTA (canonical)**, and **Compressed: No** (or decompress the file to place the FASTA in `data/raw/non_amps/`).
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
| APD | 5,496 | 5,494 | 5,494 |
| DRAMP | 4,159 | 4,049 | 4,010 |
| dbAMP | 7,625 | 7,622 | 7,619 |
| DBAASP | 1,977 | 1,860 | 1,775 |
| AMPDB | 8,038 | 5,593 | 5,593 |
| UniProt reviewed (Swiss-Prot) | 17,637 | 16,191 | 16,191 |
| UniProt unreviewed (TrEMBL) | 743,594 | 656,985 | 656,985 |
