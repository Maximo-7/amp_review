"""
build_base_dataset.py
=====================
Builds the base complete dataset (without physicochemical annotations or tool
predictions) and the evaluation dataset folder from AMP databases, UniProt
sequences and AMP prediction tool training sets.

Expected project layout (paths relative to project root):
    data/raw/abps/               -- FASTA files from AMP databases
    data/raw/non_amps/           -- UniProt FASTA files (reviewed + unreviewed)
    data/raw/tools/              -- Training (and test) sets of AMP prediction tools
    data/interim/                -- Intermediate output directory
    data/processed/              -- Output directory
    scripts/                     -- This script lives here

Run from the project root:
    python scripts/build_base_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW           = ROOT / "data" / "raw"
ABP_DIR     = RAW / "abps"
NON_AMP_DIR   = RAW / "non_amps"
TOOLS_DIR     = RAW / "tools"

INTERIM       = ROOT / "data" / "interim"
PROCESSED     = ROOT / "data" / "processed"
EVAL_DIR      = PROCESSED / "evaluation_dataset"

INTERIM.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fasta_to_df(fasta_file):
    """
    Reads a FASTA file and returns a DataFrame with columns:
        ID       -- first token of the header after '>'
        Name     -- remainder of the header (may be empty)
        Sequence -- sequence as a plain string

    The ID/Name split uses the first space or pipe character as delimiter.
    """
    import re
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.description.strip()
        parts = re.split(r"[ |]", header, maxsplit=1)
        seq_id = parts[0]
        name   = parts[1] if len(parts) > 1 else ""
        records.append({"ID": seq_id, "Name": name, "Sequence": str(record.seq)})
    return pd.DataFrame(records)


def fasta_to_df_2(fasta_file):
    """
    Like fasta_to_df but splits on the first *space* only.
    Required for UniProt headers that may contain pipe characters in the ID.
    """
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.description.strip()
        parts  = header.split(" ", maxsplit=1)
        seq_id = parts[0]
        name   = parts[1] if len(parts) > 1 else ""
        records.append({"ID": seq_id, "Name": name, "Sequence": str(record.seq)})
    return pd.DataFrame(records)


def fasta_to_df_pepnet(fasta_file):
    """
    Reads PepNet FASTA files whose headers encode the label as the last
    character after a pipe: e.g. '>trAMP3252|1' -> ID='trAMP3252', Label=1.
    """
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.description.strip()
        parts = header.split("|", maxsplit=1)
        seq_id = parts[0]
        label = int(parts[1]) if len(parts) > 1 else ""
        records.append({"ID": seq_id, "Label": label, "Sequence": str(record.seq)})
    return pd.DataFrame(records)


def df_to_fasta(df, output_file):
    """Writes a DataFrame with columns ID and Sequence to a FASTA file."""
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['ID']}\n{row['Sequence']}\n")


def strip_upper_unique_by_sequence(dataframe):
    """
    Strips whitespace, uppercases sequences and drops duplicate sequences.
    Returns a reset-indexed copy.
    """
    df = dataframe.copy()
    df["Sequence"] = df["Sequence"].str.strip().str.upper()
    df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
    return df


# ===========================================================================
# 1. AMP DATABASES  (positive / ABP sequences)
# ===========================================================================

print("=== Loading AMP databases ===")

# -- AMPDB --
ampdb_anti_gram_n = fasta_to_df(ABP_DIR / "ampdb_agn.fasta")
ampdb_anti_gram_p = fasta_to_df(ABP_DIR / "ampdb_agp.fasta")
df_ampdb = strip_upper_unique_by_sequence(
    pd.concat([ampdb_anti_gram_n, ampdb_anti_gram_p], ignore_index=True)
)

# -- APD --
df_apd = fasta_to_df(ABP_DIR / "apd.fasta")
df_apd = df_apd[df_apd["Sequence"].str.strip() != ""]
df_apd = strip_upper_unique_by_sequence(df_apd).drop(columns=["Name"])  # Name is empty

# -- dbAMP --
df_dbamp = strip_upper_unique_by_sequence(
    fasta_to_df(ABP_DIR / "dbamp.fasta")
).drop(columns=["Name"])  # Name is empty

# -- DRAMP --
df_dramp = strip_upper_unique_by_sequence(
    fasta_to_df(ABP_DIR / "dramp.fasta")
).drop(columns=["Name"])  # Name is empty

# -- DBAASP --
df_dbaasp = strip_upper_unique_by_sequence(
    fasta_to_df(ABP_DIR / "dbaasp.fasta")
)

# -- Merge all ABP databases via outer join on Sequence --
# Each database gets prefixed ID and Name columns so provenance is preserved.
abp_dfs = {
    "AMPDB":  df_ampdb,
    "APD":    df_apd,
    "dbAMP":  df_dbamp,
    "DRAMP":  df_dramp,
    "DBAASP": df_dbaasp,
}
for name, df in abp_dfs.items():
    abp_dfs[name] = df.rename(columns={"ID": f"{name}_ID"})
    if "Name" in abp_dfs[name].columns:
        abp_dfs[name] = abp_dfs[name].rename(columns={"Name": f"{name}_name"})

df_abps = None
for df in abp_dfs.values():
    df_abps = df if df_abps is None else df_abps.merge(df, on="Sequence", how="outer")

# Move Sequence to first position for readability
df_abps = df_abps[["Sequence"] + [c for c in df_abps.columns if c != "Sequence"]]

print(f"  ABPs from databases: {len(df_abps)}")


# ===========================================================================
# 2. UNIPROT REVIEWED (negative / non-AMP sequences)
# ===========================================================================
# Sequences were filtered in UniProt by excluding entries containing terms
# such as 'defensin', 'antimicrobial' or 'antibiotic'. Only reviewed (Swiss-Prot)
# entries with lengths 5-255 aa were retained.

print("=== Loading UniProt reviewed sequences ===")

df_uniprot = fasta_to_df_2(
    NON_AMP_DIR / "uniprot_reviewed.fasta"
)
df_uniprot = df_uniprot.rename(columns={"ID": "Swiss-Prot_ID", "Name": "Swiss-Prot_name"})
df_uniprot = strip_upper_unique_by_sequence(df_uniprot)

# Flag sequences that also appear in the ABP set
df_uniprot["From_UniProt_in_ABPs"] = df_uniprot["Sequence"].isin(df_abps["Sequence"])
df_uniprot = df_uniprot[["Sequence", "Swiss-Prot_ID", "Swiss-Prot_name", "From_UniProt_in_ABPs"]]

print(f"  UniProt reviewed sequences: {len(df_uniprot)}")


# ===========================================================================
# 3. TOOL TRAINING DATASETS
# ===========================================================================
# Training sequences for AMP prediction tools are collected and merged.
# Ground-truth labels: 1 = AMP, 0 = non-AMP, -1 = present in both classes.

print("=== Loading tool training datasets ===")

# -- AMP Scanner --
amp_scanner_segs = {
    "AMP":    ["AMP.tr.fa",    "AMP.eval.fa"],
    "nonAMP": ["DECOY.tr.fa",  "DECOY.eval.fa"],
}
df_AMP_Scanner_AMP    = pd.concat(
    [fasta_to_df(TOOLS_DIR / "amp_scanner" / f).drop(["Name"], axis=1)
     for f in amp_scanner_segs["AMP"]], ignore_index=True)
df_AMP_Scanner_nonAMP = pd.concat(
    [fasta_to_df(TOOLS_DIR / "amp_scanner" / f).drop(["Name"], axis=1)
     for f in amp_scanner_segs["nonAMP"]], ignore_index=True)
df_AMP_Scanner_AMP["AMP_Scanner_ground_truth"]    = 1
df_AMP_Scanner_nonAMP["AMP_Scanner_ground_truth"] = 0
df_AMP_Scanner = strip_upper_unique_by_sequence(
    pd.concat([df_AMP_Scanner_AMP, df_AMP_Scanner_nonAMP], ignore_index=True)
    .rename(columns={"ID": "AMP_Scanner_ID"})
)

# -- Macrel (trained on AmPEP dataset; sequences lack identifiers) --
df_Macrel_AMP    = fasta_to_df(TOOLS_DIR / "macrel" / "M_model_train_AMP_sequence.fasta").drop(["ID", "Name"], axis=1)
df_Macrel_nonAMP = fasta_to_df(TOOLS_DIR / "macrel" / "M_model_train_nonAMP_sequence.fasta").drop(["ID", "Name"], axis=1)
df_Macrel_AMP["Macrel_ground_truth"]    = 1
df_Macrel_nonAMP["Macrel_ground_truth"] = 0
df_Macrel = strip_upper_unique_by_sequence(
    pd.concat([df_Macrel_AMP, df_Macrel_nonAMP], ignore_index=True)
)

# -- amPEPpy --
df_amPEPpy_AMP    = strip_upper_unique_by_sequence(
    fasta_to_df(TOOLS_DIR / "ampeppy" / "M_model_train_AMP_sequence.numbered.fasta")
    .drop(["Name"], axis=1)
    .rename(columns={"ID": "amPEPpy_AMP_ID"})
)
df_amPEPpy_nonAMP = strip_upper_unique_by_sequence(
    fasta_to_df(TOOLS_DIR / "ampeppy" / "M_model_train_nonAMP_sequence.numbered.proplen.subsample.fasta")
    .drop(["Name"], axis=1)
    .rename(columns={"ID": "amPEPpy_nonAMP_ID"})
)
df_amPEPpy = (
    pd.concat([df_amPEPpy_AMP, df_amPEPpy_nonAMP], ignore_index=True)
    .drop_duplicates(subset="Sequence")[["Sequence"]]
    .merge(df_amPEPpy_AMP,    on="Sequence", how="outer")
    .merge(df_amPEPpy_nonAMP, on="Sequence", how="outer")
)
df_amPEPpy["amPEPpy_ground_truth"] = np.select(
    [
        df_amPEPpy["amPEPpy_AMP_ID"].notna()  & df_amPEPpy["amPEPpy_nonAMP_ID"].isna(),
        df_amPEPpy["amPEPpy_AMP_ID"].isna()   & df_amPEPpy["amPEPpy_nonAMP_ID"].notna(),
        df_amPEPpy["amPEPpy_AMP_ID"].notna()  & df_amPEPpy["amPEPpy_nonAMP_ID"].notna(),
    ],
    [1, 0, -1],
    default=np.nan,
).astype(int)

# -- LMPred --
df_LMPred = pd.concat(
    [pd.read_csv(TOOLS_DIR / "lmpred" / "X_train.csv"),
     pd.read_csv(TOOLS_DIR / "lmpred" / "X_val.csv")],
    ignore_index=True,
)
df_LMPred.columns = ["Sequence_length", "Sequence", "LMPred_ID"]
LMPred_labels = pd.concat(
    [pd.read_csv(TOOLS_DIR / "lmpred" / "y_train.csv", header=None),
     pd.read_csv(TOOLS_DIR / "lmpred" / "y_val.csv",   header=None)],
    ignore_index=True,
)
df_LMPred["LMPred_ground_truth"] = LMPred_labels[0].astype(int)
df_LMPred = strip_upper_unique_by_sequence(
    df_LMPred.drop(columns=["Sequence_length"])  # added globally later
)

# -- AMPlify --
amplify_amp    = fasta_to_df(TOOLS_DIR / "amplify" / "AMPlify_AMP_train_common.fa").drop(["Name"], axis=1)
amplify_nonamp = fasta_to_df(TOOLS_DIR / "amplify" / "AMPlify_non_AMP_train_balanced.fa").drop(["Name"], axis=1)
amplify_amp["AMPlify_ground_truth"]    = 1
amplify_nonamp["AMPlify_ground_truth"] = 0
df_AMPlify = strip_upper_unique_by_sequence(
    pd.concat([amplify_amp, amplify_nonamp], ignore_index=True)
    .rename(columns={"ID": "AMPlify_ID"})
)

# -- AMP-BERT --
df_AMP_BERT = pd.read_csv(TOOLS_DIR / "amp_bert" / "all_veltri.csv")
df_AMP_BERT.columns = ["AMP-BERT_ID", "Sequence", "Sequence_length", "AMP-BERT_ground_truth"]
df_AMP_BERT["AMP-BERT_ground_truth"] = df_AMP_BERT["AMP-BERT_ground_truth"].astype(int)
df_AMP_BERT = strip_upper_unique_by_sequence(
    df_AMP_BERT.drop(columns=["Sequence_length"])  # added globally later
)

# -- AMPFinder --
ampfinder_amp    = fasta_to_df(TOOLS_DIR / "ampfinder" / "D1" / "3594-Samp.fasta").drop(["Name"], axis=1)
ampfinder_nonamp = fasta_to_df(TOOLS_DIR / "ampfinder" / "D1" / "3925-Snonamp.fasta").drop(["Name"], axis=1)
ampfinder_amp["AMPFinder_ground_truth"]    = 1
ampfinder_nonamp["AMPFinder_ground_truth"] = 0
df_AMPFinder = strip_upper_unique_by_sequence(
    pd.concat([ampfinder_amp, ampfinder_nonamp], ignore_index=True)
    .rename(columns={"ID": "AMPFinder_ID"})
)

# -- AGRAMP --
agramp_amp   = fasta_to_df(TOOLS_DIR / "agramp" / "AMP_train.txt").drop(["Name"], axis=1)
agramp_namp1 = fasta_to_df(TOOLS_DIR / "agramp" / "NOAMP1_train.txt").drop(["Name"], axis=1)
agramp_namp2 = fasta_to_df(TOOLS_DIR / "agramp" / "NOAMP2_train.txt").drop(["Name"], axis=1)
agramp_namp3 = fasta_to_df(TOOLS_DIR / "agramp" / "NOAMP3_train.txt").drop(["Name"], axis=1)
agramp_amp["AGRAMP_ground_truth"] = 1
df_AGRAMP_nonAMP = pd.concat([agramp_namp1, agramp_namp2, agramp_namp3], ignore_index=True)
df_AGRAMP_nonAMP["AGRAMP_ground_truth"] = 0
df_AGRAMP = strip_upper_unique_by_sequence(
    pd.concat([agramp_amp, df_AGRAMP_nonAMP], ignore_index=True)
    .rename(columns={"ID": "AGRAMP_ID"})
)

# -- PepNet --
pepnet_train = fasta_to_df_pepnet(TOOLS_DIR / "pepnet" / "AMP" / "data_split_train.txt")
pepnet_val   = fasta_to_df_pepnet(TOOLS_DIR / "pepnet" / "AMP" / "data_split_valid.txt")
df_PepNet = strip_upper_unique_by_sequence(
    pd.concat([pepnet_train, pepnet_val], ignore_index=True)
    .rename(columns={"ID": "PepNet_ID", "Label": "PepNet_ground_truth"})
)

# -- KT-AMPpred (AMP classifier only; no ID column) --
df_KT_AMPpred = strip_upper_unique_by_sequence(
    pd.read_csv(TOOLS_DIR / "kt_amppred" / "amp_train.tsv", sep="\t")
    .drop(["index"], axis=1)
    .rename(columns={"label": "KT-AMPpred_ground_truth", "text": "Sequence"})
)

# -- PLAPD --
df_PLAPD = strip_upper_unique_by_sequence(
    pd.read_csv(TOOLS_DIR / "plapd" / "training_data.csv")
    .rename(columns={"Seq": "Sequence", "Label": "PLAPD_ground_truth"})
)

# -- DLFea4AMPGen (ABP-MPB model) --
df_DLFea4AMPGen = strip_upper_unique_by_sequence(
    pd.read_csv(TOOLS_DIR / "dlfea4ampgen" / "ABP" / "train.csv")
    .rename(columns={"id": "DLFea4AMPGen_ID", "seq": "Sequence", "label": "DLFea4AMPGen_ground_truth"})
)

# -- MultiAMP --
# Each sample is a single-sequence .fas file; the label is the last token of the header.
multi_amp_path = TOOLS_DIR / "multiamp"
seqs_dict = {"train_amp": set(), "train_nonamp": set()}
for folder in multi_amp_path.iterdir():
    if not folder.is_dir() or folder.name not in seqs_dict:
        continue
    for file in folder.iterdir():
        if file.suffix != ".fas":
            continue
        lines = [l.strip() for l in file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        label    = lines[0].split("|")[-1]   # "1" or "0"
        sequence = lines[1]
        split_key = "train_amp" if label == "1" else "train_nonamp"
        seqs_dict[split_key].add(sequence)

df_MultiAMP = strip_upper_unique_by_sequence(pd.concat([
    pd.DataFrame({"Sequence": list(seqs_dict["train_amp"]),    "MultiAMP_ground_truth": 1}),
    pd.DataFrame({"Sequence": list(seqs_dict["train_nonamp"]), "MultiAMP_ground_truth": 0}),
], ignore_index=True))

# -- Merge all tool training datasets via outer join on Sequence --
tools_dfs = {
    "AMP_Scanner":   df_AMP_Scanner,
    "Macrel":        df_Macrel,
    "amPEPpy":       df_amPEPpy,
    "LMPred":        df_LMPred,
    "AMPlify":       df_AMPlify,
    "AMP-BERT":      df_AMP_BERT,
    "AMPFinder":     df_AMPFinder,
    "AGRAMP":        df_AGRAMP,
    "PepNet":        df_PepNet,
    "KT-AMPpred":    df_KT_AMPpred,
    "PLAPD":         df_PLAPD,
    "DLFea4AMPGen":  df_DLFea4AMPGen,
    "MultiAMP":      df_MultiAMP,
}
df_tools = None
for df in tools_dfs.values():
    df_tools = df if df_tools is None else df_tools.merge(df, on="Sequence", how="outer")

print(f"  Tool training sequences: {len(df_tools)}")


# ===========================================================================
# 4. COMPLETE BASE DATASET  (ABPs + UniProt + tool training sets)
# ===========================================================================

print("=== Building complete base dataset ===")

complete_dataset = (
    df_abps
    .merge(df_uniprot, on="Sequence", how="outer")
    .merge(df_tools,   on="Sequence", how="outer")
)

# Standard amino-acid alphabet (20 canonical residues)
aa20 = set("ACDEFGHIKLMNPQRSTVWY")

# Identify ABPs that can be used for evaluation (not in any tool training set,
# standard length, only canonical residues)
abps_for_evaluation = complete_dataset[
    complete_dataset["Sequence"].isin(df_abps["Sequence"])
    & ~complete_dataset["Sequence"].isin(df_tools["Sequence"])
    & complete_dataset["Sequence"].str.len().between(5, 255)
    & complete_dataset["Sequence"].apply(lambda s: set(s).issubset(aa20))
]

non_amps_for_evaluation = complete_dataset[
    complete_dataset["Sequence"].isin(df_uniprot["Sequence"])
    & ~complete_dataset["Sequence"].isin(df_tools["Sequence"])
    & complete_dataset["Sequence"].str.len().between(5, 255)
    & ~complete_dataset["Sequence"].isin(df_abps["Sequence"])
    & complete_dataset["Sequence"].apply(lambda s: set(s).issubset(aa20))
]

print(f"  ABPs for evaluation:      {len(abps_for_evaluation)}")
print(f"  Non-AMPs for evaluation:  {len(non_amps_for_evaluation)}")

# ---------------------------------------------------------------------------
# Balance: add new unreviewed UniProt (TrEMBL) sequences to match ABP count.
# ---------------------------------------------------------------------------
n_extra = len(abps_for_evaluation) - len(non_amps_for_evaluation)
if n_extra > 0:
    print(f"  Adding {n_extra} TrEMBL sequences to balance the evaluation dataset …")
    df_uniprot_unreviewed = fasta_to_df_2(
        NON_AMP_DIR / "uniprot_unreviewed.fasta"
    )
    df_uniprot_unreviewed = df_uniprot_unreviewed.rename(
        columns={"ID": "TrEMBL_ID", "Name": "TrEMBL_name"}
    )
    df_uniprot_unreviewed = strip_upper_unique_by_sequence(df_uniprot_unreviewed)

    # Keep only sequences not already in the complete dataset
    df_uniprot_unreviewed = df_uniprot_unreviewed[
        ~df_uniprot_unreviewed["Sequence"].isin(complete_dataset["Sequence"])
    ]
    # Keep only valid sequences (length and alphabet)
    df_uniprot_unreviewed = df_uniprot_unreviewed[
        df_uniprot_unreviewed["Sequence"].str.len().between(5, 255)
        & df_uniprot_unreviewed["Sequence"].apply(lambda s: set(s).issubset(aa20))
    ]

    df_uniprot_unreviewed_selected = df_uniprot_unreviewed.sample(n=n_extra, random_state=356)
    complete_dataset = pd.concat([complete_dataset, df_uniprot_unreviewed_selected], ignore_index=True)

# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------
complete_dataset["Sequence_length"]    = complete_dataset["Sequence"].str.len()
complete_dataset["From_tools"]         = complete_dataset["Sequence"].isin(df_tools["Sequence"])
complete_dataset["Only_alphabetic"]    = complete_dataset["Sequence"].str.isalpha()
complete_dataset["ID"]                 = [f"ARES_{i + 1}" for i in range(len(complete_dataset))]
complete_dataset["ABP_from_databases"] = complete_dataset["Sequence"].isin(df_abps["Sequence"])
complete_dataset["Only_20aa"]          = complete_dataset["Sequence"].apply(lambda s: set(s).issubset(aa20))
complete_dataset["From_UniProt_in_ABPs"] = complete_dataset["From_UniProt_in_ABPs"].fillna(False)

# Sequences that are labeled as AMP in at least one tool training set but are
# NOT in any of the curated ABP databases
complete_dataset["AMP_not_ABP"] = (
    (
        complete_dataset["amPEPpy_ground_truth"].isin([-1, 1])
        | (complete_dataset["AMP_Scanner_ground_truth"] == 1)
        | (complete_dataset["Macrel_ground_truth"]      == 1)
        | (complete_dataset["LMPred_ground_truth"]      == 1)
        | (complete_dataset["AMPlify_ground_truth"]     == 1)
        | (complete_dataset["AMP-BERT_ground_truth"]    == 1)
        | (complete_dataset["AMPFinder_ground_truth"]   == 1)
        | (complete_dataset["AGRAMP_ground_truth"]      == 1)
        | (complete_dataset["PepNet_ground_truth"]      == 1)
        | (complete_dataset["KT-AMPpred_ground_truth"]  == 1)
        | (complete_dataset["PLAPD_ground_truth"]       == 1)
        | (complete_dataset["DLFea4AMPGen_ground_truth"]== 1)
        | (complete_dataset["MultiAMP_ground_truth"]    == 1)
    )
    & (~complete_dataset["ABP_from_databases"])
)

complete_dataset["Standard_sequence"] = (
    complete_dataset["Sequence_length"].between(5, 255)
    & complete_dataset["Only_20aa"]
)

# Sequences eligible for evaluation: standard and not in any tool training set
complete_dataset["For_evaluation"] = (
    complete_dataset["Standard_sequence"]
    & (~complete_dataset["From_tools"])
)

print(f"  Complete base dataset size: {len(complete_dataset)}")

print("=== Saving complete base dataset ===")
complete_dataset.to_csv(INTERIM / "complete_dataset_base.csv", index=False)

# ===========================================================================
# 5. EVALUATION DATASET  (derived outputs)
# ===========================================================================
# The evaluation dataset consists of:
#   - ABPs from the 5 AMP databases
#   - UniProt reviewed and unreviewed sequences not in the ABP set
# All must be standard sequences not present in any tool training set.

print("=== Building evaluation dataset ===")

evaluation_dataset = complete_dataset[complete_dataset["For_evaluation"]].copy()

# Drop columns that carry no information in the evaluation subset
evaluation_dataset = evaluation_dataset.dropna(axis=1, how="all")
evaluation_dataset = evaluation_dataset.drop(
    columns=[c for c in ["From_tools", "Only_20aa", "Only_alphabetic",
                          "AMP_not_ABP", "Standard_sequence", "For_evaluation"]
             if c in evaluation_dataset.columns]
)

print(f"  Evaluation dataset size: {len(evaluation_dataset)}")

print("=== Saving evaluation dataset files ===")

# -- Base CSV --
evaluation_dataset.to_csv(EVAL_DIR / "evaluation_dataset.csv", index=False)

# -- DLFea4AMPGen input (id + seq, no length) --
(evaluation_dataset[["ID", "Sequence"]]
 .rename(columns={"ID": "id", "Sequence": "seq"})
 .to_csv(EVAL_DIR / "x_test_maximo_wo_length.csv", index=False))

# -- Standard feature table (ID + Sequence + length) --
x_test = evaluation_dataset[["ID", "Sequence"]].copy()
x_test["Sequence_length"] = x_test["Sequence"].str.len()
x_test.to_csv(EVAL_DIR / "x_test_maximo.csv", index=False)

# -- Target variable --
(evaluation_dataset[["ABP_from_databases"]]
 .astype(int)
 .to_csv(EVAL_DIR / "y_test_maximo.csv", index=False))

# -- FASTA outputs --
df_to_fasta(evaluation_dataset, EVAL_DIR / "evaluation_dataset.fasta")
df_to_fasta(
    evaluation_dataset[evaluation_dataset["Sequence_length"] >= 10],
    EVAL_DIR / "evaluation_dataset_geq_10aa.fasta",  # required by AMP Scanner
)


# ===========================================================================

print("Done.")
print(f"  Evaluation dataset    → {EVAL_DIR}")
print(f"  Complete base dataset → {INTERIM / 'complete_dataset_base.csv'}")
