"""
annotate_dataset.py
===================
Annotates the base complete dataset with physicochemical properties and tool
predictions, producing the final complete_dataset.csv.

Expected project layout (paths relative to project root):
    data/interim/complete_dataset_base.csv    -- output of build_base_dataset.py
    data/processed/evaluation_dataset/        -- output of build_base_dataset.py
    results/tools_predictions/                -- prediction files per tool
    scripts/analyze_amps_csv_new.py           -- physicochemical annotation script
    scripts/                                  -- This script lives here

Run from the project root:
    python scripts/annotate_dataset.py
"""

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INTERIM        = ROOT / "data" / "interim"
PROCESSED      = ROOT / "data" / "processed"
EVAL_DIR       = PROCESSED / "evaluation_dataset"
ANNOT_SCRIPT   = ROOT / "scripts" / "analyze_amps_csv_new.py"
RESULTS        = ROOT / "results" / "tools_predictions"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def df_to_fasta(df, output_file):
    """Writes a DataFrame with columns ID and Sequence to a FASTA file."""
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['ID']}\n{row['Sequence']}\n")


# ===========================================================================
# Load base dataset
# ===========================================================================

print("=== Loading complete base dataset ===")
complete_dataset = pd.read_csv(INTERIM / "complete_dataset_base.csv")
print(f"  Rows: {len(complete_dataset)}")


# ===========================================================================
# 1. PHYSICOCHEMICAL ANNOTATION  (via external script)
# ===========================================================================

print("=== Annotating physicochemical properties ===")

with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp_fasta, \
     tempfile.NamedTemporaryFile(suffix=".csv",   delete=False) as tmp_annot:
    tmp_fasta_path = tmp_fasta.name
    tmp_annot_path = tmp_annot.name

try:
    # Write the complete dataset to a temporary FASTA
    df_to_fasta(complete_dataset, tmp_fasta_path)

    # Run annotation script
    result = subprocess.run(
        [sys.executable, str(ANNOT_SCRIPT),
         "-i", tmp_fasta_path,
         "-o", tmp_annot_path],
        check=True, capture_output=True, text=True,
    )

    complete_dataset_annotated = pd.read_csv(tmp_annot_path)

    # Merge annotations (only sequences with canonical residues are annotated)
    complete_dataset = complete_dataset.merge(
        complete_dataset_annotated[complete_dataset["Only_20aa"]],
        on=["Sequence", "ID"],
        how="left",
    )
    print("  Annotation complete.")

except subprocess.CalledProcessError as exc:
    print(f"  WARNING: annotation script failed – skipping physicochemical properties.\n"
          f"  stderr: {exc.stderr}", file=sys.stderr)
finally:
    os.unlink(tmp_fasta_path)
    os.unlink(tmp_annot_path)


# ===========================================================================
# 2. TOOL PREDICTIONS
# ===========================================================================

print("=== Loading tool predictions ===")

# -- AGRAMP --
agramp_predictions = pd.read_csv(RESULTS / "agramp" / "3gram_9_letter_predictions.tsv", sep="\t")
agramp_predictions = agramp_predictions.drop(columns=["SeqID", "Prob_NOAMP", "Peptide"])
agramp_predictions.columns = ["AGRAMP_pred_prob", "AGRAMP_pred_label", "ID"]
agramp_predictions["AGRAMP_pred_label"] = agramp_predictions["AGRAMP_pred_label"].map({"AMP": 1, "NOAMP": 0})

# -- AMP Scanner --
amp_scanner_predictions = pd.read_csv(RESULTS / "amp_scanner" / "amp_scanner_predictions.csv")
amp_scanner_predictions = amp_scanner_predictions[["SeqID", "Prediction_Class", "Prediction_Probability"]]
amp_scanner_predictions.columns = ["ID", "AMP_Scanner_pred_label", "AMP_Scanner_pred_prob"]
amp_scanner_predictions["AMP_Scanner_pred_label"] = amp_scanner_predictions["AMP_Scanner_pred_label"].map({"AMP": 1, "Non-AMP": 0})
# Some IDs end with "*" (AMP Scanner special cases due to sequence length)
amp_scanner_predictions["ID"] = amp_scanner_predictions["ID"].str.rstrip("*")

# -- AMP-BERT --
amp_bert_predictions = pd.read_csv(RESULTS / "amp_bert" / "amp_bert_predictions.csv")
amp_bert_predictions = amp_bert_predictions[["ID", "AMP-BERT_pred_label", "AMP-BERT_pred_prob"]]

# -- amPEPpy --
ampeppy_predictions = pd.read_csv(RESULTS / "ampeppy" / "ampeppy_predictions.tsv", sep="\t")
ampeppy_predictions = ampeppy_predictions[["seq_id", "probability_AMP", "predicted"]]
ampeppy_predictions.columns = ["ID", "amPEPpy_pred_prob", "amPEPpy_pred_label"]
ampeppy_predictions["amPEPpy_pred_label"] = ampeppy_predictions["amPEPpy_pred_label"].map({"AMP": 1, "nonAMP": 0})

# -- AMPFinder --
ampfinder_predictions = pd.read_csv(RESULTS / "ampfinder" / "ampfinder_predictions.csv")
ampfinder_predictions = ampfinder_predictions[["ID", "AMPFinder_pred_prob", "AMPFinder_pred_label"]]

# -- AMPlify --
# Filename: AMPlify_balanced_results_*.tsv (timestamp-dependent; resolved with glob)
amplify_files = glob.glob(str(RESULTS / "amplify" / "AMPlify_balanced_results_*.tsv"))
amplify_predictions = pd.read_csv(amplify_files[0], sep="\t")
amplify_predictions = amplify_predictions[["Sequence_ID", "Probability_score", "Prediction"]]
amplify_predictions.columns = ["ID", "AMPlify_pred_prob", "AMPlify_pred_label"]
amplify_predictions["AMPlify_pred_label"] = amplify_predictions["AMPlify_pred_label"].map({"AMP": 1, "non-AMP": 0})

# -- CAMPR4 (ANN, RF, SVM classifiers) --
campr4_ann_predictions = pd.read_csv(RESULTS / "campr4" / "ann_predictions.tsv", sep="\t")
campr4_ann_predictions.columns = ["ID", "CAMPR4_ann_pred_label", "CAMPR4_ann_pred_prob"]
campr4_ann_predictions["CAMPR4_ann_pred_label"] = campr4_ann_predictions["CAMPR4_ann_pred_label"].map({"AMP": 1, "NAMP": 0})

campr4_rf_predictions = pd.read_csv(RESULTS / "campr4" / "rf_predictions.tsv", sep="\t")
campr4_rf_predictions.columns = ["ID", "CAMPR4_rf_pred_label", "CAMPR4_rf_pred_prob"]
campr4_rf_predictions["CAMPR4_rf_pred_label"] = campr4_rf_predictions["CAMPR4_rf_pred_label"].map({"AMP": 1, "NAMP": 0})

campr4_svm_predictions = pd.read_csv(RESULTS / "campr4" / "svm_predictions.tsv", sep="\t")
campr4_svm_predictions.columns = ["ID", "CAMPR4_svm_pred_label", "CAMPR4_svm_pred_prob"]
campr4_svm_predictions["CAMPR4_svm_pred_label"] = campr4_svm_predictions["CAMPR4_svm_pred_label"].map({"AMP": 1, "NAMP": 0})

# -- DLFea4AMPGen --
# According to the repository documentation: label 0 = positive (ABP), 1 = negative.
dlfea4ampgen_predictions = pd.read_csv(RESULTS / "dlfea4ampgen" / "dlfea4ampgen_predictions.csv")
dlfea4ampgen_predictions = dlfea4ampgen_predictions.drop(columns=["Unnamed: 0", "seq"])
dlfea4ampgen_predictions.columns = ["ID", "DLFea4AMPGen_pred_label", "DLFea4AMPGen_pred_prob"]
dlfea4ampgen_predictions["DLFea4AMPGen_pred_label"] = dlfea4ampgen_predictions["DLFea4AMPGen_pred_label"].map({0: 1, 1: 0})
dlfea4ampgen_predictions["DLFea4AMPGen_pred_prob"]  = 1 - dlfea4ampgen_predictions["DLFea4AMPGen_pred_prob"]  # probability of ABP

# -- KT-AMPpred --
kt_amppred_predictions = pd.read_csv(RESULTS / "kt_amppred" / "kt_amppred_predictions.csv")
kt_amppred_predictions = kt_amppred_predictions[["ID", "Prob_AMP", "Class"]]
kt_amppred_predictions.columns = ["ID", "KT-AMPpred_pred_prob", "KT-AMPpred_pred_label"]
kt_amppred_predictions["KT-AMPpred_pred_label"] = kt_amppred_predictions["KT-AMPpred_pred_label"].map({"AMP": 1, "non-AMP": 0})

# -- LMPred --
lmpred_predictions = pd.read_csv(RESULTS / "lmpred" / "lmpred_predictions.csv")
lmpred_predictions = lmpred_predictions[["ID", "LMPred_pred_prob", "LMPred_pred_label"]]

# -- Ma et al. (2022) --
ma_attention = pd.read_csv(RESULTS / "ma_et_al" / "att_predictions.txt",  header=None)[0].tolist()
ma_bert      = pd.read_csv(RESULTS / "ma_et_al" / "bert_predictions.txt", header=None)[0].tolist()
ma_lstm      = pd.read_csv(RESULTS / "ma_et_al" / "lstm_predictions.txt", header=None)[0].tolist()
labels_df = pd.read_csv(
    RESULTS / "ma_et_al" / "ma_et_al_result.txt",
    sep=";", skiprows=2, usecols=[0, 1],
    names=["ID", "Ma_et_al_pred_label"],
)
labels_df["ID"] = labels_df["ID"].str.lstrip(">")
ma_et_al_predictions = pd.DataFrame({
    "ID":                      labels_df["ID"],
    "Ma_et_al_pred_label":     labels_df["Ma_et_al_pred_label"],
    "Ma_et_al_att_pred_prob":  ma_attention,
    "Ma_et_al_bert_pred_prob": ma_bert,
    "Ma_et_al_lstm_pred_prob": ma_lstm,
})

# -- Macrel --
macrel_predictions = pd.read_csv(
    RESULTS / "macrel" / "macrel_predictions.tsv", sep="\t", skiprows=1
)
macrel_predictions = macrel_predictions.drop(columns=["Sequence", "AMP_family", "Hemolytic", "Hemolytic_probability"])
macrel_predictions.columns = ["ID", "Macrel_pred_label", "Macrel_pred_prob"]
macrel_predictions["Macrel_pred_label"] = macrel_predictions["Macrel_pred_label"].astype(int)

# -- MultiAMP --
multiamp_predictions = pd.read_csv(RESULTS / "multiamp" / "multiamp_predictions.csv")
multiamp_predictions = multiamp_predictions.drop(columns=["sequence", "length"])
multiamp_predictions.columns = ["ID", "MultiAMP_pred_prob", "MultiAMP_pred_label"]
multiamp_predictions["MultiAMP_pred_label"] = multiamp_predictions["MultiAMP_pred_label"].map({"AMP": 1, "non-AMP": 0})

# -- PepNet (joined on Sequence, no ID column) --
pepnet_predictions = pd.read_csv(
    RESULTS / "pepnet" / "AMP_prediction_result.csv", usecols=[1, 2, 3]
)
pepnet_predictions.columns = ["Sequence", "PepNet_pred_prob", "PepNet_pred_label"]

# -- PLAPD --
plapd_predictions = pd.read_csv(RESULTS / "plapd" / "plapd_predictions.csv")
plapd_predictions = plapd_predictions.drop(columns=["Sequence"])
plapd_predictions.columns = ["ID", "PLAPD_pred_prob", "PLAPD_pred_label"]
plapd_predictions["PLAPD_pred_label"] = plapd_predictions["PLAPD_pred_label"].map({"AMP": 1, "non-AMP": 0})

# -- PyAMPA --
pyampa_predictions = pd.read_csv(RESULTS / "pyampa" / "ampvalidate_predictions.csv")
pyampa_predictions = pyampa_predictions.drop(columns=["Sequence"])
pyampa_predictions.columns = ["ID", "PyAMPA_pred_prob", "PyAMPA_pred_label"]
pyampa_predictions["PyAMPA_pred_label"] = pyampa_predictions["PyAMPA_pred_label"].map({"AMP": 1, "non-AMP": 0})

# -- Merge all predictions into the complete dataset --
complete_dataset = (
    complete_dataset
    .merge(agramp_predictions[["AGRAMP_pred_label", "AGRAMP_pred_prob", "ID"]], on="ID", how="left")
    .merge(amp_bert_predictions[["AMP-BERT_pred_label", "AMP-BERT_pred_prob", "ID"]], on="ID", how="left")
    .merge(amp_scanner_predictions[["AMP_Scanner_pred_label", "AMP_Scanner_pred_prob", "ID"]], on="ID", how="left")
    .merge(ampeppy_predictions[["amPEPpy_pred_label", "amPEPpy_pred_prob", "ID"]], on="ID", how="left")
    .merge(ampfinder_predictions[["AMPFinder_pred_label", "AMPFinder_pred_prob", "ID"]], on="ID", how="left")
    .merge(amplify_predictions[["AMPlify_pred_label", "AMPlify_pred_prob", "ID"]], on="ID", how="left")
    .merge(campr4_ann_predictions[["CAMPR4_ann_pred_label", "CAMPR4_ann_pred_prob", "ID"]], on="ID", how="left")
    .merge(campr4_rf_predictions[["CAMPR4_rf_pred_label", "CAMPR4_rf_pred_prob", "ID"]], on="ID", how="left")
    .merge(campr4_svm_predictions[["CAMPR4_svm_pred_label", "CAMPR4_svm_pred_prob", "ID"]], on="ID", how="left")
    .merge(dlfea4ampgen_predictions[["DLFea4AMPGen_pred_label", "DLFea4AMPGen_pred_prob", "ID"]], on="ID", how="left")
    .merge(kt_amppred_predictions[["KT-AMPpred_pred_label", "KT-AMPpred_pred_prob", "ID"]], on="ID", how="left")
    .merge(lmpred_predictions[["LMPred_pred_label", "LMPred_pred_prob", "ID"]], on="ID", how="left")
    .merge(ma_et_al_predictions[["Ma_et_al_pred_label", "Ma_et_al_att_pred_prob", "Ma_et_al_bert_pred_prob", "Ma_et_al_lstm_pred_prob", "ID"]], on="ID", how="left")
    .merge(macrel_predictions[["Macrel_pred_label", "Macrel_pred_prob", "ID"]], on="ID", how="left")
    .merge(multiamp_predictions[["MultiAMP_pred_label", "MultiAMP_pred_prob", "ID"]], on="ID", how="left")
    .merge(pepnet_predictions[["PepNet_pred_label", "PepNet_pred_prob", "Sequence"]], on="Sequence", how="left")
    .merge(plapd_predictions[["PLAPD_pred_label", "PLAPD_pred_prob", "ID"]], on="ID", how="left")
    .merge(pyampa_predictions[["PyAMPA_pred_label", "PyAMPA_pred_prob", "ID"]], on="ID", how="left")
)


# ===========================================================================
# 3. COLUMN TYPES AND ORDERING
# ===========================================================================

cols_to_int = [
    # Ground truths
    "AGRAMP_ground_truth", "AMP-BERT_ground_truth", "AMP_Scanner_ground_truth",
    "amPEPpy_ground_truth", "AMPFinder_ground_truth", "AMPlify_ground_truth",
    "DLFea4AMPGen_ground_truth", "KT-AMPpred_ground_truth", "LMPred_ground_truth",
    "Macrel_ground_truth", "MultiAMP_ground_truth", "PepNet_ground_truth",
    "PLAPD_ground_truth",
    # Predicted labels
    "AGRAMP_pred_label", "AMP-BERT_pred_label", "AMP_Scanner_pred_label",
    "amPEPpy_pred_label", "AMPFinder_pred_label", "AMPlify_pred_label",
    "CAMPR4_ann_pred_label", "CAMPR4_rf_pred_label", "CAMPR4_svm_pred_label",
    "DLFea4AMPGen_pred_label", "KT-AMPpred_pred_label", "LMPred_pred_label",
    "Ma_et_al_pred_label", "Macrel_pred_label", "MultiAMP_pred_label",
    "PepNet_pred_label", "PLAPD_pred_label", "PyAMPA_pred_label",
]
complete_dataset[cols_to_int] = complete_dataset[cols_to_int].astype("Int64")

cols_order = [
    # Sequence and database IDs
    "Sequence",
    "AMPDB_ID", "AMPDB_name",
    "APD_ID",
    "dbAMP_ID",
    "DRAMP_ID",
    "DBAASP_ID", "DBAASP_name",
    "Swiss-Prot_ID", "Swiss-Prot_name",
    "TrEMBL_ID", "TrEMBL_name",
    "From_UniProt_in_ABPs",
    # Tool training IDs and ground truths (chronological by publication year)
    "AMP_Scanner_ID", "AMP_Scanner_ground_truth",
    "Macrel_ID", "Macrel_ground_truth",
    "amPEPpy_AMP_ID", "amPEPpy_nonAMP_ID", "amPEPpy_ground_truth",
    "LMPred_ID", "LMPred_ground_truth",
    "AMPlify_ID", "AMPlify_ground_truth",
    "AMP-BERT_ID", "AMP-BERT_ground_truth",
    "AMPFinder_ID", "AMPFinder_ground_truth",
    "AGRAMP_ID", "AGRAMP_ground_truth",
    "PepNet_ID", "PepNet_ground_truth",
    "KT-AMPpred_ground_truth",
    "PLAPD_ground_truth",
    "DLFea4AMPGen_ID", "DLFea4AMPGen_ground_truth",
    "MultiAMP_ground_truth",
    # Dataset metadata
    "Sequence_length",
    "From_tools",
    "Only_alphabetic",
    "ID",
    "ABP_from_databases",
    "Only_20aa",
    "AMP_not_ABP",
    "Standard_sequence",
    "For_evaluation",
    # Physicochemical properties
    "Molecular Weight",
    "Net Charge (pH 7)",
    "Aromaticity",
    "Instability Index",
    "Isoelectric Point",
    "GRAVY",
    "Boman Index",
    # Predictions (chronological by publication year)
    "AMP_Scanner_pred_label", "AMP_Scanner_pred_prob",
    "Macrel_pred_label", "Macrel_pred_prob",
    "amPEPpy_pred_label", "amPEPpy_pred_prob",
    "LMPred_pred_label", "LMPred_pred_prob",
    "AMPlify_pred_label", "AMPlify_pred_prob",
    "Ma_et_al_pred_label", "Ma_et_al_att_pred_prob", "Ma_et_al_bert_pred_prob", "Ma_et_al_lstm_pred_prob",
    "CAMPR4_ann_pred_label", "CAMPR4_ann_pred_prob",
    "CAMPR4_rf_pred_label", "CAMPR4_rf_pred_prob",
    "CAMPR4_svm_pred_label", "CAMPR4_svm_pred_prob",
    "AMP-BERT_pred_label", "AMP-BERT_pred_prob",
    "AMPFinder_pred_label", "AMPFinder_pred_prob",
    "PyAMPA_pred_label", "PyAMPA_pred_prob",
    "AGRAMP_pred_label", "AGRAMP_pred_prob",
    "PepNet_pred_label", "PepNet_pred_prob",
    "KT-AMPpred_pred_label", "KT-AMPpred_pred_prob",
    "PLAPD_pred_label", "PLAPD_pred_prob",
    "DLFea4AMPGen_pred_label", "DLFea4AMPGen_pred_prob",
    "MultiAMP_pred_label", "MultiAMP_pred_prob",
]
# Only reorder columns that actually exist (guards against missing prediction files)
cols_order = [c for c in cols_order if c in complete_dataset.columns]
complete_dataset = complete_dataset[cols_order]


# ===========================================================================
# 4. FINAL OUTPUT
# ===========================================================================

print("=== Saving complete dataset ===")
complete_dataset.to_csv(PROCESSED / "complete_dataset.csv", index=False)

print("Done.")
print(f"  Complete dataset → {PROCESSED / 'complete_dataset.csv'}")
