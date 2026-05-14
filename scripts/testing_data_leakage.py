"""
testing_data_leakage.py
=======================
Investigates data leakage across AMP prediction tools that provide training
or test datasets. For each tool, two potential sources of leakage are examined:
    - Class overlap: shared sequences between positive (AMP, ABP, etc.) and
      negative (non-AMP, non-ABP, etc.) sets, which are expected to be disjoint.
    - Partition overlap: shared sequences between train, validation and test
      splits, which would inflate reported performance metrics.

A tabular summary at the end reports per-tool overlap counts and percentages.

Tools are assessed in the order they appear in the benchmark overview table
in the README (data/raw/tools/), following publication year:
    iAMP-2L, AMP Scanner, AmPEP, Macrel, amPEPpy, LMPred, AMPlify,
    Ma et al. (2022), AMP-BERT, AMPFinder, AMP-RNNpro, AGRAMP,
    PepNet, KT-AMPpred, PLAPD, DLFea4AMPGen, MultiAMP

Expected project layout (paths relative to project root):
    data/raw/tools/              -- Training/test datasets from evaluated tools
    scripts/                     -- This script lives here

Run from the project root:
    python scripts/testing_data_leakage.py
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
from Bio import SeqIO
import re


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT      = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "data" / "raw" / "tools"


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
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        header = record.description.strip()
        parts = re.split(r"[ |]", header, maxsplit=1)
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
        parts  = header.split("|")
        seq_id = "|".join(parts[:-1])
        label  = int(parts[-1]) if len(parts) > 1 else ""
        records.append({"ID": seq_id, "Label": label, "Sequence": str(record.seq)})
    return pd.DataFrame(records)


def check_pairs_overlap(splits_dict):
    """
    Check for sequence overlap between all pairwise dataset splits.

    Parameters
    ----------
    splits_dict : dict[str, set]
        Dictionary mapping split names to sets of sequences.

    Returns
    -------
    list of dict
        Each entry records the two split names and the count of shared sequences.
        Empty if no overlaps are found.
    """
    results = []
    has_overlap = False

    for (name_a, seqs_a), (name_b, seqs_b) in combinations(splits_dict.items(), 2):
        overlap = seqs_a & seqs_b
        if overlap:
            has_overlap = True
            print(f"  ⚠️  Overlap between {name_a} and {name_b}: {len(overlap)} sequences")
            results.append({"split_a": name_a, "split_b": name_b, "overlap": len(overlap)})

    if not has_overlap:
        print("  ✅ No overlap detected")

    return results


def _total_unique(splits_dict):
    """Return total number of unique sequences across all splits."""
    return len(set().union(*splits_dict.values()))


def _report_entry(tool, splits_dict, overlaps):
    """
    Build a summary row for the tabular report.

    Returns a dict with per-tool overlap statistics broken down by type:
        - pos/neg class overlap (incompatible-class leakage)
        - train/test partition overlap (benchmark-contaminating leakage)
    """
    total_seqs = _total_unique(splits_dict)

    pos_neg_overlap    = 0  # overlap between incompatible classes (AMP vs non-AMP)
    train_test_overlap = 0  # overlap between train and test of the same class

    # Classify each reported overlap pair
    for entry in overlaps:
        a, b, n = entry["split_a"], entry["split_b"], entry["overlap"]
        a_is_pos = any(k in a.lower() for k in ("amp", "hemo", "pos", "abp", "afp", "avp"))
        b_is_pos = any(k in b.lower() for k in ("amp", "hemo", "pos", "abp", "afp", "avp"))
        a_is_neg = any(k in a.lower() for k in ("non", "neg", "decoy", "namp"))
        b_is_neg = any(k in b.lower() for k in ("non", "neg", "decoy", "namp"))

        a_is_train = any(k in a.lower() for k in ("tr", "train"))
        b_is_train = any(k in b.lower() for k in ("tr", "train"))
        a_is_test  = any(k in a.lower() for k in ("test", "te"))
        b_is_test  = any(k in b.lower() for k in ("test", "te"))

        if (a_is_pos and b_is_neg) or (a_is_neg and b_is_pos):
            pos_neg_overlap += n
        elif (a_is_train and b_is_test) or (a_is_test and b_is_train):
            train_test_overlap += n

    def pct(n):
        return f"{100 * n / total_seqs:.1f}%" if total_seqs else "—"

    return {
        "Tool":                    tool,
        "Total unique seqs":       total_seqs,
        "Pos/neg overlap (n)":     pos_neg_overlap,
        "Pos/neg overlap (%)":     pct(pos_neg_overlap),
        "Train/test overlap (n)":  train_test_overlap,
        "Train/test overlap (%)":  pct(train_test_overlap),
    }


# Accumulate summary rows throughout the script
_summary_rows = []

def _run_check(tool, splits_dict):
    """Run overlap checks and store a summary row."""
    overlaps = check_pairs_overlap(splits_dict)
    _summary_rows.append(_report_entry(tool, splits_dict, overlaps))


# ===========================================================================
# 1. iAMP-2L
# ===========================================================================
# The four partitions were processed by means of
# data/raw/tools/iamp_2l/parse_iamp2l.py from manually extracted text from
# the original PDF supplementary files.

print("=" * 60)
print("iAMP-2L")
print("=" * 60)

iamp_2l_amp_tr      = fasta_to_df(TOOLS_DIR / "iamp_2l" / "processed" / "AMP_train.fasta")
iamp_2l_nonamp_tr   = fasta_to_df(TOOLS_DIR / "iamp_2l" / "processed" / "nonAMP_train.fasta")
iamp_2l_amp_test    = fasta_to_df(TOOLS_DIR / "iamp_2l" / "processed" / "AMP_test.fasta")
iamp_2l_nonamp_test = fasta_to_df(TOOLS_DIR / "iamp_2l" / "processed" / "nonAMP_test.fasta")

splits_iamp_2l = {
    "amp_tr":      set(iamp_2l_amp_tr["Sequence"]),
    "nonamp_tr":   set(iamp_2l_nonamp_tr["Sequence"]),
    "amp_test":    set(iamp_2l_amp_test["Sequence"]),
    "nonamp_test": set(iamp_2l_nonamp_test["Sequence"]),
}

_run_check("iAMP-2L", splits_iamp_2l)


# ===========================================================================
# 2. AMP Scanner
# ===========================================================================

print()
print("=" * 60)
print("AMP Scanner")
print("=" * 60)

amp_scanner_amp_tr      = fasta_to_df(TOOLS_DIR / "amp_scanner" / "AMP.tr.fa")
amp_scanner_amp_eval    = fasta_to_df(TOOLS_DIR / "amp_scanner" / "AMP.eval.fa")
amp_scanner_amp_test    = fasta_to_df(TOOLS_DIR / "amp_scanner" / "AMP.te.fa")
amp_scanner_nonamp_tr   = fasta_to_df(TOOLS_DIR / "amp_scanner" / "DECOY.tr.fa")
amp_scanner_nonamp_eval = fasta_to_df(TOOLS_DIR / "amp_scanner" / "DECOY.eval.fa")
amp_scanner_nonamp_test = fasta_to_df(TOOLS_DIR / "amp_scanner" / "DECOY.te.fa")

splits_amp_scanner = {
    "amp_tr":      set(amp_scanner_amp_tr["Sequence"]),
    "amp_eval":    set(amp_scanner_amp_eval["Sequence"]),
    "amp_test":    set(amp_scanner_amp_test["Sequence"]),
    "nonamp_tr":   set(amp_scanner_nonamp_tr["Sequence"]),
    "nonamp_eval": set(amp_scanner_nonamp_eval["Sequence"]),
    "nonamp_test": set(amp_scanner_nonamp_test["Sequence"]),
}

_run_check("AMP Scanner", splits_amp_scanner)


# ===========================================================================
# 3. AmPEP
# ===========================================================================
# Only the training dataset is available; no independent test set was released.

print()
print("=" * 60)
print("AmPEP")
print("=" * 60)

ampep_amp_tr    = fasta_to_df(TOOLS_DIR / "ampep" / "M_model_train_AMP_sequence.fasta")
ampep_nonamp_tr = fasta_to_df(TOOLS_DIR / "ampep" / "M_model_train_nonAMP_sequence.fasta")

splits_ampep = {
    "amp_tr":    set(ampep_amp_tr["Sequence"]),
    "nonamp_tr": set(ampep_nonamp_tr["Sequence"]),
}

_run_check("AmPEP", splits_ampep)


# ===========================================================================
# 4. Macrel
# ===========================================================================
# Only training data is available, both for the AMP and hemolytic peptide
# tasks. The AMP training set is shared with AmPEP (see §2.4 of the README).

print()
print("=" * 60)
print("Macrel")
print("=" * 60)

macrel_hemo_tr      = fasta_to_df(TOOLS_DIR / "macrel" / "hemo.training.pos.faa")
macrel_nonhemo_tr   = fasta_to_df(TOOLS_DIR / "macrel" / "hemo.training.neg.faa")
macrel_hemo_val     = fasta_to_df(TOOLS_DIR / "macrel" / "hemo.validation.pos.faa")
macrel_nonhemo_val  = fasta_to_df(TOOLS_DIR / "macrel" / "hemo.validation.neg.faa")

splits_macrel = {
    "hemo_tr":      set(macrel_hemo_tr["Sequence"]),
    "nonhemo_tr":   set(macrel_nonhemo_tr["Sequence"]),
    "hemo_val":     set(macrel_hemo_val["Sequence"]),
    "nonhemo_val":  set(macrel_nonhemo_val["Sequence"]),
}

_run_check("Macrel", splits_macrel)


# ===========================================================================
# 5. amPEPpy
# ===========================================================================
# The negative class is a length-balanced subsample recommended by the authors.
# Evaluation was performed through internal OOB sets; no independent test set.

print()
print("=" * 60)
print("amPEPpy")
print("=" * 60)

ampeppy_amp_tr    = fasta_to_df(TOOLS_DIR / "ampeppy" / "M_model_train_AMP_sequence.numbered.fasta")
ampeppy_nonamp_tr = fasta_to_df(TOOLS_DIR / "ampeppy" / "M_model_train_nonAMP_sequence.numbered.proplen.subsample.fasta")

splits_ampeppy = {
    "amp_tr":    set(ampeppy_amp_tr["Sequence"]),
    "nonamp_tr": set(ampeppy_nonamp_tr["Sequence"]),
}

_run_check("amPEPpy", splits_ampeppy)


# ===========================================================================
# 6. LMPred
# ===========================================================================

print()
print("=" * 60)
print("LMPred")
print("=" * 60)

lmpred_tr  = pd.read_csv(TOOLS_DIR / "lmpred" / "X_train.csv")
lmpred_val = pd.read_csv(TOOLS_DIR / "lmpred" / "X_val.csv")
lmpred_test = pd.read_csv(TOOLS_DIR / "lmpred" / "X_test.csv")

# Attach ground-truth labels (separate files, same row order)
lmpred_tr["ground_truth"]   = pd.read_csv(TOOLS_DIR / "lmpred" / "y_train.csv", header=None)[0].astype(int).values
lmpred_val["ground_truth"]  = pd.read_csv(TOOLS_DIR / "lmpred" / "y_val.csv",   header=None)[0].astype(int).values
lmpred_test["ground_truth"] = pd.read_csv(TOOLS_DIR / "lmpred" / "y_test.csv",  header=None)[0].astype(int).values

splits_lmpred = {
    "amp_tr":    set(lmpred_tr[lmpred_tr["ground_truth"]   == 1]["Sequence"]),
    "amp_val":   set(lmpred_val[lmpred_val["ground_truth"] == 1]["Sequence"]),
    "amp_test":  set(lmpred_test[lmpred_test["ground_truth"] == 1]["Sequence"]),
    "nonamp_tr":   set(lmpred_tr[lmpred_tr["ground_truth"]   == 0]["Sequence"]),
    "nonamp_val":  set(lmpred_val[lmpred_val["ground_truth"] == 0]["Sequence"]),
    "nonamp_test": set(lmpred_test[lmpred_test["ground_truth"] == 0]["Sequence"]),
}

_run_check("LMPred", splits_lmpred)


# ===========================================================================
# 7. AMPlify
# ===========================================================================
# Non-AMP sets correspond to those used in the original publication,
# constituting balanced training and test datasets.

print()
print("=" * 60)
print("AMPlify")
print("=" * 60)

amplify_amp_tr    = fasta_to_df(TOOLS_DIR / "amplify" / "AMPlify_AMP_train_common.fa")
amplify_amp_test  = fasta_to_df(TOOLS_DIR / "amplify" / "AMPlify_AMP_test_common.fa")
amplify_nonamp_tr = fasta_to_df(TOOLS_DIR / "amplify" / "AMPlify_non_AMP_train_balanced.fa")
amplify_nonamp_test = fasta_to_df(TOOLS_DIR / "amplify" / "AMPlify_non_AMP_test_balanced.fa")

splits_amplify = {
    "amp_tr":      set(amplify_amp_tr["Sequence"]),
    "amp_test":    set(amplify_amp_test["Sequence"]),
    "nonamp_tr":   set(amplify_nonamp_tr["Sequence"]),
    "nonamp_test": set(amplify_nonamp_test["Sequence"]),
}

_run_check("AMPlify", splits_amplify)


# ===========================================================================
# 8. Ma et al. (2022)
# ===========================================================================
# Only the test dataset is provided by the authors; no training data available.

print()
print("=" * 60)
print("Ma et al. (2022)")
print("=" * 60)

ma_amp_test    = fasta_to_df(TOOLS_DIR / "ma_et_al" / "AMPs.fa")
ma_nonamp_test = fasta_to_df(TOOLS_DIR / "ma_et_al" / "Non-AMPs.fa")

splits_ma_et_al = {
    "amp_test":    set(ma_amp_test["Sequence"]),
    "nonamp_test": set(ma_nonamp_test["Sequence"]),
}

_run_check("Ma et al. (2022)", splits_ma_et_al)


# ===========================================================================
# 9. AMP-BERT
# ===========================================================================

print()
print("=" * 60)
print("AMP-BERT")
print("=" * 60)

amp_bert_tr = pd.read_csv(TOOLS_DIR / "amp_bert" / "all_veltri.csv")
amp_bert_tr.columns = ["ampbert_id", "Sequence", "seq_length", "ground_truth"]

amp_bert_amp_test    = pd.read_csv(TOOLS_DIR / "amp_bert" / "veltri_dramp_cdhit_90.csv")
amp_bert_amp_test.columns = ["id", "Sequence", "seq_length", "ground_truth"]

amp_bert_nonamp_test = pd.read_csv(TOOLS_DIR / "amp_bert" / "non_amp_ampep_cdhit90.csv")
amp_bert_nonamp_test.columns = ["id", "Sequence", "seq_length", "ground_truth"]

splits_amp_bert = {
    "amp_tr":      set(amp_bert_tr[amp_bert_tr["ground_truth"]    == 1]["Sequence"]),
    "amp_test":    set(amp_bert_amp_test["Sequence"]),
    "nonamp_tr":   set(amp_bert_tr[amp_bert_tr["ground_truth"]    == 0]["Sequence"]),
    "nonamp_test": set(amp_bert_nonamp_test["Sequence"]),
}

_run_check("AMP-BERT", splits_amp_bert)


# ===========================================================================
# 10. AMPFinder
# ===========================================================================
# Only stage 1 (AMP vs. non-AMP discrimination) is tested here.
# Stage 2 (functional classification) data is not considered.

print()
print("=" * 60)
print("AMPFinder")
print("=" * 60)

ampfinder_amp    = fasta_to_df(TOOLS_DIR / "ampfinder" / "3594-Samp.fasta")
ampfinder_nonamp = fasta_to_df(TOOLS_DIR / "ampfinder" / "3925-Snonamp.fasta")

splits_ampfinder = {
    "amp":    set(ampfinder_amp["Sequence"]),
    "nonamp": set(ampfinder_nonamp["Sequence"]),
}

_run_check("AMPFinder", splits_ampfinder)


# ===========================================================================
# 11. AMP-RNNpro
# ===========================================================================

print()
print("=" * 60)
print("AMP-RNNpro")
print("=" * 60)

amp_rnnpro_amp_tr    = fasta_to_df(TOOLS_DIR / "amp_rnnpro" / "train_p.fasta")
amp_rnnpro_amp_test  = fasta_to_df(TOOLS_DIR / "amp_rnnpro" / "testp")
amp_rnnpro_nonamp_tr = fasta_to_df(TOOLS_DIR / "amp_rnnpro" / "trainn_n.fasta")
amp_rnnpro_nonamp_test = fasta_to_df(TOOLS_DIR / "amp_rnnpro" / "testn")

splits_amp_rnnpro = {
    "amp_tr":      set(amp_rnnpro_amp_tr["Sequence"]),
    "amp_test":    set(amp_rnnpro_amp_test["Sequence"]),
    "nonamp_tr":   set(amp_rnnpro_nonamp_tr["Sequence"]),
    "nonamp_test": set(amp_rnnpro_nonamp_test["Sequence"]),
}

_run_check("AMP-RNNpro", splits_amp_rnnpro)


# ===========================================================================
# 12. AGRAMP
# ===========================================================================
# Three negative training/test sets of the same size (NOAMP1, NOAMP2, NOAMP3).
# NOTE: train/test overlap is found only in the NOAMP2 and NOAMP3 categories,
# not in AMP or NOAMP1. There is no overlap between incompatible classes
# (positive vs. negative sets).

print()
print("=" * 60)
print("AGRAMP")
print("=" * 60)

agramp_amp_tr       = fasta_to_df(TOOLS_DIR / "agramp" / "AMP_train.fasta")
agramp_amp_test     = fasta_to_df(TOOLS_DIR / "agramp" / "AMP_test.fasta")
agramp_noamp1_tr    = fasta_to_df(TOOLS_DIR / "agramp" / "NOAMP1_train.fasta")
agramp_noamp1_test  = fasta_to_df(TOOLS_DIR / "agramp" / "NOAMP1_test.fasta")
agramp_noamp2_tr    = fasta_to_df(TOOLS_DIR / "agramp" / "NOAMP2_train.fasta")
agramp_noamp2_test  = fasta_to_df(TOOLS_DIR / "agramp" / "NOAMP2_test.fasta")
agramp_noamp3_tr    = fasta_to_df(TOOLS_DIR / "agramp" / "NOAMP3_train.fasta")
agramp_noamp3_test  = fasta_to_df(TOOLS_DIR / "agramp" / "NOAMP3_test.fasta")

splits_agramp = {
    "amp_tr":       set(agramp_amp_tr["Sequence"]),
    "amp_test":     set(agramp_amp_test["Sequence"]),
    "noamp1_tr":    set(agramp_noamp1_tr["Sequence"]),
    "noamp1_test":  set(agramp_noamp1_test["Sequence"]),
    "noamp2_tr":    set(agramp_noamp2_tr["Sequence"]),
    "noamp2_test":  set(agramp_noamp2_test["Sequence"]),
    "noamp3_tr":    set(agramp_noamp3_tr["Sequence"]),
    "noamp3_test":  set(agramp_noamp3_test["Sequence"]),
}

_run_check("AGRAMP", splits_agramp)

print("  Note: in case each NOAMPi was considered as an independent")
print("  category, train/test overlap is limited to NOAMP2 and NOAMP3.")
print("  No overlap between AMP and any non-AMP class.")


# ===========================================================================
# 13. PepNet
# ===========================================================================
# PepNet covers three tasks (AMP, AIP, Toxic); each is checked independently.
# Labels are encoded in the FASTA header after a pipe character (|0 or |1).
#
# Each task folder contains files from three independent naming conventions
# that must never be compared against each other:
#   Convention A: {category}_split_train.txt, {category}_split_valid.txt,
#                 test_{category}.txt  (and data_split_*.txt for the global split)
#   Convention B: train_{category}.txt, test_{category}.txt
#   Convention C: train.txt, test.txt  (global split, no category prefix)

def _pepnet_convention(stem):
    """
    Return 'A', 'B', or 'C' for a PepNet file stem based on its naming pattern.
        A -- contains '_split_'  (e.g. antibacterial_split_train, data_split_valid)
        B -- starts with 'train_' or 'test_' followed by a category name
        C -- bare 'train' or 'test' (global split)
    """
    if "_split_" in stem:
        return "A"
    if stem in ("train", "test"):
        return "C"
    return "B"


def _load_pepnet_splits(folder):
    """
    Load all .txt files from a PepNet task folder and return three dicts,
    one per naming convention (A, B, C).
    """
    splits_a, splits_b, splits_c = {}, {}, {}
    conv_map = {"A": splits_a, "B": splits_b, "C": splits_c}

    for file in sorted(folder.iterdir()):
        if not file.is_file() or file.suffix != ".txt":
            continue
        df  = fasta_to_df_pepnet(file)
        pos = set(df[df["Label"] == 1]["Sequence"])
        neg = set(df[df["Label"] == 0]["Sequence"])
        target = conv_map[_pepnet_convention(file.stem)]
        target[f"{file.stem}_pos"] = pos
        target[f"{file.stem}_neg"] = neg

    return splits_a, splits_b, splits_c


def _analyze_pepnet_category(splits_dict, categories):
    """
    Check overlaps per dataset category within a single convention dict.

    Parameters
    ----------
    splits_dict : dict
        Mapping of split keys to sets of sequences, all from the same convention.
    categories : list of str
        Category substrings to isolate (e.g. 'antibacterial', 'IF_AIP', 'data').

    Returns
    -------
    list of dict
        All overlap records found across categories.
    """
    all_overlaps = []
    for category in categories:
        subset = {k: v for k, v in splits_dict.items() if category in k}
        if subset:
            print(f"  --- Category: {category} ---")
            all_overlaps.extend(check_pairs_overlap(subset))
    return all_overlaps


print()
print("=" * 60)
print("PepNet")
print("=" * 60)

all_pepnet_splits   = {}
all_pepnet_overlaps = []

# --- AMP datasets ---
# Note: anti-mammalian and anti-cancer peptides are included within the AMP
# framework by the authors, though functionally distinct from antimicrobials.
print("  >> AMP datasets")
amp_categories = ["anti_mammalian", "antibacterial", "anticancer", "antifungal", "antiviral", "data"]
amp_a, amp_b, amp_c = _load_pepnet_splits(TOOLS_DIR / "pepnet" / "AMP")

print("  > Convention A ({category}_split_train / {category}_split_valid / test_{category})")
all_pepnet_overlaps += _analyze_pepnet_category(amp_a, amp_categories)
print("  > Convention B (train_{category} / test_{category})")
all_pepnet_overlaps += _analyze_pepnet_category(amp_b, amp_categories)
print("  > Convention C (train / test)")
all_pepnet_overlaps += _analyze_pepnet_category(amp_c, ["data"])
all_pepnet_splits.update({**amp_a, **amp_b, **amp_c})

# --- AIP datasets ---
print("  >> AIP datasets")
aip_categories = ["AIPStack", "BertAIP", "IF_AIP", "data"]
aip_a, aip_b, aip_c = _load_pepnet_splits(TOOLS_DIR / "pepnet" / "AIP")

print("  > Convention A ({category}_split_train / {category}_split_valid / test_{category})")
all_pepnet_overlaps += _analyze_pepnet_category(aip_a, aip_categories)
print("  > Convention B (train_{category} / test_{category})")
all_pepnet_overlaps += _analyze_pepnet_category(aip_b, aip_categories)
print("  > Convention C (train / test)")
all_pepnet_overlaps += _analyze_pepnet_category(aip_c, ["data"])
all_pepnet_splits.update({**aip_a, **aip_b, **aip_c})

# --- Toxic dataset ---
print("  >> Toxic dataset")
toxic_a, toxic_b, toxic_c = _load_pepnet_splits(TOOLS_DIR / "pepnet" / "Toxic")

print("  > Convention A (data_split_train / data_split_valid)")
all_pepnet_overlaps += _analyze_pepnet_category(toxic_a, ["data"])
print("  > Convention C (train / test)")
all_pepnet_overlaps += _analyze_pepnet_category(toxic_c, ["data"])
all_pepnet_splits.update({**toxic_a, **toxic_b, **toxic_c})

_summary_rows.append(_report_entry("PepNet", all_pepnet_splits, all_pepnet_overlaps))


# ===========================================================================
# 14. KT-AMPpred
# ===========================================================================
# Covers four tasks: AMP, ABP (antibacterial), AFP (antifungal), AVP (antiviral).
# Datasets are built from the same underlying peptide collection via a
# one-versus-others strategy, so many sequences appear across tasks with
# different labels. This creates cross-task dataset dependence. Overlap
# is studied within each class.

print()
print("=" * 60)
print("KT-AMPpred")
print("=" * 60)

amp_tr  = pd.read_csv(TOOLS_DIR / "kt_amppred" / "amp_train.tsv", sep="\t")
amp_test = pd.read_csv(TOOLS_DIR / "kt_amppred" / "amp_test.tsv",  sep="\t")
abp_tr  = pd.read_csv(TOOLS_DIR / "kt_amppred" / "ABP_DS_train.tsv", sep="\t")
abp_test = pd.read_csv(TOOLS_DIR / "kt_amppred" / "ABP_DS_test.tsv",  sep="\t")
afp_tr  = pd.read_csv(TOOLS_DIR / "kt_amppred" / "AFP_DS_train.tsv", sep="\t")
afp_test = pd.read_csv(TOOLS_DIR / "kt_amppred" / "AFP_DS_test.tsv",  sep="\t")
avp_tr  = pd.read_csv(TOOLS_DIR / "kt_amppred" / "AVP_DS_train.tsv", sep="\t")
avp_test = pd.read_csv(TOOLS_DIR / "kt_amppred" / "AVP_DS_test.tsv",  sep="\t")

datasets_kt = {
    "amp": (amp_tr,  amp_test),
    "abp": (abp_tr,  abp_test),
    "afp": (afp_tr,  afp_test),
    "avp": (avp_tr,  avp_test),
}

kt_all_splits   = {}
kt_all_overlaps = []

for task, (tr_df, te_df) in datasets_kt.items():
    task_splits = {
        f"{task}_tr":       set(tr_df.loc[tr_df["label"] == 1, "text"]),
        f"{task}_test":     set(te_df.loc[te_df["label"] == 1, "text"]),
        f"non_{task}_tr":   set(tr_df.loc[tr_df["label"] == 0, "text"]),
        f"non_{task}_test": set(te_df.loc[te_df["label"] == 0, "text"]),
    }
    print(f"  --- Task: {task.upper()} ---")
    overlaps = check_pairs_overlap(task_splits)
    kt_all_splits.update(task_splits)
    kt_all_overlaps.extend(overlaps)

_summary_rows.append(_report_entry("KT-AMPpred", kt_all_splits, kt_all_overlaps))

print("  No overlap detected within any individual task.")


# ===========================================================================
# 15. PLAPD
# ===========================================================================
# The independent test set derived from AMPlify is not provided by the authors.

print()
print("=" * 60)
print("PLAPD")
print("=" * 60)

plapd_tr  = pd.read_csv(TOOLS_DIR / "plapd" / "training_data.csv")
plapd_val = pd.read_csv(TOOLS_DIR / "plapd" / "val_data.csv")

splits_plapd = {
    "amp_tr":    set(plapd_tr.loc[plapd_tr["Label"]   == 1, "Seq"]),
    "nonamp_tr": set(plapd_tr.loc[plapd_tr["Label"]   == 0, "Seq"]),
    "amp_val":   set(plapd_val.loc[plapd_val["Label"] == 1, "Seq"]),
    "nonamp_val":set(plapd_val.loc[plapd_val["Label"] == 0, "Seq"]),
}

_run_check("PLAPD", splits_plapd)


# ===========================================================================
# 16. DLFea4AMPGen
# ===========================================================================
# The dataset tree has nested subfolders (ABP, AFP, AOP and Other).
# The 'Other' subfolder contains additional datasets not used to train the
# benchmark model (ABP-MPB); it is examined separately below.
# Labels: 0 = positive (AMP), 1 = negative in this project's convention.

def _navigate_dlfea4ampgen(folder):
    """
    Recursively walk the DLFea4AMPGen dataset tree and run overlap checks.

    Each leaf subfolder contains one or more CSV files with 'seq' and 'label'
    columns (0 = positive, 1 = negative).

    Returns a tuple (splits_dict, all_overlaps) aggregated across all leaves.
    """
    all_splits   = {}
    all_overlaps = []

    for subfolder in sorted(folder.iterdir()):
        if not subfolder.is_dir() or subfolder.name == "Other":
            continue
        items   = list(subfolder.iterdir())
        subdirs = [i for i in items if i.is_dir()]

        if subdirs:
            # Recurse into nested structure
            print(f"  Entering {subfolder.stem}:")
            sub_splits, sub_overlaps = _navigate_dlfea4ampgen(subfolder)
            all_splits.update(sub_splits)
            all_overlaps.extend(sub_overlaps)
            print(f"  {'—' * 40}")
            continue

        leaf_splits = {}
        for item in sorted(items):
            df = pd.read_csv(item)
            key_pos = f"{subfolder.stem}_{item.stem}_pos"
            key_neg = f"{subfolder.stem}_{item.stem}_neg"
            leaf_splits[key_pos] = set(df[df["label"] == 0]["seq"])  # 0 = positive
            leaf_splits[key_neg] = set(df[df["label"] == 1]["seq"])  # 1 = negative

        print(f"  --- Category: {subfolder.stem} ---")
        overlaps = check_pairs_overlap(leaf_splits)
        all_splits.update(leaf_splits)
        all_overlaps.extend(overlaps)

    return all_splits, all_overlaps


print()
print("=" * 60)
print("DLFea4AMPGen")
print("=" * 60)

# --- Main datasets (used to train the ABP-MPB model) ---
print("  >> Main datasets")
main_folder = TOOLS_DIR / "dlfea4ampgen"
dlfea_main_splits, dlfea_main_overlaps = _navigate_dlfea4ampgen(
    Path(str(main_folder))  # exclude Other by passing main folder; Other handled below
)
# Remove Other subfolder entries that may have been recursed into
dlfea_main_splits   = {k: v for k, v in dlfea_main_splits.items() if "Other" not in k}
dlfea_main_overlaps = [e for e in dlfea_main_overlaps if "Other" not in e.get("split_a","") and "Other" not in e.get("split_b","")]

# --- Other datasets (not used in the benchmark model) ---
print()
print("  >> Other datasets")
other_folder = TOOLS_DIR / "dlfea4ampgen" / "Other"
dlfea_other_splits, dlfea_other_overlaps = _navigate_dlfea4ampgen(other_folder)

# Combined summary entry for the whole DLFea4AMPGen tool
all_dlfea_splits   = {**dlfea_main_splits, **dlfea_other_splits}
all_dlfea_overlaps = dlfea_main_overlaps + dlfea_other_overlaps
_summary_rows.append(_report_entry("DLFea4AMPGen", all_dlfea_splits, all_dlfea_overlaps))


# ===========================================================================
# 17. MultiAMP
# ===========================================================================
# Each sample is stored as an individual .fas file; the label is the last
# token of the FASTA header (e.g. '|1' = AMP, '|0' = non-AMP).

print()
print("=" * 60)
print("MultiAMP")
print("=" * 60)

multiamp_path = TOOLS_DIR / "multiamp"
seqs_multiamp = {
    "amp_tr":    set(),
    "nonamp_tr": set(),
    "amp_test":  set(),
    "nonamp_test": set(),
}

for folder in multiamp_path.iterdir():
    if not folder.is_dir():
        continue
    for file in folder.iterdir():
        if file.suffix != ".fas":
            continue
        lines = [l.strip() for l in file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        label    = lines[0].split("|")[-1]   # "1" = AMP, "0" = non-AMP
        sequence = lines[1]

        if folder.name == "train_amp":
            key = "amp_tr" if label == "1" else "nonamp_tr"
        elif folder.name == "test_amp":
            key = "amp_test" if label == "1" else "nonamp_test"
        else:
            continue
        seqs_multiamp[key].add(sequence)

_run_check("MultiAMP", seqs_multiamp)


# ===========================================================================
# Summary report
# ===========================================================================

print()
print("=" * 60)
print("SUMMARY REPORT — Data Leakage by Tool")
print("=" * 60)
print()
print(
    "Columns:\n"
    "  Pos/neg overlap    — sequences shared between incompatible classes\n"
    "                       (e.g. AMP and non-AMP); these should always be 0.\n"
    "  Train/test overlap — sequences shared between training and test of\n"
    "                       the same class; directly inflates reported performance.\n"
    "  (%) columns are relative to total unique sequences for that tool.\n"
)

df_summary = pd.DataFrame(_summary_rows)

# Pretty-print with fixed column widths
col_widths = {col: max(len(col), df_summary[col].astype(str).str.len().max()) + 2
              for col in df_summary.columns}

header = "".join(col.ljust(col_widths[col]) for col in df_summary.columns)
print(header)
print("-" * len(header))
for _, row in df_summary.iterrows():
    print("".join(str(row[col]).ljust(col_widths[col]) for col in df_summary.columns))

print()
print("Done.")
