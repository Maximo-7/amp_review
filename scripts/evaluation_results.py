"""
evaluation_results.py
Computes evaluation metrics and generates analysis plots for AMP prediction tools.

Analyses:
    0 - Balanced accuracy bar chart (ordered highest to lowest)
    1 - Pairwise agreement matrix (lower triangular heatmap)
    2 - Prediction correctness heatmap with clustering (peptides × tools)
    3 - Correlation between peptide difficulty and physicochemical properties
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef, roc_auc_score, roc_curve
)

# ─── Configuration ────────────────────────────────────────────────────────────

DATASET_PATH               = "/home/amaximo/amp_project_alvaro/data/processed/complete_dataset_final.csv"
OUTPUT_METRICS_PATH        = "/home/amaximo/amp_project_alvaro/results/evaluation/evaluation_metrics.csv"
OUTPUT_ROC_PATH            = "/home/amaximo/amp_project_alvaro/results/evaluation/roc_curves.png"
OUTPUT_BEST_PREDICTED_PATH = "/home/amaximo/amp_project_alvaro/results/evaluation/best_predicted_sequences.csv"
OUTPUT_ANALYSIS_0_PATH     = "/home/amaximo/amp_project_alvaro/results/evaluation/analysis_0_balanced_accuracy.png"
OUTPUT_ANALYSIS_1_PATH     = "/home/amaximo/amp_project_alvaro/results/evaluation/analysis_1_agreement_matrix.png"
OUTPUT_ANALYSIS_2_PATH     = "/home/amaximo/amp_project_alvaro/results/evaluation/analysis_2_correctness_heatmap.png"
OUTPUT_ANALYSIS_3_PATH     = "/home/amaximo/amp_project_alvaro/results/evaluation/analysis_3_property_correlations.png"

# Tools and their (label_col, prob_col) — None prob means binary only
TOOLS = {
    "AMP Scanner":      ("AMP_Scanner_pred_label",  "AMP_Scanner_pred_prob"),
    "Macrel":           ("Macrel_pred_label",        "Macrel_pred_prob"),
    "amPEPpy":          ("amPEPpy_pred_label",       "amPEPpy_pred_prob"),
    "LMPred":           ("LMPred_pred_label",        "LMPred_pred_prob"),
    "AMPlify":          ("AMPlify_pred_label",       "AMPlify_pred_prob"),
    "Ma et al. (2022)": ("Ma_et_al_pred_label",      None),
    "CAMPR4 ANN":       ("CAMPR4_ann_pred_label",    "CAMPR4_ann_pred_prob"),
    "CAMPR4 RF":        ("CAMPR4_rf_pred_label",     "CAMPR4_rf_pred_prob"),
    "CAMPR4 SVM":       ("CAMPR4_svm_pred_label",    "CAMPR4_svm_pred_prob"),
    "AMP-BERT":         ("AMP-BERT_pred_label",      "AMP-BERT_pred_prob"),
    "AMPFinder":        ("AMPFinder_pred_label",     "AMPFinder_pred_prob"),
    "PyAMPA":           ("PyAMPA_pred_label",        "PyAMPA_pred_prob"),
    "AGRAMP":           ("AGRAMP_pred_label",        "AGRAMP_pred_prob"),
    "PepNet":           ("PepNet_pred_label",        "PepNet_pred_prob"),
    "KT-AMPpred":       ("KT-AMPpred_pred_label",    "KT-AMPpred_pred_prob"),
    "PLAPD":            ("PLAPD_pred_label",         "PLAPD_pred_prob"),
    "DLFea4AMPGen":     ("DLFea4AMPGen_pred_label",  "DLFea4AMPGen_pred_prob"),
    "MultiAMP":         ("MultiAMP_pred_label",      "MultiAMP_pred_prob"),
}

MA_ET_AL_PROB_COLS = [
    "Ma_et_al_att_pred_prob",
    "Ma_et_al_bert_pred_prob",
    "Ma_et_al_lstm_pred_prob",
]

PHYSICOCHEMICAL_PROPS = [
    "Sequence_length",
    "Molecular Weight",
    "Net Charge (pH 7)",
    "Aromaticity",
    "Instability Index",
    "Isoelectric Point",
    "GRAVY",
    "Boman Index",
]

# ─── Load data ────────────────────────────────────────────────────────────────

df      = pd.read_csv(DATASET_PATH, low_memory=False)
eval_df = df[df["For_evaluation"]].copy()
y_true  = eval_df["ABP_from_databases"].astype(int).values

# ─── Metrics computation ──────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity       = tp / (tp + fn) * 100
    specificity       = tn / (tn + fp) * 100
    accuracy          = accuracy_score(y_true, y_pred) * 100
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred) * 100
    f1                = f1_score(y_true, y_pred) * 100
    mcc               = matthews_corrcoef(y_true, y_pred) * 100
    auroc             = roc_auc_score(y_true, y_prob) * 100 if y_prob is not None else np.nan
    return sensitivity, specificity, accuracy, balanced_accuracy, f1, auroc, mcc


rows     = []
roc_data = {}

for tool, (label_col, prob_col) in TOOLS.items():
    cols_needed = [label_col] + ([prob_col] if prob_col else MA_ET_AL_PROB_COLS)
    mask   = eval_df[cols_needed].notna().all(axis=1)
    sub    = eval_df[mask]
    yt     = y_true[mask.values]
    y_pred = sub[label_col].astype(int).values

    if tool == "Ma et al. (2022)":
        y_prob = sub[MA_ET_AL_PROB_COLS].mean(axis=1).values
    elif prob_col is not None:
        y_prob = sub[prob_col].values
    else:
        y_prob = None

    sens, spec, acc, bacc, f1, auroc, mcc = compute_metrics(yt, y_pred, y_prob)
    rows.append({
        "Model":             tool,
        "Sensitivity":       round(sens, 2),
        "Specificity":       round(spec, 2),
        "Accuracy":          round(acc, 2),
        "Balanced Accuracy": round(bacc, 2),
        "F1":                round(f1, 2),
        "AUROC":             round(auroc, 2) if not np.isnan(auroc) else np.nan,
        "MCC":               round(mcc, 2),
    })

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(yt, y_prob)
        roc_data[tool] = (fpr, tpr, auroc)

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False)
print(metrics_df.to_string(index=False))

# ─── ROC curves ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))
colors  = cm.tab20(np.linspace(0, 1, len(roc_data)))

for (tool, (fpr, tpr, auroc)), color in zip(roc_data.items(), colors):
    ax.plot(fpr, tpr, lw=1.5, color=color, label=f"{tool} (AUROC = {auroc:.1f}%)")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC curves", fontsize=14)
ax.legend(loc="lower right", fontsize=8)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(OUTPUT_ROC_PATH, dpi=150)
plt.close()
print(f"ROC curves saved to {OUTPUT_ROC_PATH}")

# ─── Best-predicted sequences (extended with per-tool correctness) ────────────

label_cols     = [label_col for label_col, _ in TOOLS.values()]
available_cols = [c for c in label_cols if c in eval_df.columns]

pred_matrix    = eval_df[available_cols].apply(pd.to_numeric, errors="coerce")
correct_matrix = pred_matrix.eq(eval_df["ABP_from_databases"].astype(int).values, axis=0)

n_tools_with_pred = pred_matrix.notna().sum(axis=1)
n_correct         = correct_matrix.sum(axis=1)
n_wrong           = (pred_matrix.notna() & ~correct_matrix).sum(axis=1)

eval_df = eval_df.copy()
eval_df["n_tools_predicted"] = n_tools_with_pred
eval_df["n_tools_correct"]   = n_correct
eval_df["n_tools_wrong"]     = n_wrong

# Add per-tool correctness columns (1 = correct, 0 = wrong, NaN = no prediction)
for tool_name, (label_col, _) in TOOLS.items():
    if label_col in eval_df.columns:
        col_name = f"{tool_name}_correct"
        pred_vals = pd.to_numeric(eval_df[label_col], errors="coerce")
        eval_df[col_name] = np.where(
            pred_vals.isna(),
            np.nan,
            (pred_vals == eval_df["ABP_from_databases"].astype(int)).astype(float)
        )

correctness_cols = [f"{t}_correct" for t in TOOLS if f"{t}_correct" in eval_df.columns]

best_predicted = (
    eval_df[eval_df["n_tools_predicted"] > 0]
    .sort_values(["n_tools_wrong", "n_tools_correct"], ascending=[True, False])
    [["ID", "Sequence", "ABP_from_databases",
      "n_tools_predicted", "n_tools_correct", "n_tools_wrong"]
     + correctness_cols]
)

best_predicted.to_csv(OUTPUT_BEST_PREDICTED_PATH, index=False)
print(f"\nTop 10 best-predicted sequences:")
print(best_predicted.head(10).to_string(index=False))

# ─── Analysis 0: Balanced accuracy bar chart ─────────────────────────────────

metrics_sorted = metrics_df.sort_values("Balanced Accuracy", ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(
    metrics_sorted["Model"],
    metrics_sorted["Balanced Accuracy"],
    color=cm.tab20(np.linspace(0, 1, len(metrics_sorted)))
)
ax.set_xlabel("Tool", fontsize=12)
ax.set_ylabel("Balanced Accuracy (%)", fontsize=12)
ax.set_title("Balanced Accuracy by Tool (ordered highest to lowest)", fontsize=14)
ax.set_ylim([0, 100])
ax.axhline(50, color="gray", linestyle="--", lw=1, label="Random classifier (50%)")
ax.legend(fontsize=9)
plt.xticks(rotation=45, ha="right", fontsize=9)

for bar, val in zip(bars, metrics_sorted["Balanced Accuracy"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_ANALYSIS_0_PATH, dpi=150)
plt.close()
print(f"Analysis 0 saved to {OUTPUT_ANALYSIS_0_PATH}")

# ─── Analysis 1: Pairwise agreement matrix ───────────────────────────────────
# % of identical predictions between each pair of tools
# Only sequences where both tools have predictions are considered
# Tools ordered by balanced accuracy (highest to lowest, as in analysis 0)

tool_order  = metrics_sorted["Model"].tolist()
n_tools     = len(tool_order)
agreement   = pd.DataFrame(np.nan, index=tool_order, columns=tool_order)

for i, tool_i in enumerate(tool_order):
    label_i = TOOLS[tool_i][0]
    if label_i not in eval_df.columns:
        continue
    for j, tool_j in enumerate(tool_order):
        if j > i:
            break
        label_j = TOOLS[tool_j][0]
        if label_j not in eval_df.columns:
            continue
        mask    = eval_df[label_i].notna() & eval_df[label_j].notna()
        if mask.sum() == 0:
            continue
        preds_i = pd.to_numeric(eval_df.loc[mask, label_i])
        preds_j = pd.to_numeric(eval_df.loc[mask, label_j])
        pct     = (preds_i == preds_j).mean() * 100
        agreement.loc[tool_i, tool_j] = pct

# Mask upper triangle
mask_upper = np.triu(np.ones_like(agreement, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(
    agreement.astype(float),
    mask=mask_upper,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    vmin=50,
    vmax=100,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Agreement (%)"}
)
ax.set_title("Pairwise prediction agreement between tools (%)\n"
             "(tools ordered by balanced accuracy, highest to lowest)", fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_ANALYSIS_1_PATH, dpi=150)
plt.close()
print(f"Analysis 1 saved to {OUTPUT_ANALYSIS_1_PATH}")

# ─── Analysis 2: Correctness heatmap with clustering ─────────────────────────
# Binary matrix: rows = peptides, columns = tools
# 1 = correct prediction, 0 = wrong, NaN = no prediction

correctness_data = {}
for tool_name, (label_col, _) in TOOLS.items():
    if label_col not in eval_df.columns:
        continue
    pred_vals = pd.to_numeric(eval_df[label_col], errors="coerce")
    correctness_data[tool_name] = np.where(
        pred_vals.isna(),
        np.nan,
        (pred_vals == eval_df["ABP_from_databases"].astype(int)).astype(float)
    )

correctness_df = pd.DataFrame(correctness_data, index=eval_df.index)

# Drop peptides with no predictions at all
correctness_df = correctness_df.dropna(how="all")

# For clustering: fill NaN with 0.5 (neutral) — NaN means no prediction
correctness_filled = correctness_df.fillna(0.5)

g = sns.clustermap(
    correctness_filled,
    cmap=sns.color_palette(["#d73027", "#fee08b", "#1a9850"], as_cmap=True),
    figsize=(16, 14),
    row_cluster=True,
    col_cluster=True,
    xticklabels=True,
    yticklabels=False,
    linewidths=0,
    cbar_pos=None,           # disable automatic colorbar — we add a discrete legend
    dendrogram_ratio=(0.1, 0.15),
)
g.ax_heatmap.set_xlabel("Tool", fontsize=16)
g.ax_heatmap.set_ylabel(f"Peptide (n={len(correctness_filled)})", fontsize=16)
g.ax_heatmap.set_xticklabels(
    g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=15
)

# Title above the figure (not overlapping heatmap)
g.figure.suptitle(
    "Prediction correctness by peptide and tool (clustered rows and columns)",
    fontsize=22,
    y=1.02
)

# Discrete legend outside the heatmap
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#d73027", label="Incorrect (0)"),
    Patch(facecolor="#fee08b", label="No prediction (0.5)"),
    Patch(facecolor="#1a9850", label="Correct (1)"),
]
g.ax_heatmap.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(1.05, 1.0),
    frameon=True,
    fontsize=18,
    title="Correctness",
    title_fontsize=18,
)

plt.savefig(OUTPUT_ANALYSIS_2_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis 2 saved to {OUTPUT_ANALYSIS_2_PATH}")

# ─── Analysis 3: Correlation between peptide difficulty and properties ────────
# Difficulty = proportion of tools that predicted incorrectly (error rate)
# Only considers tools that have a prediction for each peptide

correctness_df_valid = correctness_df.copy()

# Error rate per peptide: mean of wrong predictions (0) among available tools
# NaN predictions are excluded from the mean
error_rate = correctness_df_valid.apply(
    lambda row: 1 - row.dropna().mean() if row.dropna().size > 0 else np.nan,
    axis=1
)
eval_df = eval_df.copy()
eval_df["error_rate"] = error_rate

# Available physicochemical properties
available_props = [p for p in PHYSICOCHEMICAL_PROPS if p in eval_df.columns]

# Compute Spearman correlations
corr_results = []
for prop in available_props:
    prop_vals = pd.to_numeric(eval_df[prop], errors="coerce")
    mask      = eval_df["error_rate"].notna() & prop_vals.notna()
    if mask.sum() < 10:
        continue
    rho, pval = stats.spearmanr(eval_df.loc[mask, "error_rate"], prop_vals[mask])
    corr_results.append({
        "Property":       prop,
        "Spearman rho":   rho,
        "p-value":        pval,
        "Significant":    pval < 0.05,
    })

corr_df = pd.DataFrame(corr_results).sort_values("Spearman rho")

def format_pval(p):
    """Format p-value: scientific notation if very small, otherwise 3 decimal places."""
    if p == 0.0:
        return "p<1e-300"
    elif p < 0.001:
        exp = int(np.floor(np.log10(p)))
        return f"p=10^{exp}"
    else:
        return f"p={p:.3f}"

fig, ax = plt.subplots(figsize=(12, 6))
colors_bar = ["#d73027" if sig else "#92c5de"
              for sig in corr_df["Significant"]]
bars = ax.barh(corr_df["Property"], corr_df["Spearman rho"], color=colors_bar)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Spearman ρ", fontsize=12)
ax.set_title("Correlation between peptide error rate and physicochemical properties\n"
             "(red = p < 0.05)", fontsize=13)

# Extend x-axis to leave room for labels
xmin, xmax = ax.get_xlim()
ax.set_xlim(xmin - 0.12, xmax + 0.12)

# Add p-value annotations outside the bars
for bar, (_, row) in zip(bars, corr_df.iterrows()):
    x      = bar.get_width()
    label  = format_pval(row["p-value"])
    # Place label beyond the bar end with a small gap, on the outer side
    if x >= 0:
        xpos = xmax + 0.01
        ha   = "left"
    else:
        xpos = xmin - 0.01
        ha   = "right"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha=ha, fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_ANALYSIS_3_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis 3 saved to {OUTPUT_ANALYSIS_3_PATH}")

# Save correlation table
corr_df.to_csv(
    OUTPUT_ANALYSIS_3_PATH.replace(".png", ".csv"),
    index=False
)
print(f"Analysis 3 correlation table saved.")
