"""
evaluation_results.py
=====================
Computes evaluation metrics and generates analysis plots for AMP prediction tools.

Expected project layout (paths relative to project root):
    data/processed/complete_dataset.csv    -- output of annotate_dataset.py
    results/evaluation/                    -- output directory for all plots and tables

Analyses:
    0  - Balanced accuracy bar chart (ordered highest to lowest)
    1  - Pairwise agreement matrix (hierarchical clustering heatmap,
         tool labels coloured by ML vs DL)
    2  - Prediction correctness heatmap with clustering (peptides x tools),
         plus side columns for true class and physicochemical properties
    3a - Spearman correlation between peptide error rate and physicochemical
         properties
    3b - Logistic regression for the top-6 models (balanced accuracy):
         coefficients of physicochemical properties as predictors of each
         model's predicted label

Run from the project root:
    python scripts/evaluation_results.py
"""

import sys
import warnings

# scipy's dendrogram uses Python recursion for tree traversal; the default
# limit of 1000 is too shallow for datasets with hundreds of peptides.
sys.setrecursionlimit(100_000)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Patch, FancyBboxPatch
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import pdist
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH               = ROOT / "data" / "processed" / "complete_dataset.csv"
OUTPUT_METRICS_PATH        = ROOT / "results" / "evaluation" / "evaluation_metrics.csv"
OUTPUT_ROC_PATH            = ROOT / "results" / "evaluation" / "roc_curves.png"
OUTPUT_BEST_PREDICTED_PATH = ROOT / "results" / "evaluation" / "best_predicted_sequences.csv"
OUTPUT_ANALYSIS_0_PATH     = ROOT / "results" / "evaluation" / "analysis_0_balanced_accuracy.png"
OUTPUT_ANALYSIS_1_PATH     = ROOT / "results" / "evaluation" / "analysis_1_agreement_matrix.png"
OUTPUT_ANALYSIS_2_PATH     = ROOT / "results" / "evaluation" / "analysis_2_correctness_heatmap.png"
OUTPUT_ANALYSIS_3A_PATH    = ROOT / "results" / "evaluation" / "analysis_3a_property_correlations.png"
OUTPUT_ANALYSIS_3B_PATH    = ROOT / "results" / "evaluation" / "analysis_3b_logreg_top6.png"
OUTPUT_ANALYSIS_3B_CSV     = ROOT / "results" / "evaluation" / "analysis_3b_logreg_top6.csv"
OUTPUT_FIGURE_DIR          = ROOT / "results" / "evaluation" / "figure"

(ROOT / "results" / "evaluation").mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

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

DL_TOOLS = {
    "AMP Scanner", "LMPred", "AMPlify", "Ma et al. (2022)",
    "CAMPR4 ANN", "AMP-BERT", "AMPFinder",
    "PepNet", "KT-AMPpred", "PLAPD", "DLFea4AMPGen", "MultiAMP",
}

UNKNOWN_TRAINING_TOOLS = {
    "Ma et al. (2022)", "CAMPR4 ANN", "CAMPR4 RF", "CAMPR4 SVM", "PyAMPA",
}

# Tools that only predict a subset of peptides due to sequence length restrictions
LENGTH_RESTRICTED_TOOLS = {"AMPlify", "AMP Scanner"}

MA_ET_AL_PROB_COLS = [
    "Ma_et_al_att_pred_prob",
    "Ma_et_al_bert_pred_prob",
    "Ma_et_al_lstm_pred_prob",
]

PHYSICOCHEMICAL_PROPS = [
    "Sequence_length",
    "Net Charge (pH 7)",
    "Aromaticity",
    "Instability Index",
    "Isoelectric Point",
    "GRAVY",
    "Boman Index",
]
# Molecular Weight excluded: heavily correlated with Sequence Length.

# Human-readable display names (used in legend and axis labels)
PROP_DISPLAY_NAMES = {
    "Sequence_length":   "Sequence Length",
    "Net Charge (pH 7)": "Net Charge (pH 7)",
    "Aromaticity":       "Aromaticity",
    "Instability Index": "Instability Index",
    "Isoelectric Point": "Isoelectric Point",
    "GRAVY":             "GRAVY",
    "Boman Index":       "Boman Index",
}

COLOR_DL = "#e07b39"   # warm orange  -> deep learning
COLOR_ML = "#4a90d9"   # steel blue   -> machine learning

# AMP/non-AMP: clearly distinct from both DL (orange) and ML (blue)
COLOR_AMP    = "#6a0dad"   # purple  (avoids clash with red/green heatmap)
COLOR_NONAMP = "#b8860b"   # dark gold (avoids clash with red/green heatmap)

PROP_CMAPS = ["viridis", "plasma", "coolwarm", "YlOrBr", "PuBu", "RdYlGn", "BrBG", "PuOr"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df      = pd.read_csv(DATASET_PATH, low_memory=False)
eval_df = df[df["For_evaluation"]].copy()
y_true  = eval_df["ABP_from_databases"].astype(int).values

# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
colors  = cm.tab20(np.linspace(0, 1, len(roc_data)))

roc_data_sorted = dict(sorted(roc_data.items(), key=lambda x: x[1][2], reverse=True))
for (tool, (fpr, tpr, auroc)), color in zip(roc_data_sorted.items(), colors):
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

# ---------------------------------------------------------------------------
# Best-predicted sequences
# ---------------------------------------------------------------------------

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

for tool_name, (label_col, _) in TOOLS.items():
    if label_col in eval_df.columns:
        col_name  = f"{tool_name}_correct"
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

# ===========================================================================
# Analysis 0: Balanced accuracy bar chart
# ===========================================================================

metrics_sorted = metrics_df.sort_values("Balanced Accuracy", ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))
bar_colors = [COLOR_DL if t in DL_TOOLS else COLOR_ML for t in metrics_sorted["Model"]]
bars = ax.bar(
    metrics_sorted["Model"],
    metrics_sorted["Balanced Accuracy"],
    color=bar_colors,
)
ax.set_xlabel("Tool", fontsize=12)
ax.set_ylabel("Balanced Accuracy (%)", fontsize=12)
ax.set_title("Balanced Accuracy by Tool (ordered highest to lowest)", fontsize=14)
ax.set_ylim([0, 110])
ax.axhline(50, color="gray", linestyle="--", lw=1)

legend_handles = [
    mpatches.Patch(color=COLOR_DL, label="Deep Learning (DL)"),
    mpatches.Patch(color=COLOR_ML, label="Machine Learning (ML)"),
    mlines.Line2D([], [], color="gray", linestyle="--", lw=1.5,
                  label="Random classifier (50%)"),
    mlines.Line2D([], [], color="black", marker="$\u2605$", linestyle="none",
                  markersize=10, label="Training dataset not available"),
    mlines.Line2D([], [], color="black", marker="$\u2716$", linestyle="none",
                  markersize=7, label="Seq. length restrictions (subset only)"),
]
ax.legend(handles=legend_handles, fontsize=9)
plt.xticks(rotation=45, ha="right", fontsize=9)

for bar, (_, row) in zip(bars, metrics_sorted.iterrows()):
    val  = row["Balanced Accuracy"]
    tool = row["Model"]
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    if tool in UNKNOWN_TRAINING_TOOLS:
        ax.text(bar.get_x() + bar.get_width() / 2, val + 4.0,
                "\u2605", ha="center", va="bottom", fontsize=12, color="black")
    if tool in LENGTH_RESTRICTED_TOOLS:
        ax.text(bar.get_x() + bar.get_width() / 2, val + 4.0,
                "\u2716", ha="center", va="bottom", fontsize=12, color="black")

plt.tight_layout()
plt.savefig(OUTPUT_ANALYSIS_0_PATH, dpi=150)
plt.close()
print(f"Analysis 0 saved to {OUTPUT_ANALYSIS_0_PATH}")

# ===========================================================================
# Analysis 1: Pairwise agreement matrix
# ===========================================================================

tool_order = metrics_sorted["Model"].tolist()
n_tools    = len(tool_order)

agreement = pd.DataFrame(np.nan, index=tool_order, columns=tool_order)

for i, tool_i in enumerate(tool_order):
    label_i = TOOLS[tool_i][0]
    if label_i not in eval_df.columns:
        continue
    for j, tool_j in enumerate(tool_order):
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

agreement_sym = agreement.copy()

arr = agreement_sym.to_numpy(copy=True)          # writeable copy
np.fill_diagonal(arr, 100.0)
for i in range(n_tools):
    for j in range(i + 1, n_tools):
        arr[i, j] = arr[j, i]
agreement_sym = pd.DataFrame(arr, index=tool_order, columns=tool_order)

dist_matrix = 100.0 - agreement_sym.fillna(50).values
np.fill_diagonal(dist_matrix, 0.0)
condensed   = pdist(dist_matrix)
Z           = linkage(condensed, method="average")
order_idx   = leaves_list(Z)

agreement_clustered = agreement_sym.iloc[order_idx, :].iloc[:, order_idx]
mask_upper = np.triu(np.ones_like(agreement_clustered, dtype=bool), k=1)

# Use the minimum observed agreement value (floored to nearest 5 %) as vmin
# so the colourbar reflects the actual data range rather than starting at 0.
_a1_vals = agreement_clustered.values[~np.triu(np.ones_like(agreement_clustered, dtype=bool))]
_a1_vals = _a1_vals[~np.isnan(_a1_vals)]
_a1_vmin = int(np.floor(_a1_vals.min() / 5) * 5)
_a1_ticks = list(range(_a1_vmin, 101, 25)) if (100 - _a1_vmin) % 25 == 0     else sorted(set([_a1_vmin] + list(range(0, 101, 25)) + [100]))

fig, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(
    agreement_clustered.astype(float),
    mask=mask_upper,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    vmin=_a1_vmin,
    vmax=100,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 10},
    cbar_kws={"label": "Agreement (%)", "ticks": _a1_ticks},
)

ax.set_title(
    "Pairwise prediction agreement between tools (%)\n(hierarchical clustering)",
    fontsize=19,
)

for tick_label in ax.get_xticklabels():
    tick_label.set_color(COLOR_DL if tick_label.get_text() in DL_TOOLS else COLOR_ML)
for tick_label in ax.get_yticklabels():
    tick_label.set_color(COLOR_DL if tick_label.get_text() in DL_TOOLS else COLOR_ML)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15)

legend_handles = [
    mpatches.Patch(color=COLOR_DL, label="Deep Learning (DL)"),
    mpatches.Patch(color=COLOR_ML, label="Machine Learning (ML)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=15, frameon=True)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Agreement (%)", fontsize=15)

plt.tight_layout()
plt.savefig(OUTPUT_ANALYSIS_1_PATH, dpi=150)
plt.close()
print(f"Analysis 1 saved to {OUTPUT_ANALYSIS_1_PATH}")

# ===========================================================================
# Analysis 2: Correctness heatmap — clustermap with dendrograms for peptides
#             and tools (Ward linkage), side columns for true class and
#             physicochemical properties, legend drawn inside a dedicated axes.
# ===========================================================================

# ── Build per-peptide correctness matrix ─────────────────────────────────
correctness_data = {}
for tool_name, (label_col, _) in TOOLS.items():
    if label_col not in eval_df.columns:
        continue
    pred_vals = pd.to_numeric(eval_df[label_col], errors="coerce")
    correctness_data[tool_name] = np.where(
        pred_vals.isna(),
        np.nan,
        (pred_vals == eval_df["ABP_from_databases"].astype(int)).astype(float),
    )

correctness_df = pd.DataFrame(correctness_data, index=eval_df.index)
correctness_df = correctness_df.dropna(how="all")

eval_sub = eval_df.loc[correctness_df.index].copy()
correctness_filled = correctness_df.fillna(0.5)

# ── Hierarchical clustering ───────────────────────────────────────────────
row_dist  = pdist(correctness_filled.values, metric="euclidean")
Z_rows    = linkage(row_dist, method="ward")
row_order = leaves_list(Z_rows)

col_dist  = pdist(correctness_filled.values.T, metric="euclidean")
Z_cols    = linkage(col_dist, method="ward")
col_order = leaves_list(Z_cols)

correctness_plot = correctness_filled.iloc[row_order, col_order]
eval_plot        = eval_sub.iloc[row_order]
tools_ordered    = [list(correctness_df.columns)[c] for c in col_order]

# ── Side annotations ──────────────────────────────────────────────────────
available_props = [p for p in PHYSICOCHEMICAL_PROPS if p in eval_plot.columns]

side_annot_raw = pd.DataFrame(index=eval_plot.index)
side_annot_raw["Ground truth"] = eval_plot["ABP_from_databases"].astype(float).values

# Physicochemical side strips are normalised to [5th, 95th] percentile so
# that outliers don't compress the colour range for the majority of peptides.
# Values below p5 map to 0 (coldest colour) and above p95 map to 1 (hottest).
prop_stats = {}
for prop in available_props:
    vals = pd.to_numeric(eval_plot[prop], errors="coerce")
    p5,  p95  = np.nanpercentile(vals, 5),  np.nanpercentile(vals, 95)
    vmid = np.nanmedian(vals)
    prop_stats[prop] = (p5, vmid, p95)   # stored as (lo, mid, hi) for legend
    norm_vals = (vals - p5) / (p95 - p5 + 1e-12)
    side_annot_raw[prop] = norm_vals.clip(0, 1)  # clip outliers to endpoints

# ── GridSpec layout ───────────────────────────────────────────────────────
n_side    = 1 + len(available_props)
n_tools_c = len(tools_ordered)

w_left_dendro = 3
w_heatmap     = n_tools_c
w_side        = 1
w_legend      = 14

col_widths  = [w_left_dendro, w_heatmap] + [w_side] * n_side + [w_legend]
row_heights = [2, 20]    # top row taller so title fits inside it without overlap

fig_width  = 26 + n_side * 0.6 + w_legend * 0.5
fig_height = 22

fig = plt.figure(figsize=(fig_width, fig_height))
gs  = gridspec.GridSpec(
    2,
    2 + n_side + 1,
    height_ratios=row_heights,
    width_ratios=col_widths,
    hspace=0.02,
    wspace=0.03,
)

ax_top_dendro   = fig.add_subplot(gs[0, 1])
ax_left_dendro  = fig.add_subplot(gs[1, 0])
ax_main         = fig.add_subplot(gs[1, 1])
ax_sides        = [fig.add_subplot(gs[1, 2 + k]) for k in range(n_side)]
ax_legend       = fig.add_subplot(gs[:, 2 + n_side])
ax_legend.set_axis_off()

DENDRO_LW = 0.7

# ── Title: drawn inside the top-dendrogram axes row, above the dendrogram ──
ax_top_dendro.set_title(
    "Prediction correctness by peptide (rows) and tool (columns)",
    fontsize=18,
    fontweight="bold",
    pad=6,
)

# ── Top dendrogram ────────────────────────────────────────────────────────
dendrogram(
    Z_cols,
    ax=ax_top_dendro,
    orientation="top",
    no_labels=True,
    color_threshold=0,
    above_threshold_color="black",
    link_color_func=lambda k: "black",
)
for coll in ax_top_dendro.collections:
    coll.set_linewidth(DENDRO_LW)
ax_top_dendro.set_axis_off()

# ── Left dendrogram ───────────────────────────────────────────────────────
dendrogram(
    Z_rows,
    ax=ax_left_dendro,
    orientation="left",
    no_labels=True,
    color_threshold=0,
    above_threshold_color="black",
    link_color_func=lambda k: "black",
)
for coll in ax_left_dendro.collections:
    coll.set_linewidth(DENDRO_LW)
ax_left_dendro.set_axis_off()
ax_left_dendro.invert_yaxis()

# ── Main heatmap ──────────────────────────────────────────────────────────
cmap_main = mcolors.LinearSegmentedColormap.from_list(
    "correctness", ["#d73027", "#fee08b", "#1a9850"]
)
ax_main.imshow(
    correctness_plot.values,
    aspect="auto",
    cmap=cmap_main,
    vmin=0,
    vmax=1,
    interpolation="none",
)
ax_main.set_yticks([])
ax_main.set_xticks(range(len(tools_ordered)))
ax_main.set_xticklabels(tools_ordered, rotation=45, ha="right", fontsize=13)
ax_main.tick_params(axis="x", bottom=True, top=False, labelbottom=True)

for tick_label in ax_main.get_xticklabels():
    tick_label.set_color(COLOR_DL if tick_label.get_text() in DL_TOOLS else COLOR_ML)

# ── Side annotation: Ground truth ─────────────────────────────────────────
cmap_class = mcolors.ListedColormap([COLOR_NONAMP, COLOR_AMP])
ax_sides[0].imshow(
    side_annot_raw["Ground truth"].values.reshape(-1, 1),
    aspect="auto",
    cmap=cmap_class,
    vmin=0,
    vmax=1,
)
ax_sides[0].set_xticks([0])
ax_sides[0].set_xticklabels(["Ground truth"], rotation=45, ha="right", fontsize=11)
ax_sides[0].set_yticks([])

# ── Side annotations: physicochemical properties ──────────────────────────
for k, prop in enumerate(available_props):
    ax_s         = ax_sides[k + 1]
    prop_data    = side_annot_raw[prop].values.reshape(-1, 1)
    display_name = PROP_DISPLAY_NAMES.get(prop, prop)
    ax_s.imshow(
        prop_data,
        aspect="auto",
        cmap=PROP_CMAPS[k % len(PROP_CMAPS)],
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    ax_s.set_xticks([0])
    ax_s.set_xticklabels([display_name], rotation=45, ha="right", fontsize=11)
    ax_s.set_yticks([])

# ===========================================================================
# Legend — drawn entirely inside ax_legend using its own coordinate system
# (axes fraction 0-1).  Gradient bars are true matplotlib Axes added with
# inset_axes (bbox_transform=ax_legend.transAxes), which guarantees that
# their pixel positions are correct regardless of figure size.
# ===========================================================================

from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes

# We split the legend axes vertically into a data-structure that describes
# each block.  Heights are in "units" that we'll map to axes-fraction at the
# end, once we know the total.

FSEC  = 14    # section-title font size
FENT  = 13    # entry font size
FNUM  = 11    # gradient tick-label font size

# Helper: draw a thin horizontal separator line across the legend axes
def _hline(ax, y, lw=0.8, color="#aaaaaa"):
    ax.axhline(y, xmin=0.05, xmax=0.95, color=color, linewidth=lw,
               transform=ax.transData, clip_on=False)

# We build the legend top-to-bottom using a running y position in axes
# coordinates (1 = top, 0 = bottom).  All drawing uses ax_legend.transAxes.

ax = ax_legend
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# ── Estimate total height needed so we can set the y-step sizes ────────────
# Swatch blocks: 3 entries for correctness, 2 for tool type, 2 for ground truth
# Gradient bars show [p5, median, p95] range (outlier-robust normalisation)
# Gradient block: len(available_props) bars
# We'll use a simple uniform grid: divide [0,1] into logical rows.

n_swatch_rows = 3 + 2 + 2   # correctness + tool type + ground truth entries
n_grad_rows   = len(available_props)
# Approximate row heights in normalised units (will be scaled to fit)
# We give section headers 1.4x a normal row, gradient bars 1.8x
unit = 1.0 / (
    4 * 1.4 +           # 4 section headers (correctness, tool, gt, properties)
    n_swatch_rows * 1.0 +
    n_grad_rows * 2.2 +
    5 * 0.4             # inter-block gaps
)
H_SEC  = 1.4 * unit   # section-header height
H_ROW  = 1.0 * unit   # swatch-entry height
H_GRAD = 2.2 * unit   # gradient bar slot height (name + bar + ticks)
H_GAP  = 0.4 * unit   # gap between blocks

y = 1.0 - 0.015       # start near top with a small margin

def _section_header(ax, y, text):
    """Draw a shaded section-header rectangle and bold label. Returns new y."""
    h = H_SEC
    # shaded background strip
    rect = plt.Rectangle((0.02, y - h), 0.96, h,
                          facecolor="#e0e0e0", edgecolor="#888888",
                          linewidth=0.8, transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)
    ax.text(0.5, y - h / 2, text,
            ha="center", va="center", fontsize=FSEC, fontweight="bold",
            transform=ax.transAxes, clip_on=False)
    return y - h

def _swatch_entry(ax, y, color, label):
    """Draw one swatch + label row. Returns new y."""
    h  = H_ROW
    yc = y - h / 2
    # swatch rectangle
    sw = plt.Rectangle((0.06, yc - h * 0.32), 0.10, h * 0.64,
                        facecolor=color, edgecolor="#555555", linewidth=0.5,
                        transform=ax.transAxes, clip_on=False)
    ax.add_patch(sw)
    ax.text(0.20, yc, label,
            ha="left", va="center", fontsize=FENT,
            transform=ax.transAxes, clip_on=False)
    return y - h

def _gradient_entry(ax, y, prop, vmin, vmean, vmax, cmap_name):
    """
    Draw one gradient bar with property name above and p5/median/p95 below.
    The bar is a real Axes added as an inset, anchored to ax.transAxes.
    Colour range corresponds to [p5, p95]; outliers are clipped to endpoints.
    Returns new y.
    """
    display_name = PROP_DISPLAY_NAMES.get(prop, prop)
    h     = H_GRAD
    name_frac = 0.30   # fraction of h for the name
    bar_frac  = 0.38   # fraction of h for the bar image
    tick_frac = 0.32   # fraction of h for tick labels

    # Property name
    y_name = y - name_frac * h / 2
    ax.text(0.5, y - name_frac * h / 2, display_name,
            ha="center", va="center", fontsize=FENT, style="italic",
            transform=ax.transAxes, clip_on=False)

    # Gradient bar as inset axes
    bar_bottom = y - name_frac * h - bar_frac * h   # bottom of bar in axes coords
    bar_height = bar_frac * h
    bar_left   = 0.06
    bar_width  = 0.88

    ax_bar = ax.inset_axes(
        [bar_left, bar_bottom, bar_width, bar_height],
        transform=ax.transAxes,
    )
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_bar.imshow(gradient, aspect="auto",
                  cmap=plt.get_cmap(cmap_name), origin="lower")
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 127.5, 255])
    ax_bar.set_xticklabels(
        [f"{vmin:.3g}", f"{vmean:.3g}", f"{vmax:.3g}"],
        fontsize=FNUM,
    )
    ax_bar.tick_params(axis="x", direction="out", length=2.5, pad=1.5)
    for spine in ax_bar.spines.values():
        spine.set_linewidth(0.5)

    return y - h

# ── Block 1: Prediction correctness ──────────────────────────────────────
y = _section_header(ax, y, "Prediction correctness")
y = _swatch_entry(ax, y, "#1a9850", "Correct")
y = _swatch_entry(ax, y, "#fee08b", "No prediction")
y = _swatch_entry(ax, y, "#d73027", "Incorrect")
y -= H_GAP

# ── Block 2: Tool type ────────────────────────────────────────────────────
y = _section_header(ax, y, "Tool type")
y = _swatch_entry(ax, y, COLOR_DL, "Deep Learning (DL)")
y = _swatch_entry(ax, y, COLOR_ML, "Machine Learning (ML)")
y -= H_GAP

# ── Block 3: Ground truth ─────────────────────────────────────────────────
y = _section_header(ax, y, "Ground truth")
y = _swatch_entry(ax, y, COLOR_AMP,    "AMP")
y = _swatch_entry(ax, y, COLOR_NONAMP, "non-AMP")
y -= H_GAP

# ── Block 4: Physicochemical properties (gradient bars) ───────────────────
y = _section_header(ax, y, "Physicochemical properties")
for k, prop in enumerate(available_props):
    vmin, vmean, vmax = prop_stats[prop]
    y = _gradient_entry(ax, y, prop, vmin, vmean, vmax,
                        PROP_CMAPS[k % len(PROP_CMAPS)])

# ── Save ──────────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_ANALYSIS_2_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis 2 saved to {OUTPUT_ANALYSIS_2_PATH}")

# ===========================================================================
# Analysis 3a: Spearman correlation
# ===========================================================================

correctness_df_valid = correctness_df.copy()
error_rate = correctness_df_valid.apply(
    lambda row: 1 - row.dropna().mean() if row.dropna().size > 0 else np.nan,
    axis=1,
)
eval_df = eval_df.copy()
eval_df["error_rate"] = error_rate

available_props_global = [p for p in PHYSICOCHEMICAL_PROPS if p in eval_df.columns]

corr_results = []
for prop in available_props_global:
    prop_vals = pd.to_numeric(eval_df[prop], errors="coerce")
    mask      = eval_df["error_rate"].notna() & prop_vals.notna()
    if mask.sum() < 10:
        continue
    rho, pval = stats.spearmanr(eval_df.loc[mask, "error_rate"], prop_vals[mask])
    corr_results.append({
        "Property":     prop,
        "Spearman rho": rho,
        "p-value":      pval,
        "Significant":  pval < 0.05,
    })

corr_df = pd.DataFrame(corr_results).sort_values("Spearman rho")

fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ["#d73027" if sig else "#92c5de" for sig in corr_df["Significant"]]
ax.barh(corr_df["Property"], corr_df["Spearman rho"], color=colors_bar)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Spearman \u03c1", fontsize=12)
ax.set_title(
    "Correlation between peptide error rate and physicochemical properties\n"
    "(red = p < 0.05)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(OUTPUT_ANALYSIS_3A_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis 3a saved to {OUTPUT_ANALYSIS_3A_PATH}")

corr_df.to_csv(str(OUTPUT_ANALYSIS_3A_PATH).replace(".png", ".csv"), index=False)
print("Analysis 3a correlation table saved.")

# ===========================================================================
# Analysis 3b: Logistic regression for top-6 models
# ===========================================================================

top6_tools         = metrics_sorted.head(6)["Model"].tolist()
available_props_lr = [p for p in PHYSICOCHEMICAL_PROPS if p in eval_df.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

all_coef_rows = []

for ax_idx, tool_name in enumerate(top6_tools):
    ax        = axes_flat[ax_idx]
    label_col = TOOLS[tool_name][0]

    if label_col not in eval_df.columns:
        ax.set_title(f"{tool_name}\n(no data)", fontsize=11)
        ax.axis("off")
        continue

    cols_needed = [label_col] + available_props_lr
    sub = eval_df[cols_needed].copy()
    sub[label_col] = pd.to_numeric(sub[label_col], errors="coerce")
    for p in available_props_lr:
        sub[p] = pd.to_numeric(sub[p], errors="coerce")
    sub = sub.dropna()

    if sub[label_col].nunique() < 2 or len(sub) < 20:
        ax.set_title(f"{tool_name}\n(insufficient data)", fontsize=11)
        ax.axis("off")
        continue

    X      = sub[available_props_lr].values
    y      = sub[label_col].astype(int).values

    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    X_sm  = sm.add_constant(X_std, has_constant="add")
    model = sm.Logit(y, X_sm)
    try:
        result = model.fit(method="bfgs", maxiter=200, disp=False)
    except Exception:
        result = model.fit(method="newton", maxiter=200, disp=False)

    coef_df = pd.DataFrame({
        "Model":       tool_name,
        "Property":    available_props_lr,
        "Coefficient": result.params[1:],
        "Std_Error":   result.bse[1:],
        "z_stat":      result.tvalues[1:],
        "p_value":     result.pvalues[1:],
    }).sort_values("Coefficient")

    all_coef_rows.append(coef_df)

    bar_colors_lr = []
    for c, p in zip(coef_df["Coefficient"], coef_df["p_value"]):
        if p < 0.05:
            bar_colors_lr.append("#d73027" if c > 0 else "#4393c3")
        else:
            bar_colors_lr.append("#f4a582" if c > 0 else "#92c5de")

    ax.barh(coef_df["Property"], coef_df["Coefficient"], color=bar_colors_lr)
    ax.axvline(0, color="black", lw=0.8)

    is_dl       = tool_name in DL_TOOLS
    label_type  = "DL" if is_dl else "ML"
    title_color = COLOR_DL if is_dl else COLOR_ML
    ax.set_title(f"{tool_name}  [{label_type}]", fontsize=11,
                 color=title_color, fontweight="bold")
    ax.set_xlabel("Standardised coefficient", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    bacc_val = metrics_df.loc[metrics_df["Model"] == tool_name, "Balanced Accuracy"].values[0]
    ax.text(
        0.98, 0.02, f"Bal. Acc. = {bacc_val:.1f}%",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray",
    )

for k in range(len(top6_tools), 6):
    axes_flat[k].axis("off")

sig_handles = [
    mpatches.Patch(color="#d73027", label="p < 0.05, positive"),
    mpatches.Patch(color="#4393c3", label="p < 0.05, negative"),
    mpatches.Patch(color="#f4a582", label="p \u2265 0.05, positive"),
    mpatches.Patch(color="#92c5de", label="p \u2265 0.05, negative"),
]
fig.legend(handles=sig_handles, loc="lower center", ncol=4, fontsize=9,
           title="Coefficient direction & significance",
           title_fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    "Logistic regression: physicochemical predictors of each top-6 model's predicted label\n"
    "(standardised Wald coefficients)",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUT_ANALYSIS_3B_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Analysis 3b saved to {OUTPUT_ANALYSIS_3B_PATH}")

if all_coef_rows:
    pd.concat(all_coef_rows, ignore_index=True).to_csv(OUTPUT_ANALYSIS_3B_CSV, index=False)
    print(f"Analysis 3b CSV saved to {OUTPUT_ANALYSIS_3B_CSV}")

# ===========================================================================
# Combined paper figure  (fully vectorial — all content re-drawn natively)
# ===========================================================================
# Layout (row-major order):
#   Row 0 (top)    : [A] ROC curves        | [B] Analysis 0 (balanced accuracy)
#   Row 1 (middle) : [C] Analysis 1        | [D] Analysis 3a (Spearman ρ)
#   Row 2 (bottom) : [E] Analysis 2 heatmap (spans both columns)
#
# Analysis 2 is itself a complex nested GridSpec; it is rebuilt inside a
# SubplotSpec cell using GridSpecFromSubplotSpec so that its internal axes
# (dendrograms, heatmap, side strips, legend) are proper matplotlib Axes
# rather than rasterised images.
# ===========================================================================

FIGURE_PDF_PATH = OUTPUT_FIGURE_DIR / "combined_figure.pdf"
FIGURE_PNG_PATH = OUTPUT_FIGURE_DIR / "combined_figure.png"

_PANEL_KW = dict(fontsize=26, fontweight="bold", va="top", ha="left")

# ── Outer grid: 3 rows × 2 columns ─────────────────────────────────────────
# Symmetric left/right margins are set via subplots_adjust
fig_paper = plt.figure(figsize=(26, 34))
fig_paper.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.05)

gs_outer = gridspec.GridSpec(
    3, 2,
    figure=fig_paper,
    height_ratios=[3, 3, 5],   # bottom row taller for heatmap
    hspace=0.28,
    wspace=0.28,
)

# ---------------------------------------------------------------------------
# Panel A – ROC curves
# ---------------------------------------------------------------------------
ax_A = fig_paper.add_subplot(gs_outer[0, 0])

roc_colors = cm.tab20(np.linspace(0, 1, len(roc_data)))
roc_data_sorted = dict(sorted(roc_data.items(), key=lambda x: x[1][2], reverse=True))
for (tool, (fpr, tpr, auroc)), color in zip(roc_data_sorted.items(), roc_colors):
    ax_A.plot(fpr, tpr, lw=1.5, color=color, label=f"{tool} (AUROC = {auroc:.1f}%)")

ax_A.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax_A.set_xlabel("False Positive Rate", fontsize=13)
ax_A.set_ylabel("True Positive Rate", fontsize=13)
ax_A.set_title("ROC curves", fontsize=16, fontweight="bold")
ax_A.legend(loc="lower right", fontsize=8)
ax_A.tick_params(axis="both", labelsize=11)
ax_A.set_xlim([0, 1])
ax_A.set_ylim([0, 1.02])
ax_A.text(-0.12, 1.04, "A", transform=ax_A.transAxes, **_PANEL_KW)

# ---------------------------------------------------------------------------
# Panel B – Analysis 0: balanced accuracy bar chart
# ---------------------------------------------------------------------------
ax_B = fig_paper.add_subplot(gs_outer[0, 1])

bar_colors_B = [COLOR_DL if t in DL_TOOLS else COLOR_ML for t in metrics_sorted["Model"]]
bars_B = ax_B.bar(
    metrics_sorted["Model"],
    metrics_sorted["Balanced Accuracy"],
    color=bar_colors_B,
)
ax_B.set_xlabel("Tool", fontsize=13)
ax_B.set_ylabel("Balanced Accuracy (%)", fontsize=13)
ax_B.set_title("Balanced Accuracy by Tool (ordered highest to lowest)", fontsize=16, fontweight="bold")
ax_B.set_ylim([0, 110])
ax_B.axhline(50, color="gray", linestyle="--", lw=1)
ax_B.tick_params(axis="y", labelsize=11)

legend_handles_B = [
    mpatches.Patch(color=COLOR_DL, label="Deep Learning (DL)"),
    mpatches.Patch(color=COLOR_ML, label="Machine Learning (ML)"),
    mlines.Line2D([], [], color="gray", linestyle="--", lw=1.5,
                  label="Random classifier (50%)"),
    mlines.Line2D([], [], color="black", marker="$\u2605$", linestyle="none",
                  markersize=9, label="Training dataset not available"),
    mlines.Line2D([], [], color="black", marker="$\u2716$", linestyle="none",
                  markersize=7, label="Seq. length restrictions (subset only)"),
]
ax_B.legend(handles=legend_handles_B, fontsize=9)
ax_B.set_xticks(range(len(metrics_sorted)))
ax_B.set_xticklabels(metrics_sorted["Model"], rotation=45, ha="right", fontsize=9)

for bar, (_, row) in zip(bars_B, metrics_sorted.iterrows()):
    val  = row["Balanced Accuracy"]
    tool = row["Model"]
    ax_B.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
              f"{val:.1f}", ha="center", va="bottom", fontsize=6)
    if tool in UNKNOWN_TRAINING_TOOLS:
        ax_B.text(bar.get_x() + bar.get_width() / 2, val + 4.0,
                  "\u2605", ha="center", va="bottom", fontsize=10, color="black")
    if tool in LENGTH_RESTRICTED_TOOLS:
        ax_B.text(bar.get_x() + bar.get_width() / 2, val + 4.0,
                  "\u2716", ha="center", va="bottom", fontsize=10, color="black")

ax_B.text(-0.10, 1.04, "B", transform=ax_B.transAxes, **_PANEL_KW)

# ---------------------------------------------------------------------------
# Panel C – Analysis 1: pairwise agreement heatmap
# ---------------------------------------------------------------------------
ax_C = fig_paper.add_subplot(gs_outer[1, 0])

sns.heatmap(
    agreement_clustered.astype(float),
    mask=mask_upper,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    vmin=_a1_vmin,
    vmax=100,
    linewidths=0.4,
    ax=ax_C,
    annot_kws={"size": 6},
    cbar_kws={"label": "Agreement (%)", "ticks": _a1_ticks},
)
ax_C.set_title(
    "Pairwise prediction agreement (%)\n(hierarchical clustering)",
    fontsize=16, fontweight="bold",
)
for tl in ax_C.get_xticklabels():
    tl.set_color(COLOR_DL if tl.get_text() in DL_TOOLS else COLOR_ML)
for tl in ax_C.get_yticklabels():
    tl.set_color(COLOR_DL if tl.get_text() in DL_TOOLS else COLOR_ML)
ax_C.set_xticklabels(ax_C.get_xticklabels(), rotation=45, ha="right", fontsize=9)
ax_C.set_yticklabels(ax_C.get_yticklabels(), rotation=0, fontsize=9)

legend_handles_C = [
    mpatches.Patch(color=COLOR_DL, label="Deep Learning (DL)"),
    mpatches.Patch(color=COLOR_ML, label="Machine Learning (ML)"),
]
ax_C.legend(handles=legend_handles_C, loc="upper right", fontsize=10, frameon=True)

cbar_C = ax_C.collections[0].colorbar
cbar_C.ax.tick_params(labelsize=9)
cbar_C.set_label("Agreement (%)", fontsize=11)

ax_C.text(-0.13, 1.04, "C", transform=ax_C.transAxes, **_PANEL_KW)

# ---------------------------------------------------------------------------
# Panel D – Analysis 3a: Spearman correlations
# ---------------------------------------------------------------------------
ax_D = fig_paper.add_subplot(gs_outer[1, 1])

colors_bar_D = ["#d73027" if sig else "#92c5de" for sig in corr_df["Significant"]]
ax_D.barh(corr_df["Property"], corr_df["Spearman rho"], color=colors_bar_D)
ax_D.axvline(0, color="black", lw=0.8)
ax_D.set_xlabel("Spearman \u03c1", fontsize=13)
ax_D.set_title(
    "Correlation: peptide error rate vs\nphysicochemical properties  (red = p < 0.05)",
    fontsize=16, fontweight="bold",
)
ax_D.tick_params(axis="both", labelsize=11)
ax_D.text(-0.10, 1.04, "D", transform=ax_D.transAxes, **_PANEL_KW)

# ---------------------------------------------------------------------------
# Panel E – Analysis 2: correctness heatmap (rebuilt natively)
# Uses GridSpecFromSubplotSpec to nest the same internal layout (left
# dendrogram | heatmap | side strips | legend) inside the bottom cell.
# ---------------------------------------------------------------------------

# The bottom cell spans both content columns
ss_E = gs_outer[2, :]

n_side_E    = 1 + len(available_props)
n_tools_E   = len(tools_ordered)

w_ld  = 2          # left dendrogram
w_hm  = n_tools_E  # heatmap
w_s   = 1          # each side strip
w_leg = 12         # legend

col_widths_E  = [w_ld, w_hm] + [w_s] * n_side_E + [w_leg]
row_heights_E = [2, 18]

gs_E = gridspec.GridSpecFromSubplotSpec(
    2,
    2 + n_side_E + 1,
    subplot_spec=ss_E,
    height_ratios=row_heights_E,
    width_ratios=col_widths_E,
    hspace=0.02,
    wspace=0.03,
)

ax_E_top_dend  = fig_paper.add_subplot(gs_E[0, 1])
ax_E_left_dend = fig_paper.add_subplot(gs_E[1, 0])
ax_E_main      = fig_paper.add_subplot(gs_E[1, 1])
ax_E_sides     = [fig_paper.add_subplot(gs_E[1, 2 + k]) for k in range(n_side_E)]
ax_E_legend    = fig_paper.add_subplot(gs_E[:, 2 + n_side_E])
ax_E_legend.set_axis_off()

# Panel label E – placed on the left dendrogram row
ax_E_top_dend.set_title(
    "Prediction correctness by peptide (rows) and tool (columns)",
    fontsize=16, fontweight="bold", pad=5,
)
ax_E_top_dend.text(-0.07, 1.35, "E", transform=ax_E_top_dend.transAxes, **_PANEL_KW)

_DLWE = 0.6

dendrogram(Z_cols, ax=ax_E_top_dend, orientation="top", no_labels=True,
           color_threshold=0, above_threshold_color="black",
           link_color_func=lambda k: "black")
for coll in ax_E_top_dend.collections:
    coll.set_linewidth(_DLWE)
ax_E_top_dend.set_axis_off()

dendrogram(Z_rows, ax=ax_E_left_dend, orientation="left", no_labels=True,
           color_threshold=0, above_threshold_color="black",
           link_color_func=lambda k: "black")
for coll in ax_E_left_dend.collections:
    coll.set_linewidth(_DLWE)
ax_E_left_dend.set_axis_off()
ax_E_left_dend.invert_yaxis()

cmap_main_E = mcolors.LinearSegmentedColormap.from_list(
    "correctness", ["#d73027", "#fee08b", "#1a9850"]
)
ax_E_main.imshow(
    correctness_plot.values, aspect="auto",
    cmap=cmap_main_E, vmin=0, vmax=1, interpolation="none",
)
ax_E_main.set_yticks([])
ax_E_main.set_xticks(range(len(tools_ordered)))
ax_E_main.set_xticklabels(tools_ordered, rotation=45, ha="right", fontsize=11)
ax_E_main.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
for tl in ax_E_main.get_xticklabels():
    tl.set_color(COLOR_DL if tl.get_text() in DL_TOOLS else COLOR_ML)

cmap_class_E = mcolors.ListedColormap([COLOR_NONAMP, COLOR_AMP])
ax_E_sides[0].imshow(
    side_annot_raw["Ground truth"].values.reshape(-1, 1),
    aspect="auto", cmap=cmap_class_E, vmin=0, vmax=1,
)
ax_E_sides[0].set_xticks([0])
ax_E_sides[0].set_xticklabels(["Ground truth"], rotation=45, ha="right", fontsize=10)
ax_E_sides[0].set_yticks([])

for k, prop in enumerate(available_props):
    ax_s = ax_E_sides[k + 1]
    ax_s.imshow(
        side_annot_raw[prop].values.reshape(-1, 1),
        aspect="auto", cmap=PROP_CMAPS[k % len(PROP_CMAPS)],
        vmin=0, vmax=1, interpolation="none",
    )
    ax_s.set_xticks([0])
    ax_s.set_xticklabels(
        [PROP_DISPLAY_NAMES.get(prop, prop)], rotation=45, ha="right", fontsize=10
    )
    ax_s.set_yticks([])

# Re-draw the legend inside ax_E_legend using the same helper functions
# defined during Analysis 2 (they close over ax via the 'ax' name, so we
# temporarily rebind the module-level 'ax' variable they reference).
_ax_saved = ax                              # save the old binding
ax = ax_E_legend                            # point helpers at the new axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

y = 1.0 - 0.015
y = _section_header(ax, y, "Prediction correctness")
y = _swatch_entry(ax, y, "#1a9850", "Correct")
y = _swatch_entry(ax, y, "#fee08b", "No prediction")
y = _swatch_entry(ax, y, "#d73027", "Incorrect")
y -= H_GAP
y = _section_header(ax, y, "Tool type")
y = _swatch_entry(ax, y, COLOR_DL, "Deep Learning (DL)")
y = _swatch_entry(ax, y, COLOR_ML, "Machine Learning (ML)")
y -= H_GAP
y = _section_header(ax, y, "Ground truth")
y = _swatch_entry(ax, y, COLOR_AMP,    "AMP")
y = _swatch_entry(ax, y, COLOR_NONAMP, "non-AMP")
y -= H_GAP
y = _section_header(ax, y, "Physicochemical properties")
for k, prop in enumerate(available_props):
    vmin, vmean, vmax = prop_stats[prop]
    y = _gradient_entry(ax, y, prop, vmin, vmean, vmax, PROP_CMAPS[k % len(PROP_CMAPS)])

ax = _ax_saved                              # restore

# ── Save both formats ────────────────────────────────────────────────────────
fig_paper.savefig(FIGURE_PDF_PATH)
fig_paper.savefig(FIGURE_PNG_PATH, dpi=300)
plt.close(fig_paper)
print(f"Combined figure saved to:\n  {FIGURE_PDF_PATH}\n  {FIGURE_PNG_PATH}")
