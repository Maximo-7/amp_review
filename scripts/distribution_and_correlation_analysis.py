"""
distribution_and_correlation_analysis.py
=========================================
Analyses the distribution of physicochemical properties of ABPs and their
correlation across and against non-AMPs, and produces an UpSet plot for the
database overlap of standard ABPs.

Expected project layout (paths relative to project root):
    data/processed/complete_dataset.csv  -- output of annotate_dataset.py

Output figures (PNG + PDF) are written to:
    results/sequences_analysis/
      distributions_per_database/
      net_charge_per_database/
      ampdb_anomalous/
      distributions_abps_vs_non_amps/
      net_charge_abps_vs_non_amps/
      correlation_heatmaps/
      upset_plot/

Run from the project root:
    python scripts/distribution_and_correlation_analysis.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from upsetplot import plot as upset_plot


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_DIR   = ROOT / "results" / "sequences_analysis"


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.dpi":       150,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, folder_name: str, stem: str) -> None:
    """Save *fig* as both PNG and PDF inside OUT_DIR / folder_name / stem."""
    out = OUT_DIR / folder_name
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    print(f"  Saved → {out / stem}.[png|pdf]")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("=== Loading complete dataset ===")
complete_dataset = pd.read_csv(
    PROCESSED / "complete_dataset.csv", low_memory=False
)
complete_dataset = complete_dataset.rename(columns={"Sequence_length": "Sequence Length"})

df_abps_standard = complete_dataset[
    (complete_dataset["ABP_from_databases"] == True) &
    (complete_dataset["Standard_sequence"]  == True)
]

# non-AMPs: not an ABP from databases, present in Swiss-Prot OR TrEMBL (if
# the column exists), standard sequence
has_trembl = (
    ~complete_dataset["TrEMBL_ID"].isnull()
    if "TrEMBL_ID" in complete_dataset.columns
    else pd.Series(False, index=complete_dataset.index)
)
df_non_amps_standard = complete_dataset[
    (complete_dataset["ABP_from_databases"] == False) &
    (~complete_dataset["Swiss-Prot_ID"].isnull() | has_trembl) &
    (complete_dataset["Standard_sequence"]  == True)
]

print(f"  ABPs (standard):     {len(df_abps_standard)}")
print(f"  non-AMPs (standard): {len(df_non_amps_standard)}")


# ---------------------------------------------------------------------------
# Per-database subsets
# ---------------------------------------------------------------------------

amp_databases = ["AMPDB", "APD", "dbAMP", "DRAMP", "DBAASP"]

dfs_abps: dict[str, pd.DataFrame] = {}
for db in amp_databases:
    col = f"{db}_ID"
    dfs_abps[db] = df_abps_standard[~df_abps_standard[col].isnull()]


# ---------------------------------------------------------------------------
# Shared property metadata
# ---------------------------------------------------------------------------

PROPERTIES = [
    "Sequence Length",
    "Molecular Weight",
    "Net Charge (pH 7)",
    "Aromaticity",
    "Instability Index",
    "Isoelectric Point",
    "GRAVY",
    "Boman Index",
]

XLABELS = [
    "Number of amino acids",
    "Mass (Da)",
    "Electric charge",
    "Sum of relative frequencies",
    "Instability",
    "pH",
    "Hydrophobicity (GRAVY)",
    "Free interaction energy (kcal/mol)",
]


# ===========================================================================
# Figure 1 – Density distributions per AMP database  (KDE only, no histograms)
# ===========================================================================

print("\n=== Figure 1: density distributions per AMP database ===")

# Dynamic x-limits for Net Charge and Instability Index
dynamic_props_db = ["Net Charge (pH 7)", "Instability Index"]
prop_limits_db: dict[str, tuple[float, float]] = {}
for prop in dynamic_props_db:
    global_min = df_abps_standard[prop].min()
    global_max = df_abps_standard[prop].max()
    margin = 0.05 * (global_max - global_min)
    prop_limits_db[prop] = (global_min - margin, global_max + margin)

fig, axes = plt.subplots(4, 2, figsize=(12, 16))
axes = axes.flatten()

for i, (ax, prop, xlabel) in enumerate(zip(axes, PROPERTIES, XLABELS)):
    for db_name, df in dfs_abps.items():
        df[prop].plot.density(ax=ax, label=db_name, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(prop)

    if prop in ("Sequence Length", "Molecular Weight"):
        ax.set_xlim(left=0)
    if prop == "Aromaticity":
        ax.set_xlim(left=0, right=1)
    if prop in prop_limits_db:
        ax.set_xlim(prop_limits_db[prop])

    ax.legend()
    ax.text(
        0.06 if i < 2 else 0.02, 0.98, chr(65 + i),
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="top", ha="left",
    )

plt.tight_layout()
save_figure(fig, "distributions_per_database", "distributions_per_database")
plt.close(fig)


# ===========================================================================
# Figure 2 – Individual KDE + histogram for net charge, per database
# ===========================================================================

print("\n=== Figure 2: net charge per database (KDE + histogram) ===")

net_charge_limits = (-20, 20)

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()
colours_db = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

for i, (ax, (db_name, df), colour) in enumerate(zip(axes, dfs_abps.items(), colours_db)):
    df["Net Charge (pH 7)"].plot.density(ax=ax, alpha=0.9, color=colour)

    bins = np.arange(
        net_charge_limits[0] - 0.5,
        net_charge_limits[1] + 1.5,
        1,
    )
    ax.hist(
        df["Net Charge (pH 7)"],
        bins=bins,
        density=True,
        alpha=0.35,
        color=colour,
        edgecolor=colour,
        linewidth=0.8,
    )

    ax.set_xlabel("Electric charge")
    ax.set_ylabel("Density")
    ax.set_title(f"Net Charge (pH 7) {db_name}")
    ax.set_xlim(net_charge_limits)
    ax.text(
        0.02, 0.98, chr(65 + i),
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="top", ha="left",
    )

plt.tight_layout()
save_figure(fig, "net_charge_per_database", "net_charge_per_database")
plt.close(fig)


# ===========================================================================
# Figure 3 – Anomalous sequences from AMPDB
# ===========================================================================

print("\n=== Figure 3: AMPDB anomalous sequences ===")

df_ampdb          = dfs_abps["AMPDB"]
df_ampdb_anom     = df_ampdb[df_ampdb["Net Charge (pH 7)"] < -5]
df_ampdb_non_anom = df_ampdb[df_ampdb["Net Charge (pH 7)"] >= -5]
dfs_ampdb = {
    "(Charge < -5)": df_ampdb_anom,
    "(Charge ≥ -5)": df_ampdb_non_anom,
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes = axes.flatten()
colours_ampdb = ["tomato", "tab:olive"]

for ax, (db_name, df), colour, label in zip(
    axes, dfs_ampdb.items(), colours_ampdb, ["A", "B"]
):
    df["Sequence Length"].plot.density(ax=ax, alpha=0.9, color=colour)
    ax.set_xlabel("Number of amino acids")
    ax.set_ylabel("Density")
    ax.set_title(f"Sequence Length {db_name}")
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="top", ha="left",
    )

plt.tight_layout()
save_figure(fig, "ampdb_anomalous", "ampdb_sequence_length")
plt.close(fig)

bpi_mask = df_ampdb_anom["AMPDB_name"].str.startswith(
    "Bactericidal permeability-increasing protein (BPI)"
)
print(f"  Number of sequences annotated as BPI: {bpi_mask.sum()}")
print(f"  Number of anomalous sequences:        {len(df_ampdb_anom)}")
print(f"  Mean sequence length – BPI class:     {df_ampdb_anom[bpi_mask]['Sequence Length'].mean():.1f}")
print(f"  Mean sequence length – non-BPI:       {df_ampdb_anom[~bpi_mask]['Sequence Length'].mean():.1f}")


# ===========================================================================
# Figure 4 – Density distributions: ABPs vs non-AMPs  (KDE only, no histograms)
# ===========================================================================

print("\n=== Figure 4: density distributions ABPs vs non-AMPs ===")

dfs_standard = {"ABPs": df_abps_standard, "non-AMPs": df_non_amps_standard}
df_reference  = pd.concat([df_abps_standard, df_non_amps_standard], ignore_index=True)

# Dynamic x-limits (1 % margin)
dynamic_props_std = ["Net Charge (pH 7)", "Instability Index"]
prop_limits_std: dict[str, tuple[float, float]] = {}
for prop in dynamic_props_std:
    global_min = df_reference[prop].min()
    global_max = df_reference[prop].max()
    margin = 0.01 * (global_max - global_min)
    prop_limits_std[prop] = (global_min - margin, global_max + margin)

fig, axes = plt.subplots(4, 2, figsize=(12, 16))
axes = axes.flatten()

for i, (ax, prop, xlabel) in enumerate(zip(axes, PROPERTIES, XLABELS)):
    data_abp     = dfs_standard["ABPs"][prop].dropna()
    data_non_amp = dfs_standard["non-AMPs"][prop].dropna()
    ks_stat, p_value = ks_2samp(data_abp, data_non_amp)
    print(f"  {prop}: KS p-value = {p_value:.2e}")

    for label, df in dfs_standard.items():
        df[prop].plot.density(ax=ax, label=label, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(prop)

    if prop in ("Sequence Length", "Molecular Weight"):
        ax.set_xlim(left=0)
    if prop == "Aromaticity":
        ax.set_xlim(left=0, right=1)
    if prop in prop_limits_std:
        ax.set_xlim(prop_limits_std[prop])

    ax.legend()
    ax.text(
        0.06 if i < 2 else 0.02, 0.98, chr(65 + i),
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="top", ha="left",
    )

plt.tight_layout()
save_figure(fig, "distributions_abps_vs_non_amps", "distributions_abps_vs_non_amps")
plt.close(fig)


# ===========================================================================
# Figure 5 – Net charge: ABPs vs non-AMPs  (KDE + histogram)
# ===========================================================================

print("\n=== Figure 5: net charge ABPs vs non-AMPs (KDE + histogram) ===")

net_charge_limits = (-20, 20)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes = axes.flatten()
colours_std = ["tab:blue", "tab:orange"]

for i, (ax, (label, df), colour) in enumerate(zip(axes, dfs_standard.items(), colours_std)):
    df["Net Charge (pH 7)"].plot.density(ax=ax, alpha=0.9, color=colour)

    bins = np.arange(
        net_charge_limits[0] - 0.5,
        net_charge_limits[1] + 1.5,
        1,
    )
    ax.hist(
        df["Net Charge (pH 7)"],
        bins=bins,
        density=True,
        alpha=0.35,
        color=colour,
        edgecolor=colour,
        linewidth=0.8,
    )

    ax.set_xlabel("Electric charge")
    ax.set_ylabel("Density")
    ax.set_title(f"Net Charge (pH 7) {label}")
    ax.set_xlim(net_charge_limits)
    ax.text(
        0.02, 0.98, chr(65 + i),
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="top", ha="left",
    )

plt.tight_layout()
save_figure(fig, "net_charge_abps_vs_non_amps", "net_charge_abps_vs_non_amps")
plt.close(fig)


# ===========================================================================
# Figure 6 – Spearman correlation heatmaps
# ===========================================================================

print("\n=== Figure 6: correlation heatmaps ===")

df_abps_prop     = df_abps_standard[PROPERTIES]
df_non_amps_prop = df_non_amps_standard[PROPERTIES]

abps_corr     = df_abps_prop.corr(method="spearman").round(2)
non_amps_corr = df_non_amps_prop.corr(method="spearman").round(2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# k=1 excludes only the upper triangle, leaving the diagonal visible
mask = np.triu(np.ones_like(abps_corr, dtype=bool), k=1)

heatmap_kwargs = dict(
    mask=mask,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.4,
    linecolor="white",
    annot_kws={"size": 8.5},
)

sns.heatmap(abps_corr, ax=axes[0], cbar=False, **heatmap_kwargs)
sns.heatmap(
    non_amps_corr,
    ax=axes[1],
    cbar=True,
    cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
    **heatmap_kwargs,
)

for i, (ax, title) in enumerate(zip(axes, ["ABPs", "non-AMPs"])):
    ax.set_title(title, fontsize=13, fontweight="semibold", pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(
        -0.06, 1.02, chr(65 + i),
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="bottom", ha="left",
    )

fig.suptitle(
    "Spearman correlation of physicochemical properties",
    fontsize=14, y=1.01,
)
plt.tight_layout()
save_figure(fig, "correlation_heatmaps", "correlation_heatmaps")
plt.close(fig)


# ===========================================================================
# Figure 7 – UpSet plot of standard ABPs from databases
# ===========================================================================

print("\n=== Figure 7: UpSet plot ===")

df_upset = complete_dataset[
    complete_dataset["ABP_from_databases"] &
    complete_dataset["Standard_sequence"]
][["Sequence"]].copy()

# Boolean membership columns – use .loc to avoid chained-assignment issues
for db in amp_databases:
    col = f"{db}_ID"
    df_upset[db] = complete_dataset.loc[df_upset.index, col].notna().values

df_upset = df_upset.set_index(amp_databases)

# upsetplot calls fillna(..., inplace=True) internally; suppress the pandas
# FutureWarning that originates inside the library (not fixable from user code)
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*inplace method.*",
    )
    fig = plt.figure(figsize=(12, 5))
    upset_plot(df_upset, show_counts=True, fig=fig)

save_figure(fig, "upset_plot", "upset_plot")
plt.close(fig)


# ---------------------------------------------------------------------------
print("\nDone.")
