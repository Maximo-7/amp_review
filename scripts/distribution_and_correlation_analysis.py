"""
distribution_and_correlation_analysis.py
=========================================
Analyses the distribution of physicochemical properties of ABPs and their
correlation across and against non-AMPs, and produces an UpSet plot for the
database overlap of standard ABPs.

Also performs a detailed Bactericidal Permeability-Increasing protein (BPI)
incidence analysis within AMPDB, motivated by the pressence of anomalous
negatively charged peptides, and extends it to the other AMP databases.

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

# ---------------------------------------------------------------------------
# BPI incidence analysis
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 1 – Retrieve canonical BPI sequences from UniProt via UniProt API
#
# Strategy: query UniProt (all entries, no reviewed filter) for entries whose
# recommended protein name is exactly "Bactericidal permeability-increasing
# protein".  The returned sequences form the ground-truth BPI reference set,
# which is then used to annotate ALL five databases via exact sequence
# matching — including the three databases (APD, dbAMP, DRAMP) that carry no
# name field.
# ---------------------------------------------------------------------------

import re
import time
import requests

_UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
_SEQ_LEN_MIN = 5
_SEQ_LEN_MAX = 255

# Exact protein name query: field "protein_name" in all UniProt entries
# (no reviewed:true filter).  Wrapping in quotes enforces phrase matching
# in the UniProt query language.
_BPI_QUERY = (
    'protein_name:"Bactericidal permeability-increasing protein" '
    f"AND length:[{_SEQ_LEN_MIN} TO {_SEQ_LEN_MAX}]"
)

def _fetch_uniprot_bpi_sequences(retries: int = 3, pause: float = 2.0) -> set[str]:
    """
    Query UniProt (all entries, no reviewed filter) for entries whose
    recommended protein name is exactly 'Bactericidal permeability-increasing
    protein' and return their sequences as a set of strings.

    Only sequences with length in [_SEQ_LEN_MIN, _SEQ_LEN_MAX] are retained,
    matching the length range of the dataset.

    Uses UniProt REST pagination (link-header cursor) to handle any result size.
    Falls back to an empty set after *retries* consecutive failures so that the
    rest of the script can still run (with a warning).
    """
    params = {
        "query":  _BPI_QUERY,
        "format": "json",
        "fields": "sequence,protein_name",
        "size":   500,
    }
    sequences: set[str] = set()
    url: str | None = _UNIPROT_SEARCH

    while url:
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, params=params if url == _UNIPROT_SEARCH else None,
                                    timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                print(f"    [UniProt] attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    time.sleep(pause)
        else:
            print("  WARNING: could not reach UniProt after all retries. "
                  "BPI UniProt set will be empty.")
            return sequences

        data = resp.json()
        for entry in data.get("results", []):
            seq = entry.get("sequence", {}).get("value", "")
            if seq:
                sequences.add(seq.upper())

        # Follow Link: <url>; rel="next" pagination header
        link_header = resp.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        url = match.group(1) if match else None
        params = None  # pagination URL already contains all query params

    return sequences

print("\n  Fetching canonical BPI sequences from UniProt …")
uniprot_bpi_seqs: set[str] = _fetch_uniprot_bpi_sequences()
print(f"  UniProt BPI sequences retrieved (length {_SEQ_LEN_MIN}–{_SEQ_LEN_MAX} aa): {len(uniprot_bpi_seqs)}")

# ---------------------------------------------------------------------------
# Step 2 – Annotate AMPDB subsets using UniProt BPI sequences
#
# For AMPDB (which has a name column) we use TWO complementary masks:
#   a) sequence-based  – exact match against the UniProt BPI set
#   b) name-based      – substring against the canonical name, since
#                         these names also contain information on the
#                         source organism
#
# We report both and take their union as the definitive BPI mask.
# ---------------------------------------------------------------------------

_BPI_CANONICAL = "Bactericidal permeability-increasing protein"

def _name_bpi_mask(series: pd.Series) -> pd.Series:
    """
    Return a boolean mask that is True when the name field contains the
    canonical BPI name as a substring (case-insensitive).  Using contains
    rather than exact equality accommodates entries whose stored name appends
    extra context (e.g. organism suffix, accession) while still excluding
    clearly unrelated proteins.
    """
    return series.str.contains(_BPI_CANONICAL, case=False, na=False)

def _seq_bpi_mask(df: pd.DataFrame) -> pd.Series:
    """Return True for rows whose sequence is in the UniProt BPI set."""
    return df["Sequence"].str.upper().isin(uniprot_bpi_seqs)

# AMPDB anomalous subset
bpi_name_mask_anom = _name_bpi_mask(df_ampdb_anom["AMPDB_name"])
bpi_seq_mask_anom  = _seq_bpi_mask(df_ampdb_anom)
bpi_mask_anom      = bpi_name_mask_anom | bpi_seq_mask_anom          # union

# AMPDB non-anomalous subset
bpi_name_mask_non_anom = _name_bpi_mask(df_ampdb_non_anom["AMPDB_name"])
bpi_seq_mask_non_anom  = _seq_bpi_mask(df_ampdb_non_anom)
bpi_mask_non_anom      = bpi_name_mask_non_anom | bpi_seq_mask_non_anom  # union

# Subsets
df_bpi_anom         = df_ampdb_anom[bpi_mask_anom]
df_non_bpi_anom     = df_ampdb_anom[~bpi_mask_anom]
df_bpi_non_anom     = df_ampdb_non_anom[bpi_mask_non_anom]
df_non_bpi_non_anom = df_ampdb_non_anom[~bpi_mask_non_anom]

# How many name-inferred BPI sequences across the full 
# AMPDB set are also present in the UniProt BPI reference set?
bpi_name_mask_all     = _name_bpi_mask(df_ampdb["AMPDB_name"])
bpi_seq_mask_all = _seq_bpi_mask(df_ampdb)
bpi_name_and_seq_mask_all = bpi_name_mask_all & bpi_seq_mask_all
n_name_also_in_uniprot = bpi_name_and_seq_mask_all.sum()
n_name_total = bpi_name_mask_all.sum()

# How many new peptides includes the UniProt BPI reference set
# to the name-inferred BPI sequences in AMPDB?

n_new_uniprot_all = (bpi_seq_mask_all & ~bpi_name_mask_all).sum()

# ---------------------------------------------------------------------------
# Step 4 – Cross-database BPI annotation via exact sequence matching
#
# All five databases (AMPDB, APD, dbAMP, DRAMP, DBAASP) are assessed by
# checking whether each sequence is present in the UniProt BPI set.
# This is the only reliable method for APD, dbAMP, and DRAMP, which have
# no name field in the dataset.
# ---------------------------------------------------------------------------

other_dbs = ["APD", "dbAMP", "DRAMP", "DBAASP"]
df_dbaasp  = dfs_abps["DBAASP"]

# All AMPDB BPI sequences (anomalous ∪ non-anomalous), for cross-db lookup
bpi_mask_all      = bpi_name_mask_all | bpi_seq_mask_all
bpi_seqs_all      = set(df_ampdb[bpi_mask_all]["Sequence"])

db_intersections: dict[str, int] = {}
for db in other_dbs:
    db_seqs = set(dfs_abps[db]["Sequence"])
    db_intersections[db] = len(bpi_seqs_all & db_seqs)

# Cross-reference: how many sequences in each db match UniProt BPI directly
db_uniprot_hits: dict[str, int] = {}
for db in amp_databases:
    db_seqs = dfs_abps[db]["Sequence"]
    db_uniprot_hits[db] = int(db_seqs.isin(uniprot_bpi_seqs).sum())

# ---------------------------------------------------------------------------
# Step 5 – DBAASP independent name-based study (complementary
# comparison; uses broad mask to enumerate BPI-related names)
# ---------------------------------------------------------------------------

dbaasp_broad_mask = (
    df_dbaasp["DBAASP_name"].str.contains(
        _BPI_CANONICAL, case=False, na=False
    ) |
    df_dbaasp["DBAASP_name"].str.contains("BPI", case=True, na=False)
)
df_dbaasp_bpi_list = df_dbaasp[dbaasp_broad_mask]["DBAASP_name"].value_counts()

# ---------------------------------------------------------------------------
# Print organised report
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  BPI INCIDENCE REPORT")
print("=" * 70)

# --- Table 1: BPI counts within AMPDB anomalous / non-anomalous subsets ---
header1 = (
    f"{'Subset':<22} {'Total':>7} {'Mean len':>9} {'BPI':>7} {'non-BPI':>9} "
    f"{'BPI %':>8} {'Mean len BPI':>14} {'Mean len non-BPI':>17}"
)
sep1 = "-" * len(header1)

print(
    f"\nTable 1 – BPI incidence within AMPDB charge-based subsets\n"
    f"  Detection: name substring match OR UniProt sequence match (union)\n"
    f"  Name match: '{_BPI_CANONICAL}' as substring (case-insensitive)\n"
    f"  Sequence match: against {len(uniprot_bpi_seqs)} UniProt BPI sequences "
    f"(length {_SEQ_LEN_MIN}–{_SEQ_LEN_MAX} aa)\n"
    f"  Name-inferred BPI sequences in AMPDB also found in UniProt set: "
    f"{n_name_also_in_uniprot} / {n_name_total}\n"
    f"  UniProt-determined BPI sequences in AMPDB not name-inferred: "
    f"{n_new_uniprot_all}"
)
print(sep1)
print(header1)
print(sep1)

for subset_label, df_sub, bpi_mask_sub in [
    ("AMPDB (Charge < -5)",  df_ampdb_anom,     bpi_mask_anom),
    ("AMPDB (Charge ≥ -5)",  df_ampdb_non_anom, bpi_mask_non_anom),
]:
    n_total   = len(df_sub)
    n_bpi     = int(bpi_mask_sub.sum())
    n_non_bpi = n_total - n_bpi
    pct_bpi   = 100 * n_bpi / n_total if n_total > 0 else float("nan")

    mean_len = df_sub["Sequence Length"].mean()
    mean_len_bpi = (
        df_sub[bpi_mask_sub]["Sequence Length"].mean()
        if n_bpi > 0 else float("nan")
    )
    mean_len_non_bpi = (
        df_sub[~bpi_mask_sub]["Sequence Length"].mean()
        if n_non_bpi > 0 else float("nan")
    )

    print(
        f"{subset_label:<22} {n_total:>7} {mean_len:>9.1f} {n_bpi:>7} {n_non_bpi:>9} "
        f"{pct_bpi:>7.1f}% {mean_len_bpi:>14.1f} {mean_len_non_bpi:>17.1f}"
    )

print(sep1)

# --- Table 2a: Cross-database intersection for all AMPDB BPI sequences ------
n_bpi_all = len(bpi_seqs_all)

header2 = f"{'Database':<10} {'Shared seqs':>12} {'% of BPI (all)':>15}"
sep2    = "-" * len(header2)

print(f"\nTable 2a – Presence of all AMPDB BPI sequences in other databases")
print(f"  (Base: {n_bpi_all} BPI sequences across the full AMPDB subset)")
print(f"  Overlap determined by exact sequence match.")
print(sep2)
print(header2)
print(sep2)

for db, n_shared in db_intersections.items():
    pct = 100 * n_shared / n_bpi_all if n_bpi_all > 0 else float("nan")
    print(f"{db:<10} {n_shared:>12} {pct:>14.1f}%")

print(sep2)

# --- Table 2b: UniProt-direct BPI hits in each database (sequence match) ---
n_uniprot_bpi = len(uniprot_bpi_seqs)

header2b = (
    f"{'Database':<18} {'UniProt BPI hits':>20} "
    f"{'% of UniProt BPI set':>23}"
)
sep2b = "-" * len(header2b)

print(
    f"\nTable 2b – BPI sequences (by direct UniProt exact-sequence match) "
    f"in each database"
)
print(
    f"  UniProt BPI reference set: {n_uniprot_bpi} sequence(s)\n"
    f"  Query: protein_name:\"{_BPI_CANONICAL}\"\n"
    f"  This covers APD, dbAMP and DRAMP, which have no name annotation."
)
print(sep2b)
print(header2b)
print(sep2b)

for db, n_hits in db_uniprot_hits.items():
    pct = 100 * n_hits / n_uniprot_bpi if n_uniprot_bpi > 0 else float("nan")
    print(f"{db:<10} {n_hits:>19} {pct:>22.1f}%")

print(sep2b)

# --- Table 3: DBAASP independent BPI name study ----------------------------
n_dbaasp_broad = int(dbaasp_broad_mask.sum())

print(f"\nTable 3 – BPI-related entries in DBAASP (name-based, all DBAASP sequences)")
print(f"  Listed: names containing '{_BPI_CANONICAL}' (case-insensitive) or 'BPI' (case-sensitive)")
print(f"  Counted as BPI: names containing '{_BPI_CANONICAL}' only")

col1_w  = max(len(n) for n in df_dbaasp_bpi_list.index) + 2
header3 = f"{'DBAASP_name':<{col1_w}} {'Count':>7}"
sep3    = "-" * len(header3)
print(sep3)
print(header3)
print(sep3)
for name, count in df_dbaasp_bpi_list.items():
    print(f"{name:<{col1_w}} {count:>7}")
print(sep3)
print(f"  Total BPI-related entries: {n_dbaasp_broad}")
print(
    "  None of these correspond to the full BPI protein; all are either\n"
    "  peptide fragments derived from BPI, BPI-related family members\n"
    "  (e.g. BPIFA2, BPI fold-containing), or LBP/BPI peptides.\n"
    "  DBAASP therefore contains no BPI entries in the strict sense."
)
print("=" * 70)


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