"""
overlap_analysis.py
====================
Analyses the overlap between AMP source databases (APD, DRAMP, dbAMP,
DBAASP, AMPDB) for the standard ABPs in the complete dataset, producing
a three-panel Venn diagram distinguishing the aggregated set and the
natural and synthetic subpopulations.

Natural/synthetic membership for APD, DRAMP and DBAASP is read from
per-database FASTA files; for dbAMP it is read from a CSV and filtered on
the "Source" column, where only synthetic membership is provided.
AMPDB has no natural/synthetic classification and is therefore
excluded from the natural and synthetic panels.

Expected project layout (paths relative to project root):
    data/processed/complete_dataset.csv                    -- output of annotate_dataset.py
    data/raw/natural_and_synthetic_amps/
        apd_natural.fasta, apd_synthetic.fasta
        dramp_natural.fasta, dramp_synthetic.fasta
        dbamp_all.csv
        dbaasp_ribosomal_1.fasta, dbaasp_ribosomal_2.fasta, dbaasp_non_ribosomal.fasta
        dbaasp_synthetic_*.fasta

Output figures (PNG + PDF) are written to:
    results/sequences_analysis/
      overlap_abps/

Run from the project root:
    python scripts/overlap_analysis.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
import pandas as pd
from venn import venn as venn_diagram
from venn import generate_colors


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
NAT_AND_SYN = ROOT / "data" / "raw" / "natural_and_synthetic_amps"
PROCESSED   = ROOT / "data" / "processed"
OUT_DIR     = ROOT / "results" / "sequences_analysis"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fasta_to_df_utf_8(fasta_file):
    """Parse a UTF-8 FASTA file into a DataFrame with ID, Name and Sequence
    columns.

    Each record's header (the text following '>') is split on the first
    space or '|' into an ID and an optional Name; multi-line sequences are
    joined. Records with no sequence (metadata-only or trailing empty
    blocks) are dropped.
    """
    records = []
    text = Path(fasta_file).read_text(encoding="utf-8")
    # split only on '>' that starts a line, so '>' inside names is left alone
    blocks = re.split(r"(?m)^>", text)
    for record in blocks:
        if not record.strip():
            continue
        lines = record.split("\n")
        header = lines[0].strip()
        parts = re.split(r"[ |]", header, maxsplit=1)
        seq_id = parts[0]
        name = parts[1].strip() if len(parts) > 1 else ""
        sequence = "\n".join(lines[1:]).strip()
        if not sequence:          # drop metadata-only / trailing empty blocks
            continue
        records.append({"ID": seq_id, "Name": name, "Sequence": sequence})
    return pd.DataFrame(records)

def strip_upper_unique_by_sequence(dataframe):
    """
    Strips whitespace, uppercases sequences and drops duplicate sequences.
    Returns a reset-indexed copy.
    """
    df = dataframe.copy()
    df["Sequence"] = df["Sequence"].str.strip().str.upper()
    df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
    return df


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

print("=== Loading data ===")

complete_dataset = pd.read_csv(
    PROCESSED / "complete_dataset.csv", low_memory=False
)

# -- APD --
df_apd_natural = strip_upper_unique_by_sequence(
    fasta_to_df_utf_8(NAT_AND_SYN / "apd_natural.fasta")
).drop(columns=["Name"])  # Name is empty

df_apd_synthetic = strip_upper_unique_by_sequence(
    fasta_to_df_utf_8(NAT_AND_SYN / "apd_synthetic.fasta")
).drop(columns=["Name"])  # Name is empty

# -- DRAMP --
df_dramp_natural = strip_upper_unique_by_sequence(
    fasta_to_df_utf_8(NAT_AND_SYN / "dramp_natural.fasta")
).drop(columns=["Name"])  # Name is empty

df_dramp_synthetic = strip_upper_unique_by_sequence(
    fasta_to_df_utf_8(NAT_AND_SYN / "dramp_synthetic.fasta")
).drop(columns=["Name"])  # Name is empty

# -- dbAMP --
df_dbamp_all = pd.read_csv(NAT_AND_SYN / "dbamp_all.csv", sep=";", usecols=["Source", "Seq"])
df_dbamp_synthetic = df_dbamp_all[df_dbamp_all["Source"].str.contains("synthetic|newly-designed", case=False, na=False)]
df_dbamp_synthetic = strip_upper_unique_by_sequence(
    df_dbamp_synthetic.rename(columns={"Seq": "Sequence"})
)

# -- DBAASP --
dbaasp_natural_files = [
    NAT_AND_SYN / "dbaasp_ribosomal_1.fasta",
    NAT_AND_SYN / "dbaasp_ribosomal_2.fasta",
    NAT_AND_SYN / "dbaasp_non_ribosomal.fasta",
]
dbaasp_synthetic_files = sorted(NAT_AND_SYN.glob("dbaasp_synthetic_*.fasta"))

df_dbaasp_natural = strip_upper_unique_by_sequence(
    pd.concat([fasta_to_df_utf_8(f) for f in dbaasp_natural_files], ignore_index=True)
)
df_dbaasp_synthetic = strip_upper_unique_by_sequence(
    pd.concat([fasta_to_df_utf_8(f) for f in dbaasp_synthetic_files], ignore_index=True)
)

# ---------------------------------------------------------------------------
# Characterize subsets
# ---------------------------------------------------------------------------

amp_databases = ["APD", "DRAMP", "dbAMP", "DBAASP", "AMPDB"]

df_standard_abps = complete_dataset[
    complete_dataset["ABP_from_databases"] &
    complete_dataset["Standard_sequence"]
].copy()

venn_sets = {
    db: set(df_standard_abps[df_standard_abps[f"{db}_ID"].notna()]["Sequence"])
    for db in amp_databases
}

# Natural panel: 3 ellipses (AMPDB and dbAMP have no natural classification)
venn_sets_natural = {
    "APD":    venn_sets["APD"]    & set(df_apd_natural["Sequence"]),
    "DRAMP":  venn_sets["DRAMP"]  & set(df_dramp_natural["Sequence"]),
    "DBAASP": venn_sets["DBAASP"] & set(df_dbaasp_natural["Sequence"]),
}

# Synthetic panel: 4 ellipses (AMPDB has no synthetic classification)
venn_sets_synthetic = {
    "APD":    venn_sets["APD"]    & set(df_apd_synthetic["Sequence"]),
    "DRAMP":  venn_sets["DRAMP"]  & set(df_dramp_synthetic["Sequence"]),
    "dbAMP":  venn_sets["dbAMP"]  & set(df_dbamp_synthetic["Sequence"]),
    "DBAASP": venn_sets["DBAASP"] & set(df_dbaasp_synthetic["Sequence"]),
}

# ---------------------------------------------------------------------------
# Figure – A) Venn diagram of database overlap, B) natural ABPs, C) synthetic ABPs
# ---------------------------------------------------------------------------

print("\n=== Figure: Venn diagrams for all / natural / synthetic ABPs ===")

FONTSIZE        = 16   # petal numbers
LABEL_FONTSIZE  = 26   # panel letters A/B/C
LEGEND_FONTSIZE = 18   # shared legend

master_colors = dict(zip(amp_databases, generate_colors(n_colors=len(amp_databases))))

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

panels = [
    (venn_sets,           axes[0]),
    (venn_sets_natural,   axes[1]),
    (venn_sets_synthetic, axes[2]),
]

for dataset_dict, ax in panels:
    colors = [master_colors[db] for db in dataset_dict.keys()]
    venn_diagram(
        dataset_dict, ax=ax, cmap=colors,
        fontsize=FONTSIZE, legend_loc=None,
    )
    # crop tightly around the drawn shapes to remove blank margins
    xs, ys = [], []
    for patch in ax.patches:
        ext = patch.get_path().get_extents(transform=patch.get_patch_transform())
        xs.extend([ext.x0, ext.x1])
        ys.extend([ext.y0, ext.y1])
    pad = 0.03
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

legend_handles = [
    Patch(facecolor=to_rgba(master_colors[db], alpha=0.7), edgecolor="black", label=db)
    for db in amp_databases
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(amp_databases),
    bbox_to_anchor=(0.5, -0.02),
    fontsize=LEGEND_FONTSIZE,
    frameon=False,
)

fig.subplots_adjust(wspace=0.05, bottom=0.12, left=0.01, right=0.99, top=0.90)

# Finalize axes positions (aspect='equal' shrinks each box differently) before reading them
fig.canvas.draw()

# Shared y for all letters (figure coords); x = each panel's own left edge
LETTER_Y = 0.90
for ax, letter in zip(axes, "ABC"):
    pos = ax.get_position()
    fig.text(pos.x0, LETTER_Y, letter, fontweight="bold", fontsize=LABEL_FONTSIZE,
              ha="left", va="bottom")

save_figure(fig, "overlap_abps", "venn_all_natural_synthetic")
plt.close(fig)
