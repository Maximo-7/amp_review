#!/usr/bin/env bash
# download_all.sh
# Downloads training/test datasets for all tools that support automated retrieval.
# Run from the root of the repository: bash data/raw/tools/download_all.sh
#
# NOT handled here (manual steps required):
#   - iAMP-2L  : sequences are embedded in PDF supplementary files (see README § 2.1)
#   - AGRAMP   : dataset server requires right-click download (see README § 2.14)

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# TOOLS_DIR now points to data/raw/tools/ (location of this script)

clone_sparse() {
    # clone_sparse <repo_url> <tmpdir> <commit> <sparse_path> <dest_dir>
    local repo="$1" tmpdir="$2" commit="$3" sparse="$4" dest="$5"
    mkdir -p "$dest"
    git clone --no-checkout --filter=blob:none "$repo" "$tmpdir"
    git -C "$tmpdir" sparse-checkout set "$sparse"
    git -C "$tmpdir" checkout "$commit"
    cp -r "$tmpdir/$sparse"/. "$dest/"
    rm -rf "$tmpdir"
}

checkout_files() {
    # checkout_files <repo_url> <tmpdir> <commit> <dest_dir> [file1 file2 ...]
    local repo="$1" tmpdir="$2" commit="$3" dest="$4"
    shift 4
    mkdir -p "$dest"
    git clone --no-checkout --filter=blob:none "$repo" "$tmpdir"
    git -C "$tmpdir" checkout "$commit" -- "$@"
    for f in "$@"; do
        cp "$tmpdir/$f" "$dest/"
    done
    rm -rf "$tmpdir"
}

# ----------------------------------------------------------------------------
# 2.2  AMP Scanner
# ----------------------------------------------------------------------------
echo "[2.2] AMP Scanner"
clone_sparse \
    https://github.com/dan-veltri/amp-scanner-v2.git \
    /tmp/amp-scanner-v2 \
    933052e2365631fe93098892120ee535e0ba381a \
    original-dataset \
    "$TOOLS_DIR/amp_scanner"

# ----------------------------------------------------------------------------
# 2.3  AmPEP
# ----------------------------------------------------------------------------
echo "[2.3] AmPEP"
checkout_files \
    https://github.com/ShirleyWISiu/AmPEP.git \
    /tmp/AmPEP \
    066e9c42dfebf9e08d67295b5a15493218ee194d \
    "$TOOLS_DIR/ampep" \
    M_model_train_AMP_sequence.zip \
    M_model_train_nonAMP_sequence.zip
cd "$TOOLS_DIR/ampep"
unzip M_model_train_AMP_sequence.zip
unzip M_model_train_nonAMP_sequence.zip
rm M_model_train_AMP_sequence.zip M_model_train_nonAMP_sequence.zip
cd - > /dev/null

# ----------------------------------------------------------------------------
# 2.4  Macrel
# ----------------------------------------------------------------------------
echo "[2.4] Macrel"
mkdir -p "$TOOLS_DIR/macrel"
cp "$TOOLS_DIR/ampep/M_model_train_AMP_sequence.fasta" \
   "$TOOLS_DIR/ampep/M_model_train_nonAMP_sequence.fasta" \
   "$TOOLS_DIR/macrel/"
wget -q -O "$TOOLS_DIR/macrel/hemo.training.pos.faa" \
    'https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/main/pos.fa'
wget -q -O "$TOOLS_DIR/macrel/hemo.training.neg.faa" \
    'https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/main/neg.fa'
wget -q -O "$TOOLS_DIR/macrel/hemo.validation.pos.faa" \
    'https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/validation/pos.fa'
wget -q -O "$TOOLS_DIR/macrel/hemo.validation.neg.faa" \
    'https://webs.iiitd.edu.in/raghava/hemopi/data/HemoPI_1_dataset/validation/neg.fa'
echo "    Run build-AMP-training.py inside data/raw/tools/macrel/ with the Macrel conda environment to complete preprocessing."

# ----------------------------------------------------------------------------
# 2.5  amPEPpy
# ----------------------------------------------------------------------------
echo "[2.5] amPEPpy"
checkout_files \
    https://github.com/tlawrence3/amPEPpy.git \
    /tmp/amPEPpy \
    85aab3428b328d9fe4744052258746d8f4ba7bf6 \
    "$TOOLS_DIR/ampeppy" \
    training_data/M_model_train_AMP_sequence.numbered.fasta \
    training_data/M_model_train_nonAMP_sequence.numbered.proplen.subsample.fasta

# ----------------------------------------------------------------------------
# 2.6  LMPred
# ----------------------------------------------------------------------------
echo "[2.6] LMPred"
clone_sparse \
    https://github.com/williamdee1/LMPred_AMP_Prediction.git \
    /tmp/LMPred \
    30c7188ea5bf8e699eebfbb08cd1ae9aef1094e9 \
    LM_Pred_Dataset \
    "$TOOLS_DIR/lmpred"

# ----------------------------------------------------------------------------
# 2.7  AMPlify
# ----------------------------------------------------------------------------
echo "[2.7] AMPlify"
checkout_files \
    https://github.com/BirolLab/AMPlify.git \
    /tmp/AMPlify \
    3a07713c25b8a21ef66d31d10e121989d26d9320 \
    "$TOOLS_DIR/amplify" \
    data/AMPlify_AMP_train_common.fa \
    data/AMPlify_AMP_test_common.fa \
    data/AMPlify_non_AMP_train_balanced.fa \
    data/AMPlify_non_AMP_test_balanced.fa

# ----------------------------------------------------------------------------
# 2.8  Ma et al. (2022)
# ----------------------------------------------------------------------------
echo "[2.8] Ma et al. (2022)"
checkout_files \
    https://github.com/mayuefine/c_AMPs-prediction.git \
    /tmp/c_AMPs-prediction \
    cf7658bc5d504ba6d996fa7b152270e38275dc46 \
    "$TOOLS_DIR/ma_et_al" \
    Data/AMPs.fa \
    Data/Non-AMPs.fa

# ----------------------------------------------------------------------------
# 2.10 AMP-BERT
# ----------------------------------------------------------------------------
echo "[2.10] AMP-BERT"
checkout_files \
    https://github.com/GIST-CSBL/AMP-BERT.git \
    /tmp/AMP-BERT \
    b9ba228180b6edc04f39cac4724281af5f031db5 \
    "$TOOLS_DIR/amp_bert" \
    non_amp_ampep_cdhit90.csv \
    veltri_dramp_cdhit_90.csv \
    all_veltri.csv

# ----------------------------------------------------------------------------
# 2.11 AMPFinder
# ----------------------------------------------------------------------------
echo "[2.11] AMPFinder"
checkout_files \
    https://github.com/abcair/AMPFinder.git \
    /tmp/AMPFinder \
    666b173d62c59627ac37ea7508467ee82cd9ec67 \
    "$TOOLS_DIR/ampfinder" \
    data/D1/3594-Samp.fasta \
    data/D1/3925-Snonamp.fasta

# ----------------------------------------------------------------------------
# 2.12 AMP-RNNpro
# ----------------------------------------------------------------------------
echo "[2.12] AMP-RNNpro"
checkout_files \
    https://github.com/Shazzad-Shaon3404/Antimicrobials_.git \
    /tmp/Antimicrobials_ \
    a2946913193422cc3fb05d7578eb37d09f646884 \
    "$TOOLS_DIR/amp_rnnpro" \
    train_p.fasta \
    trainn_n.fasta \
    testp \
    testn \
    README.md

# ----------------------------------------------------------------------------
# 2.15 PepNet
# ----------------------------------------------------------------------------
echo "[2.15] PepNet"
mkdir -p "$TOOLS_DIR/pepnet"
wget -q -O /tmp/pepnet_datasets.tar.gz \
    'https://zenodo.org/records/13223516/files/datasets.tar.gz?download=1'
tar -xzf /tmp/pepnet_datasets.tar.gz -C "$TOOLS_DIR/pepnet/"
mv "$TOOLS_DIR/pepnet/datasets/"* "$TOOLS_DIR/pepnet/"
rmdir "$TOOLS_DIR/pepnet/datasets/"
rm /tmp/pepnet_datasets.tar.gz

# ----------------------------------------------------------------------------
# 2.16 KT-AMPpred
# ----------------------------------------------------------------------------
echo "[2.16] KT-AMPpred"
clone_sparse \
    https://github.com/liangxiaodata/AMPpred.git \
    /tmp/AMPpred \
    b4a7276e73b212b1be98256e0c9aa236caa20540 \
    data \
    "$TOOLS_DIR/kt_amppred"

# ----------------------------------------------------------------------------
# 2.17 PLAPD
# ----------------------------------------------------------------------------
echo "[2.17] PLAPD"
checkout_files \
    https://github.com/lichaozhang2/PLAPD.git \
    /tmp/PLAPD \
    5f5c6ef4b21adc9e297d240e41bc1e2ca065c54b \
    "$TOOLS_DIR/plapd" \
    data/datasets/AMP/training_data.csv \
    data/datasets/AMP/val_data.csv

# ----------------------------------------------------------------------------
# 2.18 DLFea4AMPGen
# ----------------------------------------------------------------------------
echo "[2.18] DLFea4AMPGen"
clone_sparse \
    https://github.com/hgao12345/DLFea4AMPGen.git \
    /tmp/DLFea4AMPGen \
    6ec4a46a206f2501e0c29abf453ae6e0ddb5227e \
    Dataset \
    "$TOOLS_DIR/dlfea4ampgen"

# ----------------------------------------------------------------------------
# 2.19 MultiAMP
# ----------------------------------------------------------------------------
echo "[2.19] MultiAMP"
mkdir -p "$TOOLS_DIR/multiamp"
wget -q -O /tmp/multiamp_data.tar.gz \
    'https://huggingface.co/jiayi11/multi_amp/resolve/7c2b1b86304b62e9d0ff4d186c6be6ca82e02e28/data.tar.gz'
tar -xzf /tmp/multiamp_data.tar.gz -C "$TOOLS_DIR/multiamp/"
rm /tmp/multiamp_data.tar.gz

echo ""
echo "Done. Remember to run build-AMP-training.py manually for Macrel (§ 2.4)."
