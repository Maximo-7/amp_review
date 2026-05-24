#!/usr/bin/env bash
# download_models.sh
# Downloads model files for all tools that support automated retrieval.
# Run from the root of the repository: bash scripts/download_models.sh
#
# NOT handled here (manual steps required):
#   - pepnet    : model files are extracted from the tool dataset archive
#                 already downloaded by download_tools_datasets.sh (see README § Model Acquisition)
#   - prot_t5   : requires huggingface_hub Python package (see README § Model Acquisition)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/modelos"

# ----------------------------------------------------------------------------
# AMPFinder random forest model
# ----------------------------------------------------------------------------
echo "[ampfinder] AMPFinder random forest"
mkdir -p "$MODELS_DIR/ampfinder"
wget -q -O /tmp/AMPFinder.identify.zip \
    'https://github.com/abcair/AMPFinder/raw/main/qt5/model/AMPFinder.identify.zip'
unzip -q /tmp/AMPFinder.identify.zip -d "$MODELS_DIR/ampfinder/"
rm /tmp/AMPFinder.identify.zip

# ----------------------------------------------------------------------------
# Ma et al. (2022) BERT model
# ----------------------------------------------------------------------------
echo "[ma_et_al] Ma et al. (2022) BERT model"
mkdir -p "$MODELS_DIR/ma_et_al"
wget -q -O "$MODELS_DIR/ma_et_al/bert.zip" \
    'https://www.dropbox.com/sh/o58xdznyi6ulyc6/AABLckEnxP54j2X7BrGybhyea?dl=1'
unzip -q -o "$MODELS_DIR/ma_et_al/bert.zip" -d "$MODELS_DIR/ma_et_al/" || true
rm "$MODELS_DIR/ma_et_al/bert.zip"
echo "    Verifying integrity..."
echo "990d14de053d8080fcca33d712d647b6  $MODELS_DIR/ma_et_al/bert.bin" | md5sum -c -

# ----------------------------------------------------------------------------
# PyAMPA AMPValidate model and vectorizer
# ----------------------------------------------------------------------------
echo "[pyampa] PyAMPA AMPValidate"
mkdir -p "$MODELS_DIR/pyampa"
wget -q -O "$MODELS_DIR/pyampa/AMPValidate.pkl" \
    'https://github.com/SysBioUAB/PyAMPA/raw/main/AMPValidate.pkl'
wget -q -O "$MODELS_DIR/pyampa/amp_validate_vectorizer.pkl" \
    'https://github.com/SysBioUAB/PyAMPA/raw/main/amp_validate_vectorizer.pkl'

# ----------------------------------------------------------------------------
# MultiAMP sequence-only model
# ----------------------------------------------------------------------------
echo "[multiamp] MultiAMP"
mkdir -p "$MODELS_DIR/multiamp"
wget -q -O "$MODELS_DIR/multiamp/best_model_overall.pth" \
    'https://huggingface.co/jiayi11/multi_amp/resolve/main/checkpoints/best_model_overall.pth'

# ----------------------------------------------------------------------------
# DLFea4AMPGen checkpoint  (downloaded as ABP_Model.ckpt, renamed on the fly)
# ----------------------------------------------------------------------------
echo "[dlfea4ampgen] DLFea4AMPGen"
mkdir -p "$MODELS_DIR/dlfea4ampgen"
wget -q -O "$MODELS_DIR/dlfea4ampgen/ABP_Best_Model.ckpt" \
    'https://zenodo.org/records/16545412/files/ABP_Model.ckpt?download=1'

echo ""
echo "Done."
echo ""
echo "Still required (manual steps — see README § Model Acquisition):"
echo "  - models/pepnet/        : copy from data/raw/tools/pepnet/ (after running download_tools_datasets.sh)"
echo "  - models/prot_t5_xl_*/  : huggingface_hub snapshot_download"
