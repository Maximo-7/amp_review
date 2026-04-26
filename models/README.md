# Models utilized for predictions (incomplete)

This folder contains model files not cloned via Git in the Docker images utilized in this review. These files are already included in their respective Docker image developped for the evaluation pipeline. This documentation is only included for end-to-end reproducibility.
- (*) Models trained for this review.
- (+) Models downloaded from external sources (GitHub, Dropbox, Zenodo or HuggingFace).

```
models/
├── amp_bert/                                           *
│   ├── config.json
│   ├── model.safetensors
│   └── training_args.bin
├── ampfinder/                                          +
│   └── AMPFinder.identify.rf
├── dlfea4ampgen/                                       +
│   └── ABP_Best_Model.ckpt
├── kt_amppred/                                         *
│   ├── classifier_weights.pth
│   ├── config.json
│   └── pytorch_model.bin
├── lmpred/                                             *
│   └── T5XL_UNI_best_model.epoch06-loss0.28.keras
├── ma_et_al/                                           +
│   └── bert.bin
├── multiamp/                                           +
│   └── best_model_overall.pth
├── pepnet/                                             +
│   ├── checkpoints/
│   │   └── 2024_03_27_19_58_59_951/
│   │       └── model/
│   │           └── model_final.pth
│   └── properties.pkl
├── plapd/                                              *
│   └── my_best_model_without_embedding.pth
├── prot_t5_xl_half_uniref50-enc/                       +
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── spiece.model
│   └── tokenizer_config.json
├── pyampa/                                             +
│   ├── amp_validate_vectorizer.pkl
│   └── AMPValidate.pkl
└── README.md
```

## Trained models (*)

These models were trained as part of this review because the original authors
did not publish pre-trained weights.

### AMP-BERT (`models/amp_bert/`)
Fine-tuned `BertForSequenceClassification` on the `prot_bert_bfd` backbone,
trained on `all_veltri.csv` (available in the AMP-BERT repository).
Training details: 15 epochs, lr=5e-5, batch size=1, gradient accumulation=64,
fp16, weight decay=0.1, seed=0.

To retrain from scratch, set `params.amp_bert_skip_training = false` in
`nextflow.config` and run the pipeline.

### LMPred (`models/lmpred/`)
Two-layer CNN trained on T5XL-UNI embeddings of the LMPred dataset.
Training details: see `modules/lmpred_training.nf` and `docker/lmpred/lmpred_train.py`.

To retrain from scratch, set `params.lmpred_skip_training = false` in
`nextflow.config` and run the pipeline.

## Downloaded models (+)

### Ma et al. (2022) BERT model (`models/ma_et_al/bert.bin`)
Source: https://www.dropbox.com/sh/o58xdznyi6ulyc6/AABLckEnxP54j2X7BrGybhyea?dl=0

```bash
# 1. Download the ZIP from Dropbox (alternative: download manually)
wget -O models/ma_et_al/bert.zip \
    "https://www.dropbox.com/sh/o58xdznyi6ulyc6/AABLckEnxP54j2X7BrGybhyea?dl=1"
# 2. Unzip
unzip models/ma_et_al/bert.zip -d models/ma_et_al/
rm models/ma_et_al/bert.zip
# 3. Verify integrity (hash provided by the authors)
echo "990d14de053d8080fcca33d712d647b6  models/ma_et_al/bert.bin" | md5sum -c -
```

### AMPFinder random forest model (`models/ampfinder/AMPFinder.identify.rf`)
First model of the AMPFinder pipeline, identifies AMP/non-AMP.
Source: https://github.com/abcair/AMPFinder

```bash
# Download the ZIP and extract
wget -O models/ampfinder/AMPFinder.identify.zip \
    "https://github.com/abcair/AMPFinder/raw/main/qt5/model/AMPFinder.identify.zip"
unzip models/ampfinder/AMPFinder.identify.zip -d models/ampfinder/
rm models/ampfinder/AMPFinder.identify.zip
```

### PepNet trained model and properties dictionary (`models/pepnet/`)
Trained AMP classification model from the PepNet authors and dictionary
for physicochemical characterization of amino acids during preprocessing.
Source: https://zenodo.org/records/11363310

Download and extract the Zenodo archive, then copy the `AMP/checkpoints/` folder
and `properties.pkl` to `models/pepnet/checkpoints/`. Only the standard mode
checkpoint is required:

```bash
# After downloading and extracting the Zenodo archive:
cp -r /path/to/zenodo/AMP/checkpoints/2024_03_27_19_58_59_951 \
    models/pepnet/checkpoints/
cp /path/to/zenodo/properties.pkl \
    models/pepnet
```

### ProtT5-XL-UniRef50 half-precision encoder (`models/prot_t5_xl_half_uniref50-enc/`)
Encoder-only, half-precision version of ProtT5-XL-UniRef50.
Used by PepNet for feature extraction.
Source: https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc

```bash
# Download using huggingface_hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Rostlab/prot_t5_xl_half_uniref50-enc',
    local_dir='models/prot_t5_xl_half_uniref50-enc'
)
"
```

### PyAMPA AMPValidate model and vectorizer (`models/pyampa/`)
AMPValidate originally classifies peptidic chains extracted from larger proteins.
In this review it has been adapted to predict directly on raw peptides.
Source: https://github.com/SysBioUAB/PyAMPA

Download manually to `models/pyampa/`:
- https://github.com/SysBioUAB/PyAMPA/blob/main/AMPValidate.pkl
- https://github.com/SysBioUAB/PyAMPA/blob/main/amp_validate_vectorizer.pkl
