FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        libhdf5-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.4.0+cu118 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir \
    h5py==3.11.0 \
    transformers==4.43.2 \
    sentencepiece==0.2.0 \
    prettytable==3.10.2 \
    scikit-learn==1.3.2 \
    torchnet==0.0.4 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    tqdm==4.67.3

# PepNet repository
RUN git clone https://github.com/hjy23/PepNet.git /app/pepnet \
    && cd /app/pepnet \
    && git checkout 77add79f573e5f541bbe7125e03ebdfb222f0a43

# ProtTrans embedder script
# Source: https://github.com/agemagician/ProtTrans
COPY docker/pepnet/prott5_embedder.py /app/prottrans/prott5_embedder.py

# ProtT5-XL-UniRef50 half-precision encoder (downloaded from HuggingFace)
COPY models/prot_t5_xl_half_uniref50-enc /app/prottrans/prot_t5_xl_half_uniref50-enc

# PepNet trained model
COPY models/pepnet/checkpoints /app/pepnet/datasets/AMP/checkpoints

# PepNet properties dictionary
COPY models/pepnet/properties.pkl /app/pepnet/datasets/properties.pkl

# Adjust routes
RUN sed -i 's|../datasets/properties.pkl|/app/pepnet/datasets/properties.pkl|' \
    /app/pepnet/script/preprocess.py

RUN sed -i 's|../datasets|/app/pepnet/datasets|g' \
    /app/pepnet/script/predict.py
