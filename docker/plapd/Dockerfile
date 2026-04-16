FROM condaforge/mambaforge:24.9.2-0

RUN mamba create -y -n plapd -c conda-forge \
        python=3.8 \
        pip \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/plapd/bin:$PATH"

RUN pip install --no-cache-dir \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    tqdm==4.67.3 \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    safetensors==0.5.3 \
    sentencepiece==0.2.0 \
    huggingface-hub==0.36.2 \
    filelock==3.16.1 \
    regex==2024.11.6 \
    pyyaml==6.0.3 \
    biopython==1.83 \
    fair-esm==2.0.0

COPY docker/plapd/MyTrain.py /app/plapd/MyTrain.py
COPY docker/plapd/MyModel.py /app/plapd/MyModel.py
