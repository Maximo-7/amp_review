FROM condaforge/mambaforge:24.9.2-0

RUN mamba create -y -n kt_amppred -c conda-forge \
        python=3.7 \
        pip \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/kt_amppred/bin:$PATH"

# PyTorch 1.11.0 with CUDA 11.3
RUN pip install --no-cache-dir \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    torchaudio==0.11.0+cu113 \
    --index-url https://download.pytorch.org/whl/cu113

RUN pip install --no-cache-dir \
    biopython==1.81 \
    numpy==1.21.6 \
    pandas==1.3.5 \
    transformers==4.27.3 \
    sentencepiece==0.1.99

COPY models/prot_t5_xl_half_uniref50-enc /app/kt_amppred/prot_t5_xl_half_uniref50-enc
COPY docker/kt_amppred/main_amp.py /app/kt_amppred/main_amp.py
COPY docker/kt_amppred/kt_amppred_predict.py /app/kt_amppred/kt_amppred_predict.py
