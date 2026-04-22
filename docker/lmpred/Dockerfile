FROM tensorflow/tensorflow:2.16.1-gpu

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y libnccl2 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --no-deps \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

RUN pip install --no-cache-dir \
    filelock typing-extensions sympy networkx jinja2 fsspec \
    transformers==4.40.0 sentencepiece==0.2.0 \
    pandas==2.2.2 numpy==1.26.4 tqdm==4.66.4 \
    matplotlib==3.9.0 scikit-learn==1.5.0

WORKDIR /app/lmpred
COPY models/lmpred /app/lmpred/model
COPY docker/lmpred/lmpred_create_embeddings.py .
COPY docker/lmpred/lmpred_train.py .
COPY docker/lmpred/lmpred_predict.py .
