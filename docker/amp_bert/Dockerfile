FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
        python3-pip \
        python3-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.6.0 \
    transformers==4.57.1 \
    sentencepiece==0.2.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    tqdm==4.66.4 \
    scikit-learn==1.5.0 \
    accelerate==0.30.0

WORKDIR /app/amp_bert

COPY docker/amp_bert/amp_bert_predict.py ./amp_bert_predict.py
COPY docker/amp_bert/amp_bert_train.py ./amp_bert_train.py
