FROM tensorflow/tensorflow:2.20.0-gpu

RUN pip install --no-cache-dir \
    torch==2.3.0 \
    transformers==4.40.0 \
    sentencepiece==0.2.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    tqdm==4.66.4 \
    matplotlib==3.9.0 \
    scikit-learn==1.5.0

WORKDIR /app/lmpred

COPY docker/lmpred/lmpred_create_embeddings.py .
COPY docker/lmpred/lmpred_train.py .
COPY docker/lmpred/lmpred_predict.py .
