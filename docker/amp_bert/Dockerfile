FROM tensorflow/tensorflow:2.20.0-gpu

RUN pip install --no-cache-dir \
    torch==2.3.0 \
    transformers==4.57.1 \
    sentencepiece==0.2.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    tqdm==4.66.4 \
    scikit-learn==1.5.0

WORKDIR /app/amp_bert

COPY docker/amp_bert/amp_bert_predict.py ./amp_bert_predict.py
COPY docker/amp_bert/amp_bert_train.py ./amp_bert_train.py
