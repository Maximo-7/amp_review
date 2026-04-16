FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==3.0.1 \
    biopython==1.86 \
    scikit-learn==1.2.2

COPY models/pyampa/AMPValidate.pkl /app/pyampa/models/pyampa/AMPValidate.pkl
COPY models/pyampa/amp_validate_vectorizer.pkl /app/pyampa/models/pyampa/amp_validate_vectorizer.pkl

COPY docker/pyampa/ampvalidate_predict.py /app/pyampa/ampvalidate_predict.py