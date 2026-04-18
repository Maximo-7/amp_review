FROM python:3.9-bullseye

RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    fair-esm==2.0.0 \
    biopython==1.85 \
    torch-geometric==2.6.1 \
    pytorch-crf==0.7.2 \
    scikit-learn==1.6.1 \
    pandas==2.3.3 \
    tqdm==4.67.3 \
    numpy==2.0.2 \
    networkx==3.2.1 \
    scipy==1.13.1

RUN git clone https://github.com/jiayili11/multi-amp.git /app/multiamp \
    && cd /app/multiamp \
    && git checkout 1a7625f8d7f2dad282e8f0414aa80aa64649cbeb

COPY models/multiamp/best_model_overall.pth /app/multiamp/checkpoints/best_model_overall.pth
