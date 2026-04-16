FROM condaforge/mambaforge:24.9.2-0

RUN mamba create -y -n ma_et_al_att_lstm -c conda-forge \
        python=3.6 \
        pip \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/ma_et_al_att_lstm/bin:$PATH"

RUN pip install --no-cache-dir \
    tensorflow==1.14.0 \
    keras==2.2.4 \
    numpy==1.16.2 \
    h5py==2.9.0

# GitHub repository
RUN git clone https://github.com/mayuefine/c_AMPs-prediction.git /app/ma_et_al \
    && cd /app/ma_et_al \
    && git checkout cf7658bc5d504ba6d996fa7b152270e38275dc46
COPY docker/ma_et_al_att_lstm/result_maximo.pl /app/ma_et_al/script/result_maximo.pl

WORKDIR /app/ma_et_al

# Fix: format.pl and Attention.py indentation
RUN sed -i '2,62s/^   //' /app/ma_et_al/script/format.pl \
    && sed -i '2,54s/^   //' /app/ma_et_al/script/Attention.py

# Adjust models routes (and correct the typo 'Moldes' for lstm)
RUN sed -i 's|../Models/att.h5|/app/ma_et_al/Models/att.h5|' \
    script/prediction_attention.py \
    && sed -i 's|../Moldes/lstm.h5|/app/ma_et_al/Models/lstm.h5|' \
    script/prediction_lstm.py
