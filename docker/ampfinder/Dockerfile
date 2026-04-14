FROM condaforge/mambaforge:24.9.2-0

RUN mamba create -y -n AMPFinder -c conda-forge \
        python=3.8 \
        biopython=1.83 \
        numpy=1.24.4 \
        pip \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/AMPFinder/bin:$PATH"

RUN pip install --no-cache-dir \
    pandas==2.0.3 \
    scikit-learn==1.1.2 \
    scipy==1.10.1 \
    blosum==2.2.0 \
    propy3==1.1.1 \
    protlearn==0.0.3 \
    joblib==1.3.2

COPY models/ampfinder/AMPFinder.identify.rf \
     /app/ampfinder/models/ampfinder/AMPFinder.identify.rf

COPY docker/ampfinder/run_cli_maximo.py /app/ampfinder/run_cli_maximo.py
