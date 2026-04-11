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

COPY tools/AMPFinder/qt5/model/AMPFinder.identify.zip \
     /app/ampfinder/qt5/model/AMPFinder.identify.zip

COPY docker/ampfinder/run_cli_maximo.py /app/ampfinder/run_cli_maximo.py

# Install unzip
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

RUN unzip /app/ampfinder/qt5/model/AMPFinder.identify.zip \
    -d /app/ampfinder/qt5/model \
    && rm /app/ampfinder/qt5/model/AMPFinder.identify.zip
