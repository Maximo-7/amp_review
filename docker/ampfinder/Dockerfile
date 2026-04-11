FROM condaforge/mambaforge:24.9.2-0

ENV DEBIAN_FRONTEND=noninteractive

# Install libGL
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN mamba create -y -n AMPFinder -c conda-forge \
        python=3.8 \
        biopython=1.83 \
        numpy=1.24.4 \
        pip \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/AMPFinder/bin:$PATH"

RUN pip install --no-cache-dir \
    pyside2==5.15.2.1 \
    qt-material==2.14 \
    protobuf==3.20.0 \
    tensorflow==2.8.0 \
    transformers==4.46.3 \
    datasets==3.1.0 \
    pandas==2.0.3 \
    scikit-learn==1.1.2 \
    scipy==1.10.1 \
    matplotlib==3.7.5 \
    xgboost==2.1.4 \
    tqdm==4.67.3 \
    blosum==2.2.0 \
    propy3==1.1.1 \
    protlearn==0.0.3 \
    sentencepiece==0.2.0

# PyTorch for CPU, recommended by the authors
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    torchaudio==2.4.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY tools/AMPFinder /app/ampfinder

COPY docker/ampfinder/run_cli_maximo.py /app/ampfinder/qt5/run_cli_maximo.py

# Unzip models files
WORKDIR /app/ampfinder/qt5/model

# Install unzip
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

RUN unzip AMPFinder.fun.deep.model.zip \
    && rm AMPFinder.fun.deep.model.zip \
    && unzip AMPFinder.identify.zip \
    && rm AMPFinder.identify.zip

WORKDIR /app/ampfinder/qt5

# Fix syntax error (depends on the blosum package version)
RUN sed -i 's/mat\[x+y\]/mat[x][y]/' \
    main_windows.py

# Adjust models routes
RUN sed -i 's|./model/AMPFinder.identify.rf|/app/ampfinder/qt5/model/AMPFinder.identify.rf|' \
    main_windows.py \
    && sed -i 's|./model/AMPFinder.fun.deep.model|/app/ampfinder/qt5/model/AMPFinder.fun.deep.model|' \
    main_windows.py