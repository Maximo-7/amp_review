FROM condaforge/mambaforge:24.9.2-0

RUN mamba create -y -n amPEP -c conda-forge python=3.8 pandas numpy biopython scikit-learn \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/amPEP/bin:$PATH"
WORKDIR /app/amPEPpy

COPY tools/amPEPpy /app/amPEPpy

RUN /opt/conda/envs/amPEP/bin/python setup.py install