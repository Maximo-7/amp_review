FROM condaforge/mambaforge:24.9.2-0

RUN mamba create -y -n amplify -c bioconda python=3.6 amplify \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/amplify/bin:$PATH"