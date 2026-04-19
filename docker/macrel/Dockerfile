FROM condaforge/mambaforge:24.9.2-0

ENV PATH="/opt/conda/envs/env_macrel/bin:$PATH"

RUN mamba create -y -n env_macrel -c bioconda macrel \
    && mamba clean -afy
