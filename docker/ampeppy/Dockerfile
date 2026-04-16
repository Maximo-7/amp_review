FROM condaforge/mambaforge:24.9.2-0

RUN mamba create -y -n amPEP -c conda-forge python=3.8 pandas numpy biopython scikit-learn \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/amPEP/bin:$PATH"
WORKDIR /app/ampeppy

# amPEPpy repository
RUN git clone https://github.com/tlawrence3/amPEPpy.git . \
    && git checkout 85aab3428b328d9fe4744052258746d8f4ba7bf6

RUN /opt/conda/envs/amPEP/bin/python setup.py install