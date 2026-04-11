FROM condaforge/mambaforge:24.9.2-0

RUN mamba create -y -n ma_et_al_bert -c conda-forge \
        python=3.6 \
        pip \
    && mamba clean -afy

ENV PATH="/opt/conda/envs/ma_et_al_bert/bin:$PATH"

COPY tools/c_AMPs-prediction /app/ma_et_al
COPY models/ma_et_al/bert.bin /app/ma_et_al/Models/bert.bin

WORKDIR /app/ma_et_al

RUN pip install --no-cache-dir \
    tensorflow==1.14.0 \
    numpy==1.18.1 \
    pandas==1.0.2 \
    scikit-learn==0.22.1 \
    torch==1.10.2 \
    pytorch-pretrained-bert==0.6.1 \
    boto3 requests regex tqdm

# Install bert_sklearn 0.2.0 from authors directly
RUN cp -r /app/ma_et_al/bert_sklearn \
    /opt/conda/envs/ma_et_al_bert/lib/python3.6/site-packages/bert_sklearn

# Fix: disable GPU
RUN sed -i 's/os.environ\["CUDA_VISIBLE_DEVICES"\] = "1"/os.environ["CUDA_VISIBLE_DEVICES"] = ""/' \
    script/prediction_bert.py

# Fix: force CPU for bert model
RUN sed -i '/model = load_model("..\/Models\/bert.bin")/a\model.device = __import__("torch").device("cpu")' \
    script/prediction_bert.py

# Adjust model route
RUN sed -i 's|model = load_model("../Models/bert.bin")|model = load_model("/app/ma_et_al/Models/bert.bin")|' \
    script/prediction_bert.py

# Fix: map_location in bert_sklearn
RUN sed -i \
    's/state = torch.load(restore_file)/state = torch.load(restore_file, map_location=torch.device("cpu"))/' \
    /opt/conda/envs/ma_et_al_bert/lib/python3.6/site-packages/bert_sklearn/sklearn.py \
    && sed -i \
    's/state = torch.load(filename)/state = torch.load(filename, map_location=torch.device("cpu"))/' \
    /opt/conda/envs/ma_et_al_bert/lib/python3.6/site-packages/bert_sklearn/sklearn.py